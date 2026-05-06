"""In-play short-cycle CLI: collect / train / backtest / deploy.

The user's workflow on the *collector* machine is:

1. Pick a coin that's currently active (e.g. AIOTUSDT mid-pump).
2. ``python -m backend.cli.active collect --symbols AIOTUSDT --duration 4h``
   -> spins up the runtime in collect-only mode, exits cleanly when the
   duration elapses.
3. ``python -m backend.cli.active train --symbols AIOTUSDT --window 4h``
   -> trains a per-symbol LightGBM model on the last 4h of snapshots.
4. ``python -m backend.cli.active backtest --symbols AIOTUSDT --window 1h
   --models-dir ./models``
   -> replays the last 1h through the freshly-trained model with honest
   taker round-trip costs.
5. If backtest's ``avg_net_bp`` is positive on a horizon, the user
   manually rsyncs/scps the model dir to the VPS and starts the bot
   there. ``deploy`` prints the exact paths and a ready-to-paste rsync
   command - actual transport is deliberately left to the user
   ("Деплой мной вручную").

This module is a thin wrapper: it does NOT duplicate the train / backtest
logic, it shells out to the existing modules with the right defaults
for the in-play workflow (per-symbol mode, short window, honest costs).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("backend.cli.active")


def _duration_to_seconds(spec: str) -> int:
    s = spec.strip().lower()
    if not s:
        raise ValueError("duration must not be empty")
    if s.endswith("ms"):
        return max(int(s[:-2]) // 1000, 1)
    if s.endswith("h"):
        return int(float(s[:-1]) * 3600)
    if s.endswith("m"):
        return int(float(s[:-1]) * 60)
    if s.endswith("s"):
        return int(float(s[:-1]))
    raise ValueError(f"duration '{spec}' must end in s/m/h (e.g. 30m, 4h)")


def _ensure_symbols(symbols: list[str] | None) -> list[str]:
    if not symbols:
        raise SystemExit(
            "ERROR: --symbols is required for in-play commands. "
            "Pick the coin you intend to trade right now, e.g. "
            "--symbols AIOTUSDT"
        )
    return [s.strip().upper() for s in symbols if s.strip()]


# ----- collect -------------------------------------------------------


async def _run_collect_until(symbols: list[str], duration_s: int) -> int:
    """Start the runtime in COLLECT mode for ``duration_s`` seconds, then exit.

    We import here so ``--help`` doesn't pull in the websockets / pydantic
    init cost on every invocation.
    """
    from backend.config import Mode, settings  # noqa: PLC0415
    from backend.runtime import Runtime  # noqa: PLC0415

    settings.mode = Mode.COLLECT
    settings.symbols = tuple(symbols)
    rt = Runtime()
    await rt.start()
    for s in symbols:
        await rt.add_symbol(s)

    log.info(
        "active.collect: collecting %s for %ds (until %s)",
        symbols,
        duration_s,
        time.strftime("%H:%M:%S", time.localtime(time.time() + duration_s)),
    )

    stop = asyncio.Event()

    def _on_signal() -> None:
        log.info("active.collect: caught signal, stopping early")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            # Windows event loops don't support add_signal_handler.
            pass

    try:
        await asyncio.wait_for(stop.wait(), timeout=duration_s)
    except TimeoutError:
        pass
    finally:
        await rt.stop()
    log.info("active.collect: done")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    symbols = _ensure_symbols(args.symbols)
    duration_s = _duration_to_seconds(args.duration)
    return asyncio.run(_run_collect_until(symbols, duration_s))


# ----- train ---------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    """Forward to ``backend.ml.train`` with in-play defaults."""
    symbols = _ensure_symbols(args.symbols)
    cmd = [
        sys.executable,
        "-m",
        "backend.ml.train",
        "--symbols",
        *symbols,
        "--mode",
        "per-symbol",
        "--window",
        args.window,
    ]
    if args.horizons:
        cmd += ["--horizons", *args.horizons]
    if args.models_dir:
        cmd += ["--models-dir", str(args.models_dir)]
    if args.no_symbol_feature:
        cmd.append("--no-symbol-feature")
    log.info("active.train: %s", " ".join(cmd))
    return subprocess.call(cmd)


# ----- backtest ------------------------------------------------------


def cmd_backtest(args: argparse.Namespace) -> int:
    symbols = _ensure_symbols(args.symbols)
    models_dir = _resolve_per_symbol_models_dir(args.models_dir, symbols[0])
    cmd = [
        sys.executable,
        "-m",
        "backend.ml.backtest",
        "--symbols",
        *symbols,
        "--window",
        args.window,
        "--models-dir",
        str(models_dir),
    ]
    if args.horizons:
        cmd += ["--horizons", *args.horizons]
    log.info("active.backtest: %s", " ".join(cmd))
    return subprocess.call(cmd)


def _resolve_per_symbol_models_dir(root: Path, symbol: str) -> Path:
    """Auto-resolve ``models/`` -> ``models/per_symbol/{SYM}/`` when present."""
    candidate = root / "per_symbol" / symbol.upper()
    if candidate.exists() and any(candidate.glob("h*")):
        return candidate
    return root


# ----- deploy --------------------------------------------------------


def cmd_deploy(args: argparse.Namespace) -> int:
    """Print rsync-ready paths for manual deploy.

    The user explicitly asked to keep deploy as a manual step ("Деплой мной
    вручную"). This command does NOT execute any transport - it just
    summarises what to copy where.
    """
    symbols = _ensure_symbols(args.symbols)
    sym = symbols[0]
    src = _resolve_per_symbol_models_dir(args.models_dir, sym)
    if not src.exists():
        log.error("active.deploy: %s does not exist - did you train first?", src)
        return 1

    print("\n=========== READY TO DEPLOY ===========")
    print(f"  symbol:        {sym}")
    print(f"  source dir:    {src}")
    print(f"  vps target:    {args.target}:{args.remote_dir}")
    print()
    print("  Suggested rsync:")
    print(f"  rsync -avz --delete '{src}/' '{args.target}:{args.remote_dir.rstrip('/')}/'")
    print()
    print("  Then on the VPS:")
    print(f"    OPUS_MODELS_DIR={args.remote_dir} opus restart")
    print()
    return 0


# ----- main ----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backend.cli.active",
        description="In-play short-cycle workflow: collect -> train -> backtest -> deploy.",
    )
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="collect snapshots for N hours then stop")
    c.add_argument("--symbols", nargs="+", required=True)
    c.add_argument(
        "--duration",
        default="4h",
        help="how long to collect (e.g. 30m, 2h, 4h). Default 4h.",
    )
    c.set_defaults(func=cmd_collect)

    t = sub.add_parser("train", help="train per-symbol model on the last N hours")
    t.add_argument("--symbols", nargs="+", required=True)
    t.add_argument("--window", default="4h", help="data window, e.g. 4h. Default 4h.")
    t.add_argument("--horizons", nargs="+", default=None)
    t.add_argument("--models-dir", type=Path, default=Path("./models"))
    t.add_argument("--no-symbol-feature", action="store_true")
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("backtest", help="replay the last N hours through the trained model")
    b.add_argument("--symbols", nargs="+", required=True)
    b.add_argument("--window", default="1h", help="data window, e.g. 1h. Default 1h.")
    b.add_argument("--horizons", nargs="+", default=None)
    b.add_argument("--models-dir", type=Path, default=Path("./models"))
    b.set_defaults(func=cmd_backtest)

    d = sub.add_parser("deploy", help="print rsync paths for manual VPS deploy")
    d.add_argument("--symbols", nargs="+", required=True)
    d.add_argument("--models-dir", type=Path, default=Path("./models"))
    d.add_argument(
        "--target",
        default=os.environ.get("OPUS_VPS_HOST", "user@vps.example.com"),
        help="ssh target (defaults to OPUS_VPS_HOST env)",
    )
    d.add_argument(
        "--remote-dir",
        default=os.environ.get("OPUS_VPS_MODELS_DIR", "/opt/opus/models"),
        help="remote dir where the model bundle should land",
    )
    d.set_defaults(func=cmd_deploy)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
    )
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
