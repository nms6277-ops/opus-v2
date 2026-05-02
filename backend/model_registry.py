"""Discovery helpers for locally trained model bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelBundle:
    label: str
    path: Path
    horizons: list[str]

    def to_dict(self, *, active: bool = False) -> dict:
        data = asdict(self)
        data["path"] = str(self.path)
        data["active"] = active
        return data


def _horizon_name(hdir: Path) -> str:
    return hdir.name[1:] if hdir.name.startswith("h") else hdir.name


def _valid_horizon_dirs(bundle_dir: Path) -> list[Path]:
    return sorted(
        hdir
        for hdir in bundle_dir.iterdir()
        if hdir.is_dir()
        and hdir.name.startswith("h")
        and (hdir / "model.lgb").is_file()
        and (hdir / "meta.json").is_file()
    )


def discover_model_bundles(models_root: Path) -> list[ModelBundle]:
    """Return dirs under ``models_root`` that contain h*/model.lgb + meta.json."""

    models_root = Path(models_root)
    if not models_root.is_dir():
        return []

    bundles: list[ModelBundle] = []
    for candidate in sorted(p for p in models_root.rglob("*") if p.is_dir()):
        if candidate.name.startswith("h"):
            continue
        horizon_dirs = _valid_horizon_dirs(candidate)
        if not horizon_dirs:
            continue
        bundles.append(
            ModelBundle(
                label=candidate.relative_to(models_root).as_posix(),
                path=candidate,
                horizons=[_horizon_name(h) for h in horizon_dirs],
            )
        )
    return bundles
