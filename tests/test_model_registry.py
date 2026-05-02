from pathlib import Path

from backend.model_registry import discover_model_bundles


def _model(root: Path, rel: str, horizons: tuple[str, ...]) -> Path:
    base = root / rel
    for horizon in horizons:
        hdir = base / f"h{horizon}"
        hdir.mkdir(parents=True)
        (hdir / "model.lgb").write_text("model", encoding="utf-8")
        (hdir / "meta.json").write_text("{}", encoding="utf-8")
    return base


def test_discover_model_bundles_finds_valid_horizon_dirs(tmp_path):
    models_root = tmp_path / "models"
    first = _model(models_root, "agnostic/global", ("1s", "2s"))
    second = _model(models_root, "agnostic_next/global", ("1s", "2s", "3s"))
    invalid = models_root / "broken" / "global" / "h1s"
    invalid.mkdir(parents=True)
    (invalid / "model.lgb").write_text("missing meta", encoding="utf-8")

    bundles = discover_model_bundles(models_root)

    assert [b.path for b in bundles] == [first, second]
    assert bundles[0].label == "agnostic/global"
    assert bundles[0].horizons == ["1s", "2s"]
    assert bundles[1].label == "agnostic_next/global"
    assert bundles[1].horizons == ["1s", "2s", "3s"]
