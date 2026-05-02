from pathlib import Path

import pytest

from backend.runtime import Runtime
from backend.state import AppState


class FakePredictor:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.enabled = True
        self.horizons = ["1s", "2s"]


class FakeTrader:
    def __init__(self, open_positions=None):
        self.predictor = None
        self._positions = open_positions or {}


@pytest.mark.asyncio
async def test_set_model_dir_reloads_predictor_and_updates_trader(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.runtime.Predictor", FakePredictor)
    monkeypatch.setattr("backend.runtime.settings.data_dir", tmp_path / "data")
    rt = Runtime(AppState())
    rt._trader = FakeTrader()
    model_dir = tmp_path / "models" / "agnostic_next" / "global"
    model_dir.mkdir(parents=True)

    result = await rt.set_model_dir(model_dir)

    assert result["model_dir"] == str(model_dir)
    assert result["horizons"] == ["1s", "2s"]
    assert isinstance(rt._predictor, FakePredictor)
    assert rt._trader.predictor is rt._predictor
    assert (tmp_path / "data" / "runtime_model.json").is_file()


@pytest.mark.asyncio
async def test_set_model_dir_blocks_when_trader_has_open_positions(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.runtime.Predictor", FakePredictor)
    rt = Runtime(AppState())
    rt._trader = FakeTrader(open_positions={"UBUSDT": object()})

    with pytest.raises(ValueError, match="open positions"):
        await rt.set_model_dir(tmp_path / "models" / "agnostic_next" / "global")


def test_runtime_loads_persisted_model_dir(monkeypatch, tmp_path):
    saved = tmp_path / "models" / "agnostic_next" / "global"
    saved.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "runtime_model.json").write_text(
        '{"model_dir": "' + str(saved).replace("\\", "\\\\") + '"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("backend.runtime.settings.data_dir", data_dir)

    rt = Runtime(AppState())

    assert rt.model_dir == saved
