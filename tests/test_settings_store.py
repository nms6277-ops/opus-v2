from pathlib import Path

import pytest

from backend.settings_store import RuntimeSettings, SettingsHardCaps, SettingsLimitError, SettingsStore


def test_runtime_settings_rejects_values_above_hard_caps():
    with pytest.raises(SettingsLimitError) as exc:
        SettingsStore.validate_hard_caps(
            RuntimeSettings(
                leverage=25,
                max_live_symbols=99,
                daily_loss_limit_usd=10.0,
                loss_12h_limit_usd=10.0,
                symbol_loss_limit_usd=5.0,
                probation_notional_usd=50.0,
                active_notional_usd=500.0,
            ),
            SettingsHardCaps(
                hard_max_leverage=10,
                hard_max_live_symbols=5,
                hard_daily_loss_usd=2.0,
                hard_12h_loss_usd=2.0,
                hard_symbol_loss_usd=0.30,
                hard_notional_usd=20.0,
            ),
        )

    msg = str(exc.value)
    assert "leverage=25 exceeds OPUS_HARD_MAX_LEVERAGE=10" in msg
    assert "active_notional_usd=500.0 exceeds OPUS_HARD_MAX_NOTIONAL_USD=20.0" in msg


def test_settings_store_load_clamps_existing_file_for_backward_compatibility(tmp_path: Path):
    path = tmp_path / "runtime_settings.json"
    path.write_text(
        """
        {
          "leverage": 25,
          "max_live_symbols": 99,
          "daily_loss_limit_usd": 10.0,
          "loss_12h_limit_usd": 10.0,
          "symbol_loss_limit_usd": 5.0,
          "probation_notional_usd": 50.0,
          "active_notional_usd": 500.0
        }
        """,
        encoding="utf-8",
    )
    store = SettingsStore(
        path=path,
        hard_caps=SettingsHardCaps(
            hard_max_leverage=10,
            hard_max_live_symbols=5,
            hard_daily_loss_usd=2.0,
            hard_12h_loss_usd=2.0,
            hard_symbol_loss_usd=0.30,
            hard_notional_usd=20.0,
        ),
    )

    loaded = store.load()

    assert loaded.leverage == 10
    assert loaded.max_live_symbols == 5
    assert loaded.active_notional_usd == 20.0


def test_settings_store_save_rejects_values_above_hard_caps(tmp_path: Path):
    path = tmp_path / "runtime_settings.json"
    store = SettingsStore(
        path=path,
        hard_caps=SettingsHardCaps(
            hard_max_leverage=10,
            hard_max_live_symbols=5,
            hard_daily_loss_usd=2.0,
            hard_12h_loss_usd=2.0,
            hard_symbol_loss_usd=0.30,
            hard_notional_usd=20.0,
        ),
    )

    with pytest.raises(SettingsLimitError):
        store.save(RuntimeSettings(active_notional_usd=50.0))


def test_runtime_settings_can_be_clamped_for_legacy_load():
    runtime_settings = RuntimeSettings(
        leverage=25,
        max_live_symbols=99,
        daily_loss_limit_usd=10.0,
        loss_12h_limit_usd=10.0,
        symbol_loss_limit_usd=5.0,
        probation_notional_usd=50.0,
        active_notional_usd=500.0,
    )
    capped = SettingsStore.clamp_to_hard_caps(
        runtime_settings,
        SettingsHardCaps(
            hard_max_leverage=10,
            hard_max_live_symbols=5,
            hard_daily_loss_usd=2.0,
            hard_12h_loss_usd=2.0,
            hard_symbol_loss_usd=0.30,
            hard_notional_usd=20.0,
        ),
    )
    assert capped.leverage == 10
    assert capped.max_live_symbols == 5
    assert capped.daily_loss_limit_usd == 2.0
    assert capped.loss_12h_limit_usd == 2.0
    assert capped.symbol_loss_limit_usd == 0.30
    assert capped.probation_notional_usd == 20.0
    assert capped.active_notional_usd == 20.0


def test_runtime_settings_defaults_are_vps_safe():
    runtime_settings = RuntimeSettings()
    assert runtime_settings.leverage == 10
    assert runtime_settings.max_live_symbols == 5
    assert runtime_settings.daily_loss_limit_usd == 2.0
    assert runtime_settings.loss_12h_limit_usd == 2.0
    assert runtime_settings.symbol_loss_limit_usd == 0.30
    assert runtime_settings.probation_notional_usd == 6.0
    assert runtime_settings.active_notional_usd == 20.0
    assert runtime_settings.probation_trades == 7
    assert runtime_settings.cooldown_hours == 12


def test_settings_store_persists_and_clamps(tmp_path: Path):
    path = tmp_path / "runtime_settings.json"
    store = SettingsStore(
        path=path,
        hard_caps=SettingsHardCaps(
            hard_max_leverage=10,
            hard_max_live_symbols=5,
            hard_daily_loss_usd=2.0,
            hard_12h_loss_usd=2.0,
            hard_symbol_loss_usd=0.30,
            hard_notional_usd=20.0,
        ),
    )

    saved = store.save(
        RuntimeSettings(
            leverage=10,
            max_live_symbols=5,
            daily_loss_limit_usd=2.0,
            loss_12h_limit_usd=2.0,
            symbol_loss_limit_usd=0.30,
            probation_notional_usd=10.0,
            active_notional_usd=20.0,
        )
    )

    assert saved.leverage == 10
    assert saved.max_live_symbols == 5
    assert saved.daily_loss_limit_usd == 2.0
    assert saved.loss_12h_limit_usd == 2.0
    assert saved.symbol_loss_limit_usd == 0.30
    assert saved.probation_notional_usd == 10.0
    assert saved.active_notional_usd == 20.0
    assert store.load() == saved
    assert store.snapshot()["active_notional_usd"] == 20.0
