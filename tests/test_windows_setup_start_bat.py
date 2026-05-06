from pathlib import Path


def test_setup_start_windows_bat_is_self_contained():
    script = Path("scripts/setup_start_windows.bat")

    text = script.read_text(encoding="utf-8")

    assert "powershell" not in text.lower()
    assert "-m backend.main" in text
    assert "pip install -e" in text
    assert "--python" in text
    assert "OPUS_BASE_PYTHON" in text
    assert "polars[rtcompat]>=1.34" in text
    assert "--force-reinstall" in text
    assert "--no-cache-dir" in text
    assert "hasattr(pl, 'DataFrame')" in text
    assert "polars-runtime-32" in text
    assert 'if /I "%~1"=="--python"' in text
    assert 'if not exist "%PY%"' in text
    assert '"%BASE_PY%" -m venv .venv' in text
    assert "%RC%" not in text
    assert ".env.example" in text


def test_pyproject_uses_polars_compat_runtime():
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"polars[rtcompat]>=1.34"' in text
    assert '"polars>=1.7"' not in text
