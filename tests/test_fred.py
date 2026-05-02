from __future__ import annotations

import pytest

from liqsub import fred


def _fred_csv(series_id: str, value: str = "1.23") -> str:
    return f"observation_date,{series_id}\n2026-04-29,{value}\n"


def test_download_fred_series_fetches_targeted_series(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_fetch(series_id: str, timeout: int = 45) -> str:
        calls.append(f"{series_id}:{timeout}")
        return _fred_csv(series_id)

    monkeypatch.setattr(fred, "fetch_fred_csv_text", fake_fetch)

    manifest = fred.download_fred_series(
        tmp_path,
        series_keys=["bill_yield_1mo"],
        timeout=7,
        retries=0,
    )

    assert calls == ["DGS1MO:7"]
    row = manifest.loc[manifest["series_key"] == "bill_yield_1mo"].iloc[0]
    assert row["status"] == "downloaded"
    assert int(row["rows"]) == 1
    assert len(manifest) == len(fred.FRED_SERIES)
    path = tmp_path / "fred" / "bill_yield_1mo__DGS1MO.csv"
    assert path.exists()
    parsed = fred.parse_fred_csv(path.read_text(encoding="utf-8"), series_id="DGS1MO")
    assert parsed.loc[0, "value"] == 1.23


def test_download_fred_series_keeps_existing_cache_after_optional_failure(tmp_path, monkeypatch) -> None:
    fred_dir = tmp_path / "fred"
    fred_dir.mkdir()
    cache = fred_dir / "bill_yield_1mo__DGS1MO.csv"
    cache.write_text(_fred_csv("DGS1MO", value="4.56"), encoding="utf-8")

    def fake_fetch(series_id: str, timeout: int = 45) -> str:
        raise RuntimeError("timeout")

    monkeypatch.setattr(fred, "fetch_fred_csv_text", fake_fetch)

    manifest = fred.download_fred_series(tmp_path, series_keys=["bill_yield_1mo"], retries=1)

    row = manifest.loc[manifest["series_key"] == "bill_yield_1mo"].iloc[0]
    assert str(row["status"]).startswith("kept_existing_cache_after_failed_refresh")
    assert int(row["cache_rows"]) == 1
    parsed = fred.parse_fred_csv(cache.read_text(encoding="utf-8"), series_id="DGS1MO")
    assert parsed.loc[0, "value"] == 4.56


def test_download_fred_series_fails_missing_required_series(tmp_path, monkeypatch) -> None:
    def fake_fetch(series_id: str, timeout: int = 45) -> str:
        raise RuntimeError("timeout")

    monkeypatch.setattr(fred, "fetch_fred_csv_text", fake_fetch)

    with pytest.raises(RuntimeError, match="required FRED downloads failed: reserves"):
        fred.download_fred_series(tmp_path, series_keys=["reserves"], retries=0)


def test_download_fred_series_rejects_unknown_series_key(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown FRED series"):
        fred.download_fred_series(tmp_path, series_keys=["not_a_series"])


def test_build_monthly_fred_panel_uses_refreshed_optional_rates(tmp_path, monkeypatch) -> None:
    fred_dir = tmp_path / "fred"
    fred_dir.mkdir()
    (fred_dir / "bill_yield_1mo__DGS1MO.csv").write_text(_fred_csv("DGS1MO", "3.87"), encoding="utf-8")
    (fred_dir / "bill_yield_6mo__DGS6MO.csv").write_text(_fred_csv("DGS6MO", "3.73"), encoding="utf-8")
    (fred_dir / "cash_assets__CASACBW027SBOG.csv").write_text(
        _fred_csv("CASACBW027SBOG", "3000"),
        encoding="utf-8",
    )

    panel = fred.build_monthly_fred_panel(tmp_path)

    assert set(["bill_yield_1mo", "bill_yield_6mo", "cash_assets"]).issubset(panel.columns)
    assert panel.loc[0, "bill_yield_1mo"] == 3.87
    assert panel.loc[0, "bill_yield_6mo"] == 3.73
    assert panel.loc[0, "cash_assets"] == 3_000_000.0
