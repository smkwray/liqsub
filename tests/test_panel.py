from __future__ import annotations

import pandas as pd

from liqsub.panel import build_monthly_plumbing_panel, classify_on_rrp_regime


def test_classify_on_rrp_regime_thresholds() -> None:
    out = classify_on_rrp_regime(pd.Series([100_000, 500_000, 1_500_000]))
    assert list(out) == ["scarce", "transition", "abundant"]


def test_tga_uses_dts_with_h41_row_fallback(tmp_path) -> None:
    fred_dir = tmp_path / "fred"
    dts_dir = tmp_path / "fiscaldata"
    fred_dir.mkdir()
    dts_dir.mkdir()
    (fred_dir / "tga_wednesday__WDTGAL.csv").write_text(
        "observation_date,WDTGAL\n2005-09-28,100\n2005-10-26,200\n",
        encoding="utf-8",
    )
    (dts_dir / "dts_operating_cash_balance.csv").write_text(
        "record_date,account_type,close_today_bal,open_today_bal\n"
        "2005-10-26,Treasury General Account (TGA),300,\n",
        encoding="utf-8",
    )
    panel = build_monthly_plumbing_panel(tmp_path)
    values = dict(zip(panel["month"].dt.strftime("%Y-%m"), panel["tga"]))
    assert values["2005-09"] == 100
    assert values["2005-10"] == 300
