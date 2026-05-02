from __future__ import annotations

from pathlib import Path

import pandas as pd

from liqsub.buycurve import derive_monthly_treasury_supply, validate_buycurve_panel


def test_missing_buycurve_panel_is_blocked(tmp_path: Path) -> None:
    report = validate_buycurve_panel(tmp_path / "missing.csv")
    assert report.status == "blocked"
    assert report.errors


def test_buycurve_validation_catches_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "panel.csv"
    pd.DataFrame({"month": ["2024-01-01"]}).to_csv(path, index=False)
    report = validate_buycurve_panel(path)
    assert report.status == "failed"
    assert "missing required columns" in report.errors[0]


def test_buycurve_validation_catches_negative_issuance(tmp_path: Path) -> None:
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        [
            {
                "month": "2024-01-01",
                "security_type": "Bill",
                "maturity_bucket": "0-3m",
                "auction_count": 1,
                "accepted_amount_sum": -1.0,
                "offering_amount_sum": 70.0,
                "weighted_maturity_years": 0.25,
                "bill_share_by_accepted_amount": 0.7,
            }
        ]
    ).to_csv(path, index=False)
    report = validate_buycurve_panel(path)
    assert report.status == "failed"
    assert any("accepted_amount_sum contains negative values" in error for error in report.errors)


def test_buycurve_validation_catches_invalid_bill_share(tmp_path: Path) -> None:
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        [
            {
                "month": "2024-01-01",
                "security_type": "Bill",
                "maturity_bucket": "0-3m",
                "auction_count": 1,
                "accepted_amount_sum": 70.0,
                "offering_amount_sum": 70.0,
                "weighted_maturity_years": 0.25,
                "bill_share_by_accepted_amount": 1.2,
            }
        ]
    ).to_csv(path, index=False)
    report = validate_buycurve_panel(path)
    assert report.status == "failed"
    assert any("bill_share_by_accepted_amount contains values outside [0, 1]" in error for error in report.errors)


def test_derive_monthly_treasury_supply(tmp_path: Path) -> None:
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        [
            {
                "month": "2024-01-01",
                "security_type": "Bill",
                "maturity_bucket": "0-3m",
                "auction_count": 1,
                "accepted_amount_sum": 70.0,
                "offering_amount_sum": 70.0,
                "weighted_maturity_years": 0.25,
                "bill_share_by_accepted_amount": 0.7,
            },
            {
                "month": "2024-01-01",
                "security_type": "Note",
                "maturity_bucket": "2y",
                "auction_count": 1,
                "accepted_amount_sum": 30.0,
                "offering_amount_sum": 30.0,
                "weighted_maturity_years": 2.0,
                "bill_share_by_accepted_amount": 0.7,
            },
        ]
    ).to_csv(path, index=False)
    out = derive_monthly_treasury_supply(path)
    assert out.loc[0, "gross_bill_issuance"] == 0.00007
    assert out.loc[0, "coupon_issuance"] == 0.00003
    assert out.loc[0, "bill_share"] == 0.7
    assert out.loc[0, "weighted_maturity_years"] == 0.775
