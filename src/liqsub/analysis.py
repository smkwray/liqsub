from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd

from liqsub.config import project_as_of_date
from liqsub.panel import classify_on_rrp_regime
from liqsub.paths import ensure_dir, relative_to_root


ANALYSIS_COLUMNS = [
    "gross_bill_issuance",
    "coupon_issuance",
    "bill_share",
    "weighted_maturity_years",
    "deposits",
    "domestic_deposits",
    "total_mmf_assets",
    "mmf_treasury_holdings",
    "mmf_repo_holdings",
    "mmf_on_rrp_exposure",
    "on_rrp",
    "reserves",
    "tga",
    "iorb_rate",
    "sofr",
    "bill_yield",
]

CHANGE_COLUMNS = [
    "deposits",
    "domestic_deposits",
    "total_mmf_assets",
    "mmf_treasury_holdings",
    "mmf_repo_holdings",
    "mmf_on_rrp_exposure",
    "on_rrp",
    "reserves",
    "tga",
]

OUTCOME_CHANGES = [
    "d_deposits",
    "d_total_mmf_assets",
    "d_mmf_treasury_holdings",
    "d_on_rrp",
    "d_reserves",
    "d_tga",
]

OUTCOME_TO_LEVEL = {
    "d_deposits": "deposits",
    "d_total_mmf_assets": "total_mmf_assets",
    "d_mmf_treasury_holdings": "mmf_treasury_holdings",
    "d_on_rrp": "on_rrp",
    "d_reserves": "reserves",
    "d_tga": "tga",
}

ON_RRP_THRESHOLD_SPECS = {
    "fixed_250b_1000b": (250_000.0, 1_000_000.0),
    "low_100b_750b": (100_000.0, 750_000.0),
    "wide_500b_1500b": (500_000.0, 1_500_000.0),
}

TERMINAL_COMPLETE_MIN_NON_NULL_SHARE = 0.90

INTERNAL_EVIDENCE_GATE_COLUMNS = [
    "design_path",
    "status",
    "claim_use",
    "evidence_basis",
    "primary_artifacts",
    "binding_blockers",
    "next_design_step",
]

INTERNAL_CLAIM_ARTIFACT_COLUMNS = [
    "claim_id",
    "claim_text",
    "claim_strength",
    "artifact_paths",
    "permitted_use",
    "forbidden_upgrade",
]

STRONG_CANDIDATE_NARRATIVE_COLUMNS = [
    "sample",
    "outcome",
    "event_month",
    "priority",
    "direction_pattern",
    "h0_beta",
    "h0_t",
    "h3_beta",
    "h3_t",
    "h6_beta",
    "h6_t",
    "remaining_isolated_bill_shock_events",
    "event_regime",
    "high_rate_regime",
    "gross_bill_issuance",
    "bill_supply_shock_resid_100b",
    "bill_share",
    "bill_yield",
    "tga",
    "reserves",
    "on_rrp",
    "total_mmf_assets",
    "mmf_treasury_holdings",
    "d_mmf_treasury_holdings",
    "pre3_d_mmf_treasury_holdings",
    "post3_d_mmf_treasury_holdings",
    "post6_d_mmf_treasury_holdings",
    "external_calendar_check",
    "narrative_status",
]

EXTERNAL_CALENDAR_EVIDENCE_COLUMNS = [
    "event_month",
    "evidence_status",
    "evidence_topic",
    "official_source",
    "source_url",
    "calendar_note",
    "panel_regime",
    "panel_bill_shock_resid_100b",
    "panel_gross_bill_issuance",
    "panel_bill_share",
    "panel_tga",
    "panel_on_rrp",
    "panel_mmf_treasury_holdings",
    "use_in_narrative",
]

CANDIDATE_TABLE_COLUMNS = [
    "sample",
    "outcome",
    "event_month",
    "candidate_status",
    "priority",
    "direction_pattern",
    "h0_beta",
    "h0_t",
    "h3_beta",
    "h3_t",
    "h6_beta",
    "h6_t",
    "event_regime",
    "high_rate_regime",
    "bill_supply_shock_resid_100b",
    "gross_bill_issuance",
    "bill_share",
    "tga",
    "on_rrp",
    "mmf_treasury_holdings",
    "impact_d_mmf_treasury_holdings",
    "pre3_d_mmf_treasury_holdings",
    "post3_d_mmf_treasury_holdings",
    "post6_d_mmf_treasury_holdings",
    "external_evidence_topics",
    "external_source_count",
    "external_source_urls",
    "required_next_step",
]

COLUMN_METADATA = {
    "gross_bill_issuance": {
        "source_family": "buycurve",
        "unit": "usd_millions",
        "native_frequency": "auction",
        "monthly_aggregation": "sum accepted amount across bills",
    },
    "coupon_issuance": {
        "source_family": "buycurve",
        "unit": "usd_millions",
        "native_frequency": "auction",
        "monthly_aggregation": "sum accepted amount across notes/bonds/tips/frn",
    },
    "bill_share": {
        "source_family": "buycurve",
        "unit": "share",
        "native_frequency": "auction",
        "monthly_aggregation": "bill accepted amount divided by total accepted amount",
    },
    "weighted_maturity_years": {
        "source_family": "buycurve",
        "unit": "years",
        "native_frequency": "auction",
        "monthly_aggregation": "accepted-amount weighted average maturity",
    },
    "tga_wednesday": {
        "source_family": "fred_h41",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "last monthly observation",
    },
    "tga_week_average": {
        "source_family": "fred_h41",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "monthly average of weekly observations",
    },
    "tga_dts": {
        "source_family": "fiscaldata_dts",
        "unit": "usd_millions",
        "native_frequency": "business_day",
        "monthly_aggregation": "last available daily closing balance in calendar month",
    },
    "tga": {
        "source_family": "derived",
        "unit": "usd_millions",
        "native_frequency": "monthly",
        "monthly_aggregation": "DTS TGA when available, otherwise H.4.1 fallback",
    },
    "reserves": {
        "source_family": "fred_h41",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "last monthly observation",
    },
    "fed_treasury_holdings": {
        "source_family": "fred_h41",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "last monthly observation",
    },
    "on_rrp": {
        "source_family": "fred_on_rrp_operations",
        "unit": "usd_millions",
        "native_frequency": "daily",
        "monthly_aggregation": "arithmetic mean over calendar month",
    },
    "deposits": {
        "source_family": "fred_h8",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "last monthly observation",
    },
    "deposits_nsa": {
        "source_family": "fred_h8",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "last monthly observation",
    },
    "domestic_deposits": {
        "source_family": "fred_h8",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "last monthly observation",
    },
    "bank_treasury_agency_securities": {
        "source_family": "fred_h8",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "last monthly observation",
    },
    "total_bank_credit": {
        "source_family": "fred_h8",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "last monthly observation",
    },
    "retail_mmf_assets": {
        "source_family": "fred_mmf",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "last monthly observation",
    },
    "institutional_mmf_assets": {
        "source_family": "fred_mmf",
        "unit": "usd_millions",
        "native_frequency": "weekly",
        "monthly_aggregation": "last monthly observation",
    },
    "iorb_rate": {
        "source_family": "fred_rates",
        "unit": "percent",
        "native_frequency": "daily",
        "monthly_aggregation": "arithmetic mean over calendar month",
    },
    "fed_funds": {
        "source_family": "fred_rates",
        "unit": "percent",
        "native_frequency": "daily",
        "monthly_aggregation": "arithmetic mean over calendar month",
    },
    "sofr": {
        "source_family": "fred_rates",
        "unit": "percent",
        "native_frequency": "daily",
        "monthly_aggregation": "arithmetic mean over calendar month",
    },
    "bill_yield_3mo": {
        "source_family": "fred_rates",
        "unit": "percent",
        "native_frequency": "daily",
        "monthly_aggregation": "arithmetic mean over calendar month",
    },
    "bill_yield": {
        "source_family": "derived",
        "unit": "percent",
        "native_frequency": "monthly",
        "monthly_aggregation": "alias of monthly mean 3-month bill yield",
    },
    "total_mmf_assets": {
        "source_family": "ofr_stfm",
        "unit": "usd_millions",
        "native_frequency": "monthly",
        "monthly_aggregation": "reported monthly value",
    },
    "mmf_treasury_holdings": {
        "source_family": "ofr_stfm",
        "unit": "usd_millions",
        "native_frequency": "monthly",
        "monthly_aggregation": "reported monthly value",
    },
    "mmf_repo_holdings": {
        "source_family": "ofr_stfm",
        "unit": "usd_millions",
        "native_frequency": "monthly",
        "monthly_aggregation": "reported monthly value",
    },
    "mmf_treasury_repo_holdings": {
        "source_family": "ofr_stfm",
        "unit": "usd_millions",
        "native_frequency": "monthly",
        "monthly_aggregation": "reported monthly value",
    },
    "mmf_on_rrp_exposure": {
        "source_family": "ofr_stfm",
        "unit": "usd_millions",
        "native_frequency": "monthly",
        "monthly_aggregation": "reported monthly value",
    },
    "mmf_agency_holdings": {
        "source_family": "ofr_stfm",
        "unit": "usd_millions",
        "native_frequency": "monthly",
        "monthly_aggregation": "reported monthly value",
    },
    "on_rrp_regime": {
        "source_family": "derived",
        "unit": "category",
        "native_frequency": "monthly",
        "monthly_aggregation": "classified from ON RRP levels",
    },
}


@dataclass(frozen=True)
class OLSResult:
    outcome: str
    sample: str
    predictor: str
    nobs: int
    beta: float
    se_hc1: float
    t_hc1: float
    r2: float


def load_monthly_panel(root: Path) -> pd.DataFrame:
    path = root / "data" / "clean" / "monthly_liquidity_substitution_panel.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    panel = pd.read_csv(path)
    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")
    return panel.sort_values("month").reset_index(drop=True)


def _as_of_timestamp(as_of_date: object | None = None, root: Path | None = None) -> pd.Timestamp:
    if as_of_date is None:
        return pd.Timestamp(project_as_of_date(root))
    return pd.Timestamp(as_of_date).normalize()


def _row_non_null_share(frame: pd.DataFrame, *, date_column: str) -> pd.Series:
    value_columns = [column for column in frame.columns if column != date_column]
    if not value_columns:
        return pd.Series(0.0, index=frame.index)
    return frame[value_columns].notna().mean(axis=1)


def _terminal_incomplete_mask(
    frame: pd.DataFrame,
    *,
    date_column: str,
    min_non_null_share: float = TERMINAL_COMPLETE_MIN_NON_NULL_SHARE,
) -> pd.Series:
    if date_column not in frame.columns or frame.empty:
        return pd.Series(False, index=frame.index)
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    shares = _row_non_null_share(frame, date_column=date_column)
    sorted_index = dates.sort_values(ascending=False).index
    mask = pd.Series(False, index=frame.index)
    for idx in sorted_index:
        if pd.isna(dates.loc[idx]):
            continue
        if shares.loc[idx] >= min_non_null_share:
            break
        mask.loc[idx] = True
    return mask


def estimation_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if "baseline_estimation_use" not in panel.columns:
        return panel.copy()
    return panel.loc[panel["baseline_estimation_use"] == "include"].copy()


def add_analysis_fields(panel: pd.DataFrame, *, as_of_date: object | None = None) -> pd.DataFrame:
    out = panel.copy()
    as_of = _as_of_timestamp(as_of_date)
    month_series = pd.to_datetime(out["month"], errors="coerce")
    source_non_null_share = _row_non_null_share(out, date_column="month")
    source_terminal_incomplete = _terminal_incomplete_mask(out, date_column="month")
    if "on_rrp" in out.columns:
        out["on_rrp_regime_fixed"] = classify_on_rrp_regime(out["on_rrp"])
        for spec_name, (scarce_cutoff, abundant_cutoff) in ON_RRP_THRESHOLD_SPECS.items():
            out[f"on_rrp_regime_{spec_name}"] = classify_on_rrp_regime_with_thresholds(
                out["on_rrp"],
                scarce_cutoff=scarce_cutoff,
                abundant_cutoff=abundant_cutoff,
            )
        nonnull = pd.to_numeric(out["on_rrp"], errors="coerce").dropna()
        if len(nonnull) >= 12 and nonnull.nunique() >= 3:
            low, high = nonnull.quantile([1 / 3, 2 / 3])
            out["on_rrp_regime_tercile"] = pd.cut(
                out["on_rrp"],
                bins=[float("-inf"), low, high, float("inf")],
                labels=["low", "middle", "high"],
            ).astype("string")
        else:
            out["on_rrp_regime_tercile"] = pd.Series(pd.NA, index=out.index, dtype="string")

    if "fed_treasury_holdings" in out.columns:
        out["fed_treasury_holdings_12m_change"] = pd.to_numeric(
            out["fed_treasury_holdings"], errors="coerce"
        ).diff(12)
        out["qe_qt_regime"] = np.select(
            [
                out["fed_treasury_holdings_12m_change"] > 50_000,
                out["fed_treasury_holdings_12m_change"] < -50_000,
            ],
            ["qe", "qt"],
            default="flat_or_unknown",
        )

    if "iorb_rate" in out.columns:
        out["high_rate_regime"] = np.where(out["iorb_rate"] >= 2.0, "high_rate", "low_rate")
        out.loc[out["iorb_rate"].isna(), "high_rate_regime"] = pd.NA

    if {"bill_yield", "iorb_rate"}.issubset(out.columns):
        out["bill_iorb_spread"] = out["bill_yield"] - out["iorb_rate"]
    if {"sofr", "iorb_rate"}.issubset(out.columns):
        out["sofr_iorb_spread"] = out["sofr"] - out["iorb_rate"]

    for column in CHANGE_COLUMNS:
        if column in out.columns:
            out[f"d_{column}"] = pd.to_numeric(out[column], errors="coerce").diff()
    if "gross_bill_issuance" in out.columns:
        out["gross_bill_issuance_100b"] = out["gross_bill_issuance"] / 100_000.0
    if "coupon_issuance" in out.columns:
        out["coupon_issuance_100b"] = out["coupon_issuance"] / 100_000.0
    out["month_of_year"] = out["month"].dt.month
    out["time_index"] = np.arange(len(out), dtype=float)
    for month in range(2, 13):
        out[f"month_{month:02d}"] = (out["month_of_year"] == month).astype(float)
    for outcome in OUTCOME_CHANGES:
        if outcome in out.columns:
            out[f"lag1_{outcome}"] = out[outcome].shift(1)
            out[f"pre3_{outcome}"] = out[outcome].shift(1).rolling(3, min_periods=2).sum()
    out["is_future_or_scheduled"] = month_series > as_of
    out["terminal_non_null_share"] = source_non_null_share
    out["is_terminal_incomplete"] = source_terminal_incomplete
    out["terminal_completeness_status"] = "complete_or_historical"
    out.loc[source_terminal_incomplete, "terminal_completeness_status"] = "terminal_incomplete"
    out.loc[out["is_future_or_scheduled"], "terminal_completeness_status"] = "future_or_scheduled"
    out["baseline_estimation_use"] = "include"
    out.loc[
        out["is_future_or_scheduled"] | out["is_terminal_incomplete"],
        "baseline_estimation_use",
    ] = "exclude_from_causal_baseline"
    out = add_bill_supply_shock_fields(out)
    excluded = out["baseline_estimation_use"] != "include"
    shock_columns = [
        "large_gross_bill_shock",
        "large_residual_bill_shock",
        "isolated_bill_shock",
        "bill_supply_shock_resid_100b",
    ]
    for column in shock_columns:
        if column in out.columns:
            out.loc[excluded, column] = pd.NA
    return out


def classify_on_rrp_regime_with_thresholds(
    on_rrp_usd_millions: pd.Series,
    *,
    scarce_cutoff: float,
    abundant_cutoff: float,
) -> pd.Series:
    values = pd.to_numeric(on_rrp_usd_millions, errors="coerce")
    return pd.Series(
        pd.cut(
            values,
            bins=[float("-inf"), scarce_cutoff, abundant_cutoff, float("inf")],
            labels=["scarce", "transition", "abundant"],
        ),
        index=values.index,
        dtype="string",
    )


def add_bill_supply_shock_fields(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "gross_bill_issuance_100b" not in out.columns:
        return out
    out["lag1_gross_bill_issuance_100b"] = out["gross_bill_issuance_100b"].shift(1)
    predictors = ["time_index", "lag1_gross_bill_issuance_100b", *_seasonal_predictors()]
    required = ["gross_bill_issuance_100b", *predictors]
    if not set(required).issubset(out.columns):
        return out
    df = out[required].dropna()
    if len(df) < 36:
        out["bill_supply_shock_resid_100b"] = np.nan
        out["bill_supply_shock_z"] = np.nan
        out["large_positive_bill_shock"] = False
        out["isolated_large_positive_bill_shock"] = False
        return out

    beta, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(df)), df[predictors].to_numpy(dtype=float)]),
        df["gross_bill_issuance_100b"].to_numpy(dtype=float),
        rcond=None,
    )
    fitted = np.column_stack([np.ones(len(df)), df[predictors].to_numpy(dtype=float)]) @ beta
    resid = df["gross_bill_issuance_100b"].to_numpy(dtype=float) - fitted
    out["bill_supply_shock_resid_100b"] = np.nan
    out.loc[df.index, "bill_supply_shock_resid_100b"] = resid
    resid_series = pd.Series(resid, index=df.index)
    resid_std = resid_series.std(ddof=1)
    out["bill_supply_shock_z"] = out["bill_supply_shock_resid_100b"] / resid_std if resid_std else np.nan
    threshold = max(float(resid_series.quantile(0.90)), float(resid_std or 0.0))
    out["bill_supply_shock_threshold_100b"] = threshold
    out["large_positive_bill_shock"] = out["bill_supply_shock_resid_100b"] >= threshold
    out["isolated_large_positive_bill_shock"] = _isolate_events(
        out["large_positive_bill_shock"],
        values=out["bill_supply_shock_resid_100b"],
        min_gap=3,
    )
    return out


def _isolate_events(mask: pd.Series, *, min_gap: int, values: pd.Series | None = None) -> pd.Series:
    isolated = pd.Series(False, index=mask.index)
    event_indices = [int(idx) for idx in mask[mask.fillna(False)].index]
    if not event_indices:
        return isolated
    clusters: list[list[int]] = []
    current = [event_indices[0]]
    for idx in event_indices[1:]:
        if idx - current[-1] <= min_gap:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)
    value_series = values if values is not None else pd.Series(0.0, index=mask.index)
    for cluster in clusters:
        value_slice = pd.to_numeric(value_series.loc[cluster], errors="coerce")
        if value_slice.notna().any():
            selected = int(value_slice.idxmax())
        else:
            selected = cluster[0]
        isolated.loc[selected] = True
    return isolated


def sample_windows(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in ANALYSIS_COLUMNS:
        if column not in panel.columns:
            rows.append({"column": column, "status": "missing", "non_null": 0})
            continue
        mask = panel[column].notna()
        rows.append(
            {
                "column": column,
                "status": "ok" if mask.any() else "empty",
                "non_null": int(mask.sum()),
                "first_month": panel.loc[mask, "month"].min().date().isoformat() if mask.any() else "",
                "last_month": panel.loc[mask, "month"].max().date().isoformat() if mask.any() else "",
            }
        )
    return pd.DataFrame(rows)


def source_metadata_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column, metadata in COLUMN_METADATA.items():
        rows.append(
            {
                "column": column,
                "status": "present" if column in panel.columns else "missing",
                **metadata,
            }
        )
    return pd.DataFrame(rows)


def coverage_qa(panel: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(set(COLUMN_METADATA) | set(ANALYSIS_COLUMNS) | set(panel.columns))
    rows: list[dict[str, object]] = []
    total_months = len(panel)
    for column in columns:
        if column == "month":
            continue
        metadata = COLUMN_METADATA.get(column, {})
        if column not in panel.columns:
            rows.append(
                {
                    "column": column,
                    "status": "missing",
                    "source_family": metadata.get("source_family", ""),
                    "unit": metadata.get("unit", ""),
                    "non_null": 0,
                    "missing": total_months,
                    "coverage_share": 0.0,
                }
            )
            continue
        mask = panel[column].notna()
        first_month = panel.loc[mask, "month"].min() if mask.any() else pd.NaT
        last_month = panel.loc[mask, "month"].max() if mask.any() else pd.NaT
        rows.append(
            {
                "column": column,
                "status": "ok" if mask.any() else "empty",
                "source_family": metadata.get("source_family", "derived_or_unmapped"),
                "unit": metadata.get("unit", ""),
                "native_frequency": metadata.get("native_frequency", ""),
                "monthly_aggregation": metadata.get("monthly_aggregation", ""),
                "non_null": int(mask.sum()),
                "missing": int((~mask).sum()),
                "coverage_share": float(mask.mean()) if total_months else 0.0,
                "first_month": first_month.date().isoformat() if pd.notna(first_month) else "",
                "last_month": last_month.date().isoformat() if pd.notna(last_month) else "",
                "max_internal_gap_months": _max_internal_gap_months(panel.loc[mask, "month"]),
                "leading_missing_months": _leading_missing_months(mask),
                "trailing_missing_months": _trailing_missing_months(mask),
            }
        )
    return pd.DataFrame(rows)


def future_row_qa(panel: pd.DataFrame, *, as_of_date: object | None = None) -> pd.DataFrame:
    if "month" not in panel.columns:
        return pd.DataFrame()
    today = _as_of_timestamp(as_of_date)
    working = panel.copy()
    if "terminal_non_null_share" not in working.columns:
        working["terminal_non_null_share"] = _row_non_null_share(working, date_column="month")
    if "is_terminal_incomplete" not in working.columns:
        working["is_terminal_incomplete"] = _terminal_incomplete_mask(working, date_column="month")
    if "is_future_or_scheduled" not in working.columns:
        working["is_future_or_scheduled"] = pd.to_datetime(working["month"], errors="coerce") > today
    if "baseline_estimation_use" not in working.columns:
        working["baseline_estimation_use"] = "include"
        working.loc[
            working["is_future_or_scheduled"] | working["is_terminal_incomplete"],
            "baseline_estimation_use",
        ] = "exclude_from_causal_baseline"
    rows = working.loc[working["is_future_or_scheduled"] | working["is_terminal_incomplete"]].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "future_or_scheduled",
                "terminal_incomplete",
                "terminal_non_null_share",
                "non_null_columns",
                "baseline_estimation_use",
            ]
        )
    return pd.DataFrame(
        [
            {
                "month": row.month.date().isoformat(),
                "future_or_scheduled": bool(row.is_future_or_scheduled),
                "terminal_incomplete": bool(row.is_terminal_incomplete),
                "terminal_non_null_share": float(row.terminal_non_null_share),
                "non_null_columns": int(
                    pd.Series(row._asdict()).drop(labels=["Index"], errors="ignore").notna().sum()
                ),
                "baseline_estimation_use": str(row.baseline_estimation_use),
            }
            for row in rows.itertuples()
        ]
    )


def regime_classification_qa(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regime_columns = [
        "on_rrp_regime",
        "on_rrp_regime_fixed",
        *[f"on_rrp_regime_{spec_name}" for spec_name in ON_RRP_THRESHOLD_SPECS],
        "on_rrp_regime_tercile",
        "qe_qt_regime",
        "high_rate_regime",
    ]
    total_months = len(panel)
    for regime_col in regime_columns:
        if regime_col not in panel.columns:
            rows.append({"regime_variable": regime_col, "status": "missing", "n_months": 0})
            continue
        values = panel[regime_col]
        for regime, group in panel.dropna(subset=[regime_col]).groupby(regime_col, observed=True):
            event_count = (
                int(group["isolated_large_positive_bill_shock"].fillna(False).sum())
                if "isolated_large_positive_bill_shock" in group.columns
                else 0
            )
            rows.append(
                {
                    "regime_variable": regime_col,
                    "status": "ok",
                    "regime": regime,
                    "n_months": int(len(group)),
                    "share_of_panel": float(len(group) / total_months) if total_months else 0.0,
                    "n_isolated_bill_shock_events": event_count,
                    "first_month": group["month"].min().date().isoformat(),
                    "last_month": group["month"].max().date().isoformat(),
                    "missing_months_for_variable": int(values.isna().sum()),
                }
            )
    return pd.DataFrame(rows)


def _max_internal_gap_months(months: pd.Series) -> int:
    if len(months) < 2:
        return 0
    periods = pd.to_datetime(months).dt.to_period("M").sort_values()
    gaps = [int(right.ordinal - left.ordinal - 1) for left, right in zip(periods.iloc[:-1], periods.iloc[1:])]
    return max(gaps, default=0)


def _leading_missing_months(mask: pd.Series) -> int:
    if not mask.any():
        return int(len(mask))
    return int(mask.idxmax()) if isinstance(mask.index, pd.RangeIndex) else int((~mask.loc[: mask.idxmax()]).sum())


def _trailing_missing_months(mask: pd.Series) -> int:
    if not mask.any():
        return int(len(mask))
    reversed_values = mask.iloc[::-1].reset_index(drop=True)
    return int(reversed_values.idxmax())


def descriptive_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in ANALYSIS_COLUMNS + [f"d_{column}" for column in CHANGE_COLUMNS]:
        if column not in panel.columns:
            continue
        values = pd.to_numeric(panel[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "column": column,
                "nobs": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def regime_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regime_columns = [
        "on_rrp_regime_fixed",
        *[f"on_rrp_regime_{spec_name}" for spec_name in ON_RRP_THRESHOLD_SPECS],
        "on_rrp_regime_tercile",
        "qe_qt_regime",
        "high_rate_regime",
    ]
    for regime_col in regime_columns:
        if regime_col not in panel.columns:
            continue
        for regime, group in panel.dropna(subset=[regime_col]).groupby(regime_col, observed=True):
            row: dict[str, object] = {
                "regime_variable": regime_col,
                "regime": regime,
                "n_months": int(len(group)),
                "first_month": group["month"].min().date().isoformat(),
                "last_month": group["month"].max().date().isoformat(),
            }
            for column in ["gross_bill_issuance", "bill_share", "on_rrp", "deposits", "total_mmf_assets"]:
                if column in group.columns:
                    row[f"mean_{column}"] = float(pd.to_numeric(group[column], errors="coerce").mean())
            rows.append(row)
    return pd.DataFrame(rows)


def regime_threshold_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec_name, (scarce_cutoff, abundant_cutoff) in ON_RRP_THRESHOLD_SPECS.items():
        regime_col = f"on_rrp_regime_{spec_name}"
        if regime_col not in panel.columns:
            continue
        for regime, group in panel.dropna(subset=[regime_col]).groupby(regime_col, observed=True):
            event_count = (
                int(group["isolated_large_positive_bill_shock"].fillna(False).sum())
                if "isolated_large_positive_bill_shock" in group.columns
                else 0
            )
            rows.append(
                {
                    "threshold_spec": spec_name,
                    "scarce_cutoff_usd_millions": scarce_cutoff,
                    "abundant_cutoff_usd_millions": abundant_cutoff,
                    "regime": regime,
                    "n_months": int(len(group)),
                    "n_isolated_bill_shock_events": event_count,
                    "mean_on_rrp": float(pd.to_numeric(group["on_rrp"], errors="coerce").mean()),
                    "mean_bill_share": float(pd.to_numeric(group["bill_share"], errors="coerce").mean()),
                    "mean_gross_bill_issuance": float(
                        pd.to_numeric(group["gross_bill_issuance"], errors="coerce").mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "gross_bill_issuance",
        "bill_supply_shock_resid_100b",
        "bill_share",
        "d_deposits",
        "d_total_mmf_assets",
        "d_mmf_treasury_holdings",
        "d_on_rrp",
        "d_reserves",
        "d_tga",
        "bill_iorb_spread",
    ]
    available = [column for column in columns if column in panel.columns]
    corr = panel[available].corr(min_periods=24)
    rows = []
    for left in available:
        for right in available:
            if left >= right:
                continue
            value = corr.loc[left, right]
            if pd.notna(value):
                rows.append({"left": left, "right": right, "correlation": float(value)})
    return pd.DataFrame(rows)


def bill_shock_events(panel: pd.DataFrame) -> pd.DataFrame:
    if "isolated_large_positive_bill_shock" not in panel.columns:
        return pd.DataFrame()
    events = panel.loc[panel["isolated_large_positive_bill_shock"].fillna(False)].copy()
    columns = [
        "month",
        "gross_bill_issuance",
        "bill_supply_shock_resid_100b",
        "bill_supply_shock_z",
        "bill_share",
        "on_rrp",
        "on_rrp_regime_fixed",
        "on_rrp_regime_tercile",
        "high_rate_regime",
    ]
    available = [column for column in columns if column in events.columns]
    return events[available].reset_index(drop=True)


def event_study_by_event(panel: pd.DataFrame) -> pd.DataFrame:
    if "isolated_large_positive_bill_shock" not in panel.columns:
        return pd.DataFrame()
    events = panel.index[panel["isolated_large_positive_bill_shock"].fillna(False)].tolist()
    rows: list[dict[str, object]] = []
    for event_idx in events:
        event = panel.loc[event_idx]
        for outcome in OUTCOME_CHANGES:
            if outcome not in panel.columns:
                continue
            for label, start, end in [
                ("pre3", event_idx - 3, event_idx - 1),
                ("impact", event_idx, event_idx),
                ("post3", event_idx + 1, event_idx + 3),
                ("post6", event_idx + 1, event_idx + 6),
            ]:
                if start < 0 or end >= len(panel):
                    continue
                window = panel.loc[start:end, outcome].dropna()
                if window.empty:
                    continue
                rows.append(
                    {
                        "event_month": event["month"].date().isoformat(),
                        "event_regime": event.get("on_rrp_regime_fixed", pd.NA),
                        "event_high_rate_regime": event.get("high_rate_regime", pd.NA),
                        "event_shock_resid_100b": event.get("bill_supply_shock_resid_100b", np.nan),
                        "outcome": outcome,
                        "window": label,
                        "n_obs_in_window": int(len(window)),
                        "sum_change": float(window.sum()),
                        "mean_change": float(window.mean()),
                    }
                )
    return pd.DataFrame(rows)


def event_study_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    grouped = event_rows.groupby(["outcome", "window"], observed=True)
    return (
        grouped.agg(
            n_events=("event_month", "nunique"),
            mean_sum_change=("sum_change", "mean"),
            median_sum_change=("sum_change", "median"),
            mean_change=("mean_change", "mean"),
        )
        .reset_index()
        .sort_values(["outcome", "window"])
    )


def local_projection_diagnostics(panel: pd.DataFrame, *, min_nobs: int = 48) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    horizons = [0, 3, 6]
    specs = {
        "shock_only": ["bill_supply_shock_resid_100b"],
        "shock_plus_bill_share": ["bill_supply_shock_resid_100b", "bill_share"],
        "shock_plus_pretrend": None,
        "shock_plus_rate": ["bill_supply_shock_resid_100b", "bill_share", "bill_yield"],
    }
    for outcome in OUTCOME_CHANGES:
        if outcome not in panel.columns:
            continue
        for horizon in horizons:
            target = _future_cumulative_change(panel[outcome], horizon)
            target_col = f"f{horizon}_{outcome}"
            work = panel.copy()
            work[target_col] = target
            for spec_name, base_predictors in specs.items():
                predictors = (
                    ["bill_supply_shock_resid_100b", "bill_share", f"pre3_{outcome}"]
                    if spec_name == "shock_plus_pretrend"
                    else list(base_predictors or [])
                )
                required = [target_col, *predictors]
                if not set(required).issubset(work.columns):
                    continue
                df = work[required].dropna()
                if len(df) < min_nobs:
                    rows.append(
                        {
                            "outcome": outcome,
                            "horizon_months": horizon,
                            "spec": spec_name,
                            "status": "skipped_too_few_observations",
                            "nobs": int(len(df)),
                            "min_nobs": min_nobs,
                        }
                    )
                    continue
                beta, se, r2 = _ols_hc1(df, target_col, predictors)
                for idx, predictor in enumerate(["intercept", *predictors]):
                    rows.append(
                        {
                            "outcome": outcome,
                            "horizon_months": horizon,
                            "spec": spec_name,
                            "status": "estimated",
                            "predictor": predictor,
                            "nobs": int(len(df)),
                            "beta": float(beta[idx]),
                            "se_hc1": float(se[idx]),
                            "t_hc1": float(beta[idx] / se[idx]) if se[idx] else np.nan,
                            "r2": r2,
                            "interpretation": "descriptive local-projection diagnostic; not causal",
                        }
                    )
    return pd.DataFrame(rows)


def regime_interaction_diagnostics(panel: pd.DataFrame, *, min_nobs: int = 48) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_regime_col = "on_rrp_regime_fixed"
    if base_regime_col not in panel.columns:
        return pd.DataFrame()
    work = panel.copy()
    work["rrp_abundant_dummy"] = (work[base_regime_col] == "abundant").astype(float)
    work["rrp_transition_dummy"] = (work[base_regime_col] == "transition").astype(float)
    work["shock_x_abundant"] = work["bill_supply_shock_resid_100b"] * work["rrp_abundant_dummy"]
    work["shock_x_transition"] = work["bill_supply_shock_resid_100b"] * work["rrp_transition_dummy"]
    predictors = [
        "bill_supply_shock_resid_100b",
        "rrp_abundant_dummy",
        "rrp_transition_dummy",
        "shock_x_abundant",
        "shock_x_transition",
        "bill_share",
    ]
    for outcome in OUTCOME_CHANGES:
        if outcome not in work.columns:
            continue
        required = [outcome, *predictors]
        df = work[required].dropna()
        if len(df) < min_nobs:
            rows.append(
                {
                    "outcome": outcome,
                    "status": "skipped_too_few_observations",
                    "nobs": int(len(df)),
                    "min_nobs": min_nobs,
                }
            )
            continue
        beta, se, r2 = _ols_hc1(df, outcome, predictors)
        for idx, predictor in enumerate(["intercept", *predictors]):
            rows.append(
                {
                    "outcome": outcome,
                    "status": "estimated",
                    "predictor": predictor,
                    "nobs": int(len(df)),
                    "beta": float(beta[idx]),
                    "se_hc1": float(se[idx]),
                    "t_hc1": float(beta[idx] / se[idx]) if se[idx] else np.nan,
                    "r2": r2,
                    "interpretation": "descriptive regime-interaction diagnostic; not causal",
                }
            )
    return pd.DataFrame(rows)


def _future_cumulative_change(series: pd.Series, horizon: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if horizon == 0:
        return values
    return values.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))


def sample_recommendations(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_name, columns in _sample_candidate_columns().items():
        mask = _sample_mask(panel, columns)
        available = [column for column in columns if column in panel.columns]
        rows.append(
            {
                "sample": sample_name,
                "required_columns": ",".join(columns),
                "available_required_columns": ",".join(available),
                "n_months": int(mask.sum()),
                "first_month": panel.loc[mask, "month"].min().date().isoformat() if mask.any() else "",
                "last_month": panel.loc[mask, "month"].max().date().isoformat() if mask.any() else "",
                "recommended_use": _sample_use(sample_name, int(mask.sum())),
            }
        )
    return pd.DataFrame(rows)


def design_readiness_table(panel: pd.DataFrame, *, min_nobs: int = 48) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    samples = _sample_candidate_columns()
    for sample_name, sample_columns in samples.items():
        sample_mask = _sample_mask(panel, sample_columns)
        sample = panel.loc[sample_mask].copy()
        for outcome in OUTCOME_CHANGES:
            required = [outcome, "bill_supply_shock_resid_100b", "bill_share"]
            if not set(required).issubset(sample.columns):
                rows.append(
                    {
                        "sample": sample_name,
                        "outcome": outcome,
                        "status": "missing_required_columns",
                        "recommended_next_step": "add_or_repair_required_source_columns",
                    }
                )
                continue
            df = sample[required].dropna()
            nobs = int(len(df))
            event_count = _sample_event_count(sample)
            residual_pretrend = _max_abs_residual_pretrend_t(sample, outcome)
            lp_estimable = _lp_estimable(sample, outcome, min_nobs=min_nobs)
            level_column = OUTCOME_TO_LEVEL.get(outcome, "")
            coverage_share = float(sample[level_column].notna().mean()) if level_column in sample.columns and len(sample) else 0.0
            status, next_step = _design_status(
                nobs=nobs,
                event_count=event_count,
                residual_pretrend_t=residual_pretrend,
                lp_estimable=lp_estimable,
                sample_name=sample_name,
            )
            rows.append(
                {
                    "sample": sample_name,
                    "outcome": outcome,
                    "status": status,
                    "nobs": nobs,
                    "sample_months": int(sample_mask.sum()),
                    "outcome_level_column": level_column,
                    "outcome_coverage_share_within_sample": coverage_share,
                    "isolated_bill_shock_events": event_count,
                    "max_abs_residual_pretrend_t": residual_pretrend,
                    "lp_h0_h3_h6_estimable": lp_estimable,
                    "recommended_next_step": next_step,
                }
            )
    return pd.DataFrame(rows)


def _sample_candidate_columns() -> dict[str, list[str]]:
    return {
        "full_treasury_deposits": [
            "gross_bill_issuance",
            "bill_share",
            "deposits",
            "domestic_deposits",
            "bill_yield",
        ],
        "post_2005_tga": ["gross_bill_issuance", "bill_share", "deposits", "tga", "bill_yield"],
        "post_2010_mmf_holdings": [
            "gross_bill_issuance",
            "bill_share",
            "deposits",
            "total_mmf_assets",
            "mmf_treasury_holdings",
            "on_rrp",
            "reserves",
            "tga",
        ],
        "post_2021_rate_controls": [
            "gross_bill_issuance",
            "bill_share",
            "deposits",
            "total_mmf_assets",
            "mmf_treasury_holdings",
            "on_rrp",
            "iorb_rate",
            "sofr",
        ],
    }


def _sample_mask(panel: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in panel.columns]
    if not available or len(available) != len(columns):
        return pd.Series(False, index=panel.index)
    return panel[available].notna().all(axis=1)


def _sample_event_count(sample: pd.DataFrame) -> int:
    if "isolated_large_positive_bill_shock" not in sample.columns:
        return 0
    return int(sample["isolated_large_positive_bill_shock"].fillna(False).sum())


def _max_abs_residual_pretrend_t(sample: pd.DataFrame, outcome: str) -> float:
    diagnostics = residual_shock_pretrend_diagnostics(sample, min_nobs=24)
    if diagnostics.empty or "shock_t_hc1" not in diagnostics.columns:
        return np.nan
    rows = diagnostics.loc[diagnostics["outcome"] == outcome, "shock_t_hc1"].dropna()
    return float(rows.abs().max()) if not rows.empty else np.nan


def _lp_estimable(sample: pd.DataFrame, outcome: str, *, min_nobs: int) -> bool:
    diagnostics = local_projection_diagnostics(sample, min_nobs=min_nobs)
    if diagnostics.empty or "predictor" not in diagnostics.columns:
        return False
    rows = diagnostics.loc[
        (diagnostics["outcome"] == outcome)
        & (diagnostics["spec"] == "shock_plus_pretrend")
        & (diagnostics["predictor"] == "bill_supply_shock_resid_100b")
        & (diagnostics["status"] == "estimated")
    ]
    return set(rows["horizon_months"]) >= {0, 3, 6}


def _design_status(
    *,
    nobs: int,
    event_count: int,
    residual_pretrend_t: float,
    lp_estimable: bool,
    sample_name: str,
) -> tuple[str, str]:
    if nobs < 48 or event_count < 5:
        return "too_sparse", "use_broader_sample_or_lower_frequency_design"
    if sample_name == "post_2021_rate_controls":
        return "diagnostic_only", "treat_as_rate_control_sensitivity_not_main_design"
    if pd.notna(residual_pretrend_t) and residual_pretrend_t >= 2.0:
        return "diagnostic_only", "add_anticipation_controls_or_find_cleaner_shock"
    if not lp_estimable:
        return "diagnostic_only", "insufficient_lp_support_across_default_horizons"
    return "candidate_design", "review_coefficients_and_event_narrative_before_public_use"


def candidate_lp_table(panel: pd.DataFrame) -> pd.DataFrame:
    readiness = design_readiness_table(panel)
    rows: list[dict[str, object]] = []
    for candidate in readiness.loc[readiness["status"] == "candidate_design"].itertuples(index=False):
        sample_columns = _sample_candidate_columns()[candidate.sample]
        sample = panel.loc[_sample_mask(panel, sample_columns)].copy()
        lp = local_projection_diagnostics(sample)
        if lp.empty or "predictor" not in lp.columns:
            continue
        shock_rows = lp.loc[
            (lp["outcome"] == candidate.outcome)
            & (lp["spec"] == "shock_plus_pretrend")
            & (lp["predictor"] == "bill_supply_shock_resid_100b")
            & (lp["status"] == "estimated")
        ]
        for lp_row in shock_rows.itertuples(index=False):
            rows.append(
                {
                    "sample": candidate.sample,
                    "outcome": candidate.outcome,
                    "horizon_months": int(lp_row.horizon_months),
                    "nobs": int(lp_row.nobs),
                    "isolated_bill_shock_events": int(candidate.isolated_bill_shock_events),
                    "max_abs_residual_pretrend_t": float(candidate.max_abs_residual_pretrend_t),
                    "shock_beta": float(lp_row.beta),
                    "shock_se_hc1": float(lp_row.se_hc1),
                    "shock_t_hc1": float(lp_row.t_hc1),
                    "r2": float(lp_row.r2),
                    "interpretation": "candidate descriptive LP row; still requires narrative/event review",
                }
            )
    return pd.DataFrame(rows)


def candidate_event_review_table(panel: pd.DataFrame) -> pd.DataFrame:
    readiness = design_readiness_table(panel)
    rows: list[dict[str, object]] = []
    for candidate in readiness.loc[readiness["status"] == "candidate_design"].itertuples(index=False):
        sample_columns = _sample_candidate_columns()[candidate.sample]
        sample = panel.loc[_sample_mask(panel, sample_columns)].reset_index(drop=True)
        event_rows = event_study_by_event(sample)
        if event_rows.empty:
            rows.append(
                {
                    "sample": candidate.sample,
                    "outcome": candidate.outcome,
                    "status": "no_event_windows",
                    "recommended_next_step": "inspect_event_definition_or_broaden_sample",
                }
            )
            continue
        outcome_events = event_rows.loc[event_rows["outcome"] == candidate.outcome]
        if outcome_events.empty:
            rows.append(
                {
                    "sample": candidate.sample,
                    "outcome": candidate.outcome,
                    "status": "no_outcome_event_windows",
                    "recommended_next_step": "inspect_outcome_coverage_around_events",
                }
            )
            continue
        window_summary = _candidate_window_summary(outcome_events)
        pre3 = window_summary.get("pre3", {})
        impact = window_summary.get("impact", {})
        post3 = window_summary.get("post3", {})
        post6 = window_summary.get("post6", {})
        pre_abs = abs(float(pre3.get("mean_sum_change", np.nan)))
        impact_abs = abs(float(impact.get("mean_sum_change", np.nan)))
        post3_abs = abs(float(post3.get("mean_sum_change", np.nan)))
        reference_values = [value for value in [impact_abs, post3_abs] if pd.notna(value)]
        reference_abs = max(reference_values) if reference_values else np.nan
        pre_to_reference = pre_abs / reference_abs if pd.notna(reference_abs) and reference_abs else np.nan
        same_sign_pre_impact = _same_sign(pre3.get("mean_sum_change"), impact.get("mean_sum_change"))
        status, next_step = _event_review_status(
            pre_to_reference=pre_to_reference,
            same_sign_pre_impact=same_sign_pre_impact,
            n_events=int(max(summary.get("n_events", 0) for summary in window_summary.values())),
        )
        rows.append(
            {
                "sample": candidate.sample,
                "outcome": candidate.outcome,
                "status": status,
                "n_events": int(max(summary.get("n_events", 0) for summary in window_summary.values())),
                "pre3_n_events": int(pre3.get("n_events", 0)),
                "impact_n_events": int(impact.get("n_events", 0)),
                "post3_n_events": int(post3.get("n_events", 0)),
                "post6_n_events": int(post6.get("n_events", 0)),
                "pre3_mean_sum_change": pre3.get("mean_sum_change", np.nan),
                "impact_mean_sum_change": impact.get("mean_sum_change", np.nan),
                "post3_mean_sum_change": post3.get("mean_sum_change", np.nan),
                "post6_mean_sum_change": post6.get("mean_sum_change", np.nan),
                "pre3_to_max_impact_or_post3_abs_ratio": pre_to_reference,
                "same_sign_pre3_and_impact": same_sign_pre_impact,
                "largest_abs_impact_event_months": _largest_abs_impact_event_months(outcome_events),
                "recommended_next_step": next_step,
            }
        )
    return pd.DataFrame(rows)


def candidate_event_month_table(panel: pd.DataFrame) -> pd.DataFrame:
    readiness = design_readiness_table(panel)
    rows: list[dict[str, object]] = []
    for candidate in readiness.loc[readiness["status"] == "candidate_design"].itertuples(index=False):
        sample_columns = _sample_candidate_columns()[candidate.sample]
        sample = panel.loc[_sample_mask(panel, sample_columns)].reset_index(drop=True)
        event_rows = event_study_by_event(sample)
        if event_rows.empty:
            continue
        outcome_events = event_rows.loc[event_rows["outcome"] == candidate.outcome]
        if outcome_events.empty:
            continue
        for event_month, group in outcome_events.groupby("event_month", observed=True):
            windows = {
                str(row.window): row
                for row in group.itertuples(index=False)
            }
            pre3 = windows.get("pre3")
            impact = windows.get("impact")
            post3 = windows.get("post3")
            post6 = windows.get("post6")
            impact_sum = float(impact.sum_change) if impact is not None else np.nan
            pre3_sum = float(pre3.sum_change) if pre3 is not None else np.nan
            post3_sum = float(post3.sum_change) if post3 is not None else np.nan
            post6_sum = float(post6.sum_change) if post6 is not None else np.nan
            reference_values = [abs(value) for value in [impact_sum, post3_sum] if pd.notna(value)]
            reference_abs = max(reference_values) if reference_values else np.nan
            pre_to_reference = abs(pre3_sum) / reference_abs if pd.notna(reference_abs) and reference_abs else np.nan
            rows.append(
                {
                    "sample": candidate.sample,
                    "outcome": candidate.outcome,
                    "event_month": event_month,
                    "event_regime": _first_non_null(group, "event_regime"),
                    "event_high_rate_regime": _first_non_null(group, "event_high_rate_regime"),
                    "event_shock_resid_100b": _first_non_null(group, "event_shock_resid_100b"),
                    "pre3_sum_change": pre3_sum,
                    "impact_sum_change": impact_sum,
                    "post3_sum_change": post3_sum,
                    "post6_sum_change": post6_sum,
                    "pre3_to_max_impact_or_post3_abs_ratio": pre_to_reference,
                    "same_sign_pre3_and_impact": _same_sign(pre3_sum, impact_sum),
                    "abs_impact_sum_change": abs(impact_sum) if pd.notna(impact_sum) else np.nan,
                    "manual_review_priority": _event_month_priority(pre_to_reference, impact_sum),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["manual_review_priority", "abs_impact_sum_change", "sample", "outcome", "event_month"],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)


def filtered_candidate_lp_table(panel: pd.DataFrame) -> pd.DataFrame:
    readiness = design_readiness_table(panel)
    dirty_events = candidate_event_month_table(panel)
    rows: list[dict[str, object]] = []
    for candidate in readiness.loc[readiness["status"] == "candidate_design"].itertuples(index=False):
        sample, excluded_months = _filtered_candidate_sample(panel, candidate, dirty_events)
        lp = local_projection_diagnostics(sample)
        if lp.empty or "predictor" not in lp.columns:
            rows.append(
                {
                    "sample": candidate.sample,
                    "outcome": candidate.outcome,
                    "status": "skipped_no_filtered_lp",
                    "excluded_event_months": len(excluded_months),
                    "remaining_isolated_bill_shock_events": _sample_event_count(sample),
                }
            )
            continue
        shock_rows = lp.loc[
            (lp["outcome"] == candidate.outcome)
            & (lp["spec"] == "shock_plus_pretrend")
            & (lp["predictor"] == "bill_supply_shock_resid_100b")
            & (lp["status"] == "estimated")
        ]
        if shock_rows.empty:
            rows.append(
                {
                    "sample": candidate.sample,
                    "outcome": candidate.outcome,
                    "status": "skipped_too_few_filtered_observations",
                    "excluded_event_months": len(excluded_months),
                    "remaining_isolated_bill_shock_events": _sample_event_count(sample),
                }
            )
            continue
        for lp_row in shock_rows.itertuples(index=False):
            rows.append(
                {
                    "sample": candidate.sample,
                    "outcome": candidate.outcome,
                    "status": "estimated",
                    "horizon_months": int(lp_row.horizon_months),
                    "nobs": int(lp_row.nobs),
                    "excluded_event_months": len(excluded_months),
                    "remaining_isolated_bill_shock_events": _sample_event_count(sample),
                    "shock_beta": float(lp_row.beta),
                    "shock_se_hc1": float(lp_row.se_hc1),
                    "shock_t_hc1": float(lp_row.t_hc1),
                    "r2": float(lp_row.r2),
                    "interpretation": "filtered descriptive LP; high-pre-movement candidate events excluded",
                }
            )
    return pd.DataFrame(rows)


def filtered_candidate_event_review_table(panel: pd.DataFrame) -> pd.DataFrame:
    readiness = design_readiness_table(panel)
    dirty_events = candidate_event_month_table(panel)
    rows: list[dict[str, object]] = []
    for candidate in readiness.loc[readiness["status"] == "candidate_design"].itertuples(index=False):
        sample, excluded_months = _filtered_candidate_sample(panel, candidate, dirty_events)
        event_rows = event_study_by_event(sample.reset_index(drop=True))
        if event_rows.empty:
            rows.append(
                {
                    "sample": candidate.sample,
                    "outcome": candidate.outcome,
                    "status": "no_filtered_event_windows",
                    "excluded_event_months": len(excluded_months),
                    "remaining_isolated_bill_shock_events": _sample_event_count(sample),
                    "recommended_next_step": "filtered_events_are_too_sparse",
                }
            )
            continue
        outcome_events = event_rows.loc[event_rows["outcome"] == candidate.outcome]
        if outcome_events.empty:
            rows.append(
                {
                    "sample": candidate.sample,
                    "outcome": candidate.outcome,
                    "status": "no_filtered_outcome_event_windows",
                    "excluded_event_months": len(excluded_months),
                    "remaining_isolated_bill_shock_events": _sample_event_count(sample),
                    "recommended_next_step": "inspect_filtered_outcome_coverage_around_events",
                }
            )
            continue
        summary = _candidate_window_summary(outcome_events)
        pre3 = summary.get("pre3", {})
        impact = summary.get("impact", {})
        post3 = summary.get("post3", {})
        post6 = summary.get("post6", {})
        pre_abs = abs(float(pre3.get("mean_sum_change", np.nan)))
        impact_abs = abs(float(impact.get("mean_sum_change", np.nan)))
        post3_abs = abs(float(post3.get("mean_sum_change", np.nan)))
        reference_values = [value for value in [impact_abs, post3_abs] if pd.notna(value)]
        reference_abs = max(reference_values) if reference_values else np.nan
        pre_to_reference = pre_abs / reference_abs if pd.notna(reference_abs) and reference_abs else np.nan
        same_sign_pre_impact = _same_sign(pre3.get("mean_sum_change"), impact.get("mean_sum_change"))
        status, next_step = _event_review_status(
            pre_to_reference=pre_to_reference,
            same_sign_pre_impact=same_sign_pre_impact,
            n_events=int(max(window.get("n_events", 0) for window in summary.values())),
        )
        rows.append(
            {
                "sample": candidate.sample,
                "outcome": candidate.outcome,
                "status": status,
                "excluded_event_months": len(excluded_months),
                "remaining_isolated_bill_shock_events": _sample_event_count(sample),
                "n_events": int(max(window.get("n_events", 0) for window in summary.values())),
                "pre3_n_events": int(pre3.get("n_events", 0)),
                "impact_n_events": int(impact.get("n_events", 0)),
                "post3_n_events": int(post3.get("n_events", 0)),
                "post6_n_events": int(post6.get("n_events", 0)),
                "pre3_mean_sum_change": pre3.get("mean_sum_change", np.nan),
                "impact_mean_sum_change": impact.get("mean_sum_change", np.nan),
                "post3_mean_sum_change": post3.get("mean_sum_change", np.nan),
                "post6_mean_sum_change": post6.get("mean_sum_change", np.nan),
                "pre3_to_max_impact_or_post3_abs_ratio": pre_to_reference,
                "same_sign_pre3_and_impact": same_sign_pre_impact,
                "largest_abs_impact_event_months": _largest_abs_impact_event_months(outcome_events),
                "recommended_next_step": next_step,
            }
        )
    return pd.DataFrame(rows)


def filtered_candidate_shortlist_table(panel: pd.DataFrame) -> pd.DataFrame:
    events = filtered_candidate_event_review_table(panel)
    lps = filtered_candidate_lp_table(panel)
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for event_row in events.itertuples(index=False):
        sample_lps = lps.loc[
            (lps["sample"] == event_row.sample)
            & (lps["outcome"] == event_row.outcome)
            & (lps["status"] == "estimated")
        ] if not lps.empty and "status" in lps.columns else pd.DataFrame()
        lp_by_horizon = {
            int(row.horizon_months): row
            for row in sample_lps.itertuples(index=False)
            if pd.notna(row.horizon_months)
        }
        h0 = lp_by_horizon.get(0)
        h3 = lp_by_horizon.get(3)
        h6 = lp_by_horizon.get(6)
        priority = _filtered_shortlist_priority(event_row, h0, h3, h6)
        rows.append(
            {
                "sample": event_row.sample,
                "outcome": event_row.outcome,
                "priority": priority,
                "event_status": event_row.status,
                "excluded_event_months": event_row.excluded_event_months,
                "remaining_isolated_bill_shock_events": event_row.remaining_isolated_bill_shock_events,
                "pre3_to_max_impact_or_post3_abs_ratio": event_row.pre3_to_max_impact_or_post3_abs_ratio,
                "largest_abs_impact_event_months": event_row.largest_abs_impact_event_months,
                "h0_beta": _row_attr(h0, "shock_beta"),
                "h0_t": _row_attr(h0, "shock_t_hc1"),
                "h3_beta": _row_attr(h3, "shock_beta"),
                "h3_t": _row_attr(h3, "shock_t_hc1"),
                "h6_beta": _row_attr(h6, "shock_beta"),
                "h6_t": _row_attr(h6, "shock_t_hc1"),
                "direction_pattern": _direction_pattern(h0, h3, h6),
                "recommended_next_step": _filtered_shortlist_next_step(priority),
            }
        )
    out = pd.DataFrame(rows)
    priority_order = {
        "strong_manual_review": 0,
        "manual_review": 1,
        "weak_or_mixed": 2,
        "still_compromised": 3,
    }
    out["priority_rank"] = out["priority"].map(priority_order).fillna(99)
    return out.sort_values(
        ["priority_rank", "sample", "outcome"],
    ).drop(columns=["priority_rank"]).reset_index(drop=True)


def on_rrp_manual_review_table(panel: pd.DataFrame) -> pd.DataFrame:
    shortlist = filtered_candidate_shortlist_table(panel)
    if shortlist.empty:
        return pd.DataFrame()
    manual = shortlist.loc[
        (shortlist["outcome"] == "d_on_rrp") & (shortlist["priority"] == "manual_review")
    ]
    if manual.empty:
        return pd.DataFrame()

    event_reviews = filtered_candidate_event_review_table(panel)
    event_months = candidate_event_month_table(panel)
    rows: list[dict[str, object]] = []
    for candidate in manual.itertuples(index=False):
        review = _single_row(event_reviews, sample=candidate.sample, outcome=candidate.outcome)
        month_rows = event_months.loc[
            (event_months["sample"] == candidate.sample)
            & (event_months["outcome"] == candidate.outcome)
            & (event_months["event_month"].isin(_split_event_months(candidate.largest_abs_impact_event_months)))
        ] if not event_months.empty else pd.DataFrame()
        max_abs_t = max(
            [
                abs(value)
                for value in [candidate.h0_t, candidate.h3_t, candidate.h6_t]
                if pd.notna(value)
            ],
            default=np.nan,
        )
        impact_mean = review.get("impact_mean_sum_change", np.nan)
        post3_mean = review.get("post3_mean_sum_change", np.nan)
        post6_mean = review.get("post6_mean_sum_change", np.nan)
        flags = ["lp_t_below_2", "no_strong_candidate_narrative"]
        if _opposite_sign(impact_mean, post3_mean) or _opposite_sign(impact_mean, post6_mean):
            flags.append("event_window_reversal")
        if int(review.get("remaining_isolated_bill_shock_events", 0) or 0) < 8:
            flags.append("sparse_filtered_events")
        rows.append(
            {
                "sample": candidate.sample,
                "outcome": candidate.outcome,
                "decision": "do_not_promote_descriptive_context_only",
                "event_months": candidate.largest_abs_impact_event_months,
                "event_regimes": _join_unique(month_rows.get("event_regime", pd.Series(dtype=object))),
                "remaining_isolated_bill_shock_events": review.get(
                    "remaining_isolated_bill_shock_events", np.nan
                ),
                "pre3_to_reference_ratio": review.get(
                    "pre3_to_max_impact_or_post3_abs_ratio", np.nan
                ),
                "impact_mean_sum_change": impact_mean,
                "post3_mean_sum_change": post3_mean,
                "post6_mean_sum_change": post6_mean,
                "h0_t": candidate.h0_t,
                "h3_t": candidate.h3_t,
                "h6_t": candidate.h6_t,
                "max_abs_lp_t": max_abs_t,
                "direction_pattern": candidate.direction_pattern,
                "decision_flags": ";".join(flags),
                "required_next_step": "build_targeted_on_rrp_event_design_before_promotion",
            }
        )
    return pd.DataFrame(rows).sort_values(["sample", "outcome"]).reset_index(drop=True)


def strong_candidate_narrative_table(panel: pd.DataFrame) -> pd.DataFrame:
    shortlist = filtered_candidate_shortlist_table(panel)
    if shortlist.empty:
        return pd.DataFrame(columns=STRONG_CANDIDATE_NARRATIVE_COLUMNS)
    rows: list[dict[str, object]] = []
    strong = shortlist.loc[shortlist["priority"] == "strong_manual_review"]
    for candidate in strong.itertuples(index=False):
        for event_month in _split_event_months(candidate.largest_abs_impact_event_months):
            context = _month_context(panel, event_month)
            rows.append(
                {
                    "sample": candidate.sample,
                    "outcome": candidate.outcome,
                    "event_month": event_month,
                    "priority": candidate.priority,
                    "direction_pattern": candidate.direction_pattern,
                    "h0_beta": candidate.h0_beta,
                    "h0_t": candidate.h0_t,
                    "h3_beta": candidate.h3_beta,
                    "h3_t": candidate.h3_t,
                    "h6_beta": candidate.h6_beta,
                    "h6_t": candidate.h6_t,
                    "remaining_isolated_bill_shock_events": candidate.remaining_isolated_bill_shock_events,
                    "event_regime": context.get("on_rrp_regime_fixed", pd.NA),
                    "high_rate_regime": context.get("high_rate_regime", pd.NA),
                    "gross_bill_issuance": context.get("gross_bill_issuance", np.nan),
                    "bill_supply_shock_resid_100b": context.get("bill_supply_shock_resid_100b", np.nan),
                    "bill_share": context.get("bill_share", np.nan),
                    "bill_yield": context.get("bill_yield", np.nan),
                    "tga": context.get("tga", np.nan),
                    "reserves": context.get("reserves", np.nan),
                    "on_rrp": context.get("on_rrp", np.nan),
                    "total_mmf_assets": context.get("total_mmf_assets", np.nan),
                    "mmf_treasury_holdings": context.get("mmf_treasury_holdings", np.nan),
                    "d_mmf_treasury_holdings": context.get("d_mmf_treasury_holdings", np.nan),
                    "pre3_d_mmf_treasury_holdings": context.get("pre3_d_mmf_treasury_holdings", np.nan),
                    "post3_d_mmf_treasury_holdings": _forward_sum(panel, event_month, "d_mmf_treasury_holdings", 3),
                    "post6_d_mmf_treasury_holdings": _forward_sum(panel, event_month, "d_mmf_treasury_holdings", 6),
                    "external_calendar_check": _calendar_check_hint(event_month),
                    "narrative_status": "needs_external_calendar_validation",
                }
            )
    return (
        pd.DataFrame(rows, columns=STRONG_CANDIDATE_NARRATIVE_COLUMNS)
        .drop_duplicates()
        .reset_index(drop=True)
    )


def external_calendar_evidence_table(panel: pd.DataFrame) -> pd.DataFrame:
    narratives = strong_candidate_narrative_table(panel)
    if narratives.empty:
        return pd.DataFrame(columns=EXTERNAL_CALENDAR_EVIDENCE_COLUMNS)
    evidence = _external_calendar_evidence()
    rows: list[dict[str, object]] = []
    for event_month in sorted(narratives["event_month"].dropna().unique()):
        context = _month_context(panel, event_month)
        month_evidence = evidence.get(event_month, [])
        if not month_evidence:
            rows.append(
                {
                    "event_month": event_month,
                    "evidence_status": "missing_external_source",
                    "evidence_topic": "manual_follow_up_required",
                    "official_source": "",
                    "source_url": "",
                    "calendar_note": "add official Treasury, Fed, OFR, or FiscalData source before public use",
                    "panel_regime": context.get("on_rrp_regime_fixed", pd.NA),
                    "panel_bill_shock_resid_100b": context.get("bill_supply_shock_resid_100b", np.nan),
                }
            )
            continue
        for item in month_evidence:
            rows.append(
                {
                    "event_month": event_month,
                    "evidence_status": "source_attached",
                    "evidence_topic": item["topic"],
                    "official_source": item["source"],
                    "source_url": item["url"],
                    "calendar_note": item["note"],
                    "panel_regime": context.get("on_rrp_regime_fixed", pd.NA),
                    "panel_bill_shock_resid_100b": context.get("bill_supply_shock_resid_100b", np.nan),
                    "panel_gross_bill_issuance": context.get("gross_bill_issuance", np.nan),
                    "panel_bill_share": context.get("bill_share", np.nan),
                    "panel_tga": context.get("tga", np.nan),
                    "panel_on_rrp": context.get("on_rrp", np.nan),
                    "panel_mmf_treasury_holdings": context.get("mmf_treasury_holdings", np.nan),
                    "use_in_narrative": "calendar_context_only_not_causal_identification",
                }
            )
    return pd.DataFrame(rows, columns=EXTERNAL_CALENDAR_EVIDENCE_COLUMNS)


def candidate_table(panel: pd.DataFrame) -> pd.DataFrame:
    narratives = strong_candidate_narrative_table(panel)
    evidence = external_calendar_evidence_table(panel)
    if narratives.empty:
        return pd.DataFrame(columns=CANDIDATE_TABLE_COLUMNS)

    rows: list[dict[str, object]] = []
    for narrative in narratives.itertuples(index=False):
        event_evidence = (
            evidence.loc[evidence["event_month"] == narrative.event_month]
            if not evidence.empty and "event_month" in evidence.columns
            else pd.DataFrame()
        )
        source_urls = _join_unique(event_evidence.get("source_url", pd.Series(dtype=object)))
        evidence_topics = _join_unique(event_evidence.get("evidence_topic", pd.Series(dtype=object)))
        source_count = int(event_evidence["source_url"].dropna().nunique()) if not event_evidence.empty else 0
        rows.append(
            {
                "sample": narrative.sample,
                "outcome": narrative.outcome,
                "event_month": narrative.event_month,
                "candidate_status": "descriptive_candidate_not_identified",
                "priority": narrative.priority,
                "direction_pattern": narrative.direction_pattern,
                "h0_beta": narrative.h0_beta,
                "h0_t": narrative.h0_t,
                "h3_beta": narrative.h3_beta,
                "h3_t": narrative.h3_t,
                "h6_beta": narrative.h6_beta,
                "h6_t": narrative.h6_t,
                "event_regime": narrative.event_regime,
                "high_rate_regime": narrative.high_rate_regime,
                "bill_supply_shock_resid_100b": narrative.bill_supply_shock_resid_100b,
                "gross_bill_issuance": narrative.gross_bill_issuance,
                "bill_share": narrative.bill_share,
                "tga": narrative.tga,
                "on_rrp": narrative.on_rrp,
                "mmf_treasury_holdings": narrative.mmf_treasury_holdings,
                "impact_d_mmf_treasury_holdings": narrative.d_mmf_treasury_holdings,
                "pre3_d_mmf_treasury_holdings": narrative.pre3_d_mmf_treasury_holdings,
                "post3_d_mmf_treasury_holdings": narrative.post3_d_mmf_treasury_holdings,
                "post6_d_mmf_treasury_holdings": narrative.post6_d_mmf_treasury_holdings,
                "external_evidence_topics": evidence_topics,
                "external_source_count": source_count,
                "external_source_urls": source_urls,
                "required_next_step": "manual_event_validation_and_identification_design_before_promotion",
            }
        )
    return (
        pd.DataFrame(rows, columns=CANDIDATE_TABLE_COLUMNS)
        .sort_values(["event_month", "sample", "outcome"])
        .reset_index(drop=True)
    )


def interpretation_candidate_table(panel: pd.DataFrame) -> pd.DataFrame:
    return candidate_table(panel)


def claim_readiness_table(panel: pd.DataFrame) -> pd.DataFrame:
    candidates = candidate_table(panel)
    shortlist = filtered_candidate_shortlist_table(panel)
    readiness = design_readiness_table(panel)
    if candidates.empty:
        if shortlist.empty:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for short in shortlist.itertuples(index=False):
            design = _single_row(readiness, sample=short.sample, outcome=short.outcome)
            blocker_flags = [
                "no_strong_candidate_narrative",
                "manual_event_validation_needed",
                "causal_identification_not_established",
            ]
            if short.priority != "strong_manual_review":
                blocker_flags.append(f"shortlist_priority_{short.priority}")
            if short.event_status != "event_pattern_reviewable":
                blocker_flags.append("event_pattern_not_reviewable")
            pre_ratio = short.pre3_to_max_impact_or_post3_abs_ratio
            if pd.notna(pre_ratio) and float(pre_ratio) > 0.35:
                blocker_flags.append("material_pre_event_movement")
            residual_pretrend = design.get("max_abs_residual_pretrend_t", np.nan)
            if pd.notna(residual_pretrend) and float(residual_pretrend) >= 2.0:
                blocker_flags.append("residual_shock_pretrend_flag")
            rows.append(
                {
                    "sample": short.sample,
                    "outcome": short.outcome,
                    "readiness_status": "blocked",
                    "event_months": short.largest_abs_impact_event_months,
                    "n_event_months": len(_split_event_months(short.largest_abs_impact_event_months)),
                    "n_external_source_urls": 0,
                    "lp_direction_pattern": short.direction_pattern,
                    "max_abs_lp_t": max(
                        [
                            abs(value)
                            for value in [short.h0_t, short.h3_t, short.h6_t]
                            if pd.notna(value)
                        ],
                        default=np.nan,
                    ),
                    "h0_min_abs_t": abs(short.h0_t) if pd.notna(short.h0_t) else np.nan,
                    "h3_min_abs_t": abs(short.h3_t) if pd.notna(short.h3_t) else np.nan,
                    "h6_min_abs_t": abs(short.h6_t) if pd.notna(short.h6_t) else np.nan,
                    "filtered_event_status": short.event_status,
                    "pre3_to_reference_ratio": pre_ratio,
                    "excluded_event_months": short.excluded_event_months,
                    "remaining_isolated_bill_shock_events": short.remaining_isolated_bill_shock_events,
                    "design_status": design.get("status", ""),
                    "max_abs_residual_pretrend_t": residual_pretrend,
                    "blocker_flags": ";".join(blocker_flags),
                    "required_next_step": _claim_next_step(blocker_flags),
                }
            )
        return pd.DataFrame(rows).sort_values(["sample", "outcome"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    grouped = candidates.groupby(["sample", "outcome"], observed=True)
    for (sample, outcome), group in grouped:
        short = _single_row(shortlist, sample=sample, outcome=outcome)
        design = _single_row(readiness, sample=sample, outcome=outcome)
        max_abs_t = max(
            [
                abs(value)
                for value in [
                    group["h0_t"].dropna().max(),
                    group["h3_t"].dropna().max(),
                    group["h6_t"].dropna().max(),
                ]
                if pd.notna(value)
            ],
            default=np.nan,
        )
        event_months = _join_unique(group["event_month"])
        blocker_flags = _claim_blocker_flags(group, short, design)
        rows.append(
            {
                "sample": sample,
                "outcome": outcome,
                "readiness_status": _claim_readiness_status(blocker_flags),
                "event_months": event_months,
                "n_event_months": int(group["event_month"].nunique()),
                "n_external_source_urls": int(
                    len(
                        {
                            url
                            for urls in group["external_source_urls"].dropna()
                            for url in str(urls).split(";")
                            if url
                        }
                    )
                ),
                "lp_direction_pattern": _join_unique(group["direction_pattern"]),
                "max_abs_lp_t": float(max_abs_t) if pd.notna(max_abs_t) else np.nan,
                "h0_min_abs_t": float(group["h0_t"].abs().min()),
                "h3_min_abs_t": float(group["h3_t"].abs().min()),
                "h6_min_abs_t": float(group["h6_t"].abs().min()),
                "filtered_event_status": short.get("event_status", ""),
                "pre3_to_reference_ratio": short.get(
                    "pre3_to_max_impact_or_post3_abs_ratio", np.nan
                ),
                "excluded_event_months": short.get("excluded_event_months", np.nan),
                "remaining_isolated_bill_shock_events": short.get(
                    "remaining_isolated_bill_shock_events", np.nan
                ),
                "design_status": design.get("status", ""),
                "max_abs_residual_pretrend_t": design.get("max_abs_residual_pretrend_t", np.nan),
                "blocker_flags": ";".join(blocker_flags),
                "required_next_step": _claim_next_step(blocker_flags),
            }
        )
    return pd.DataFrame(rows).sort_values(["readiness_status", "sample", "outcome"]).reset_index(drop=True)


def write_candidate_narrative_report(panel: pd.DataFrame, outputs: dict[str, Path]) -> None:
    candidates = _read_csv_or_empty(outputs["candidate_table"])
    claim_readiness = _read_csv_or_empty(outputs["readiness_summary"])
    shortlist = _read_csv_or_empty(outputs["filtered_candidate_shortlist"])
    evidence = _read_csv_or_empty(outputs["external_calendar_evidence"])
    if candidates.empty:
        text = "# Monthly Candidate Review Report\n\nNo strong candidate rows found.\n"
        outputs["candidate_report"].write_text(text, encoding="utf-8")
        outputs["candidate_review_report"].write_text(text, encoding="utf-8")
        return

    strong_samples = _format_strong_sample_lines(candidates)
    readiness_lines = _format_claim_readiness_lines(claim_readiness)
    event_sections = _format_candidate_event_sections(candidates, evidence)
    manual_count = (
        int((shortlist.get("priority") == "manual_review").sum()) if not shortlist.empty else 0
    )
    weak_count = (
        int((shortlist.get("priority").isin(["weak_or_mixed", "still_compromised"])).sum())
        if not shortlist.empty and "priority" in shortlist.columns
        else 0
    )
    text = "\n".join(
        [
            "# Monthly Candidate Review Report",
            "",
            "## Status",
            "",
            "- Strong rows remain descriptive candidates, not causal estimates.",
            f"- Strong sample/outcome rows: {candidates[['sample', 'outcome']].drop_duplicates().shape[0]}",
            f"- Strong event months with attached context: {candidates['event_month'].nunique()}",
            f"- Other manual-review shortlist rows: {manual_count}",
            f"- Weak, mixed, or still-compromised shortlist rows: {weak_count}",
            "",
            "## Strong Sample/Outcome Rows",
            "",
            *strong_samples,
            "",
            "## Readiness",
            "",
            *readiness_lines,
            "",
            "## Event-Month Context",
            "",
            *event_sections,
            "",
            "## Guardrails",
            "",
            "- Use this report as a triage document for event validation.",
            "- The official-source notes provide calendar context only.",
            "- Pretrend, event-window, and local-projection diagnostics still block causal language.",
            "- Any public interpretation needs a stronger shock design or explicit event validation.",
            "",
        ]
    )
    outputs["candidate_report"].write_text(text, encoding="utf-8")
    outputs["candidate_review_report"].write_text(text, encoding="utf-8")


def evidence_gate_summary(root: Path) -> pd.DataFrame:
    table_dir = root / "output" / "tables"
    monthly_claim = _read_csv_or_empty(table_dir / "monthly_readiness_summary.csv")
    monthly_on_rrp = _read_csv_or_empty(table_dir / "monthly_on_rrp_manual_review.csv")
    monthly_shortlist = _read_csv_or_empty(table_dir / "monthly_filtered_candidate_shortlist.csv")
    weekly_stable = _read_csv_or_empty(table_dir / "weekly_stability_candidates.csv")
    weekly_claim = _read_csv_or_empty(table_dir / "weekly_design_readiness.csv")
    large_final = _read_csv_or_empty(table_dir / "weekly_large_rebuild_final_review.csv")
    large_cells = _read_csv_or_empty(table_dir / "weekly_large_rebuild_cell_summary.csv")
    rows = [
        _monthly_broad_gate_row(monthly_claim),
        _monthly_on_rrp_gate_row(monthly_on_rrp),
        _monthly_mmf_gate_row(monthly_shortlist, monthly_claim),
        _weekly_broad_gate_row(weekly_stable, weekly_claim),
        _weekly_large_rebuild_gate_row(large_final, large_cells),
        _negative_evidence_gate_row(monthly_claim, weekly_stable),
    ]
    return pd.DataFrame(rows, columns=INTERNAL_EVIDENCE_GATE_COLUMNS)


def write_evidence_gate_summary(root: Path, *, include_internal: bool | None = None) -> dict[str, str]:
    table_dir = ensure_dir(root / "output" / "tables")
    report_dir = ensure_dir(root / "output" / "reports")
    table = evidence_gate_summary(root)
    table_path = table_dir / "evidence_gate_summary.csv"
    report_path = report_dir / "evidence_gate_summary.md"
    table.to_csv(table_path, index=False)
    _write_evidence_gate_report(table, report_path)
    outputs = {
        "evidence_gate_summary": relative_to_root(root, table_path),
        "evidence_gate_report": relative_to_root(root, report_path),
    }
    if include_internal is None:
        include_internal = os.environ.get("LIQSUB_WRITE_INTERNAL_REPORTS") == "1"
    if include_internal:
        internal_table_dir = ensure_dir(root / "output" / "internal" / "tables")
        internal_report_dir = ensure_dir(root / "output" / "internal" / "reports")
        internal_table_path = internal_table_dir / "internal_evidence_gate_summary.csv"
        internal_report_path = internal_report_dir / "internal_evidence_gate_summary.md"
        claim_map_path = internal_table_dir / "internal_claim_artifact_map.csv"
        chapter_path = internal_report_dir / "descriptive_negative_evidence_chapter_draft.md"
        table.to_csv(internal_table_path, index=False)
        _write_internal_evidence_gate_report(table, internal_report_path)
        claim_map = internal_claim_artifact_map(table)
        claim_map.to_csv(claim_map_path, index=False)
        _write_descriptive_negative_chapter_draft(table, claim_map, chapter_path)
        outputs.update(
            {
                "internal_evidence_gate_summary": relative_to_root(root, internal_table_path),
                "internal_evidence_gate_report": relative_to_root(root, internal_report_path),
                "internal_claim_artifact_map": relative_to_root(root, claim_map_path),
                "internal_descriptive_negative_evidence_chapter_draft": relative_to_root(root, chapter_path),
            }
        )
    return outputs


def internal_evidence_gate_summary(root: Path) -> pd.DataFrame:
    return evidence_gate_summary(root)


def write_internal_evidence_gate_summary(root: Path) -> dict[str, str]:
    return write_evidence_gate_summary(root, include_internal=True)


def _external_calendar_evidence() -> dict[str, list[dict[str, str]]]:
    path = Path(__file__).resolve().parents[2] / "data" / "manual" / "event_calendar_context.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"event_month", "topic", "source", "url", "note"}
    if not required.issubset(df.columns):
        return {}
    evidence: dict[str, list[dict[str, str]]] = {}
    for row in df.dropna(subset=["event_month", "topic", "source", "url"]).itertuples(index=False):
        event_month = str(row.event_month)
        evidence.setdefault(event_month, []).append(
            {
                "topic": str(row.topic),
                "source": str(row.source),
                "url": str(row.url),
                "note": str(row.note),
            }
        )
    return evidence


def _join_unique(values: pd.Series) -> str:
    if values.empty:
        return ""
    cleaned = [str(value) for value in values.dropna() if str(value)]
    return ";".join(dict.fromkeys(cleaned))


def _monthly_broad_gate_row(monthly_claim: pd.DataFrame) -> dict[str, object]:
    blocked = _count_value(monthly_claim, "readiness_status", "blocked")
    rows = int(len(monthly_claim))
    blockers = _join_table_values(monthly_claim, "blocker_flags")
    return {
        "design_path": "monthly_broad_bill_supply_designs",
        "status": "blocked" if rows else "no_current_candidates",
        "claim_use": "descriptive_or_negative_evidence_only",
        "evidence_basis": f"{blocked} blocked monthly readiness rows out of {rows}",
        "primary_artifacts": "monthly_readiness_summary.csv;monthly_pretrend_diagnostics.csv;monthly_residual_shock_pretrend_diagnostics.csv",
        "binding_blockers": blockers or "no claim-ready monthly rows",
        "next_design_step": "freeze as diagnostic evidence unless anticipation controls or cleaner shocks are added",
    }


def _monthly_on_rrp_gate_row(monthly_on_rrp: pd.DataFrame) -> dict[str, object]:
    statuses = _join_table_values(monthly_on_rrp, "promotion_decision")
    max_t = _series_abs_max(monthly_on_rrp, "max_abs_lp_t")
    return {
        "design_path": "monthly_on_rrp_manual_review",
        "status": "blocked",
        "claim_use": "context_only_do_not_promote",
        "evidence_basis": f"{len(monthly_on_rrp)} ON RRP manual-review rows; max_abs_lp_t={_fmt(max_t)}",
        "primary_artifacts": "monthly_on_rrp_manual_review.csv;monthly_filtered_candidate_shortlist.csv",
        "binding_blockers": statuses
        or "weak LP evidence, sparse regime splits, and event-window reversals",
        "next_design_step": "do not promote; only reopen with weekly or daily ON RRP-specific event timing",
    }


def _monthly_mmf_gate_row(
    monthly_shortlist: pd.DataFrame,
    monthly_claim: pd.DataFrame,
) -> dict[str, object]:
    mmf_shortlist = _filter_contains(monthly_shortlist, "outcome", "mmf_treasury")
    mmf_claim = _filter_contains(monthly_claim, "outcome", "mmf_treasury")
    blockers = _join_table_values(mmf_claim, "blocker_flags") or _join_table_values(
        mmf_shortlist,
        "event_status",
    )
    return {
        "design_path": "monthly_mmf_treasury_holdings_redesign_lead",
        "status": "redesign_lead_blocked_for_claims" if not mmf_shortlist.empty else "not_active",
        "claim_use": "secondary_redesign_lead_only",
        "evidence_basis": f"{len(mmf_shortlist)} MMF Treasury shortlist rows; {len(mmf_claim)} readiness rows",
        "primary_artifacts": "monthly_filtered_candidate_shortlist.csv;monthly_filtered_candidate_event_review.csv;monthly_candidate_lp_coefficients.csv",
        "binding_blockers": blockers
        or "pre-event movement and event-context validation still unresolved",
        "next_design_step": "build MMF portfolio-allocation redesign with Treasury, repo, ON RRP, agency, and total asset components",
    }


def _weekly_broad_gate_row(weekly_stable: pd.DataFrame, weekly_claim: pd.DataFrame) -> dict[str, object]:
    passing = _count_value(weekly_stable, "status", "passes_all_gates")
    blocked = _count_value(weekly_stable, "status", "blocked")
    blockers = _join_table_values(weekly_stable, "blocker_flags") or _join_table_values(
        weekly_claim,
        "blocker_flags",
    )
    return {
        "design_path": "weekly_broad_tga_rebuild_design",
        "status": "blocked",
        "claim_use": "negative_evidence_appendix",
        "evidence_basis": f"{passing} passing broad cells; {blocked} blocked broad cells",
        "primary_artifacts": "weekly_stability_candidates.csv;weekly_placebo_tests.csv;weekly_event_filter_sensitivity.csv",
        "binding_blockers": blockers or "matched placebo and bootstrap gates fail",
        "next_design_step": "keep as broad-design failure evidence; do not relax gates",
    }


def _weekly_large_rebuild_gate_row(
    large_final: pd.DataFrame,
    large_cells: pd.DataFrame,
) -> dict[str, object]:
    if large_final.empty:
        return {
            "design_path": "weekly_large_rebuild_targeted_reserves",
            "status": "not_estimated",
            "claim_use": "do_not_promote",
            "evidence_basis": "targeted final review missing",
            "primary_artifacts": "weekly_large_rebuild_final_review.csv",
            "binding_blockers": "run analyze-weekly after targeted design generation",
            "next_design_step": "estimate targeted large-rebuild final review",
        }
    final = large_final.iloc[0]
    reserves = large_cells.loc[large_cells["table_role"] == "headline_reserves_main"] if not large_cells.empty else pd.DataFrame()
    reserve_cells = _format_reserve_cells(reserves)
    return {
        "design_path": "weekly_large_rebuild_targeted_reserves",
        "status": final.get("status", "unknown"),
        "claim_use": final.get("claim_use", "do_not_promote"),
        "evidence_basis": reserve_cells or str(final.get("headline_cells", "")),
        "primary_artifacts": "weekly_large_rebuild_final_review.csv;weekly_large_rebuild_cell_summary.csv;weekly_large_rebuild_event_sign_stability.csv;weekly_large_rebuild_abnormal_changes.csv",
        "binding_blockers": final.get("binding_limitations", ""),
        "next_design_step": "use only as descriptive reserves-plumbing diagnostic with event-level heterogeneity visible",
    }


def _negative_evidence_gate_row(monthly_claim: pd.DataFrame, weekly_stable: pd.DataFrame) -> dict[str, object]:
    monthly_blocked = _count_value(monthly_claim, "readiness_status", "blocked")
    weekly_passing = _count_value(weekly_stable, "status", "passes_all_gates")
    weekly_blocked = _count_value(weekly_stable, "status", "blocked")
    return {
        "design_path": "cross_design_diagnostic_summary",
        "status": "diagnostic_summary_ready",
        "claim_use": "backend_diagnostic_summary",
        "evidence_basis": f"monthly blocked rows={monthly_blocked}; weekly broad passing cells={weekly_passing}; weekly broad blocked cells={weekly_blocked}",
        "primary_artifacts": "monthly_readiness_summary.csv;weekly_stability_candidates.csv;weekly_large_rebuild_diagnostic_report.md",
        "binding_blockers": "current broad designs fail diagnostic gates",
        "next_design_step": "keep broad diagnostics separate from targeted reserve-plumbing diagnostics",
    }


def _write_evidence_gate_report(table: pd.DataFrame, path: Path) -> None:
    targeted = _gate_row(table, "weekly_large_rebuild_targeted_reserves")
    weekly_broad = _gate_row(table, "weekly_broad_tga_rebuild_design")
    monthly_broad = _gate_row(table, "monthly_broad_bill_supply_designs")
    monthly_on_rrp = _gate_row(table, "monthly_on_rrp_manual_review")
    monthly_mmf = _gate_row(table, "monthly_mmf_treasury_holdings_redesign_lead")
    lines = [
        "# Evidence Gate Summary",
        "",
        "This is a backend diagnostics report. It classifies current monthly and weekly analysis outputs by evidence status and required next backend work.",
        "",
        "## Gate Matrix",
        "",
        "| design path | status | use | next step |",
        "| --- | --- | --- | --- |",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| "
            f"{row.design_path} | `{row.status}` | `{row.claim_use}` | "
            f"{row.next_design_step} |"
        )
    lines.extend(
        [
            "",
            "## Backend State",
            "",
            "- Monthly and weekly panels build and emit reproducible diagnostics.",
            "- Broad monthly and weekly designs are retained as diagnostic outputs because current gates block promotion.",
            "- The targeted weekly large-rebuild reserve path is a narrow diagnostic output with visible event-level limitations.",
            "- ON RRP and MMF rows remain redesign leads or appendix diagnostics, not headline backend results.",
            "",
            "## Binding Blockers",
            "",
            f"- Monthly broad designs: {_gate_value(monthly_broad, 'binding_blockers')}.",
            f"- Monthly ON RRP: {_gate_value(monthly_on_rrp, 'binding_blockers')}.",
            f"- Monthly MMF Treasury holdings: {_gate_value(monthly_mmf, 'binding_blockers')}.",
            f"- Weekly broad design: {_gate_value(weekly_broad, 'binding_blockers')}.",
            f"- Targeted large-rebuild reserves: {_gate_value(targeted, 'binding_blockers')}.",
            "",
            "## Key Artifacts",
            "",
            "- `output/tables/monthly_readiness_summary.csv`",
            "- `output/tables/monthly_pretrend_diagnostics.csv`",
            "- `output/tables/monthly_residual_shock_pretrend_diagnostics.csv`",
            "- `output/tables/weekly_stability_candidates.csv`",
            "- `output/tables/weekly_placebo_tests.csv`",
            "- `output/tables/weekly_large_rebuild_final_review.csv`",
            "- `output/tables/weekly_large_rebuild_cell_summary.csv`",
            "- `output/tables/weekly_large_rebuild_event_sign_stability.csv`",
            "",
            "## Backend Next Work",
            "",
            "1. Keep generated diagnostic schemas parseable, including zero-row outputs.",
            "2. Continue separating public backend diagnostics from internal planning artifacts.",
            "3. Add stronger tests around source metadata, schema stability, and targeted weekly matching logic.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_internal_evidence_gate_report(table: pd.DataFrame, path: Path) -> None:
    targeted = _gate_row(table, "weekly_large_rebuild_targeted_reserves")
    weekly_broad = _gate_row(table, "weekly_broad_tga_rebuild_design")
    monthly_broad = _gate_row(table, "monthly_broad_bill_supply_designs")
    monthly_on_rrp = _gate_row(table, "monthly_on_rrp_manual_review")
    monthly_mmf = _gate_row(table, "monthly_mmf_treasury_holdings_redesign_lead")
    negative = _gate_row(table, "cross_design_diagnostic_summary")
    lines = [
        "# Internal Evidence Gate Summary",
        "",
        "No current design is public-facing causal evidence. The targeted weekly large-rebuild reserve result is descriptive reserves-plumbing evidence only.",
        "",
        "## Executive Judgment",
        "",
        "The project is ready for a descriptive empirical-design writeup, not a causal liquidity-substitution claim. Broad monthly and weekly designs remain blocked by the project's own gates. The only positive public-facing material is a narrow weekly diagnostic: large TGA rebuild episodes are associated with reserve declines in the non-debt-limit main sample.",
        "",
        "## Gate Matrix",
        "",
        "| design path | status | claim use | next step |",
        "| --- | --- | --- | --- |",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| "
            f"{row.design_path} | `{row.status}` | `{row.claim_use}` | "
            f"{row.next_design_step} |"
        )
    lines.extend(
        [
            "",
            "## What Can Be Said",
            "",
            "- The current broad monthly and weekly designs do not support a causal liquidity-substitution claim.",
            "- The targeted weekly large-rebuild design can be used as descriptive reserves-plumbing evidence.",
            "- The project has a defensible negative-evidence chapter path: broad designs fail because of pretrends, sparse regime splits, bootstrap failures, and matched-placebo failures.",
            "- Monthly MMF Treasury holdings remain a secondary redesign lead, not a result.",
            "- Monthly and weekly ON RRP evidence should stay context-only or appendix-only unless a new event-timing design is built.",
            "",
            "## Binding Blockers",
            "",
            f"- Monthly broad designs: {_gate_value(monthly_broad, 'binding_blockers')}.",
            f"- Monthly ON RRP: {_gate_value(monthly_on_rrp, 'binding_blockers')}.",
            f"- Monthly MMF Treasury holdings: {_gate_value(monthly_mmf, 'binding_blockers')}.",
            f"- Weekly broad design: {_gate_value(weekly_broad, 'binding_blockers')}.",
            f"- Targeted large-rebuild reserves: {_gate_value(targeted, 'binding_blockers')}.",
            "",
            "## Targeted Large-Rebuild Evidence",
            "",
            f"- Status: `{_gate_value(targeted, 'status')}`.",
            f"- Claim use: `{_gate_value(targeted, 'claim_use')}`.",
            f"- Evidence basis: {_gate_value(targeted, 'evidence_basis')}.",
            "- Required framing: reserve-balance movement around large TGA rebuilds, not a causal bill-issuance law.",
            "- Event-level heterogeneity must stay visible, especially `event_039` positive reserve changes at tau 2 and tau 4.",
            "- ON RRP remains appendix-only, with `event_041` and `event_045` mechanism-risk flags.",
            "",
            "Key artifacts:",
            "",
            "- `output/tables/weekly_large_rebuild_final_review.csv`",
            "- `output/tables/weekly_large_rebuild_cell_summary.csv`",
            "- `output/tables/weekly_large_rebuild_event_sign_stability.csv`",
            "- `output/tables/weekly_large_rebuild_abnormal_changes.csv`",
            "- `output/reports/weekly_large_rebuild_diagnostic_writeup.md`",
            "- `output/reports/weekly_large_rebuild_reserves_event_time.svg`",
            "- `output/reports/weekly_large_rebuild_on_rrp_event_time.svg`",
            "",
            "## Chapter Path",
            "",
            f"- Negative-evidence path status: `{_gate_value(negative, 'status')}`.",
            f"- Basis: {_gate_value(negative, 'evidence_basis')}.",
            "- Recommended structure: first document broad-design failures, then present the targeted reserve diagnostic as a narrow design lead with explicit limitations.",
            "",
            "## Forbidden Upgrades",
            "",
            "- Do not say that Treasury bill issuance causally drains deposits.",
            "- Do not say that bill issuance causally drains ON RRP.",
            "- Do not say that the project establishes a stable liquidity-substitution law.",
            "- Do not promote the targeted reserve result beyond descriptive money-market plumbing evidence.",
            "",
            "## Next Empirical Steps",
            "",
            "1. If writing now, draft a descriptive/negative-evidence chapter using this gate matrix.",
            "2. If redesigning further, prioritize the monthly MMF portfolio-allocation scaffold or a cleaner ON RRP event-timing design.",
            "3. Preserve the broad weekly and monthly blocked gates unless a pre-specified redesign passes them.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def internal_claim_artifact_map(table: pd.DataFrame) -> pd.DataFrame:
    monthly_broad = _gate_row(table, "monthly_broad_bill_supply_designs")
    weekly_broad = _gate_row(table, "weekly_broad_tga_rebuild_design")
    targeted = _gate_row(table, "weekly_large_rebuild_targeted_reserves")
    mmf = _gate_row(table, "monthly_mmf_treasury_holdings_redesign_lead")
    on_rrp = _gate_row(table, "monthly_on_rrp_manual_review")
    negative = _gate_row(table, "cross_design_diagnostic_summary")
    rows = [
        {
            "claim_id": "C01",
            "claim_text": "The current broad monthly and weekly designs do not support a causal liquidity-substitution claim.",
            "claim_strength": "negative_evidence",
            "artifact_paths": _claim_artifacts(monthly_broad, weekly_broad),
            "permitted_use": "chapter analysis sentence",
            "forbidden_upgrade": "Do not infer that liquidity substitution is false in the economy.",
        },
        {
            "claim_id": "C02",
            "claim_text": "Monthly broad bill-supply designs are blocked under the project gates.",
            "claim_strength": "blocked_design_summary",
            "artifact_paths": _gate_value(monthly_broad, "primary_artifacts"),
            "permitted_use": "monthly design limitation",
            "forbidden_upgrade": "Do not promote monthly LP or event-window coefficients as causal evidence.",
        },
        {
            "claim_id": "C03",
            "claim_text": "Monthly ON RRP evidence is context-only and should not be promoted.",
            "claim_strength": "context_only",
            "artifact_paths": _gate_value(on_rrp, "primary_artifacts"),
            "permitted_use": "mechanism limitation",
            "forbidden_upgrade": "Do not claim bill issuance causally drains ON RRP.",
        },
        {
            "claim_id": "C04",
            "claim_text": "Monthly MMF Treasury holdings remain a secondary redesign lead, not a result.",
            "claim_strength": "redesign_lead",
            "artifact_paths": _gate_value(mmf, "primary_artifacts"),
            "permitted_use": "future work motivation",
            "forbidden_upgrade": "Do not claim MMFs absorb bill issuance without a portfolio redesign.",
        },
        {
            "claim_id": "C05",
            "claim_text": "The broad weekly TGA-rebuild design fails the current robustness gates.",
            "claim_strength": "negative_evidence",
            "artifact_paths": _gate_value(weekly_broad, "primary_artifacts"),
            "permitted_use": "appendix or negative-evidence exhibit",
            "forbidden_upgrade": "Do not relax weekly gates to recover broad claims.",
        },
        {
            "claim_id": "C06",
            "claim_text": "The targeted weekly large-rebuild reserve design is descriptive reserves-plumbing evidence.",
            "claim_strength": "descriptive_targeted_evidence",
            "artifact_paths": _gate_value(targeted, "primary_artifacts"),
            "permitted_use": "narrow positive diagnostic",
            "forbidden_upgrade": "Do not describe it as causal bill-issuance evidence.",
        },
        {
            "claim_id": "C07",
            "claim_text": "ON RRP remains appendix-only in the targeted large-rebuild design.",
            "claim_strength": "appendix_mechanism_diagnostic",
            "artifact_paths": "weekly_large_rebuild_cell_summary.csv;weekly_large_rebuild_event_sign_stability.csv;weekly_large_rebuild_on_rrp_event_time.svg",
            "permitted_use": "mechanism caveat",
            "forbidden_upgrade": "Do not present ON RRP as the headline absorption channel.",
        },
        {
            "claim_id": "C08",
            "claim_text": "A descriptive/negative-evidence chapter is currently defensible.",
            "claim_strength": "chapter_path",
            "artifact_paths": _gate_value(negative, "primary_artifacts"),
            "permitted_use": "chapter framing",
            "forbidden_upgrade": "Do not label any current path public-facing causal evidence.",
        },
    ]
    return pd.DataFrame(rows, columns=INTERNAL_CLAIM_ARTIFACT_COLUMNS)


def _write_descriptive_negative_chapter_draft(
    table: pd.DataFrame,
    claim_map: pd.DataFrame,
    path: Path,
) -> None:
    monthly_broad = _gate_row(table, "monthly_broad_bill_supply_designs")
    weekly_broad = _gate_row(table, "weekly_broad_tga_rebuild_design")
    targeted = _gate_row(table, "weekly_large_rebuild_targeted_reserves")
    negative = _gate_row(table, "cross_design_diagnostic_summary")
    lines = [
        "# Descriptive And Negative Evidence Chapter Draft",
        "",
        "## Evidence Position",
        "",
        "This chapter does not claim to identify a causal liquidity-substitution law. Its contribution is diagnostic: it documents why the broad monthly and weekly designs do not clear the project's evidence gates, then reports a narrower descriptive reserve-plumbing pattern around large TGA rebuild episodes.",
        "",
        "## Evidence Boundary",
        "",
        f"The evidence-gate matrix classifies the broad monthly design as `{_gate_value(monthly_broad, 'status')}` and the broad weekly design as `{_gate_value(weekly_broad, 'status')}`. The targeted weekly large-rebuild reserve design is classified as `{_gate_value(targeted, 'claim_use')}`. These statuses rule out public-facing causal language in the current draft.",
        "",
        "## Broad Monthly Designs",
        "",
        f"The monthly broad bill-supply path is blocked. The current evidence basis is: {_gate_value(monthly_broad, 'evidence_basis')}. The binding blockers are: {_gate_value(monthly_broad, 'binding_blockers')}. This supports a design limitation and negative-evidence discussion, not a causal estimate.",
        "",
        "Monthly ON RRP is weaker still. It remains context-only, with weak LP evidence, sparse regime splits, and event-window reversals. Monthly MMF Treasury holdings are more promising as a future redesign lead, but the existing tables still block claim use because pre-event movement and event validation remain unresolved.",
        "",
        "## Broad Weekly Design",
        "",
        f"The broad weekly TGA-rebuild path is also blocked. The current evidence basis is: {_gate_value(weekly_broad, 'evidence_basis')}. Its principal blocker is the failure of robustness gates, especially matched-placebo and bootstrap diagnostics. The broad weekly result should appear as an appendix or negative-evidence exhibit.",
        "",
        "## Targeted Large-Rebuild Reserve Diagnostic",
        "",
        f"The targeted large-rebuild reserve path is the strongest positive diagnostic. Its evidence basis is: {_gate_value(targeted, 'evidence_basis')}. This can be written as reserve-balance movement around large TGA rebuilds in the non-debt-limit main sample.",
        "",
        f"The limitations are binding: {_gate_value(targeted, 'binding_blockers')}. In particular, the design has a five-event main sample, keeps `event_038` appendix-only, and leaves ON RRP appendix-only. Event-level heterogeneity must remain visible, including the positive reserve changes for `event_039` at tau 2 and tau 4.",
        "",
        "## Claim-To-Artifact Discipline",
        "",
        "Every internal interpretation claim should map to `output/internal/tables/internal_claim_artifact_map.csv`. The permitted claims are:",
        "",
    ]
    for row in claim_map.itertuples(index=False):
        lines.append(f"- `{row.claim_id}`: {row.claim_text} Artifact(s): `{row.artifact_paths}`.")
    lines.extend(
        [
            "",
            "## Recommended Chapter Structure",
            "",
            "1. Introduce the intended liquidity-substitution question and explain the evidence gates.",
            "2. Present the monthly broad design failures and why pretrends/event windows block causal interpretation.",
            "3. Present the broad weekly design failure and matched-placebo/bootstrap blockers.",
            "4. Present the targeted large-rebuild reserve diagnostic as descriptive money-market plumbing evidence.",
            "5. Close with redesign priorities: MMF portfolio allocation and cleaner ON RRP event timing.",
            "",
            "## Forbidden Language",
            "",
            "- Do not say bill issuance causally drains deposits.",
            "- Do not say bill issuance causally drains ON RRP.",
            "- Do not say the current project establishes liquidity substitution.",
            "- Do not describe the targeted large-rebuild reserve result as causal.",
            "",
            "## Current Chapter Status",
            "",
            f"`{_gate_value(negative, 'status')}`: {_gate_value(negative, 'next_design_step')}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _claim_artifacts(*rows: dict[str, object]) -> str:
    artifacts: list[str] = []
    for row in rows:
        for artifact in _gate_value(row, "primary_artifacts").split(";"):
            if artifact and artifact not in artifacts:
                artifacts.append(artifact)
    return ";".join(artifacts)


def _gate_row(table: pd.DataFrame, design_path: str) -> dict[str, object]:
    if table.empty or "design_path" not in table.columns:
        return {}
    rows = table.loc[table["design_path"] == design_path]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _gate_value(row: dict[str, object], column: str) -> str:
    value = row.get(column, "")
    return "" if pd.isna(value) else str(value)


def _count_value(table: pd.DataFrame, column: str, value: str) -> int:
    if table.empty or column not in table.columns:
        return 0
    return int((table[column] == value).sum())


def _join_table_values(table: pd.DataFrame, column: str) -> str:
    if table.empty or column not in table.columns:
        return ""
    values: list[str] = []
    for value in table[column].dropna():
        for part in str(value).split(";"):
            if part and part not in values:
                values.append(part)
    return ";".join(values)


def _series_abs_max(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table.columns:
        return np.nan
    values = pd.to_numeric(table[column], errors="coerce").abs().dropna()
    return float(values.max()) if not values.empty else np.nan


def _filter_contains(table: pd.DataFrame, column: str, needle: str) -> pd.DataFrame:
    if table.empty or column not in table.columns:
        return pd.DataFrame()
    return table.loc[table[column].astype(str).str.contains(needle, case=False, na=False)]


def _format_reserve_cells(cells: pd.DataFrame) -> str:
    if cells.empty:
        return ""
    parts = []
    for row in cells.sort_values("tau").itertuples(index=False):
        parts.append(f"reserves:tau{int(row.tau)} mean={_fmt(row.mean_change)} status={row.status}")
    return ";".join(parts)


def _fmt(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    return f"{float(number):.3f}"


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _single_row(table: pd.DataFrame, *, sample: object, outcome: object) -> dict[str, object]:
    if table.empty or not {"sample", "outcome"}.issubset(table.columns):
        return {}
    rows = table.loc[(table["sample"] == sample) & (table["outcome"] == outcome)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _claim_blocker_flags(
    candidates: pd.DataFrame,
    shortlist_row: dict[str, object],
    design_row: dict[str, object],
) -> list[str]:
    blockers = ["manual_event_validation_needed", "causal_identification_not_established"]
    source_count = {
        url
        for urls in candidates["external_source_urls"].dropna()
        for url in str(urls).split(";")
        if url
    }
    if not source_count:
        blockers.append("missing_official_calendar_sources")
    if shortlist_row.get("event_status") != "event_pattern_reviewable":
        blockers.append("event_pattern_not_reviewable")
    pre_ratio = shortlist_row.get("pre3_to_max_impact_or_post3_abs_ratio", np.nan)
    if pd.notna(pre_ratio) and float(pre_ratio) > 0.35:
        blockers.append("material_pre_event_movement")
    residual_pretrend = design_row.get("max_abs_residual_pretrend_t", np.nan)
    if pd.notna(residual_pretrend) and float(residual_pretrend) >= 2.0:
        blockers.append("residual_shock_pretrend_flag")
    if candidates["h6_t"].abs().min() < 2.0:
        blockers.append("h6_lp_t_below_2")
    if candidates["direction_pattern"].nunique() > 1:
        blockers.append("direction_pattern_not_unique")
    return blockers


def _claim_readiness_status(blockers: list[str]) -> str:
    hard_blockers = {
        "no_strong_candidate_narrative",
        "missing_official_calendar_sources",
        "event_pattern_not_reviewable",
        "material_pre_event_movement",
        "residual_shock_pretrend_flag",
    }
    if hard_blockers.intersection(blockers):
        return "blocked"
    if blockers == ["manual_event_validation_needed", "causal_identification_not_established"]:
        return "descriptive_ready_only"
    return "needs_manual_validation"


def _claim_next_step(blockers: list[str]) -> str:
    if "no_strong_candidate_narrative" in blockers:
        return "review_monthly_shortlist_or_refine_candidate_thresholds"
    if "missing_official_calendar_sources" in blockers:
        return "attach_official_calendar_sources"
    if "event_pattern_not_reviewable" in blockers or "material_pre_event_movement" in blockers:
        return "refine_event_filter_or_write_explicit_anticipation_narrative"
    if "residual_shock_pretrend_flag" in blockers:
        return "add_anticipation_controls_or_find_cleaner_shock"
    if "h6_lp_t_below_2" in blockers:
        return "treat_h6_as_weak_persistence_evidence"
    return "manual_event_validation_and_identification_design_before_claim"


def _format_strong_sample_lines(candidates: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    columns = [
        "sample",
        "outcome",
        "direction_pattern",
        "h0_beta",
        "h0_t",
        "h3_beta",
        "h3_t",
        "h6_beta",
        "h6_t",
    ]
    for row in candidates[columns].drop_duplicates().itertuples(index=False):
        lines.append(
            "- "
            f"{row.sample} / {row.outcome}: {row.direction_pattern}; "
            f"h0 beta {row.h0_beta:.1f} (t={row.h0_t:.2f}), "
            f"h3 beta {row.h3_beta:.1f} (t={row.h3_t:.2f}), "
            f"h6 beta {row.h6_beta:.1f} (t={row.h6_t:.2f})"
        )
    return lines


def _format_claim_readiness_lines(claim_readiness: pd.DataFrame) -> list[str]:
    if claim_readiness.empty:
        return ["- No interpretation-readiness rows generated."]
    lines: list[str] = []
    for row in claim_readiness.itertuples(index=False):
        lines.append(
            "- "
            f"{row.sample} / {row.outcome}: {row.readiness_status}; "
            f"blockers={row.blocker_flags}; next={row.required_next_step}"
        )
    return lines


def _format_candidate_event_sections(candidates: pd.DataFrame, evidence: pd.DataFrame) -> list[str]:
    sections: list[str] = []
    for event_month, group in candidates.groupby("event_month", observed=True):
        first = group.iloc[0]
        sections.extend(
            [
                f"### {event_month}",
                "",
                (
                    f"- Regime: {first.event_regime}; bill shock residual: "
                    f"{first.bill_supply_shock_resid_100b:.2f} x $100b; "
                    f"bill share: {first.bill_share:.3f}"
                ),
                (
                    f"- MMF Treasury holdings change: impact "
                    f"{first.impact_d_mmf_treasury_holdings:.1f}, "
                    f"pre3 {first.pre3_d_mmf_treasury_holdings:.1f}, "
                    f"post3 {first.post3_d_mmf_treasury_holdings:.1f}, "
                    f"post6 {first.post6_d_mmf_treasury_holdings:.1f}"
                ),
                f"- Candidate samples: {_join_unique(group['sample'])}",
            ]
        )
        event_evidence = (
            evidence.loc[evidence["event_month"] == event_month]
            if not evidence.empty and "event_month" in evidence.columns
            else pd.DataFrame()
        )
        if event_evidence.empty:
            sections.append("- External evidence: missing official-source attachment.")
        else:
            for source in event_evidence.itertuples(index=False):
                sections.append(
                    f"- {source.evidence_topic}: {source.official_source} ({source.source_url})"
                )
        sections.append("")
    return sections


def _split_event_months(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part for part in str(value).split(";") if part]


def _month_context(panel: pd.DataFrame, event_month: str) -> dict[str, object]:
    month = pd.to_datetime(event_month, errors="coerce")
    if pd.isna(month):
        return {}
    rows = panel.loc[panel["month"] == month]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _forward_sum(panel: pd.DataFrame, event_month: str, column: str, horizon: int) -> float:
    if column not in panel.columns:
        return np.nan
    month = pd.to_datetime(event_month, errors="coerce")
    if pd.isna(month):
        return np.nan
    matches = panel.index[panel["month"] == month].tolist()
    if not matches:
        return np.nan
    event_idx = matches[0]
    start = event_idx + 1
    end = event_idx + horizon
    if end >= len(panel):
        return np.nan
    values = pd.to_numeric(panel.loc[start:end, column], errors="coerce").dropna()
    return float(values.sum()) if not values.empty else np.nan


def _calendar_check_hint(event_month: str) -> str:
    hints = {
        "2020-04-01": "check COVID-era Treasury bill issuance, Fed facilities, MMF stress, and fiscal response calendar",
        "2023-05-01": "check debt-ceiling/TGA drawdown and bill-supply calendar around spring 2023",
        "2025-07-01": "check current-cycle Treasury financing, RRP, MMF, and rate-setting calendar",
    }
    return hints.get(event_month, "check Treasury financing calendar, FOMC/rate context, and MMF flow news")


def _filtered_shortlist_priority(event_row: object, h0: object, h3: object, h6: object) -> str:
    if event_row.status != "event_pattern_reviewable":
        return "still_compromised"
    t_values = [abs(value) for value in [_row_attr(h0, "shock_t_hc1"), _row_attr(h3, "shock_t_hc1"), _row_attr(h6, "shock_t_hc1")] if pd.notna(value)]
    if not t_values:
        return "weak_or_mixed"
    same_direction = _same_lp_direction(h0, h3)
    if max(t_values) >= 2.0 and same_direction:
        return "strong_manual_review"
    if max(t_values) >= 1.0:
        return "manual_review"
    return "weak_or_mixed"


def _filtered_shortlist_next_step(priority: str) -> str:
    if priority == "strong_manual_review":
        return "write_event_narrative_and_check_external_calendar"
    if priority == "manual_review":
        return "inspect_event_months_and_sign_stability"
    if priority == "still_compromised":
        return "do_not_use_without_new_event_filter_or_controls"
    return "keep_as_descriptive_sensitivity_only"


def _row_attr(row: object, attr: str) -> float:
    if row is None:
        return np.nan
    return getattr(row, attr, np.nan)


def _same_lp_direction(left: object, right: object) -> bool:
    left_beta = _row_attr(left, "shock_beta")
    right_beta = _row_attr(right, "shock_beta")
    if pd.isna(left_beta) or pd.isna(right_beta) or left_beta == 0 or right_beta == 0:
        return False
    return bool(np.sign(left_beta) == np.sign(right_beta))


def _direction_pattern(h0: object, h3: object, h6: object) -> str:
    signs = []
    for row in [h0, h3, h6]:
        beta = _row_attr(row, "shock_beta")
        if pd.isna(beta):
            signs.append("missing")
        elif beta > 0:
            signs.append("positive")
        elif beta < 0:
            signs.append("negative")
        else:
            signs.append("zero")
    return "/".join(signs)


def _filtered_candidate_sample(
    panel: pd.DataFrame,
    candidate: object,
    dirty_events: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    sample_columns = _sample_candidate_columns()[candidate.sample]
    sample = panel.loc[_sample_mask(panel, sample_columns)].copy()
    if dirty_events.empty:
        return sample, set()
    rows = dirty_events.loc[
        (dirty_events["sample"] == candidate.sample)
        & (dirty_events["outcome"] == candidate.outcome)
        & (dirty_events["manual_review_priority"] == "high_pre_event_movement")
    ]
    excluded_months = set(rows["event_month"].dropna().astype(str))
    if not excluded_months:
        return sample, excluded_months
    month_key = sample["month"].dt.date.astype(str)
    excluded_mask = month_key.isin(excluded_months)
    if "isolated_large_positive_bill_shock" in sample.columns:
        sample.loc[excluded_mask, "isolated_large_positive_bill_shock"] = False
    if "large_positive_bill_shock" in sample.columns:
        sample.loc[excluded_mask, "large_positive_bill_shock"] = False
    if "bill_supply_shock_resid_100b" in sample.columns:
        sample.loc[excluded_mask, "bill_supply_shock_resid_100b"] = np.nan
    if "bill_supply_shock_z" in sample.columns:
        sample.loc[excluded_mask, "bill_supply_shock_z"] = np.nan
    return sample, excluded_months


def _first_non_null(group: pd.DataFrame, column: str) -> object:
    if column not in group.columns:
        return pd.NA
    values = group[column].dropna()
    return values.iloc[0] if not values.empty else pd.NA


def _event_month_priority(pre_to_reference: float, impact_sum: float) -> str:
    if pd.isna(impact_sum):
        return "missing_impact"
    if pd.notna(pre_to_reference) and pre_to_reference > 0.75:
        return "high_pre_event_movement"
    if pd.notna(pre_to_reference) and pre_to_reference > 0.35:
        return "material_pre_event_movement"
    return "reviewable"


def _candidate_window_summary(event_rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for window, group in event_rows.groupby("window", observed=True):
        summary[str(window)] = {
            "n_events": float(group["event_month"].nunique()),
            "mean_sum_change": float(group["sum_change"].mean()),
            "median_sum_change": float(group["sum_change"].median()),
        }
    return summary


def _same_sign(left: object, right: object) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    left_value = float(left)
    right_value = float(right)
    if left_value == 0 or right_value == 0:
        return False
    return bool(np.sign(left_value) == np.sign(right_value))


def _opposite_sign(left: object, right: object) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    left_value = float(left)
    right_value = float(right)
    if left_value == 0 or right_value == 0:
        return False
    return bool(np.sign(left_value) != np.sign(right_value))


def _event_review_status(
    *,
    pre_to_reference: float,
    same_sign_pre_impact: bool,
    n_events: int,
) -> tuple[str, str]:
    if n_events < 5:
        return "too_few_events", "use_broader_sample_or_lower_frequency_design"
    if pd.isna(pre_to_reference):
        return "manual_review", "inspect_missing_pre_or_post_event_windows"
    if pre_to_reference > 0.75:
        return "pre_event_movement_large", "write_event_narrative_or_find_cleaner_events"
    if same_sign_pre_impact and pre_to_reference > 0.35:
        return "pre_event_movement_material", "add_anticipation_controls_or_narrative_filter"
    return "event_pattern_reviewable", "review_event_months_before_public_use"


def _largest_abs_impact_event_months(event_rows: pd.DataFrame, *, limit: int = 3) -> str:
    impact = event_rows.loc[event_rows["window"] == "impact"].copy()
    if impact.empty:
        return ""
    impact["abs_sum_change"] = impact["sum_change"].abs()
    months = impact.sort_values("abs_sum_change", ascending=False)["event_month"].head(limit)
    return ";".join(str(month) for month in months)


def _sample_use(sample_name: str, n_months: int) -> str:
    if n_months < 36:
        return "too_sparse_for_default_regressions"
    if sample_name == "post_2021_rate_controls":
        return "rate-control diagnostic only; short sample"
    if sample_name == "post_2010_mmf_holdings":
        return "preferred plumbing mechanism sample"
    if sample_name == "post_2005_tga":
        return "preferred TGA-control sample"
    return "broad descriptive sample"


def _ols_hc1(df: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    y = df[outcome].to_numpy(dtype=float)
    x = df[predictors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = x.T @ ((residuals**2)[:, None] * x)
    nobs, k = x.shape
    scale = nobs / max(nobs - k, 1)
    vcov = scale * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residuals**2).sum()) / total if total else 0.0
    return beta, se, r2


def _seasonal_predictors() -> list[str]:
    return [f"month_{month:02d}" for month in range(2, 13)]


def _spec_predictors(spec_name: str, outcome: str) -> list[str]:
    specs = {
        "supply_only": ["gross_bill_issuance_100b", "bill_share"],
        "supply_plus_bill_yield": ["gross_bill_issuance_100b", "bill_share", "bill_yield"],
        "supply_plus_iorb": ["gross_bill_issuance_100b", "bill_share", "iorb_rate"],
        "supply_lagged_outcome": [
            "gross_bill_issuance_100b",
            "bill_share",
            f"lag1_{outcome}",
        ],
        "supply_seasonal": ["gross_bill_issuance_100b", "bill_share", *_seasonal_predictors()],
    }
    return specs[spec_name]


def first_pass_regressions(panel: pd.DataFrame, *, min_nobs: int = 36) -> pd.DataFrame:
    outcomes = OUTCOME_CHANGES
    specs = {
        "supply_only": None,
        "supply_plus_bill_yield": None,
        "supply_plus_iorb": None,
        "supply_lagged_outcome": None,
        "supply_seasonal": None,
    }
    samples: list[tuple[str, pd.DataFrame]] = [("all_available", panel)]
    if "on_rrp_regime_fixed" in panel.columns:
        for regime, group in panel.dropna(subset=["on_rrp_regime_fixed"]).groupby(
            "on_rrp_regime_fixed", observed=True
        ):
            samples.append((f"on_rrp_{regime}", group))

    rows: list[dict[str, object]] = []
    for sample_name, sample in samples:
        for spec_name in specs:
            for outcome in outcomes:
                predictors = _spec_predictors(spec_name, outcome)
                required = [outcome, *predictors]
                if not set(required).issubset(sample.columns):
                    continue
                df = sample[required].dropna()
                if len(df) < min_nobs:
                    rows.append(
                        {
                            "outcome": outcome,
                            "sample": sample_name,
                            "spec": spec_name,
                            "status": "skipped_too_few_observations",
                            "nobs": int(len(df)),
                            "min_nobs": min_nobs,
                        }
                    )
                    continue
                beta, se, r2 = _ols_hc1(df, outcome, predictors)
                for idx, predictor in enumerate(["intercept", *predictors]):
                    rows.append(
                        {
                            "outcome": outcome,
                            "sample": sample_name,
                            "spec": spec_name,
                            "status": "estimated",
                            "predictor": predictor,
                            "nobs": int(len(df)),
                            "beta": float(beta[idx]),
                            "se_hc1": float(se[idx]),
                            "t_hc1": float(beta[idx] / se[idx]) if se[idx] else np.nan,
                            "r2": r2,
                        }
                    )
    return pd.DataFrame(rows)


def pretrend_diagnostics(panel: pd.DataFrame, *, min_nobs: int = 36) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in OUTCOME_CHANGES:
        for pre_col in [f"lag1_{outcome}", f"pre3_{outcome}"]:
            required = [pre_col, "gross_bill_issuance_100b", "bill_share"]
            if not set(required).issubset(panel.columns):
                continue
            df = panel[required].dropna()
            if len(df) < min_nobs:
                rows.append(
                    {
                        "outcome": outcome,
                        "pretrend_measure": pre_col,
                        "status": "skipped_too_few_observations",
                        "nobs": int(len(df)),
                        "min_nobs": min_nobs,
                    }
                )
                continue
            beta, se, r2 = _ols_hc1(df, pre_col, ["gross_bill_issuance_100b", "bill_share"])
            rows.append(
                {
                    "outcome": outcome,
                    "pretrend_measure": pre_col,
                    "status": "estimated",
                    "nobs": int(len(df)),
                    "gross_bill_beta": float(beta[1]),
                    "gross_bill_se_hc1": float(se[1]),
                    "gross_bill_t_hc1": float(beta[1] / se[1]) if se[1] else np.nan,
                    "bill_share_beta": float(beta[2]),
                    "bill_share_se_hc1": float(se[2]),
                    "bill_share_t_hc1": float(beta[2] / se[2]) if se[2] else np.nan,
                    "r2": r2,
                    "interpretation": "nonzero pretrend coefficient flags anticipation or sample-selection risk",
                }
            )
    return pd.DataFrame(rows)


def residual_shock_pretrend_diagnostics(panel: pd.DataFrame, *, min_nobs: int = 36) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    shock_col = "bill_supply_shock_resid_100b"
    for outcome in OUTCOME_CHANGES:
        for pre_col in [f"lag1_{outcome}", f"pre3_{outcome}"]:
            required = [pre_col, shock_col, "bill_share"]
            if not set(required).issubset(panel.columns):
                continue
            df = panel[required].dropna()
            if len(df) < min_nobs:
                rows.append(
                    {
                        "outcome": outcome,
                        "pretrend_measure": pre_col,
                        "status": "skipped_too_few_observations",
                        "nobs": int(len(df)),
                        "min_nobs": min_nobs,
                    }
                )
                continue
            beta, se, r2 = _ols_hc1(df, pre_col, [shock_col, "bill_share"])
            rows.append(
                {
                    "outcome": outcome,
                    "pretrend_measure": pre_col,
                    "status": "estimated",
                    "nobs": int(len(df)),
                    "shock_beta": float(beta[1]),
                    "shock_se_hc1": float(se[1]),
                    "shock_t_hc1": float(beta[1] / se[1]) if se[1] else np.nan,
                    "bill_share_beta": float(beta[2]),
                    "bill_share_se_hc1": float(se[2]),
                    "bill_share_t_hc1": float(beta[2] / se[2]) if se[2] else np.nan,
                    "r2": r2,
                    "interpretation": "residual-shock pretrend balance check; nonzero coefficients weaken causal use",
                }
            )
    return pd.DataFrame(rows)


def write_monthly_analysis(root: Path) -> dict[str, object]:
    as_of = project_as_of_date(root)
    panel = add_analysis_fields(load_monthly_panel(root), as_of_date=as_of)
    analysis_base = estimation_panel(panel)
    table_dir = ensure_dir(root / "output" / "tables")
    report_dir = ensure_dir(root / "output" / "reports")
    derived_path = root / "data" / "clean" / "monthly_liquidity_substitution_panel_analysis.csv"
    panel.to_csv(derived_path, index=False)

    outputs = {
        "analysis_panel": derived_path,
        "source_metadata": table_dir / "monthly_source_metadata.csv",
        "coverage_qa": table_dir / "monthly_coverage_qa.csv",
        "future_row_qa": table_dir / "monthly_future_row_qa.csv",
        "regime_classification_qa": table_dir / "monthly_regime_classification_qa.csv",
        "sample_windows": table_dir / "monthly_sample_windows.csv",
        "descriptive_summary": table_dir / "monthly_descriptive_summary.csv",
        "regime_summary": table_dir / "monthly_regime_summary.csv",
        "regime_threshold_sensitivity": table_dir / "monthly_regime_threshold_sensitivity.csv",
        "sample_recommendations": table_dir / "monthly_sample_recommendations.csv",
        "design_readiness": table_dir / "monthly_design_readiness.csv",
        "candidate_lp_coefficients": table_dir / "monthly_candidate_lp_coefficients.csv",
        "candidate_event_review": table_dir / "monthly_candidate_event_review.csv",
        "candidate_event_months": table_dir / "monthly_candidate_event_months.csv",
        "filtered_candidate_lp_coefficients": table_dir
        / "monthly_filtered_candidate_lp_coefficients.csv",
        "filtered_candidate_event_review": table_dir
        / "monthly_filtered_candidate_event_review.csv",
        "filtered_candidate_shortlist": table_dir / "monthly_filtered_candidate_shortlist.csv",
        "on_rrp_manual_review": table_dir / "monthly_on_rrp_manual_review.csv",
        "strong_candidate_narratives": table_dir / "monthly_strong_candidate_narratives.csv",
        "external_calendar_evidence": table_dir / "monthly_external_calendar_evidence.csv",
        "candidate_table": table_dir / "monthly_candidate_table.csv",
        "claim_readiness": table_dir / "monthly_claim_readiness.csv",
        "readiness_summary": table_dir / "monthly_readiness_summary.csv",
        "correlations": table_dir / "monthly_correlations.csv",
        "bill_shock_events": table_dir / "monthly_bill_shock_events.csv",
        "event_study_by_event": table_dir / "monthly_event_study_by_event.csv",
        "event_study_summary": table_dir / "monthly_event_study_summary.csv",
        "first_pass_regressions": table_dir / "monthly_first_pass_regressions.csv",
        "pretrend_diagnostics": table_dir / "monthly_pretrend_diagnostics.csv",
        "residual_shock_pretrend_diagnostics": table_dir
        / "monthly_residual_shock_pretrend_diagnostics.csv",
        "local_projection_diagnostics": table_dir / "monthly_local_projection_diagnostics.csv",
        "regime_interaction_diagnostics": table_dir
        / "monthly_regime_interaction_diagnostics.csv",
        "evidence_gate_summary": table_dir / "evidence_gate_summary.csv",
        "evidence_gate_report": report_dir / "evidence_gate_summary.md",
        "report": report_dir / "monthly_mvp_report.md",
        "candidate_report": report_dir / "monthly_candidate_narrative_report.md",
        "candidate_review_report": report_dir / "monthly_candidate_review_report.md",
    }
    source_metadata_table(panel).to_csv(outputs["source_metadata"], index=False)
    coverage_qa(panel).to_csv(outputs["coverage_qa"], index=False)
    future_row_qa(panel, as_of_date=as_of).to_csv(outputs["future_row_qa"], index=False)
    regime_classification_qa(panel).to_csv(outputs["regime_classification_qa"], index=False)
    sample_windows(analysis_base).to_csv(outputs["sample_windows"], index=False)
    descriptive_summary(analysis_base).to_csv(outputs["descriptive_summary"], index=False)
    regime_summary(analysis_base).to_csv(outputs["regime_summary"], index=False)
    regime_threshold_sensitivity(analysis_base).to_csv(outputs["regime_threshold_sensitivity"], index=False)
    sample_recommendations(analysis_base).to_csv(outputs["sample_recommendations"], index=False)
    design_readiness_table(analysis_base).to_csv(outputs["design_readiness"], index=False)
    candidate_lp_table(analysis_base).to_csv(outputs["candidate_lp_coefficients"], index=False)
    candidate_event_review_table(analysis_base).to_csv(outputs["candidate_event_review"], index=False)
    candidate_event_month_table(analysis_base).to_csv(outputs["candidate_event_months"], index=False)
    filtered_candidate_lp_table(analysis_base).to_csv(outputs["filtered_candidate_lp_coefficients"], index=False)
    filtered_candidate_event_review_table(analysis_base).to_csv(outputs["filtered_candidate_event_review"], index=False)
    filtered_candidate_shortlist_table(analysis_base).to_csv(outputs["filtered_candidate_shortlist"], index=False)
    on_rrp_manual_review_table(analysis_base).to_csv(outputs["on_rrp_manual_review"], index=False)
    strong_candidate_narrative_table(analysis_base).to_csv(outputs["strong_candidate_narratives"], index=False)
    external_calendar_evidence_table(analysis_base).to_csv(outputs["external_calendar_evidence"], index=False)
    candidate_table(analysis_base).to_csv(outputs["candidate_table"], index=False)
    readiness_summary = claim_readiness_table(analysis_base)
    readiness_summary.to_csv(outputs["claim_readiness"], index=False)
    readiness_summary.to_csv(outputs["readiness_summary"], index=False)
    correlation_table(analysis_base).to_csv(outputs["correlations"], index=False)
    events = bill_shock_events(analysis_base)
    event_rows = event_study_by_event(analysis_base)
    events.to_csv(outputs["bill_shock_events"], index=False)
    event_rows.to_csv(outputs["event_study_by_event"], index=False)
    event_study_summary(event_rows).to_csv(outputs["event_study_summary"], index=False)
    first_pass_regressions(analysis_base).to_csv(outputs["first_pass_regressions"], index=False)
    pretrend_diagnostics(analysis_base).to_csv(outputs["pretrend_diagnostics"], index=False)
    residual_shock_pretrend_diagnostics(analysis_base).to_csv(
        outputs["residual_shock_pretrend_diagnostics"], index=False
    )
    local_projection_diagnostics(analysis_base).to_csv(outputs["local_projection_diagnostics"], index=False)
    regime_interaction_diagnostics(analysis_base).to_csv(
        outputs["regime_interaction_diagnostics"], index=False
    )
    _write_report(analysis_base, outputs)
    write_candidate_narrative_report(analysis_base, outputs)
    write_evidence_gate_summary(root)
    return {key: relative_to_root(root, value) for key, value in outputs.items()}


def _write_report(panel: pd.DataFrame, outputs: dict[str, Path]) -> None:
    min_month = panel["month"].min().date().isoformat()
    max_month = panel["month"].max().date().isoformat()
    regression = _read_csv_or_empty(outputs["first_pass_regressions"])
    lp = _read_csv_or_empty(outputs["local_projection_diagnostics"])
    regime_sensitivity = _read_csv_or_empty(outputs["regime_threshold_sensitivity"])
    regime_interactions = _read_csv_or_empty(outputs["regime_interaction_diagnostics"])
    coverage = _read_csv_or_empty(outputs["coverage_qa"])
    samples = _read_csv_or_empty(outputs["sample_recommendations"])
    design = _read_csv_or_empty(outputs["design_readiness"])
    candidate_lps = _read_csv_or_empty(outputs["candidate_lp_coefficients"])
    candidate_events = _read_csv_or_empty(outputs["candidate_event_review"])
    candidate_event_months = _read_csv_or_empty(outputs["candidate_event_months"])
    filtered_candidate_lps = _read_csv_or_empty(outputs["filtered_candidate_lp_coefficients"])
    filtered_candidate_events = _read_csv_or_empty(outputs["filtered_candidate_event_review"])
    filtered_candidate_shortlist = _read_csv_or_empty(outputs["filtered_candidate_shortlist"])
    on_rrp_manual_review = _read_csv_or_empty(outputs["on_rrp_manual_review"])
    strong_candidate_narratives = _read_csv_or_empty(outputs["strong_candidate_narratives"])
    external_calendar_evidence = _read_csv_or_empty(outputs["external_calendar_evidence"])
    claim_readiness = _read_csv_or_empty(outputs["readiness_summary"])
    pretrends = _read_csv_or_empty(outputs["pretrend_diagnostics"])
    residual_pretrends = _read_csv_or_empty(outputs["residual_shock_pretrend_diagnostics"])
    events = _read_csv_or_empty(outputs["bill_shock_events"])
    event_summary = _read_csv_or_empty(outputs["event_study_summary"])
    estimated = int((regression.get("status") == "estimated").sum()) if not regression.empty else 0
    lp_estimated = int((lp.get("status") == "estimated").sum()) if not lp.empty else 0
    skipped = int((regression.get("status") == "skipped_too_few_observations").sum()) if not regression.empty else 0
    pretrend_flags = 0
    if not pretrends.empty and "gross_bill_t_hc1" in pretrends.columns:
        pretrend_flags = int((pretrends["gross_bill_t_hc1"].abs() >= 2.0).sum())
    residual_pretrend_flags = 0
    if not residual_pretrends.empty and "shock_t_hc1" in residual_pretrends.columns:
        residual_pretrend_flags = int((residual_pretrends["shock_t_hc1"].abs() >= 2.0).sum())
    event_count = len(events)
    interaction_estimated = (
        int((regime_interactions.get("status") == "estimated").sum())
        if not regime_interactions.empty
        else 0
    )
    sparse_columns = 0
    if not coverage.empty and "coverage_share" in coverage.columns:
        sparse_columns = int(((coverage["status"] == "ok") & (coverage["coverage_share"] < 0.5)).sum())
    candidate_designs = 0
    diagnostic_designs = 0
    if not design.empty and "status" in design.columns:
        candidate_designs = int((design["status"] == "candidate_design").sum())
        diagnostic_designs = int((design["status"] == "diagnostic_only").sum())
    candidate_lp_rows = len(candidate_lps)
    reviewable_event_rows = 0
    if not candidate_events.empty and "status" in candidate_events.columns:
        reviewable_event_rows = int((candidate_events["status"] == "event_pattern_reviewable").sum())
    high_pre_event_month_rows = 0
    if not candidate_event_months.empty and "manual_review_priority" in candidate_event_months.columns:
        high_pre_event_month_rows = int(
            (candidate_event_months["manual_review_priority"] == "high_pre_event_movement").sum()
        )
    filtered_lp_rows = 0
    filtered_reviewable_events = 0
    if not filtered_candidate_lps.empty and "status" in filtered_candidate_lps.columns:
        filtered_lp_rows = int((filtered_candidate_lps["status"] == "estimated").sum())
    if not filtered_candidate_events.empty and "status" in filtered_candidate_events.columns:
        filtered_reviewable_events = int(
            (filtered_candidate_events["status"] == "event_pattern_reviewable").sum()
        )
    strong_shortlist_rows = 0
    if not filtered_candidate_shortlist.empty and "priority" in filtered_candidate_shortlist.columns:
        strong_shortlist_rows = int(
            (filtered_candidate_shortlist["priority"] == "strong_manual_review").sum()
        )
    on_rrp_manual_review_rows = len(on_rrp_manual_review)
    strong_narrative_rows = len(strong_candidate_narratives)
    external_evidence_rows = len(external_calendar_evidence)
    claim_readiness_rows = len(claim_readiness)
    claim_ready_rows = 0
    claim_blocked_rows = 0
    if not claim_readiness.empty and "readiness_status" in claim_readiness.columns:
        claim_ready_rows = int(
            (claim_readiness["readiness_status"] == "descriptive_ready_only").sum()
        )
        claim_blocked_rows = int((claim_readiness["readiness_status"] == "blocked").sum())
    threshold_lines = [
        (
            f"- {row.threshold_spec} {row.regime}: {int(row.n_months)} months, "
            f"{int(row.n_isolated_bill_shock_events)} isolated shock events"
        )
        for row in regime_sensitivity.itertuples(index=False)
    ]
    pre_event_warning = ""
    if not event_summary.empty:
        pre_rows = event_summary.loc[event_summary["window"] == "pre3", "mean_sum_change"].abs()
        if not pre_rows.empty and (pre_rows > 0).any():
            pre_event_warning = (
                "- Event windows also show nonzero pre-event channel movement; use them as "
                "diagnostics, not causal event-study estimates."
            )
    sample_lines = [
        f"- {row.sample}: {int(row.n_months)} months, {row.first_month} to {row.last_month}; {row.recommended_use}"
        for row in samples.itertuples(index=False)
    ]
    text = "\n".join(
        [
            "# Monthly MVP Report",
            "",
            f"- Panel rows: {len(panel)}",
            f"- Panel range: {min_month} to {max_month}",
            f"- Estimated regression rows: {estimated}",
            f"- Estimated local-projection diagnostic rows: {lp_estimated}",
            f"- Skipped regression rows for sparse samples: {skipped}",
            f"- Pretrend diagnostics with |t| >= 2 on gross bill issuance: {pretrend_flags}",
            f"- Residual-shock pretrend diagnostics with |t| >= 2: {residual_pretrend_flags}",
            f"- Isolated large residual bill-supply events: {event_count}",
            f"- Estimated regime-interaction diagnostic rows: {interaction_estimated}",
            f"- QA coverage rows with less than 50% panel coverage: {sparse_columns}",
            f"- Candidate design rows after readiness checks: {candidate_designs}",
            f"- Diagnostic-only design rows after readiness checks: {diagnostic_designs}",
            f"- Candidate LP coefficient rows for review: {candidate_lp_rows}",
            f"- Candidate event rows with reviewable event patterns: {reviewable_event_rows}",
            f"- Candidate event-month rows with high pre-event movement: {high_pre_event_month_rows}",
            f"- Filtered candidate LP coefficient rows: {filtered_lp_rows}",
            f"- Filtered candidate event rows with reviewable patterns: {filtered_reviewable_events}",
            f"- Strong filtered shortlist rows for manual review: {strong_shortlist_rows}",
            f"- ON RRP manual-review rows kept as context only: {on_rrp_manual_review_rows}",
            f"- Strong candidate narrative rows needing calendar validation: {strong_narrative_rows}",
            f"- External calendar evidence rows attached: {external_evidence_rows}",
            f"- Interpretation-readiness rows: {claim_readiness_rows}",
            f"- Descriptive-ready-only rows: {claim_ready_rows}",
            f"- Blocked interpretation rows: {claim_blocked_rows}",
            "",
            "Recommended samples:",
            "",
            *sample_lines,
            "",
            "ON RRP threshold sensitivity:",
            "",
            *threshold_lines,
            "",
            "Interpretation guardrails:",
            "",
            "- These first-pass regressions are descriptive diagnostics, not causal estimates.",
            "- Local-projection diagnostics use residual bill-supply shocks but still require a stronger identification design.",
            "- `supply_only` keeps a broader sample; `supply_plus_iorb` is mainly a post-2021 rate-control diagnostic.",
            "- Pretrend flags mean bill-supply months often occur after related liquidity-channel movement; this must be handled before causal interpretation.",
            "- Residual-shock pretrend checks are the cleaner diagnostic for event and LP designs.",
            pre_event_warning,
            "- ON RRP regime splits must be reviewed before interpreting bill-supply responses.",
            "- Abundant and transition regimes are short under every threshold variant currently tested.",
            "- Use the sample-window and QA tables before comparing columns with different coverage.",
            "",
        ]
    )
    outputs["report"].write_text(text, encoding="utf-8")
