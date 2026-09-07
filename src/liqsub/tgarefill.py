from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from liqsub.config import weekly_large_rebuild_randomization_p_max
from liqsub.paths import ensure_dir, relative_to_root


LARGE_REBUILD_DESIGN = "rebuild_ge_200b_main_no_debtlimit"
LARGE_REBUILD_THRESHOLD = 200_000.0
LARGE_REBUILD_MAIN_TAUS = (0, 2, 4)
LARGE_REBUILD_SECONDARY_TAUS = (1,)
LARGE_REBUILD_APPENDIX_TAUS = (8,)
EVENT_STUDY_OUTCOMES = ["deposits", "on_rrp", "reserves", "mmf_treasury_holdings"]
TERMINAL_COMPLETE_MIN_NON_NULL_SHARE = 0.90

LARGE_REBUILD_ROSTER_COLUMNS = [
    "targeted_design",
    "event_id",
    "event_start",
    "event_end",
    "delta_tga_event",
    "mean_bill_share",
    "manual_tags",
    "sample_role",
    "exclusion_reason",
    "debt_limit_confounded",
    "on_rrp_tau8_sign_reversal",
    "official_source",
    "source_url",
    "calendar_note",
    "mechanism_implication",
]

LARGE_REBUILD_CELL_SUMMARY_COLUMNS = [
    "targeted_design",
    "table_role",
    "outcome",
    "tau",
    "primary_cell",
    "event_count",
    "event_ids",
    "mean_change",
    "median_change",
    "same_sign_event_share",
    "dominant_sign",
    "leave_one_out_same_sign_share",
    "bootstrap_same_sign_share",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "placebo_false_positive_share",
    "pseudo_events_n",
    "matched_event_groups_n",
    "pseudo_mean_change",
    "pseudo_abs_p90",
    "worst_match_tier",
    "match_quality_status",
    "match_quality_blocker_events",
    "randomization_p_value",
    "randomization_p_max",
    "randomization_status",
    "claim_use",
    "status",
    "blocker_flags",
    "interpretation",
]

LARGE_REBUILD_EVENT_SIGN_COLUMNS = [
    "targeted_design",
    "event_id",
    "event_start",
    "sample_role",
    "outcome",
    "tau",
    "table_role",
    "level_change_from_baseline",
    "sign",
    "same_sign_as_cell_mean",
    "cell_mean_change",
    "contribution_share_of_abs_sum",
    "debt_limit_confounded",
    "on_rrp_tau8_sign_reversal",
    "claim_use",
]

LARGE_REBUILD_ABNORMAL_COLUMNS = [
    "targeted_design",
    "event_id",
    "event_start",
    "outcome",
    "tau",
    "table_role",
    "actual_change",
    "matched_pseudo_mean_change",
    "abnormal_change",
    "matched_pseudo_n",
    "match_tiers",
    "min_state_distance",
    "median_state_distance",
    "pseudo_abs_p90",
    "cell_actual_mean",
    "cell_matched_pseudo_mean",
    "cell_abnormal_mean",
    "claim_use",
]

LARGE_REBUILD_MATCH_QUALITY_COLUMNS = [
    "targeted_design",
    "event_id",
    "event_start",
    "outcome",
    "matched_pseudo_n",
    "match_tiers",
    "worst_match_tier",
    "min_state_distance",
    "median_state_distance",
    "max_state_distance",
    "min_week_distance",
    "median_week_distance",
    "max_week_distance",
    "status",
    "interpretation",
]

LARGE_REBUILD_RANDOMIZATION_COLUMNS = [
    "targeted_design",
    "outcome",
    "tau",
    "event_count",
    "matched_event_groups_n",
    "draws",
    "actual_mean_change",
    "null_mean_change",
    "null_p05",
    "null_p50",
    "null_p95",
    "two_sided_p_value",
    "placebo_design",
    "status",
    "interpretation",
]

LARGE_REBUILD_FINAL_REVIEW_COLUMNS = [
    "targeted_design",
    "review_scope",
    "status",
    "claim_use",
    "main_event_count",
    "headline_passing_cells",
    "headline_blocked_cells",
    "headline_cells",
    "appendix_cells",
    "binding_limitations",
    "required_next_step",
]

LARGE_REBUILD_BLOCKER_SUMMARY_COLUMNS = [
    "targeted_design",
    "table_role",
    "outcome",
    "tau",
    "status",
    "blocker_count",
    "primary_blocker",
    "blocker_flags",
    "randomization_p_value",
    "randomization_p_max",
    "match_quality_status",
    "next_step",
]

LARGE_REBUILD_EVENT_STUDY_COLUMNS = [
    "event_id",
    "matched_event_id",
    "event_start",
    "event_week",
    "tau",
    "outcome",
    "level",
    "baseline_level",
    "baseline_window",
    "baseline_non_null",
    "baseline_method",
    "level_change_from_baseline",
    "delta_tga_event",
    "mean_bill_share",
    "manual_tags",
]

TGAREFILL_EXPORTS = {
    "master_weekly_panel": {
        "path": Path("data/raw/tgarefill/master_weekly_panel.csv"),
        "required": [
            "date",
            "tga_weekly_average",
            "reserve_balances_weekly_wednesday",
            "reverse_repos_weekly_wednesday",
            "commercial_bank_deposits_weekly_sa",
            "bill",
            "coupon",
            "bill_share",
        ],
        "date_columns": ["date"],
        "nonnegative_columns": [
            "tga_weekly_average",
            "reserve_balances_weekly_wednesday",
            "reverse_repos_weekly_wednesday",
            "commercial_bank_deposits_weekly_sa",
            "bill",
            "coupon",
        ],
        "share_columns": ["bill_share"],
        "unique_columns": ["date"],
    },
    "event_candidates": {
        "path": Path("data/raw/tgarefill/event_candidates.csv"),
        "required": [
            "event_id",
            "baseline_date",
            "start_date",
            "end_date",
            "delta_tga_event",
            "manual_tags",
        ],
        "date_columns": ["baseline_date", "start_date", "end_date"],
        "nonnegative_columns": ["delta_tga_event"],
        "unique_columns": ["event_id"],
    },
    "auction_shock_lp": {
        "path": Path("data/raw/tgarefill/auction_shock_lp.csv"),
        "required": ["response_var", "horizon", "beta", "se_nw", "t_stat_nw", "significant_5pct", "n_obs", "regime", "shock_spec", "sample", "timing", "canonical_sample_end"],
        "numeric_columns": ["horizon", "beta", "se_nw", "t_stat_nw"],
    },
    "canonical_bill_surprise_shocks": {
        "path": Path("data/raw/tgarefill/canonical_bill_surprise_shocks.csv"),
        "required": [
            "date",
            "bill_size_surprise",
            "bill_size_surprise_announcement_week",
            "tax_receipt_surprise",
            "rapid_rebuild_flag",
            "sample",
            "canonical_sample_end",
        ],
        "date_columns": ["date"],
        "numeric_columns": [
            "bill_size_surprise",
            "bill_size_surprise_announcement_week",
            "tax_receipt_surprise",
            "rapid_rebuild_flag",
        ],
        "unique_columns": ["date"],
    },
    "promotion_robustness_summary": {
        "path": Path("data/raw/tgarefill/promotion_robustness_summary.csv"),
        "required": [
            "spec",
            "response_var",
            "response_label",
            "shock_sd_bn",
            "h4_effect_bn",
            "h4_t_stat_nw",
            "h4_significant_5pct",
        ],
        "numeric_columns": [
            "shock_sd_bn",
            "placebo_significant_count",
            "all_channel_placebo_significant_count",
            "h4_effect_bn",
            "h4_t_stat_nw",
        ],
    },
    "mmfalloc_downstream_summary": {
        "path": Path("data/raw/tgarefill/mmfalloc_downstream_summary.csv"),
        "required": ["metric", "value", "unit", "note"],
    },
    "mmfalloc_source_gates": {
        "path": Path("data/raw/tgarefill/mmfalloc_source_gates.csv"),
        "required": ["gate", "passed", "value", "threshold", "note"],
    },
    "mmfalloc_baseline": {
        "path": Path("data/raw/tgarefill/mmfalloc_baseline.csv"),
        "required": [
            "event_month",
            "horizon",
            "category",
            "delta_millions",
            "event_bill_surprise_millions",
            "delta_per_bill_surprise",
        ],
        "date_columns": ["event_month"],
        "numeric_columns": [
            "delta_millions",
            "event_bill_surprise_millions",
            "delta_per_bill_surprise",
        ],
    },
}


@dataclass(frozen=True)
class TgarefillValidation:
    status: str
    errors: tuple[str, ...]
    rows_by_export: dict[str, int]


def validate_tgarefill_exports(root: Path) -> TgarefillValidation:
    errors: list[str] = []
    rows_by_export: dict[str, int] = {}
    for name, spec in TGAREFILL_EXPORTS.items():
        path = root / spec["path"]
        if not path.exists():
            errors.append(f"missing tgarefill export {name}: {path}")
            rows_by_export[name] = 0
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            errors.append(f"{path}: file is empty")
            rows_by_export[name] = 0
            continue
        missing = [column for column in spec["required"] if column not in df.columns]
        if missing:
            errors.append(f"{path}: missing required columns: {', '.join(missing)}")
        if df.empty:
            errors.append(f"{path}: no rows")
        errors.extend(_validate_tgarefill_dates(path, df, spec))
        errors.extend(_validate_tgarefill_numeric_columns(path, df, spec))
        if name == "event_candidates":
            errors.extend(_validate_tgarefill_event_dates(path, df))
        rows_by_export[name] = int(len(df))
    return TgarefillValidation(
        status="ok" if not errors else "failed",
        errors=tuple(errors),
        rows_by_export=rows_by_export,
    )


def _validate_tgarefill_dates(path: Path, df: pd.DataFrame, spec: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for column in spec.get("date_columns", []):
        if column not in df.columns:
            continue
        dates = pd.to_datetime(df[column], errors="coerce")
        if dates.isna().any():
            errors.append(f"{path}: {column} contains unparsable dates")
        if column == "date" and not dates.is_monotonic_increasing:
            errors.append(f"{path}: {column} is not sorted ascending")
    unique_columns = list(spec.get("unique_columns", []))
    for column in unique_columns:
        if column in df.columns and df[column].duplicated().any():
            errors.append(f"{path}: duplicate values in {column}")
    return errors


def _validate_tgarefill_numeric_columns(path: Path, df: pd.DataFrame, spec: dict[str, object]) -> list[str]:
    errors: list[str] = []
    numeric_columns = set(spec.get("numeric_columns", []))
    numeric_columns.update(spec.get("nonnegative_columns", []))
    numeric_columns.update(spec.get("share_columns", []))
    for column in sorted(numeric_columns):
        if column not in df.columns:
            continue
        nonblank = df[column].dropna().astype(str).str.strip()
        nonblank = nonblank.loc[nonblank != ""]
        values = pd.to_numeric(nonblank, errors="coerce")
        if values.isna().any():
            errors.append(f"{path}: {column} contains non-numeric values")
            continue
        if values.empty:
            errors.append(f"{path}: {column} has no numeric values")
            continue
        if column in spec.get("nonnegative_columns", []) and (values < 0).any():
            errors.append(f"{path}: {column} contains negative values")
        if column in spec.get("share_columns", []) and ((values < 0) | (values > 1)).any():
            errors.append(f"{path}: {column} contains values outside [0, 1]")
    return errors


def _validate_tgarefill_event_dates(path: Path, df: pd.DataFrame) -> list[str]:
    required = {"baseline_date", "start_date", "end_date"}
    if not required.issubset(df.columns):
        return []
    dates = df[list(required)].apply(pd.to_datetime, errors="coerce")
    errors: list[str] = []
    valid = dates.notna().all(axis=1)
    if valid.any() and (dates.loc[valid, "start_date"] < dates.loc[valid, "baseline_date"]).any():
        errors.append(f"{path}: start_date precedes baseline_date")
    if valid.any() and (dates.loc[valid, "end_date"] < dates.loc[valid, "start_date"]).any():
        errors.append(f"{path}: end_date precedes start_date")
    return errors


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
    mask = pd.Series(False, index=frame.index)
    for idx in dates.sort_values(ascending=False).index:
        if pd.isna(dates.loc[idx]):
            continue
        if shares.loc[idx] >= min_non_null_share:
            break
        mask.loc[idx] = True
    return mask


def add_weekly_terminal_completeness_fields(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "week" not in out.columns:
        return out
    out["terminal_non_null_share"] = _row_non_null_share(out, date_column="week")
    out["is_terminal_incomplete"] = _terminal_incomplete_mask(out, date_column="week")
    out["terminal_completeness_status"] = "complete_or_historical"
    out.loc[out["is_terminal_incomplete"], "terminal_completeness_status"] = "terminal_incomplete"
    out["baseline_estimation_use"] = "include"
    out.loc[out["is_terminal_incomplete"], "baseline_estimation_use"] = "exclude_from_causal_baseline"
    return out


def weekly_estimation_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if "baseline_estimation_use" not in panel.columns:
        panel = add_weekly_terminal_completeness_fields(panel)
    return panel.loc[panel["baseline_estimation_use"] == "include"].copy()


def weekly_terminal_period_qa(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "week",
        "terminal_incomplete",
        "terminal_non_null_share",
        "non_null_columns",
        "baseline_estimation_use",
    ]
    if "week" not in panel.columns:
        return pd.DataFrame(columns=columns)
    working = panel.copy()
    if "terminal_non_null_share" not in working.columns:
        working = add_weekly_terminal_completeness_fields(working)
    rows = working.loc[working["is_terminal_incomplete"]].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        [
            {
                "week": row.week.date().isoformat(),
                "terminal_incomplete": bool(row.is_terminal_incomplete),
                "terminal_non_null_share": float(row.terminal_non_null_share),
                "non_null_columns": int(
                    pd.Series(row._asdict()).drop(labels=["Index"], errors="ignore").notna().sum()
                ),
                "baseline_estimation_use": str(row.baseline_estimation_use),
            }
            for row in rows.itertuples()
        ],
        columns=columns,
    )


def load_weekly_tgarefill_panel(root: Path) -> pd.DataFrame:
    path = root / TGAREFILL_EXPORTS["master_weekly_panel"]["path"]
    if not path.exists():
        return pd.DataFrame(columns=["week"])
    df = pd.read_csv(path)
    df["week"] = pd.to_datetime(df["date"], errors="coerce")
    out = pd.DataFrame(
        {
            "week": df["week"],
            "tga": _numeric(df, "tga_weekly_average"),
            "reserves": _numeric(df, "reserve_balances_weekly_wednesday"),
            "on_rrp": _numeric(df, "reverse_repos_weekly_wednesday"),
            "deposits": _numeric(df, "commercial_bank_deposits_weekly_sa"),
            "gross_bill_settlement_100b": _numeric(df, "bill") / 100_000_000_000.0,
            "coupon_settlement_100b": _numeric(df, "coupon") / 100_000_000_000.0,
            "bill_share": _numeric(df, "bill_share"),
            "mmf_treasury_holdings": _numeric(df, "mmf_treasury_holdings"),
            "mmf_repo_total": _numeric(df, "mmf_repo_total"),
        }
    )
    for column in ["tga", "reserves", "on_rrp", "deposits", "mmf_treasury_holdings", "mmf_repo_total"]:
        out[f"d_{column}"] = out[column].diff()
    out["bill_size_surprise_100b"] = _bill_surprise(out)
    out["rapid_tga_rebuild_event"] = False
    out["debt_limit_window"] = False
    out["covid_crisis_window"] = out["week"].between("2020-03-01", "2020-06-30")
    out["bank_stress_window"] = out["week"].between("2023-03-01", "2023-05-31")
    out["tax_week"] = out["week"].dt.month.isin([4, 6, 9, 12]) & out["week"].dt.day.between(10, 20)
    out["on_rrp_regime_pre"] = _weekly_on_rrp_regime(out["on_rrp"].shift(1))
    out["high_rate_regime_pre"] = _high_rate_placeholder(out)
    out = _attach_event_flags(root, out.sort_values("week").reset_index(drop=True))
    return add_weekly_terminal_completeness_fields(out)


def write_weekly_outputs(root: Path) -> dict[str, object]:
    validation = validate_tgarefill_exports(root)
    if validation.status != "ok":
        return {
            "status": validation.status,
            "errors": list(validation.errors),
            "rows_by_export": validation.rows_by_export,
        }
    clean_dir = ensure_dir(root / "data" / "clean")
    panel = load_weekly_tgarefill_panel(root)
    output = clean_dir / "weekly_liquidity_substitution_panel.csv"
    panel.to_csv(output, index=False)
    return {
        "status": "ok",
        "rows": int(len(panel)),
        "first_week": panel["week"].min().date().isoformat() if not panel.empty else "",
        "last_week": panel["week"].max().date().isoformat() if not panel.empty else "",
        "output": relative_to_root(root, output),
    }


def write_weekly_analysis(root: Path) -> dict[str, object]:
    panel_path = root / "data" / "clean" / "weekly_liquidity_substitution_panel.csv"
    if panel_path.exists():
        panel = pd.read_csv(panel_path, parse_dates=["week"])
    else:
        panel = load_weekly_tgarefill_panel(root)
    table_dir = ensure_dir(root / "output" / "tables")
    report_dir = ensure_dir(root / "output" / "reports")
    outputs = {
        "upstream_qa": table_dir / "weekly_upstream_tgarefill_qa.csv",
        "terminal_period_qa": table_dir / "weekly_terminal_period_qa.csv",
        "timing_alignment_qa": table_dir / "weekly_timing_alignment_qa.csv",
        "event_candidates_clean": table_dir / "weekly_event_candidates_clean.csv",
        "event_exclusion_log": table_dir / "weekly_event_exclusion_log.csv",
        "pretrend_balance": table_dir / "weekly_pretrend_balance.csv",
        "lp_coefficients": table_dir / "weekly_lp_coefficients.csv",
        "event_study_by_event": table_dir / "weekly_event_study_by_event.csv",
        "event_study_summary": table_dir / "weekly_event_study_summary.csv",
        "leave_one_event_out": table_dir / "weekly_leave_one_event_out.csv",
        "block_bootstrap": table_dir / "weekly_block_bootstrap.csv",
        "placebo_tests": table_dir / "weekly_placebo_tests.csv",
        "filter_sensitivity": table_dir / "weekly_event_filter_sensitivity.csv",
        "large_rebuild_review": table_dir / "weekly_large_rebuild_targeted_review.csv",
        "large_rebuild_calendar_validation": table_dir
        / "weekly_large_rebuild_calendar_validation.csv",
        "large_rebuild_event_roster": table_dir / "weekly_large_rebuild_event_roster.csv",
        "large_rebuild_cell_summary": table_dir / "weekly_large_rebuild_cell_summary.csv",
        "large_rebuild_event_sign_stability": table_dir
        / "weekly_large_rebuild_event_sign_stability.csv",
        "large_rebuild_abnormal_changes": table_dir
        / "weekly_large_rebuild_abnormal_changes.csv",
        "large_rebuild_match_quality": table_dir
        / "weekly_large_rebuild_match_quality.csv",
        "large_rebuild_randomization_inference": table_dir
        / "weekly_large_rebuild_randomization_inference.csv",
        "large_rebuild_final_review": table_dir / "weekly_large_rebuild_final_review.csv",
        "large_rebuild_blocker_summary": table_dir
        / "weekly_large_rebuild_blocker_summary.csv",
        "tgarefill_promotion_reconciliation": table_dir
        / "tgarefill_promotion_reconciliation.csv",
        "tgarefill_promotion_reconciliation_report": report_dir
        / "tgarefill_promotion_reconciliation.md",
        "large_rebuild_reserves_plot": report_dir / "weekly_large_rebuild_reserves_event_time.svg",
        "large_rebuild_on_rrp_plot": report_dir / "weekly_large_rebuild_on_rrp_event_time.svg",
        "large_rebuild_diagnostic_writeup": report_dir
        / "weekly_large_rebuild_diagnostic_writeup.md",
        "large_rebuild_diagnostic_report": report_dir
        / "weekly_large_rebuild_diagnostic_report.md",
        "evidence_gate_summary": table_dir / "evidence_gate_summary.csv",
        "evidence_gate_report": report_dir / "evidence_gate_summary.md",
        "stable_claim_candidates": table_dir / "weekly_stable_claim_candidates.csv",
        "stability_candidates": table_dir / "weekly_stability_candidates.csv",
        "outcome_claim_readiness": table_dir / "weekly_outcome_claim_readiness.csv",
        "outcome_readiness": table_dir / "weekly_outcome_readiness.csv",
        "claim_readiness": table_dir / "weekly_claim_readiness.csv",
        "design_readiness": table_dir / "weekly_design_readiness.csv",
        "report": report_dir / "weekly_identification_candidate_report.md",
    }
    weekly_upstream_qa(root, panel).to_csv(outputs["upstream_qa"], index=False)
    weekly_terminal_period_qa(panel).to_csv(outputs["terminal_period_qa"], index=False)
    analysis_panel = weekly_estimation_panel(panel)
    weekly_timing_alignment_qa(analysis_panel).to_csv(outputs["timing_alignment_qa"], index=False)
    clean, exclusions = weekly_event_candidates_clean(root, analysis_panel)
    clean.to_csv(outputs["event_candidates_clean"], index=False)
    exclusions.to_csv(outputs["event_exclusion_log"], index=False)
    weekly_pretrend_balance(analysis_panel, clean).to_csv(outputs["pretrend_balance"], index=False)
    lp = weekly_lp_coefficients(analysis_panel)
    lp.to_csv(outputs["lp_coefficients"], index=False)
    event_study = weekly_event_study_by_event(analysis_panel, clean)
    event_study.to_csv(outputs["event_study_by_event"], index=False)
    weekly_event_study_summary(event_study).to_csv(outputs["event_study_summary"], index=False)
    loo = weekly_leave_one_event_out(event_study)
    loo.to_csv(outputs["leave_one_event_out"], index=False)
    bootstrap = weekly_block_bootstrap(event_study)
    bootstrap.to_csv(outputs["block_bootstrap"], index=False)
    placebo = weekly_placebo_tests(analysis_panel, clean, event_study)
    placebo.to_csv(outputs["placebo_tests"], index=False)
    filter_sensitivity = weekly_event_filter_sensitivity(analysis_panel, clean)
    filter_sensitivity.to_csv(outputs["filter_sensitivity"], index=False)
    large_rebuild_review = weekly_large_rebuild_targeted_review(analysis_panel, clean)
    large_rebuild_review.to_csv(outputs["large_rebuild_review"], index=False)
    large_rebuild_calendar = weekly_large_rebuild_calendar_validation(root, large_rebuild_review)
    large_rebuild_calendar.to_csv(outputs["large_rebuild_calendar_validation"], index=False)
    large_rebuild_roster = weekly_large_rebuild_event_roster(root, clean)
    large_rebuild_roster.to_csv(outputs["large_rebuild_event_roster"], index=False)
    large_rebuild_cell_summary = weekly_large_rebuild_cell_summary(root, analysis_panel, clean)
    large_rebuild_cell_summary.to_csv(outputs["large_rebuild_cell_summary"], index=False)
    large_rebuild_event_sign = weekly_large_rebuild_event_sign_stability(root, analysis_panel, clean)
    large_rebuild_event_sign.to_csv(outputs["large_rebuild_event_sign_stability"], index=False)
    large_rebuild_abnormal = weekly_large_rebuild_abnormal_changes(root, analysis_panel, clean)
    large_rebuild_abnormal.to_csv(outputs["large_rebuild_abnormal_changes"], index=False)
    large_rebuild_match_quality = weekly_large_rebuild_match_quality(root, analysis_panel, clean)
    large_rebuild_match_quality.to_csv(outputs["large_rebuild_match_quality"], index=False)
    large_rebuild_randomization = weekly_large_rebuild_randomization_inference(root, analysis_panel, clean)
    large_rebuild_randomization.to_csv(
        outputs["large_rebuild_randomization_inference"], index=False
    )
    large_rebuild_final = weekly_large_rebuild_final_review(
        large_rebuild_roster,
        large_rebuild_cell_summary,
        large_rebuild_event_sign,
        large_rebuild_abnormal,
        large_rebuild_match_quality,
    )
    large_rebuild_final.to_csv(outputs["large_rebuild_final_review"], index=False)
    large_rebuild_blockers = weekly_large_rebuild_blocker_summary(large_rebuild_cell_summary)
    large_rebuild_blockers.to_csv(outputs["large_rebuild_blocker_summary"], index=False)
    promotion_reconciliation = tgarefill_promotion_reconciliation(root)
    promotion_reconciliation.to_csv(outputs["tgarefill_promotion_reconciliation"], index=False)
    write_tgarefill_promotion_reconciliation_report(
        promotion_reconciliation,
        path=outputs["tgarefill_promotion_reconciliation_report"],
    )
    weekly_large_rebuild_event_time_svg(
        root,
        analysis_panel,
        clean,
        outcome="reserves",
        path=outputs["large_rebuild_reserves_plot"],
    )
    weekly_large_rebuild_event_time_svg(
        root,
        analysis_panel,
        clean,
        outcome="on_rrp",
        path=outputs["large_rebuild_on_rrp_plot"],
    )
    weekly_large_rebuild_diagnostic_writeup(
        large_rebuild_roster,
        large_rebuild_cell_summary,
        large_rebuild_event_sign,
        large_rebuild_abnormal,
        large_rebuild_final,
        path=outputs["large_rebuild_diagnostic_writeup"],
    )
    weekly_large_rebuild_diagnostic_writeup(
        large_rebuild_roster,
        large_rebuild_cell_summary,
        large_rebuild_event_sign,
        large_rebuild_abnormal,
        large_rebuild_final,
        path=outputs["large_rebuild_diagnostic_report"],
    )
    stable = weekly_stable_claim_candidates(loo, bootstrap, placebo)
    stable.to_csv(outputs["stable_claim_candidates"], index=False)
    stable.to_csv(outputs["stability_candidates"], index=False)
    outcome_readiness = weekly_outcome_claim_readiness(analysis_panel, clean, stable)
    outcome_readiness.to_csv(outputs["outcome_claim_readiness"], index=False)
    outcome_readiness.to_csv(outputs["outcome_readiness"], index=False)
    readiness = weekly_claim_readiness(analysis_panel, clean, lp, event_study, loo, bootstrap, placebo)
    readiness.to_csv(outputs["claim_readiness"], index=False)
    readiness.to_csv(outputs["design_readiness"], index=False)
    from liqsub.analysis import write_evidence_gate_summary

    write_evidence_gate_summary(root)
    _write_weekly_report(
        analysis_panel,
        clean,
        exclusions,
        lp,
        event_study,
        loo,
        bootstrap,
        placebo,
        filter_sensitivity,
        large_rebuild_review,
        large_rebuild_calendar,
        stable,
        outcome_readiness,
        readiness,
        outputs["report"],
    )
    return {key: relative_to_root(root, value) for key, value in outputs.items()}


def weekly_upstream_qa(root: Path, panel: pd.DataFrame) -> pd.DataFrame:
    validation = validate_tgarefill_exports(root)
    rows = [
        {
            "export": name,
            "status": "present" if rows else "missing",
            "rows": rows,
        }
        for name, rows in validation.rows_by_export.items()
    ]
    rows.append(
        {
            "export": "weekly_liquidity_substitution_panel",
            "status": "built" if not panel.empty else "empty",
            "rows": int(len(panel)),
        }
    )
    return pd.DataFrame(rows)


def _observed_bool(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


PROMOTION_CHANNELS = {
    "MMF Treasury Holdings": "mmf_treasury_holdings",
    "ON RRP": "on_rrp_daily_total",
    "Bank Deposits": "commercial_bank_deposits_weekly_nsa",
    "Reserves": "reserve_balances_weekly_wednesday",
}
PROMOTION_SPECS = {
    "canonical_issue_week": ("bill_surprise", "canonical_pre_rmp_through_2025_11", "issue_week", "bill_size_surprise"),
    "same_week_announcement_timing": ("bill_surprise_announcement_week", "same_week_timing_canonical_pre_rmp", "announcement_week", "bill_size_surprise_announcement_week"),
}
# Upstream summaries round effects (billions), t statistics and shock SDs to 4 decimals.
SUMMARY_ROUNDING_ATOL = 0.000051


def _promotion_authority(summary: pd.DataFrame, lp: pd.DataFrame, shocks: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    """Reconcile copied summaries to named LP rows and observed shock scales."""
    required_lp = {"response_var", "horizon", "beta", "se_nw", "t_stat_nw", "significant_5pct", "n_obs", "regime", "shock_spec", "sample", "timing", "canonical_sample_end"}
    required_shocks = {"date", "sample", "canonical_sample_end", "bill_size_surprise", "bill_size_surprise_announcement_week"}
    if not required_lp.issubset(lp) or not required_shocks.issubset(shocks):
        return {}
    dates = pd.to_datetime(shocks["date"], errors="coerce")
    if (shocks.empty or dates.isna().any() or dates.duplicated().any()
            or not shocks["sample"].eq(PROMOTION_SPECS["canonical_issue_week"][1]).all()
            or not shocks["canonical_sample_end"].eq("2025-11-30").all()
            or dates.gt(pd.Timestamp("2025-11-30")).any()):
        return {}
    result = {}
    for spec, (shock_spec, sample, timing, shock_column) in PROMOTION_SPECS.items():
        values = pd.to_numeric(shocks[shock_column], errors="coerce")
        if not np.isfinite(values).all() or len(values) < 2:
            continue
        sd_bn = float(values.std(ddof=1) / 1000)
        if sd_bn <= 0:
            continue
        selected = lp.loc[lp.shock_spec.eq(shock_spec) & lp["sample"].eq(sample) & lp.timing.eq(timing) & lp.regime.isna()].copy()
        selected["horizon"] = pd.to_numeric(selected.horizon, errors="coerce")
        if not selected.canonical_sample_end.eq("2025-11-30").all():
            continue
        for channel, response in PROMOTION_CHANNELS.items():
            summary_rows = summary.loc[summary.spec.eq(spec) & summary.response_label.eq(channel)]
            source = selected.loc[selected.response_var.eq(response)]
            head = source.loc[source.horizon.eq(4)]
            leads = source.loc[source.horizon.lt(0)]
            if (len(summary_rows) != 1 or len(head) != 1 or len(leads) != 4
                    or set(leads.horizon) != {-4, -3, -2, -1}):
                continue
            row, h4 = summary_rows.iloc[0], head.iloc[0]
            if row.response_var != response:
                continue
            numeric = pd.to_numeric(h4[["beta", "se_nw", "t_stat_nw", "n_obs"]], errors="coerce")
            if not np.isfinite(numeric).all() or numeric.se_nw <= 0 or numeric.n_obs <= 0 or numeric.n_obs % 1:
                continue
            significant = _observed_bool(h4.significant_5pct)
            if significant is None or significant != (abs(numeric.t_stat_nw) > 1.96):
                continue
            if not np.isclose(numeric.beta / numeric.se_nw, numeric.t_stat_nw, rtol=1e-8, atol=1e-8):
                continue
            lead_t = pd.to_numeric(leads.t_stat_nw, errors="coerce")
            lead_flags = leads.significant_5pct.map(_observed_bool)
            if not np.isfinite(lead_t).all() or lead_flags.isna().any() or not lead_flags.eq(lead_t.abs().gt(1.96)).all():
                continue
            hits = int(lead_flags.sum())
            expected = {"h4_effect_bn": numeric.beta * sd_bn, "h4_t_stat_nw": numeric.t_stat_nw, "shock_sd_bn": sd_bn, "placebo_significant_count": hits}
            observed = pd.to_numeric(row[list(expected)], errors="coerce")
            if (not np.isfinite(observed).all()
                    or not np.allclose(observed, list(expected.values()), rtol=0, atol=SUMMARY_ROUNDING_ATOL)
                    or _observed_bool(row.h4_significant_5pct) is not significant):
                continue
            result[(spec, channel)] = {**expected, "n_obs": int(numeric.n_obs), "sample": sample, "timing": timing, "shock_spec": shock_spec, "significant": significant}
    return result


def tgarefill_promotion_reconciliation(root: Path) -> pd.DataFrame:
    """Import the joint association only after atomic source reconciliation."""
    names = ("promotion_robustness_summary", "canonical_bill_surprise_shocks", "auction_shock_lp")
    paths = {name: root / TGAREFILL_EXPORTS[name]["path"] for name in names}
    authority = {}
    if all(path.exists() for path in paths.values()):
        summary = pd.read_csv(paths[names[0]])
        required = set(TGAREFILL_EXPORTS[names[0]]["required"]) | {"placebo_significant_count"}
        if not required.issubset(summary):
            raise ValueError("Incomplete upstream promotion summary schema")
        if summary.duplicated(["spec", "response_label"]).any():
            raise ValueError("Duplicate upstream promotion specification/channel")
        authority = _promotion_authority(summary, pd.read_csv(paths[names[2]]), pd.read_csv(paths[names[1]]))
    positive = {"MMF Treasury Holdings": 1, "ON RRP": -1}
    complete = len(authority) == len(PROMOTION_CHANNELS) * len(PROMOTION_SPECS)
    if complete:
        for channel, sign in positive.items():
            row = authority[("canonical_issue_week", channel)]
            timing = authority[("same_week_announcement_timing", channel)]
            complete &= bool(row["significant"] and row["h4_effect_bn"] * sign > 0 and timing["placebo_significant_count"] == 0)
        complete &= all(not authority[("canonical_issue_week", channel)]["significant"] for channel in ("Bank Deposits", "Reserves"))
    rows = []
    for channel in PROMOTION_CHANNELS:
        row = authority.get(("canonical_issue_week", channel), {})
        timing = authority.get(("same_week_announcement_timing", channel), {})
        supported = complete and channel in positive
        boundary = complete and channel not in positive
        rows.append({
            "claim_id": "focused_tga_refill_bill_surprise",
            "status": "supported_focused_claim" if supported else "not_supported_as_channel" if boundary else "blocked_incomplete_evidence",
            "claim_use": "aggregate_association" if supported else "boundary_condition" if boundary else "not_available",
            "evidence_basis": "reconciled_tgarefill_canonical_pre_rmp_bill_surprise_lp",
            "channel": channel,
            "response_var": PROMOTION_CHANNELS[channel],
            "h4_effect_bn": row.get("h4_effect_bn", pd.NA),
            "h4_t_stat_nw": row.get("h4_t_stat_nw", pd.NA),
            "significant_5pct": row.get("significant", pd.NA),
            "shock_sd_bn": row.get("shock_sd_bn", pd.NA),
            "n_obs": row.get("n_obs", pd.NA),
            "sample": row.get("sample", ""), "timing": row.get("timing", ""), "shock_spec": row.get("shock_spec", ""),
            "same_week_placebo_count": timing.get("placebo_significant_count", pd.NA),
            "aggregate_complete": bool(complete),
            "pretrend_status": "missing_same_week_placebo_evidence" if not timing else "same_week_timing_has_no_significant_placebo" if timing["placebo_significant_count"] == 0 else "same_week_timing_has_remaining_placebo",
            "broad_substitution_reconciliation": "broad_liqsub_substitution_remains_blocked",
            "permitted_language": ("Bill offering-size deviations in the selected sample are associated with higher MMF Treasury holdings and lower ON RRP balances; channel samples differ." if supported else "No robust bank-deposit or reserve drain in this design." if boundary else ""),
            "forbidden_upgrade": "do_not_claim_causal_effect_final_ownership_funding_share_or_bill_deposit_substitution",
            "primary_artifacts": ";".join(str(path.relative_to(root)) for path in paths.values()),
        })
    rows.extend(_mmfalloc_validation_rows(root))
    return pd.DataFrame(rows)

def _mmfalloc_integrity(summary_values: dict[str, object], baseline: pd.DataFrame) -> bool:
    """Validate the displayed descriptive statistics, not just source feasibility."""
    try:
        if summary_values.get("claim_boundary") != "focused_tga_refill_mmf_allocation_support_only":
            return False
        start, end = pd.to_datetime([summary_values.get("sample_start"), summary_values.get("sample_end")], errors="coerce")
        count = float(summary_values.get("event_count", "nan"))
        if pd.isna(start) or pd.isna(end) or start > end or not np.isfinite(count) or count <= 0 or count % 1:
            return False
        months = pd.to_datetime(baseline["event_month"], errors="coerce")
        if (months.isna().any() or not months.eq(months.dt.to_period("M").dt.to_timestamp("M")).all()
                or not months.between(start, end).all() or months.nunique() != count):
            return False
        if baseline.assign(event_month=months).duplicated(["event_month", "category"]).any() or not baseline.horizon.eq("0_vs_minus1").all():
            return False
        categories = {"treasury_bills", "treasury_coupons", "fed_onrrp", "private_treasury_repo", "agency_repo", "agency_debt", "private_instruments", "cash_other", "total_assets"}
        if any(set(group.category) != categories for _, group in baseline.groupby(months)):
            return False
        values = baseline.copy()
        for column in TGAREFILL_EXPORTS["mmfalloc_baseline"]["numeric_columns"]:
            values[column] = pd.to_numeric(values[column], errors="coerce")
            if not np.isfinite(values[column]).all():
                return False
        shock = values.event_bill_surprise_millions
        if not shock.gt(0).all() or values.groupby(months).event_bill_surprise_millions.nunique().ne(1).any():
            return False
        if not np.allclose(values.delta_per_bill_surprise, values.delta_millions / shock, rtol=1e-9, atol=1e-9):
            return False
        table = values.assign(event_month=months).pivot(index="event_month", columns="category", values="delta_millions")
        metrics = {
            "mean_event_delta_treasury_total": (table.treasury_bills + table.treasury_coupons).mean(),
            "mean_event_delta_fed_onrrp": table.fed_onrrp.mean(),
            "mean_event_delta_repo_ex_fed": (table.private_treasury_repo + table.agency_repo).mean(),
        }
        return all(np.isclose(float(summary_values.get(key, "nan")), value, rtol=0, atol=SUMMARY_ROUNDING_ATOL) for key, value in metrics.items())
    except (ValueError, TypeError, KeyError):
        return False


def _mmfalloc_validation_rows(root: Path) -> list[dict[str, object]]:
    summary_path = root / TGAREFILL_EXPORTS["mmfalloc_downstream_summary"]["path"]
    gates_path = root / TGAREFILL_EXPORTS["mmfalloc_source_gates"]["path"]
    baseline_path = root / TGAREFILL_EXPORTS["mmfalloc_baseline"]["path"]
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    summary_values = dict(zip(summary["metric"].astype(str), summary["value"], strict=True)) if {"metric", "value"}.issubset(summary) else {}
    gates_passed = _observed_bool(summary_values.get("source_gates_passed")) is True
    gates_passed = gates_passed and "metric" in summary and not summary.duplicated("metric").any()
    if gates_path.exists() and baseline_path.exists():
        gates = pd.read_csv(gates_path)
        baseline = pd.read_csv(baseline_path)
        required_gates = {"coverage", "required_fields", "mapping", "reconciliation"}
        gates_passed = gates_passed and (
            {"gate", "passed"}.issubset(gates.columns)
            and required_gates.issubset(set(gates["gate"]))
            and not gates.duplicated("gate").any()
            and gates["passed"].map(_observed_bool).eq(True).all()
            and not baseline.empty
            and set(TGAREFILL_EXPORTS["mmfalloc_baseline"]["required"]).issubset(baseline.columns)
        )
        if gates_passed:
            for column in TGAREFILL_EXPORTS["mmfalloc_baseline"]["numeric_columns"]:
                gates_passed = gates_passed and np.isfinite(
                    pd.to_numeric(baseline[column], errors="coerce")
                ).all()
    else:
        gates_passed = False
    gates_passed = gates_passed and _mmfalloc_integrity(summary_values, baseline)
    claim_label = str(summary_values.get("claim_boundary", ""))
    status = "imported_descriptive_allocation" if gates_passed else "blocked_mmfalloc_gates"
    return [
        {
            "claim_id": "focused_tga_refill_mmfalloc_validation",
            "status": status,
            "claim_use": "descriptive_context" if gates_passed else "not_available",
            "evidence_basis": "sec_nmfp_descriptive_allocation",
            "channel": "Fund-level MMF allocation",
            "h4_effect_bn": pd.NA,
            "h4_t_stat_nw": pd.NA,
            "significant_5pct": pd.NA,
            "pretrend_status": "see_mmfalloc_placebos",
            "broad_substitution_reconciliation": (
                "broad_liqsub_substitution_remains_blocked"
            ),
            "permitted_language": (
                "Imported fund-level MMF allocation changes describe selected "
                "positive bill-surprise months; they do not identify causal funding routes."
            ) if gates_passed else "",
            "forbidden_upgrade": (
                "do_not_claim_household_behavior_final_ownership_bank_deposit_pass_through_"
                "or_general_bill_deposit_substitution"
            ),
            "primary_artifacts": (
                f"{summary_path.relative_to(root)};"
                f"{gates_path.relative_to(root)};"
                f"{baseline_path.relative_to(root)}"
            ),
            "claim_label": claim_label,
            "sample_start": summary_values.get("sample_start", ""),
            "sample_end": summary_values.get("sample_end", ""),
            "event_count": summary_values.get("event_count", ""),
            "mean_event_delta_treasury_total": summary_values.get(
                "mean_event_delta_treasury_total", ""
            ),
            "mean_event_delta_fed_onrrp": summary_values.get("mean_event_delta_fed_onrrp", ""),
            "mean_event_delta_repo_ex_fed": summary_values.get(
                "mean_event_delta_repo_ex_fed", ""
            ),
        }
    ]


def write_tgarefill_promotion_reconciliation_report(
    reconciliation: pd.DataFrame,
    *,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reconciliation.empty:
        path.write_text("# Tgarefill Promotion Reconciliation\n\nNo promotion rows available.\n", encoding="utf-8")
        return
    promoted = reconciliation.loc[reconciliation["status"] == "supported_focused_claim"]
    blocked = reconciliation.loc[reconciliation["status"] == "not_supported_as_channel"]
    mmfalloc = reconciliation.loc[
        reconciliation["status"].astype(str).isin(
            ["imported_descriptive_allocation", "blocked_mmfalloc_gates"]
        )
    ]
    lines = [
        "# Tgarefill Promotion Reconciliation",
        "",
        "## Bottom Line",
        "",
        (
            "Broad `liqsub` substitution remains blocked, but the focused `tgarefill` "
            "bill-surprise summary is imported only as aggregate association evidence "
            "when both named channels have complete qualifying rows."
        ),
        "",
        "## Promoted Channels",
        "",
    ]
    for row in promoted.itertuples(index=False):
        lines.append(
            f"- {row.channel}: h=4 effect {float(row.h4_effect_bn):+.1f}B, "
            f"NW t-stat {float(row.h4_t_stat_nw):.1f}."
        )
    lines.extend(["", "## Boundary Channels", ""])
    for row in blocked.itertuples(index=False):
        lines.append(
            f"- {row.channel}: h=4 effect {float(row.h4_effect_bn):+.1f}B, "
            f"NW t-stat {float(row.h4_t_stat_nw):.1f}; not promoted."
        )
    if not mmfalloc.empty:
        lines.extend(["", "## Descriptive MMF Allocation", ""])
        for row in mmfalloc.itertuples(index=False):
            if str(row.status) == "imported_descriptive_allocation":
                lines.append(
                    "- SEC N-MFP `mmfalloc`: source gates pass for "
                    f"{row.sample_start} through {row.sample_end}; across "
                    f"{row.event_count} selected positive bill-surprise months, "
                    f"Treasury allocations change {float(row.mean_event_delta_treasury_total):+.1f}M, "
                    f"Fed ON RRP changes {float(row.mean_event_delta_fed_onrrp):+.1f}M, "
                    f"and repo ex-Fed changes {float(row.mean_event_delta_repo_ex_fed):+.1f}M."
                )
            else:
                lines.append("- SEC N-MFP `mmfalloc`: validation gates did not pass.")
        lines.append(
            "Use this only as descriptive allocation context, separate from the canonical aggregate design."
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Do not claim general bill-deposit substitution or a stable one-for-one liquidity-substitution system.",
            "Use this only as an association with trailing-median bill offering-size deviations.",
            "Do not infer household behavior, final ownership, or bank-deposit pass-through from the MMF panel.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def weekly_timing_alignment_qa(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["tga", "reserves", "on_rrp", "deposits", "mmf_treasury_holdings"]:
        if column not in panel.columns:
            continue
        values = panel[column].dropna()
        rows.append(
            {
                "column": column,
                "non_null": int(values.size),
                "coverage_share": float(panel[column].notna().mean()) if len(panel) else 0.0,
                "first_week": panel.loc[panel[column].notna(), "week"].min().date().isoformat()
                if values.size
                else "",
                "last_week": panel.loc[panel[column].notna(), "week"].max().date().isoformat()
                if values.size
                else "",
                "max_internal_gap_weeks": _max_internal_gap_weeks(panel.loc[panel[column].notna(), "week"]),
            }
        )
    return pd.DataFrame(rows)


def weekly_event_candidates_clean(root: Path, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / TGAREFILL_EXPORTS["event_candidates"]["path"]
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    events = pd.read_csv(path)
    for column in ["baseline_date", "start_date", "end_date"]:
        events[column] = pd.to_datetime(events[column], errors="coerce")
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    available_weeks = set(pd.to_datetime(panel["week"], errors="coerce").dropna())
    for event in events.itertuples(index=False):
        reasons = []
        if pd.isna(event.start_date) or pd.isna(event.baseline_date):
            reasons.append("missing_event_date")
        if event.start_date not in available_weeks:
            reasons.append("start_week_missing_from_panel")
        tags = str(getattr(event, "manual_tags", ""))
        if "covid" in tags.lower():
            reasons.append("covid_crisis_narrative_only")
        delta_tga = pd.to_numeric(pd.Series([getattr(event, "delta_tga_event", pd.NA)]), errors="coerce").iloc[0]
        if pd.isna(delta_tga) or float(delta_tga) <= 0:
            reasons.append("non_positive_tga_rebuild")
        target = exclusions if reasons else rows
        target.append(
            {
                "event_id": event.event_id,
                "baseline_date": _date_or_blank(event.baseline_date),
                "start_date": _date_or_blank(event.start_date),
                "end_date": _date_or_blank(event.end_date),
                "delta_tga_event": delta_tga,
                "mean_bill_share": getattr(event, "mean_bill_share", pd.NA),
                "manual_tags": tags,
                "status": "excluded" if reasons else "clean_candidate",
                "exclusion_reason": ";".join(reasons),
            }
        )
    clean = pd.DataFrame(rows)
    exclusion_log = pd.DataFrame(exclusions)
    isolated, cluster_exclusions = _isolate_weekly_events(clean, min_gap_weeks=8)
    if not cluster_exclusions.empty:
        exclusion_log = pd.concat([exclusion_log, cluster_exclusions], ignore_index=True)
    return isolated, exclusion_log


def _isolate_weekly_events(events: pd.DataFrame, *, min_gap_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return events, pd.DataFrame()
    work = events.copy()
    work["start_date_dt"] = pd.to_datetime(work["start_date"], errors="coerce")
    work["delta_tga_event_numeric"] = pd.to_numeric(work["delta_tga_event"], errors="coerce")
    work = work.sort_values("start_date_dt").reset_index(drop=True)
    keep_indices: list[int] = []
    exclusion_rows: list[dict[str, object]] = []
    cluster: list[int] = []
    last_date: pd.Timestamp | None = None
    for idx, row in work.iterrows():
        start = row["start_date_dt"]
        if pd.isna(start):
            exclusion_rows.append(_weekly_event_exclusion_record(row, "missing_event_date_after_validation"))
            continue
        if not cluster:
            cluster = [idx]
            last_date = start
            continue
        assert last_date is not None
        gap_weeks = int((start - last_date).days // 7)
        if gap_weeks < min_gap_weeks:
            cluster.append(idx)
        else:
            _keep_peak_cluster_event(work, cluster, keep_indices, exclusion_rows, min_gap_weeks)
            cluster = [idx]
        last_date = start
    if cluster:
        _keep_peak_cluster_event(work, cluster, keep_indices, exclusion_rows, min_gap_weeks)
    drop_columns = ["start_date_dt", "delta_tga_event_numeric"]
    kept = work.loc[keep_indices].drop(columns=drop_columns).sort_values("start_date").reset_index(drop=True)
    exclusions = pd.DataFrame(exclusion_rows)
    return kept, exclusions


def _keep_peak_cluster_event(
    events: pd.DataFrame,
    cluster: list[int],
    keep_indices: list[int],
    exclusion_rows: list[dict[str, object]],
    min_gap_weeks: int,
) -> None:
    if len(cluster) == 1:
        keep_indices.append(cluster[0])
        return
    cluster_frame = events.loc[cluster].copy()
    peak_idx = int(cluster_frame["delta_tga_event_numeric"].fillna(float("-inf")).idxmax())
    keep_indices.append(peak_idx)
    for idx in cluster:
        if idx == peak_idx:
            continue
        row = events.loc[idx]
        peak_event_id = events.loc[peak_idx, "event_id"]
        reason = f"clustered_within_{min_gap_weeks}_weeks_of_larger_rebuild:{peak_event_id}"
        exclusion_rows.append(_weekly_event_exclusion_record(row, reason))


def _weekly_event_exclusion_record(row: pd.Series, reason: str) -> dict[str, object]:
    return {
        "event_id": row.get("event_id", ""),
        "baseline_date": row.get("baseline_date", ""),
        "start_date": row.get("start_date", ""),
        "end_date": row.get("end_date", ""),
        "delta_tga_event": row.get("delta_tga_event", pd.NA),
        "mean_bill_share": row.get("mean_bill_share", pd.NA),
        "manual_tags": row.get("manual_tags", ""),
        "status": "excluded",
        "exclusion_reason": reason,
    }


def weekly_pretrend_balance(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["d_deposits", "d_on_rrp", "d_reserves", "d_mmf_treasury_holdings"]:
        if outcome not in panel.columns:
            continue
        for event in events.itertuples(index=False):
            start = pd.to_datetime(event.start_date, errors="coerce")
            if pd.isna(start):
                continue
            window = panel.loc[(panel["week"] >= start - pd.Timedelta(weeks=4)) & (panel["week"] < start), outcome]
            rows.append(
                {
                    "event_id": event.event_id,
                    "event_start": start.date().isoformat(),
                    "outcome": outcome,
                    "pre4_non_null": int(window.notna().sum()),
                    "pre4_sum": float(window.sum()) if window.notna().any() else pd.NA,
                    "status": "ok" if window.notna().sum() >= 4 else "incomplete_pre_window",
                }
            )
    return pd.DataFrame(rows)


def weekly_placebo_tests(
    panel: pd.DataFrame,
    events: pd.DataFrame | None = None,
    actual_event_study: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if events is None or events.empty or actual_event_study is None or actual_event_study.empty:
        return _legacy_weekly_placebo_tests(panel)
    panel = panel.sort_values("week").reset_index(drop=True)
    event_starts = set(pd.to_datetime(events["start_date"], errors="coerce").dropna())
    pseudo_df = _matched_pseudo_events(panel, events, event_starts)
    pseudo_rows = weekly_event_study_by_event(panel, pseudo_df)
    actual_summary = weekly_event_study_summary(actual_event_study)
    rows: list[dict[str, object]] = []
    post_taus = [0, 1, 2, 4, 8]
    for actual in actual_summary.loc[actual_summary["tau"].isin(post_taus)].itertuples(index=False):
        if pseudo_rows.empty or "outcome" not in pseudo_rows.columns:
            pseudo = pd.Series(dtype=float)
        else:
            pseudo = pseudo_rows.loc[
                (pseudo_rows["outcome"] == actual.outcome)
                & (pseudo_rows["tau"] == actual.tau),
                "level_change_from_baseline",
            ].dropna()
        if pseudo.empty:
            rows.append(
                {
                    "outcome": actual.outcome,
                    "tau": actual.tau,
                    "status": "no_pseudo_events",
                    "actual_mean_change": actual.mean_change,
                    "pseudo_events_n": 0,
                    "placebo_design": "matched_event_pseudo",
                }
            )
            continue
        threshold = abs(float(actual.mean_change))
        false_positive_share = float((pseudo.abs() >= threshold).mean()) if threshold else 1.0
        rows.append(
            {
                "outcome": actual.outcome,
                "tau": actual.tau,
                "status": "estimated",
                "actual_mean_change": actual.mean_change,
                "pseudo_events_n": int(pseudo.size),
                "matched_event_groups_n": int(pseudo_df["matched_event_id"].nunique())
                if "matched_event_id" in pseudo_df.columns
                else 0,
                "pseudo_mean_change": float(pseudo.mean()),
                "pseudo_abs_p90": float(pseudo.abs().quantile(0.90)),
                "false_positive_share": false_positive_share,
                "placebo_design": "matched_event_pseudo",
                "interpretation": "matched pseudo-event diagnostic; requires false_positive_share < 0.10 for causal-candidate use",
            }
        )
    return pd.DataFrame(rows).sort_values(["outcome", "tau"]).reset_index(drop=True)


def _matched_pseudo_events(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    event_starts: set[pd.Timestamp],
    *,
    max_matches_per_event: int = 25,
    min_matches_per_event: int = 5,
) -> pd.DataFrame:
    pseudo_weeks = _pseudo_event_weeks(panel, event_starts)
    if not pseudo_weeks:
        return pd.DataFrame()
    feature_panel = _weekly_match_features(panel)
    candidates = feature_panel.loc[feature_panel["week"].isin(pseudo_weeks)].copy()
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        start = pd.to_datetime(event.start_date, errors="coerce")
        if pd.isna(start):
            continue
        event_feature = feature_panel.loc[feature_panel["week"] == start]
        if event_feature.empty:
            continue
        matched = _select_matched_pseudo_weeks(
            candidates,
            event_feature.iloc[0],
            max_matches=max_matches_per_event,
            min_matches=min_matches_per_event,
        )
        for match_idx, row in enumerate(matched.itertuples(index=False)):
            rows.append(
                {
                    "event_id": f"pseudo_{event.event_id}_{match_idx:03d}",
                    "matched_event_id": event.event_id,
                    "baseline_date": row.week - pd.Timedelta(weeks=1),
                    "start_date": row.week,
                    "delta_tga_event": 0.0,
                    "mean_bill_share": np.nan,
                    "manual_tags": f"pseudo;match_tier={row.match_tier}",
                }
            )
    return pd.DataFrame(rows)


def _weekly_match_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["week"] = pd.to_datetime(out["week"], errors="coerce")
    out["calendar_quarter"] = out["week"].dt.quarter
    out["calendar_month"] = out["week"].dt.month
    out["week_of_year"] = out["week"].dt.isocalendar().week.astype("Int64")
    for column in ["tax_week", "debt_limit_window", "covid_crisis_window", "bank_stress_window"]:
        if column not in out.columns:
            out[column] = False
        out[column] = out[column].fillna(False).astype(bool)
    if "on_rrp_regime_pre" not in out.columns:
        out["on_rrp_regime_pre"] = ""
    out["on_rrp_regime_pre"] = out["on_rrp_regime_pre"].fillna("").astype(str)
    return out[
        [
            "week",
            "calendar_quarter",
            "calendar_month",
            "week_of_year",
            "tax_week",
            "debt_limit_window",
            "covid_crisis_window",
            "bank_stress_window",
            "on_rrp_regime_pre",
        ]
    ]


def _select_matched_pseudo_weeks(
    candidates: pd.DataFrame,
    event_feature: pd.Series,
    *,
    max_matches: int,
    min_matches: int,
) -> pd.DataFrame:
    tiers = [
        ("month_regime_calendar", ["calendar_month", "tax_week", "debt_limit_window", "on_rrp_regime_pre"]),
        ("quarter_regime_calendar", ["calendar_quarter", "tax_week", "debt_limit_window", "on_rrp_regime_pre"]),
        ("quarter_calendar", ["calendar_quarter", "tax_week", "debt_limit_window"]),
        ("quarter_only", ["calendar_quarter"]),
        ("all_eligible", []),
    ]
    for tier, columns in tiers:
        matched = candidates.copy()
        for column in columns:
            matched = matched.loc[matched[column] == event_feature[column]]
        if len(matched) >= min_matches or tier == "all_eligible":
            matched = matched.copy()
            matched["match_tier"] = tier
            matched["week_distance"] = (
                matched["week_of_year"].astype(float) - float(event_feature["week_of_year"])
            ).abs()
            return matched.sort_values(["week_distance", "week"]).head(max_matches)
    raise AssertionError("unreachable matched placebo tier")


def _legacy_weekly_placebo_tests(panel: pd.DataFrame) -> pd.DataFrame:
    if "bill_size_surprise_100b" not in panel.columns:
        return pd.DataFrame()
    rows = []
    for outcome in ["d_deposits", "d_on_rrp", "d_reserves", "d_mmf_treasury_holdings"]:
        if outcome not in panel.columns:
            continue
        df = panel[[outcome, "bill_size_surprise_100b"]].dropna()
        if len(df) < 52:
            rows.append({"outcome": outcome, "status": "too_sparse", "nobs": int(len(df))})
            continue
        corr = df[outcome].corr(df["bill_size_surprise_100b"].shift(4))
        rows.append(
            {
                "outcome": outcome,
                "status": "diagnostic_only",
                "nobs": int(len(df)),
                "lead4_bill_surprise_corr": float(corr) if pd.notna(corr) else pd.NA,
                "interpretation": "placeholder placebo; replace with HAC pseudo-event tests",
            }
        )
    return pd.DataFrame(rows)


def weekly_lp_coefficients(panel: pd.DataFrame, *, min_nobs: int = 104) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    outcomes = ["d_deposits", "d_on_rrp", "d_reserves", "d_mmf_treasury_holdings"]
    horizons = [0, 1, 2, 4, 8]
    for outcome in outcomes:
        if outcome not in panel.columns:
            continue
        work = panel.copy()
        work[f"lag1_level_{outcome}"] = _outcome_level(panel, outcome).shift(1)
        work[f"pre4_{outcome}"] = pd.to_numeric(work[outcome], errors="coerce").shift(1).rolling(4, min_periods=4).sum()
        for horizon in horizons:
            target = f"f{horizon}_{outcome}"
            work[target] = _future_weekly_cumulative_change(work[outcome], horizon)
            predictors = [
                "bill_size_surprise_100b",
                f"pre4_{outcome}",
                f"lag1_level_{outcome}",
                "bill_share",
                "coupon_settlement_100b",
                "tax_week",
                "covid_crisis_window",
                "bank_stress_window",
                "debt_limit_window",
            ]
            required = [target, *predictors]
            if not set(required).issubset(work.columns):
                continue
            df = work[required].copy()
            for column in ["tax_week", "covid_crisis_window", "bank_stress_window", "debt_limit_window"]:
                df[column] = df[column].astype(float)
            df = df.dropna()
            if len(df) < min_nobs:
                rows.append(
                    {
                        "outcome": outcome,
                        "horizon_weeks": horizon,
                        "status": "skipped_too_few_observations",
                        "nobs": int(len(df)),
                        "min_nobs": min_nobs,
                    }
                )
                continue
            beta, se_hc1, se_hac, r2 = _ols_hc1_hac(
                df,
                outcome=target,
                predictors=predictors,
                hac_lags=max(4, horizon + 1),
            )
            for idx, predictor in enumerate(["intercept", *predictors]):
                rows.append(
                    {
                        "outcome": outcome,
                        "horizon_weeks": horizon,
                        "status": "estimated",
                        "predictor": predictor,
                        "nobs": int(len(df)),
                        "beta": float(beta[idx]),
                        "se_hc1": float(se_hc1[idx]),
                        "se_hac": float(se_hac[idx]),
                        "t_hc1": float(beta[idx] / se_hc1[idx]) if se_hc1[idx] else np.nan,
                        "t_hac": float(beta[idx] / se_hac[idx]) if se_hac[idx] else np.nan,
                        "hac_lags": max(4, horizon + 1),
                        "r2": r2,
                        "interpretation": "weekly descriptive LP with HAC errors; identification gates still required",
                    }
                )
    return pd.DataFrame(rows)


def weekly_event_study_by_event(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    outcomes = ["deposits", "on_rrp", "reserves", "mmf_treasury_holdings"]
    rows: list[dict[str, object]] = []
    panel = panel.sort_values("week").reset_index(drop=True)
    week_to_index = {week: idx for idx, week in enumerate(panel["week"])}
    for event in events.itertuples(index=False):
        start = pd.to_datetime(event.start_date, errors="coerce")
        baseline = pd.to_datetime(event.baseline_date, errors="coerce")
        if pd.isna(start) or start not in week_to_index:
            continue
        start_idx = week_to_index[start]
        baseline_idx = week_to_index.get(baseline, start_idx - 1)
        for outcome in outcomes:
            if outcome not in panel.columns:
                continue
            baseline_value = _event_baseline_value(panel, baseline_idx, outcome)
            for tau in range(-4, 9):
                idx = start_idx + tau
                if idx < 0 or idx >= len(panel):
                    continue
                value = pd.to_numeric(pd.Series([panel.loc[idx, outcome]]), errors="coerce").iloc[0]
                rows.append(
                    {
                        "event_id": event.event_id,
                        "event_start": start.date().isoformat(),
                        "event_week": panel.loc[idx, "week"].date().isoformat(),
                        "tau": tau,
                        "outcome": outcome,
                        "level": value,
                        "baseline_level": baseline_value,
                        "level_change_from_baseline": value - baseline_value
                        if pd.notna(value) and pd.notna(baseline_value)
                        else np.nan,
                        "delta_tga_event": getattr(event, "delta_tga_event", np.nan),
                        "mean_bill_share": getattr(event, "mean_bill_share", np.nan),
                        "manual_tags": getattr(event, "manual_tags", ""),
                    }
                )
    return pd.DataFrame(rows)


def weekly_event_study_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    grouped = event_rows.dropna(subset=["level_change_from_baseline"]).groupby(
        ["outcome", "tau"], observed=True
    )
    return (
        grouped.agg(
            n_events=("event_id", "nunique"),
            mean_change=("level_change_from_baseline", "mean"),
            median_change=("level_change_from_baseline", "median"),
            p25_change=("level_change_from_baseline", lambda s: float(s.quantile(0.25))),
            p75_change=("level_change_from_baseline", lambda s: float(s.quantile(0.75))),
        )
        .reset_index()
        .sort_values(["outcome", "tau"])
    )


def weekly_leave_one_event_out(event_rows: pd.DataFrame) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    post_taus = [0, 1, 2, 4, 8]
    work = event_rows.loc[event_rows["tau"].isin(post_taus)].dropna(
        subset=["level_change_from_baseline"]
    )
    for (outcome, tau), group in work.groupby(["outcome", "tau"], observed=True):
        full_mean = float(group["level_change_from_baseline"].mean())
        full_sign = _sign_label(full_mean)
        event_ids = sorted(group["event_id"].dropna().unique())
        if len(event_ids) < 2:
            rows.append(
                {
                    "outcome": outcome,
                    "tau": tau,
                    "status": "too_few_events",
                    "n_events": int(len(event_ids)),
                    "full_sample_mean": full_mean,
                    "full_sample_sign": full_sign,
                }
            )
            continue
        leave_means = []
        same_sign_count = 0
        for event_id in event_ids:
            leave = group.loc[group["event_id"] != event_id, "level_change_from_baseline"]
            leave_mean = float(leave.mean()) if not leave.empty else np.nan
            leave_means.append(leave_mean)
            if _same_nonzero_sign(full_mean, leave_mean):
                same_sign_count += 1
        leave_series = pd.Series(leave_means).dropna()
        rows.append(
            {
                "outcome": outcome,
                "tau": tau,
                "status": "estimated",
                "n_events": int(len(event_ids)),
                "full_sample_mean": full_mean,
                "full_sample_sign": full_sign,
                "same_sign_leave_one_out_share": float(same_sign_count / len(event_ids)),
                "min_leave_one_out_mean": float(leave_series.min()) if not leave_series.empty else np.nan,
                "max_leave_one_out_mean": float(leave_series.max()) if not leave_series.empty else np.nan,
                "max_abs_mean_change_from_full": float((leave_series - full_mean).abs().max())
                if not leave_series.empty
                else np.nan,
                "interpretation": "event-time robustness; requires >=80% same-sign stability for causal-candidate use",
            }
        )
    return pd.DataFrame(rows).sort_values(["outcome", "tau"]).reset_index(drop=True)


def weekly_block_bootstrap(
    event_rows: pd.DataFrame,
    *,
    n_bootstrap: int = 500,
    seed: int = 17,
) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    post_taus = [0, 1, 2, 4, 8]
    work = event_rows.loc[event_rows["tau"].isin(post_taus)].dropna(
        subset=["level_change_from_baseline"]
    )
    rng = np.random.default_rng(seed)
    for (outcome, tau), group in work.groupby(["outcome", "tau"], observed=True):
        event_values = group.groupby("event_id", observed=True)["level_change_from_baseline"].mean()
        if len(event_values) < 2:
            rows.append(
                {
                    "outcome": outcome,
                    "tau": tau,
                    "status": "too_few_events",
                    "n_events": int(len(event_values)),
                }
            )
            continue
        values = event_values.to_numpy(dtype=float)
        full_mean = float(values.mean())
        draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
        rows.append(
            {
                "outcome": outcome,
                "tau": tau,
                "status": "estimated",
                "n_events": int(len(values)),
                "n_bootstrap": n_bootstrap,
                "full_sample_mean": full_mean,
                "bootstrap_mean": float(draws.mean()),
                "bootstrap_se": float(draws.std(ddof=1)),
                "ci_lower": float(np.quantile(draws, 0.025)),
                "ci_upper": float(np.quantile(draws, 0.975)),
                "bootstrap_same_sign_share": float(
                    np.mean([_same_nonzero_sign(full_mean, draw) for draw in draws])
                ),
                "interpretation": "event-level block bootstrap; requires same-sign share >= 0.80 and CI not crossing zero for causal-candidate use",
            }
        )
    return pd.DataFrame(rows).sort_values(["outcome", "tau"]).reset_index(drop=True)


def weekly_stable_claim_candidates(
    leave_one_out: pd.DataFrame,
    bootstrap: pd.DataFrame,
    placebo: pd.DataFrame,
) -> pd.DataFrame:
    if leave_one_out.empty or bootstrap.empty or placebo.empty:
        return pd.DataFrame()
    merged = leave_one_out.merge(
        bootstrap,
        on=["outcome", "tau"],
        how="outer",
        suffixes=("_loo", "_bootstrap"),
    ).merge(
        placebo,
        on=["outcome", "tau"],
        how="outer",
        suffixes=("", "_placebo"),
    )
    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        blockers = []
        loo_share = getattr(row, "same_sign_leave_one_out_share", np.nan)
        boot_share = getattr(row, "bootstrap_same_sign_share", np.nan)
        ci_lower = getattr(row, "ci_lower", np.nan)
        ci_upper = getattr(row, "ci_upper", np.nan)
        false_positive = getattr(row, "false_positive_share", np.nan)
        if pd.isna(loo_share) or loo_share < 0.80:
            blockers.append("leave_one_out_unstable")
        if pd.isna(boot_share) or boot_share < 0.80:
            blockers.append("bootstrap_unstable")
        if pd.isna(ci_lower) or pd.isna(ci_upper) or (ci_lower <= 0 <= ci_upper):
            blockers.append("bootstrap_ci_crosses_zero")
        if pd.isna(false_positive) or false_positive >= 0.10:
            blockers.append("placebo_false_positive_high")
        rows.append(
            {
                "outcome": row.outcome,
                "tau": int(row.tau),
                "status": "passes_all_gates" if not blockers else "blocked",
                "event_mean_change": getattr(row, "full_sample_mean_loo", np.nan),
                "leave_one_out_same_sign_share": loo_share,
                "bootstrap_same_sign_share": boot_share,
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "placebo_false_positive_share": false_positive,
                "blocker_flags": ";".join(blockers),
                "claim_use": "descriptive_stable_cell" if not blockers else "do_not_promote",
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "outcome", "tau"]).reset_index(drop=True)


def weekly_event_filter_sensitivity(panel: pd.DataFrame, clean_events: pd.DataFrame) -> pd.DataFrame:
    if clean_events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for variant_name, variant_events, description in _weekly_event_filter_variants(clean_events):
        event_study = weekly_event_study_by_event(panel, variant_events)
        leave_one = weekly_leave_one_event_out(event_study)
        bootstrap = weekly_block_bootstrap(event_study, n_bootstrap=200)
        placebo = weekly_placebo_tests(panel, variant_events, event_study)
        stable = weekly_stable_claim_candidates(leave_one, bootstrap, placebo)
        pass_rows = stable.loc[stable["status"] == "passes_all_gates"] if not stable.empty else pd.DataFrame()
        event_count = int(len(variant_events))
        rows.append(
            {
                "variant": variant_name,
                "description": description,
                "clean_events_n": event_count,
                "stable_cells_n": int(len(pass_rows)),
                "stable_cells": _format_stable_cells(pass_rows),
                "min_placebo_false_positive_share": _series_min(placebo, "false_positive_share"),
                "max_placebo_false_positive_share": _series_max(placebo, "false_positive_share"),
                "min_bootstrap_same_sign_share": _series_min(bootstrap, "bootstrap_same_sign_share"),
                "bootstrap_ci_crossing_cells": _ci_cross_count(bootstrap),
                "decision": _weekly_filter_variant_decision(event_count, len(pass_rows)),
                "required_next_step": _weekly_filter_variant_next_step(event_count, len(pass_rows)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stable_cells_n", "clean_events_n", "variant"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def weekly_large_rebuild_targeted_review(panel: pd.DataFrame, clean_events: pd.DataFrame) -> pd.DataFrame:
    if clean_events.empty:
        return pd.DataFrame()
    events = clean_events.loc[
        pd.to_numeric(clean_events["delta_tga_event"], errors="coerce") >= 200_000.0
    ].reset_index(drop=True)
    if events.empty:
        return pd.DataFrame()
    event_study = weekly_event_study_by_event(panel, events)
    leave_one = weekly_leave_one_event_out(event_study)
    bootstrap = weekly_block_bootstrap(event_study, n_bootstrap=500)
    placebo = weekly_placebo_tests(panel, events, event_study)
    stable = weekly_stable_claim_candidates(leave_one, bootstrap, placebo)
    passing = stable.loc[stable["status"] == "passes_all_gates"] if not stable.empty else pd.DataFrame()
    if passing.empty:
        return pd.DataFrame(
            [
                {
                    "targeted_design": "rebuild_ge_200b",
                    "decision": "no_passing_cells",
                    "event_count": int(len(events)),
                    "required_next_step": "do_not_promote_this_targeted_design",
                }
            ]
        )

    rows: list[dict[str, object]] = []
    event_meta = events.set_index("event_id", drop=False)
    for cell in passing.itertuples(index=False):
        cell_events = event_study.loc[
            (event_study["outcome"] == cell.outcome) & (event_study["tau"] == cell.tau)
        ].dropna(subset=["level_change_from_baseline"])
        for event_row in cell_events.sort_values("event_start").itertuples(index=False):
            meta = event_meta.loc[event_row.event_id] if event_row.event_id in event_meta.index else {}
            rows.append(
                {
                    "targeted_design": "rebuild_ge_200b",
                    "decision": "small_sample_manual_review_only",
                    "outcome": cell.outcome,
                    "tau": int(cell.tau),
                    "event_id": event_row.event_id,
                    "event_start": event_row.event_start,
                    "event_week": event_row.event_week,
                    "event_count": int(len(events)),
                    "delta_tga_event": getattr(event_row, "delta_tga_event", np.nan),
                    "mean_bill_share": getattr(event_row, "mean_bill_share", np.nan),
                    "manual_tags": getattr(event_row, "manual_tags", ""),
                    "level_change_from_baseline": event_row.level_change_from_baseline,
                    "cell_mean_change": cell.event_mean_change,
                    "leave_one_out_same_sign_share": cell.leave_one_out_same_sign_share,
                    "bootstrap_same_sign_share": cell.bootstrap_same_sign_share,
                    "bootstrap_ci_lower": cell.bootstrap_ci_lower,
                    "bootstrap_ci_upper": cell.bootstrap_ci_upper,
                    "placebo_false_positive_share": cell.placebo_false_positive_share,
                    "event_span": _targeted_event_span(meta),
                    "required_next_step": "manual_validate_large_rebuild_calendar_and_mechanism",
                }
            )
    return pd.DataFrame(rows).sort_values(["outcome", "tau", "event_start"]).reset_index(drop=True)


def weekly_large_rebuild_calendar_validation(root: Path, review: pd.DataFrame) -> pd.DataFrame:
    if review.empty or "event_id" not in review.columns:
        return pd.DataFrame()
    context = _load_weekly_large_rebuild_context(root)
    rows: list[dict[str, object]] = []
    for event_id, group in review.groupby("event_id", observed=True):
        context_row = (
            context.loc[context["event_id"] == event_id].iloc[0].to_dict()
            if not context.empty and event_id in set(context["event_id"])
            else {}
        )
        on_rrp_tau8 = _event_cell_change(group, "on_rrp", 8)
        rows.append(
            {
                "targeted_design": "rebuild_ge_200b",
                "event_id": event_id,
                "event_start": _first_value(group, "event_start"),
                "event_span": _first_value(group, "event_span"),
                "delta_tga_event": _first_value(group, "delta_tga_event"),
                "mean_bill_share": _first_value(group, "mean_bill_share"),
                "manual_tags": _first_value(group, "manual_tags"),
                "on_rrp_tau8_change": on_rrp_tau8,
                "reserves_tau0_change": _event_cell_change(group, "reserves", 0),
                "reserves_tau2_change": _event_cell_change(group, "reserves", 2),
                "reserves_tau4_change": _event_cell_change(group, "reserves", 4),
                "evidence_status": context_row.get("evidence_status", "missing_context"),
                "evidence_topic": context_row.get("evidence_topic", ""),
                "official_source": context_row.get("official_source", ""),
                "source_url": context_row.get("source_url", ""),
                "calendar_note": context_row.get("calendar_note", ""),
                "mechanism_implication": context_row.get("mechanism_implication", ""),
                "validation_decision": _large_rebuild_event_validation_decision(
                    context_row,
                    on_rrp_tau8,
                    str(_first_value(group, "manual_tags")),
                ),
                "required_next_step": "manual_calendar_mechanism_review_before_narrative_use",
            }
        )
    return pd.DataFrame(rows).sort_values("event_start").reset_index(drop=True)


def weekly_large_rebuild_event_roster(root: Path, clean_events: pd.DataFrame) -> pd.DataFrame:
    if clean_events.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_ROSTER_COLUMNS)
    events = clean_events.loc[
        pd.to_numeric(clean_events["delta_tga_event"], errors="coerce") >= LARGE_REBUILD_THRESHOLD
    ].copy()
    if events.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_ROSTER_COLUMNS)
    context = _load_weekly_large_rebuild_context(root)
    context_by_event = context.set_index("event_id", drop=False) if not context.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for event in events.sort_values("start_date").itertuples(index=False):
        event_id = str(event.event_id)
        raw_tags = getattr(event, "manual_tags", "")
        manual_tags = "" if pd.isna(raw_tags) else str(raw_tags)
        context_row = (
            context_by_event.loc[event_id].to_dict()
            if not context_by_event.empty and event_id in context_by_event.index
            else {}
        )
        debt_limit_confounded = event_id == "event_038" or "debt" in manual_tags.lower()
        on_rrp_reversal = event_id in {"event_041", "event_045"}
        sample_role = "appendix" if debt_limit_confounded else "main"
        exclusion_reason = "debt_limit_confounded" if debt_limit_confounded else ""
        rows.append(
            {
                "targeted_design": LARGE_REBUILD_DESIGN,
                "event_id": event_id,
                "event_start": _date_or_blank(getattr(event, "start_date", "")),
                "event_end": _date_or_blank(getattr(event, "end_date", "")),
                "delta_tga_event": getattr(event, "delta_tga_event", np.nan),
                "mean_bill_share": getattr(event, "mean_bill_share", np.nan),
                "manual_tags": manual_tags,
                "sample_role": sample_role,
                "exclusion_reason": exclusion_reason,
                "debt_limit_confounded": bool(debt_limit_confounded),
                "on_rrp_tau8_sign_reversal": bool(on_rrp_reversal),
                "official_source": context_row.get("official_source", ""),
                "source_url": context_row.get("source_url", ""),
                "calendar_note": context_row.get("calendar_note", ""),
                "mechanism_implication": context_row.get("mechanism_implication", ""),
            }
        )
    return pd.DataFrame(rows, columns=LARGE_REBUILD_ROSTER_COLUMNS).reset_index(drop=True)


def weekly_large_rebuild_event_study(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_EVENT_STUDY_COLUMNS)
    rows: list[dict[str, object]] = []
    panel = panel.sort_values("week").reset_index(drop=True)
    week_to_index = {pd.to_datetime(week): idx for idx, week in enumerate(panel["week"])}
    for event in events.itertuples(index=False):
        start = pd.to_datetime(event.start_date, errors="coerce")
        if pd.isna(start) or start not in week_to_index:
            continue
        start_idx = week_to_index[start]
        baseline_indices = [start_idx + tau for tau in (-4, -3, -2)]
        baseline_window = ";".join(
            panel.loc[idx, "week"].date().isoformat()
            for idx in baseline_indices
            if 0 <= idx < len(panel)
        )
        for outcome in EVENT_STUDY_OUTCOMES:
            if outcome not in panel.columns:
                continue
            baseline_values = [
                pd.to_numeric(pd.Series([panel.loc[idx, outcome]]), errors="coerce").iloc[0]
                for idx in baseline_indices
                if 0 <= idx < len(panel)
            ]
            baseline_series = pd.Series(baseline_values, dtype="float64").dropna()
            baseline_value = float(baseline_series.mean()) if len(baseline_series) == 3 else np.nan
            for tau in range(-4, 9):
                idx = start_idx + tau
                if idx < 0 or idx >= len(panel):
                    continue
                value = pd.to_numeric(pd.Series([panel.loc[idx, outcome]]), errors="coerce").iloc[0]
                rows.append(
                    {
                        "event_id": event.event_id,
                        "matched_event_id": getattr(event, "matched_event_id", ""),
                        "event_start": start.date().isoformat(),
                        "event_week": panel.loc[idx, "week"].date().isoformat(),
                        "tau": tau,
                        "outcome": outcome,
                        "level": value,
                        "baseline_level": baseline_value,
                        "baseline_window": baseline_window,
                        "baseline_non_null": int(len(baseline_series)),
                        "baseline_method": "mean_tau_minus4_to_minus2",
                        "level_change_from_baseline": value - baseline_value
                        if pd.notna(value) and pd.notna(baseline_value)
                        else np.nan,
                        "delta_tga_event": getattr(event, "delta_tga_event", np.nan),
                        "mean_bill_share": getattr(event, "mean_bill_share", np.nan),
                        "manual_tags": getattr(event, "manual_tags", ""),
                    }
                )
    return pd.DataFrame(rows, columns=LARGE_REBUILD_EVENT_STUDY_COLUMNS)


def weekly_large_rebuild_cell_summary(
    root: Path,
    panel: pd.DataFrame,
    clean_events: pd.DataFrame,
) -> pd.DataFrame:
    roster = weekly_large_rebuild_event_roster(root, clean_events)
    if roster.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_CELL_SUMMARY_COLUMNS)
    events = _large_rebuild_main_events(clean_events, roster)
    if events.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_CELL_SUMMARY_COLUMNS)
    event_study = weekly_large_rebuild_event_study(panel, events)
    leave_one = weekly_leave_one_event_out(event_study)
    bootstrap = weekly_block_bootstrap(event_study, n_bootstrap=2_000, seed=29)
    placebo = _weekly_large_rebuild_state_matched_placebo(panel, events, event_study, outcome="reserves")
    randomization = weekly_large_rebuild_randomization_inference(root, panel, clean_events)
    match_quality = weekly_large_rebuild_match_quality(root, panel, clean_events)
    randomization_p_max = weekly_large_rebuild_randomization_p_max(root)
    rows: list[dict[str, object]] = []
    for outcome, taus, table_role in [
        ("reserves", LARGE_REBUILD_MAIN_TAUS, "headline_reserves_main"),
        ("reserves", LARGE_REBUILD_SECONDARY_TAUS, "secondary_reserves_if_survives"),
        ("on_rrp", LARGE_REBUILD_APPENDIX_TAUS, "appendix_on_rrp_heterogeneity"),
    ]:
        for tau in taus:
            rows.append(
                _large_rebuild_cell_row(
                    event_study,
                    leave_one,
                    bootstrap,
                    placebo,
                    randomization,
                    match_quality,
                    outcome=outcome,
                    tau=tau,
                    table_role=table_role,
                    randomization_p_max=randomization_p_max,
                )
            )
    return pd.DataFrame(rows, columns=LARGE_REBUILD_CELL_SUMMARY_COLUMNS)


def weekly_large_rebuild_event_sign_stability(
    root: Path,
    panel: pd.DataFrame,
    clean_events: pd.DataFrame,
) -> pd.DataFrame:
    roster = weekly_large_rebuild_event_roster(root, clean_events)
    if roster.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_EVENT_SIGN_COLUMNS)
    events = _large_rebuild_events_for_roster(clean_events, roster)
    if events.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_EVENT_SIGN_COLUMNS)
    event_study = weekly_large_rebuild_event_study(panel, events)
    rows: list[dict[str, object]] = []
    for outcome, taus, table_role in [
        ("reserves", LARGE_REBUILD_MAIN_TAUS, "headline_reserves_main"),
        ("on_rrp", LARGE_REBUILD_APPENDIX_TAUS, "appendix_on_rrp_heterogeneity"),
    ]:
        for tau in taus:
            group = event_study.loc[
                (event_study["outcome"] == outcome) & (event_study["tau"] == tau)
            ].dropna(subset=["level_change_from_baseline"])
            if group.empty:
                continue
            reference_group = _event_sign_reference_group(group, roster, table_role)
            cell_mean = float(reference_group["level_change_from_baseline"].mean())
            abs_sum = float(reference_group["level_change_from_baseline"].abs().sum())
            for row in group.sort_values("event_start").itertuples(index=False):
                roster_row = _roster_row(roster, row.event_id)
                change = float(row.level_change_from_baseline)
                sample_role = roster_row.get("sample_role", "")
                rows.append(
                    {
                        "targeted_design": LARGE_REBUILD_DESIGN,
                        "event_id": row.event_id,
                        "event_start": row.event_start,
                        "sample_role": sample_role,
                        "outcome": outcome,
                        "tau": int(tau),
                        "table_role": table_role,
                        "level_change_from_baseline": change,
                        "sign": _sign_label(change),
                        "same_sign_as_cell_mean": _same_nonzero_sign(change, cell_mean),
                        "cell_mean_change": cell_mean,
                        "contribution_share_of_abs_sum": abs(change) / abs_sum
                        if abs_sum and _event_in_reference_sample(table_role, sample_role)
                        else np.nan,
                        "debt_limit_confounded": bool(roster_row.get("debt_limit_confounded", False)),
                        "on_rrp_tau8_sign_reversal": bool(
                            roster_row.get("on_rrp_tau8_sign_reversal", False)
                        ),
                        "claim_use": _event_sign_claim_use(outcome, table_role, roster_row),
                    }
                )
    return pd.DataFrame(rows, columns=LARGE_REBUILD_EVENT_SIGN_COLUMNS)


def weekly_large_rebuild_abnormal_changes(
    root: Path,
    panel: pd.DataFrame,
    clean_events: pd.DataFrame,
) -> pd.DataFrame:
    roster = weekly_large_rebuild_event_roster(root, clean_events)
    if roster.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_ABNORMAL_COLUMNS)
    events = _large_rebuild_main_events(clean_events, roster)
    if events.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_ABNORMAL_COLUMNS)
    event_study = weekly_large_rebuild_event_study(panel, events)
    pseudo_events = _state_matched_pseudo_events(
        panel,
        events,
        set(pd.to_datetime(events["start_date"], errors="coerce").dropna()),
        outcome="reserves",
    )
    pseudo_study = weekly_large_rebuild_event_study(panel, pseudo_events)
    rows: list[dict[str, object]] = []
    for tau in [*LARGE_REBUILD_MAIN_TAUS, *LARGE_REBUILD_SECONDARY_TAUS]:
        actual_cell = event_study.loc[
            (event_study["outcome"] == "reserves") & (event_study["tau"] == tau)
        ].dropna(subset=["level_change_from_baseline"])
        if actual_cell.empty:
            continue
        cell_actual_mean = float(actual_cell["level_change_from_baseline"].mean())
        cell_pseudo = pseudo_study.loc[
            (pseudo_study["outcome"] == "reserves") & (pseudo_study["tau"] == tau)
        ].dropna(subset=["level_change_from_baseline"])
        cell_pseudo_mean = (
            float(cell_pseudo["level_change_from_baseline"].mean())
            if not cell_pseudo.empty
            else np.nan
        )
        for actual in actual_cell.sort_values("event_start").itertuples(index=False):
            pseudo = pseudo_study.loc[
                (pseudo_study["matched_event_id"] == actual.event_id)
                & (pseudo_study["outcome"] == "reserves")
                & (pseudo_study["tau"] == tau)
            ].dropna(subset=["level_change_from_baseline"])
            pseudo_mean = (
                float(pseudo["level_change_from_baseline"].mean()) if not pseudo.empty else np.nan
            )
            actual_change = float(actual.level_change_from_baseline)
            rows.append(
                {
                    "targeted_design": LARGE_REBUILD_DESIGN,
                    "event_id": actual.event_id,
                    "event_start": actual.event_start,
                    "outcome": "reserves",
                    "tau": int(tau),
                    "table_role": "headline_reserves_main"
                    if tau in LARGE_REBUILD_MAIN_TAUS
                    else "secondary_reserves_if_survives",
                    "actual_change": actual_change,
                    "matched_pseudo_mean_change": pseudo_mean,
                    "abnormal_change": actual_change - pseudo_mean
                    if pd.notna(pseudo_mean)
                    else np.nan,
                    "matched_pseudo_n": int(len(pseudo)),
                    "match_tiers": _match_tiers_for_event(pseudo_events, actual.event_id),
                    "min_state_distance": _state_distance_for_event(pseudo_events, actual.event_id, "min"),
                    "median_state_distance": _state_distance_for_event(
                        pseudo_events, actual.event_id, "median"
                    ),
                    "pseudo_abs_p90": float(pseudo["level_change_from_baseline"].abs().quantile(0.90))
                    if not pseudo.empty
                    else np.nan,
                    "cell_actual_mean": cell_actual_mean,
                    "cell_matched_pseudo_mean": cell_pseudo_mean,
                    "cell_abnormal_mean": cell_actual_mean - cell_pseudo_mean
                    if pd.notna(cell_pseudo_mean)
                    else np.nan,
                    "claim_use": "descriptive_targeted_plumbing_evidence"
                    if tau in LARGE_REBUILD_MAIN_TAUS
                    else "do_not_promote",
                }
            )
    return pd.DataFrame(rows, columns=LARGE_REBUILD_ABNORMAL_COLUMNS)


def weekly_large_rebuild_match_quality(
    root: Path,
    panel: pd.DataFrame,
    clean_events: pd.DataFrame,
) -> pd.DataFrame:
    roster = weekly_large_rebuild_event_roster(root, clean_events)
    if roster.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_MATCH_QUALITY_COLUMNS)
    events = _large_rebuild_main_events(clean_events, roster)
    if events.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_MATCH_QUALITY_COLUMNS)
    pseudo_events = _state_matched_pseudo_events(
        panel,
        events,
        set(pd.to_datetime(events["start_date"], errors="coerce").dropna()),
        outcome="reserves",
    )
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        start = pd.to_datetime(event.start_date, errors="coerce")
        event_rows = _pseudo_rows_for_event(pseudo_events, event.event_id)
        tiers = _match_tiers_for_event(pseudo_events, event.event_id)
        state_distances = _tag_float_series(event_rows, "state_distance")
        week_distances = _tag_float_series(event_rows, "week_distance")
        status = "ok"
        if len(event_rows) < 5:
            status = "too_few_pseudo_matches"
        elif "all_eligible_state" in tiers:
            status = "uses_fallback_match_tier"
        rows.append(
            {
                "targeted_design": LARGE_REBUILD_DESIGN,
                "event_id": event.event_id,
                "event_start": _date_or_blank(start),
                "outcome": "reserves",
                "matched_pseudo_n": int(len(event_rows)),
                "match_tiers": tiers,
                "worst_match_tier": _worst_match_tier(event_rows),
                "min_state_distance": _series_stat(state_distances, "min"),
                "median_state_distance": _series_stat(state_distances, "median"),
                "max_state_distance": _series_stat(state_distances, "max"),
                "min_week_distance": _series_stat(week_distances, "min"),
                "median_week_distance": _series_stat(week_distances, "median"),
                "max_week_distance": _series_stat(week_distances, "max"),
                "status": status,
                "interpretation": "event-level quality diagnostic for reserves state-matched pseudo-events",
            }
        )
    return pd.DataFrame(rows, columns=LARGE_REBUILD_MATCH_QUALITY_COLUMNS)


def weekly_large_rebuild_randomization_inference(
    root: Path,
    panel: pd.DataFrame,
    clean_events: pd.DataFrame,
    *,
    draws: int = 2_000,
    seed: int = 31,
) -> pd.DataFrame:
    roster = weekly_large_rebuild_event_roster(root, clean_events)
    if roster.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_RANDOMIZATION_COLUMNS)
    events = _large_rebuild_main_events(clean_events, roster)
    if events.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_RANDOMIZATION_COLUMNS)
    actual_study = weekly_large_rebuild_event_study(panel, events)
    pseudo_events = _state_matched_pseudo_events(
        panel,
        events,
        set(pd.to_datetime(events["start_date"], errors="coerce").dropna()),
        outcome="reserves",
    )
    pseudo_study = weekly_large_rebuild_event_study(panel, pseudo_events)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    event_ids = sorted(events["event_id"].astype(str).unique())
    for tau in LARGE_REBUILD_MAIN_TAUS:
        actual = actual_study.loc[
            (actual_study["outcome"] == "reserves") & (actual_study["tau"] == tau)
        ].dropna(subset=["level_change_from_baseline"])
        actual_mean = (
            float(actual["level_change_from_baseline"].astype(float).mean())
            if not actual.empty
            else np.nan
        )
        pseudo_by_event: list[np.ndarray] = []
        for event_id in event_ids:
            values = (
                pseudo_study.loc[
                    (pseudo_study["matched_event_id"].astype(str) == event_id)
                    & (pseudo_study["outcome"] == "reserves")
                    & (pseudo_study["tau"] == tau),
                    "level_change_from_baseline",
                ]
                .dropna()
                .astype(float)
                .to_numpy()
            )
            if values.size:
                pseudo_by_event.append(values)
        if pd.isna(actual_mean) or len(pseudo_by_event) < len(event_ids):
            rows.append(
                _large_rebuild_randomization_row(
                    tau=tau,
                    event_count=len(event_ids),
                    matched_event_groups_n=len(pseudo_by_event),
                    draws=0,
                    actual_mean=actual_mean,
                    null_draws=np.array([], dtype=float),
                    status="insufficient_pseudo_matches",
                )
            )
            continue
        null_draws = np.array(
            [np.mean([rng.choice(values) for values in pseudo_by_event]) for _ in range(draws)],
            dtype=float,
        )
        rows.append(
            _large_rebuild_randomization_row(
                tau=tau,
                event_count=len(event_ids),
                matched_event_groups_n=len(pseudo_by_event),
                draws=draws,
                actual_mean=actual_mean,
                null_draws=null_draws,
                status="estimated",
            )
        )
    return pd.DataFrame(rows, columns=LARGE_REBUILD_RANDOMIZATION_COLUMNS)


def _large_rebuild_randomization_row(
    *,
    tau: int,
    event_count: int,
    matched_event_groups_n: int,
    draws: int,
    actual_mean: float,
    null_draws: np.ndarray,
    status: str,
) -> dict[str, object]:
    if null_draws.size and pd.notna(actual_mean):
        p_value = float((np.abs(null_draws) >= abs(float(actual_mean))).mean())
        null_mean = float(null_draws.mean())
        null_p05 = float(np.quantile(null_draws, 0.05))
        null_p50 = float(np.quantile(null_draws, 0.50))
        null_p95 = float(np.quantile(null_draws, 0.95))
    else:
        p_value = np.nan
        null_mean = np.nan
        null_p05 = np.nan
        null_p50 = np.nan
        null_p95 = np.nan
    return {
        "targeted_design": LARGE_REBUILD_DESIGN,
        "outcome": "reserves",
        "tau": int(tau),
        "event_count": int(event_count),
        "matched_event_groups_n": int(matched_event_groups_n),
        "draws": int(draws),
        "actual_mean_change": actual_mean,
        "null_mean_change": null_mean,
        "null_p05": null_p05,
        "null_p50": null_p50,
        "null_p95": null_p95,
        "two_sided_p_value": p_value,
        "placebo_design": "state_matched_event_level_randomization",
        "status": status,
        "interpretation": (
            "event-level randomization diagnostic from state-matched pseudo-events; "
            "small-sample descriptive check, not standalone identification"
        ),
    }


def weekly_large_rebuild_final_review(
    roster: pd.DataFrame,
    cell_summary: pd.DataFrame,
    event_sign: pd.DataFrame,
    abnormal_changes: pd.DataFrame,
    match_quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if roster.empty or cell_summary.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_FINAL_REVIEW_COLUMNS)
    headline = cell_summary.loc[cell_summary["table_role"] == "headline_reserves_main"]
    passing = headline.loc[headline["status"] == "passes_targeted_descriptive_gates"]
    blocked = headline.loc[headline["status"] != "passes_targeted_descriptive_gates"]
    limitations = [
        "five_event_main_sample",
        "event_038_debt_limit_appendix_only",
        "on_rrp_appendix_only",
        "not_causal_bill_issuance_claim",
    ]
    if _has_on_rrp_reversal(roster):
        limitations.append("on_rrp_tau8_sign_reversal_flags")
    if _has_large_event_concentration(event_sign):
        limitations.append("event_contribution_concentration_review_required")
    if abnormal_changes.empty:
        limitations.append("abnormal_change_table_missing")
    if match_quality is not None and _has_match_quality_blocker(match_quality):
        limitations.append("event_match_quality_review_required")
    status = (
        "descriptive_reserves_plumbing_ready_for_writeup"
        if len(passing) == len(headline) and len(headline) > 0
        else "blocked_for_headline_writeup"
    )
    return pd.DataFrame(
        [
            {
                "targeted_design": LARGE_REBUILD_DESIGN,
                "review_scope": "weekly_large_rebuild_targeted_design",
                "status": status,
                "claim_use": "descriptive_targeted_plumbing_evidence"
                if status == "descriptive_reserves_plumbing_ready_for_writeup"
                else "do_not_promote",
                "main_event_count": int((roster["sample_role"] == "main").sum()),
                "headline_passing_cells": int(len(passing)),
                "headline_blocked_cells": int(len(blocked)),
                "headline_cells": _format_cell_list(headline),
                "appendix_cells": _format_cell_list(
                    cell_summary.loc[cell_summary["table_role"].str.startswith("appendix")]
                ),
                "binding_limitations": ";".join(limitations),
                "required_next_step": (
                    "draft_context_heavy_reserves_plumbing_narrative_with_event_level_tables"
                    if status == "descriptive_reserves_plumbing_ready_for_writeup"
                    else "do_not_write_headline_targeted_result"
                ),
            }
        ],
        columns=LARGE_REBUILD_FINAL_REVIEW_COLUMNS,
    )


def weekly_large_rebuild_blocker_summary(cell_summary: pd.DataFrame) -> pd.DataFrame:
    if cell_summary.empty:
        return pd.DataFrame(columns=LARGE_REBUILD_BLOCKER_SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in cell_summary.itertuples(index=False):
        blockers = [
            blocker
            for blocker in str(getattr(row, "blocker_flags", "") or "").split(";")
            if blocker and blocker.lower() != "nan"
        ]
        status = str(getattr(row, "status", ""))
        rows.append(
            {
                "targeted_design": getattr(row, "targeted_design", LARGE_REBUILD_DESIGN),
                "table_role": getattr(row, "table_role", ""),
                "outcome": getattr(row, "outcome", ""),
                "tau": int(getattr(row, "tau", 0)),
                "status": status,
                "blocker_count": int(len(blockers)),
                "primary_blocker": blockers[0] if blockers else "",
                "blocker_flags": ";".join(blockers),
                "randomization_p_value": getattr(row, "randomization_p_value", np.nan),
                "randomization_p_max": getattr(row, "randomization_p_max", np.nan),
                "match_quality_status": getattr(row, "match_quality_status", ""),
                "next_step": _large_rebuild_blocker_next_step(status, blockers),
            }
        )
    return pd.DataFrame(rows, columns=LARGE_REBUILD_BLOCKER_SUMMARY_COLUMNS)


def _large_rebuild_blocker_next_step(status: str, blockers: list[str]) -> str:
    if status == "passes_targeted_descriptive_gates":
        return "retain_as_passing_diagnostic_cell"
    if "randomization_p_value_high" in blockers:
        return "review_randomization_distribution_and_event_heterogeneity"
    if any(blocker.startswith("event_match_quality_") for blocker in blockers):
        return "review_event_level_pseudo_match_quality"
    if "placebo_false_positive_high" in blockers:
        return "review_state_matched_placebo_distribution"
    if "bootstrap_ci_crosses_zero" in blockers or "bootstrap_unstable" in blockers:
        return "review_event_bootstrap_and_leave_one_out_sensitivity"
    return "retain_blocked_status_until_design_gate_is_resolved"


def weekly_large_rebuild_event_time_svg(
    root: Path,
    panel: pd.DataFrame,
    clean_events: pd.DataFrame,
    *,
    outcome: str,
    path: Path,
) -> None:
    roster = weekly_large_rebuild_event_roster(root, clean_events)
    events = _large_rebuild_events_for_roster(clean_events, roster)
    if roster.empty or events.empty:
        _write_empty_svg(path, f"No {outcome} large-rebuild event-time data")
        return
    event_study = weekly_large_rebuild_event_study(panel, events)
    rows = event_study.loc[event_study["outcome"] == outcome].dropna(
        subset=["level_change_from_baseline"]
    )
    if rows.empty:
        _write_empty_svg(path, f"No {outcome} large-rebuild event-time data")
        return
    title = (
        "Large TGA rebuild event-time paths: reserves"
        if outcome == "reserves"
        else "Large TGA rebuild event-time paths: ON RRP"
    )
    _write_event_time_svg(path, rows, roster, title=title, outcome=outcome)


def weekly_large_rebuild_diagnostic_writeup(
    roster: pd.DataFrame,
    cell_summary: pd.DataFrame,
    event_sign: pd.DataFrame,
    abnormal_changes: pd.DataFrame,
    final_review: pd.DataFrame,
    *,
    path: Path,
) -> None:
    ensure_dir(path.parent)
    if final_review.empty:
        path.write_text(
            "# Weekly Large-Rebuild Diagnostic\n\nNo targeted large-rebuild review is available.\n",
            encoding="utf-8",
        )
        return
    final = final_review.iloc[0].to_dict()
    reserve_cells = cell_summary.loc[cell_summary["table_role"] == "headline_reserves_main"]
    on_rrp_cells = cell_summary.loc[cell_summary["outcome"] == "on_rrp"]
    lines = [
        "# Weekly Large-Rebuild Diagnostic",
        "",
        "## Diagnostic Status",
        "",
        f"- Status: `{final.get('status', '')}`.",
        f"- Interpretation use: `{final.get('claim_use', '')}`.",
        f"- Main event count: {int(final.get('main_event_count', 0))}.",
        f"- Binding limitations: `{final.get('binding_limitations', '')}`.",
        "",
        "This targeted reserves-plumbing diagnostic is currently blocked for headline writeup when any primary reserve horizon fails the matched-placebo, bootstrap, match-quality, or randomization gates.",
        "",
        "## Headline Reserve Cells",
        "",
        "| outcome | tau | mean change | placebo false positive | randomization p | blockers | status |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in reserve_cells.sort_values("tau").itertuples(index=False):
        lines.append(
            "| "
            f"{row.outcome} | {int(row.tau)} | {_format_number(row.mean_change)} | "
            f"{_format_number(row.placebo_false_positive_share, digits=3)} | "
            f"{_format_number(row.randomization_p_value, digits=3)} | "
            f"`{getattr(row, 'blocker_flags', '') or ''}` | "
            f"`{row.status}` |"
        )
    lines.extend(
        [
            "",
            "## Event-Level Interpretation",
            "",
            "- `event_038` is retained as appendix context only because it is debt-limit-confounded.",
            "- `event_039` is positive for reserves at tau 2 and tau 4, so the average reserve decline is not event-uniform.",
            "- `event_041` and `event_045` remain flagged for ON RRP mechanism-risk review.",
            "",
            "## Abnormal Reserve Changes",
            "",
            "| tau | actual mean | matched pseudo mean | abnormal mean |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    if not abnormal_changes.empty:
        abnormal_summary = abnormal_changes.groupby("tau", observed=True).agg(
            actual_mean=("actual_change", "mean"),
            pseudo_mean=("matched_pseudo_mean_change", "mean"),
            abnormal_mean=("abnormal_change", "mean"),
        )
        for tau, row in abnormal_summary.sort_index().iterrows():
            lines.append(
                "| "
                f"{int(tau)} | {_format_number(row.actual_mean)} | "
                f"{_format_number(row.pseudo_mean)} | {_format_number(row.abnormal_mean)} |"
            )
    lines.extend(
        [
            "",
            "## ON RRP",
            "",
        ]
    )
    if on_rrp_cells.empty:
        lines.append("No ON RRP targeted cell is available.")
    else:
        row = on_rrp_cells.iloc[0]
        lines.append(
            "ON RRP remains appendix-only: "
            f"tau {int(row['tau'])}, status `{row['status']}`, interpretation use `{row['claim_use']}`."
        )
    lines.extend(
        [
            "",
            "## Figure Outputs",
            "",
            "- `output/reports/weekly_large_rebuild_reserves_event_time.svg`",
            "- `output/reports/weekly_large_rebuild_on_rrp_event_time.svg`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def weekly_outcome_claim_readiness(
    panel: pd.DataFrame,
    clean_events: pd.DataFrame,
    stable: pd.DataFrame,
) -> pd.DataFrame:
    coverage = _weekly_outcome_coverage(panel)
    rows: list[dict[str, object]] = []
    event_count = int(len(clean_events))
    for outcome, coverage_share in coverage.items():
        blockers = []
        if event_count < 30:
            blockers.append("clean_positive_events_below_30")
        if coverage_share < 0.90:
            blockers.append("outcome_coverage_below_90pct")
        outcome_stable = stable.loc[stable["outcome"] == outcome] if not stable.empty else pd.DataFrame()
        stable_cells = int((outcome_stable["status"] == "passes_all_gates").sum()) if not outcome_stable.empty else 0
        blocked_cells = int((outcome_stable["status"] == "blocked").sum()) if not outcome_stable.empty else 0
        if stable_cells == 0:
            blockers.append("no_stable_outcome_tau_cells")
        rows.append(
            {
                "outcome": outcome,
                "readiness_status": "blocked" if blockers else "descriptive_stable_pending_manual_validation",
                "clean_positive_events_n": event_count,
                "outcome_coverage_share": coverage_share,
                "stable_cells_n": stable_cells,
                "blocked_cells_n": blocked_cells,
                "blocker_flags": ";".join(blockers),
                "claim_use": "descriptive_channel_triage" if stable_cells else "do_not_promote",
            }
        )
    return pd.DataFrame(rows).sort_values(["readiness_status", "outcome"]).reset_index(drop=True)


def weekly_claim_readiness(
    panel: pd.DataFrame,
    clean_events: pd.DataFrame,
    lp: pd.DataFrame | None = None,
    event_study: pd.DataFrame | None = None,
    leave_one_out: pd.DataFrame | None = None,
    bootstrap: pd.DataFrame | None = None,
    placebo: pd.DataFrame | None = None,
) -> pd.DataFrame:
    event_count = int(len(clean_events))
    timing = weekly_timing_alignment_qa(panel)
    min_coverage = float(timing["coverage_share"].min()) if not timing.empty else 0.0
    blockers: list[str] = []
    if event_count < 30:
        blockers.append("clean_positive_events_below_30")
    if min_coverage < 0.90:
        blockers.append("outcome_coverage_below_90pct")
    lp_estimated = 0
    if lp is None or lp.empty or "status" not in lp.columns:
        blockers.append("weekly_hac_lp_missing")
    else:
        lp_estimated = int((lp["status"] == "estimated").sum())
        if "predictor" not in lp.columns:
            blockers.append("weekly_hac_lp_shock_rows_missing")
        else:
            shock_rows = lp.loc[
                (lp["status"] == "estimated")
                & (lp["predictor"] == "bill_size_surprise_100b")
            ]
            if shock_rows.empty:
                blockers.append("weekly_hac_lp_shock_rows_missing")
    event_study_rows = 0
    if event_study is None or event_study.empty:
        blockers.append("event_time_design_not_yet_estimated")
    else:
        event_study_rows = int(len(event_study))
    leave_one_out_rows = 0
    min_same_sign_share = np.nan
    if leave_one_out is None or leave_one_out.empty:
        blockers.append("leave_one_event_out_not_estimated")
    else:
        leave_one_out_rows = int(len(leave_one_out))
        if "same_sign_leave_one_out_share" in leave_one_out.columns:
            shares = leave_one_out.loc[
                leave_one_out["status"] == "estimated",
                "same_sign_leave_one_out_share",
            ].dropna()
            if not shares.empty:
                min_same_sign_share = float(shares.min())
                if min_same_sign_share < 0.80:
                    blockers.append("leave_one_event_out_sign_instability")
    bootstrap_rows = 0
    min_bootstrap_sign_share = np.nan
    bootstrap_ci_cross_zero = np.nan
    if bootstrap is None or bootstrap.empty:
        blockers.append("block_bootstrap_not_estimated")
    else:
        bootstrap_rows = int(len(bootstrap))
        estimated = bootstrap.loc[bootstrap["status"] == "estimated"].copy()
        if estimated.empty:
            blockers.append("block_bootstrap_not_estimated")
        else:
            min_bootstrap_sign_share = float(estimated["bootstrap_same_sign_share"].dropna().min())
            bootstrap_ci_cross_zero = bool(
                ((estimated["ci_lower"] <= 0) & (estimated["ci_upper"] >= 0)).any()
            )
            if min_bootstrap_sign_share < 0.80:
                blockers.append("block_bootstrap_sign_instability")
            if bootstrap_ci_cross_zero:
                blockers.append("block_bootstrap_ci_crosses_zero")
    placebo_rows = 0
    max_placebo_false_positive_share = np.nan
    if placebo is None or placebo.empty:
        blockers.append("placebo_tests_not_estimated")
    else:
        placebo_rows = int(len(placebo))
        estimated_placebos = placebo.loc[placebo["status"] == "estimated"]
        if estimated_placebos.empty:
            blockers.append("placebo_tests_not_estimated")
        elif "false_positive_share" in estimated_placebos.columns:
            max_placebo_false_positive_share = float(
                estimated_placebos["false_positive_share"].dropna().max()
            )
            if max_placebo_false_positive_share >= 0.10:
                blockers.append("placebo_false_positive_rate_high")
    return pd.DataFrame(
        [
            {
                "design": "tgarefill_weekly_bill_surprise",
                "readiness_status": "blocked"
                if blockers
                else "causal_candidate_pending_manual_validation",
                "clean_positive_events_n": event_count,
                "min_outcome_coverage_share": min_coverage,
                "estimated_lp_rows": lp_estimated,
                "event_study_rows": event_study_rows,
                "leave_one_event_out_rows": leave_one_out_rows,
                "min_leave_one_out_same_sign_share": min_same_sign_share,
                "block_bootstrap_rows": bootstrap_rows,
                "min_bootstrap_same_sign_share": min_bootstrap_sign_share,
                "block_bootstrap_ci_crosses_zero": bootstrap_ci_cross_zero,
                "placebo_rows": placebo_rows,
                "max_placebo_false_positive_share": max_placebo_false_positive_share,
                "blocker_flags": ";".join(blockers),
                "required_next_step": "tighten_event_filters_or_restrict_claims_to_stable_outcomes",
            }
        ]
    )


def _write_weekly_report(
    panel: pd.DataFrame,
    clean_events: pd.DataFrame,
    exclusions: pd.DataFrame,
    lp: pd.DataFrame,
    event_study: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    bootstrap: pd.DataFrame,
    placebo: pd.DataFrame,
    filter_sensitivity: pd.DataFrame,
    large_rebuild_review: pd.DataFrame,
    large_rebuild_calendar: pd.DataFrame,
    stable: pd.DataFrame,
    outcome_readiness: pd.DataFrame,
    readiness: pd.DataFrame,
    path: Path,
) -> None:
    status = readiness.iloc[0]["readiness_status"] if not readiness.empty else "unknown"
    estimated_lp_rows = int((lp["status"] == "estimated").sum()) if not lp.empty else 0
    shock_lp_rows = (
        int(((lp["status"] == "estimated") & (lp["predictor"] == "bill_size_surprise_100b")).sum())
        if not lp.empty and "predictor" in lp.columns
        else 0
    )
    event_study_rows = len(event_study)
    leave_one_out_rows = len(leave_one_out)
    bootstrap_rows = len(bootstrap)
    placebo_rows = len(placebo)
    stable_rows = (
        int((stable["status"] == "passes_all_gates").sum())
        if not stable.empty and "status" in stable.columns
        else 0
    )
    sensitivity_leads = (
        int((filter_sensitivity["decision"] == "small_sample_sensitivity_only").sum())
        if not filter_sensitivity.empty and "decision" in filter_sensitivity.columns
        else 0
    )
    large_rebuild_cells = (
        int(large_rebuild_review[["outcome", "tau"]].drop_duplicates().shape[0])
        if not large_rebuild_review.empty and {"outcome", "tau"}.issubset(large_rebuild_review.columns)
        else 0
    )
    large_rebuild_context_rows = len(large_rebuild_calendar)
    outcome_ready_rows = (
        int((outcome_readiness["readiness_status"] == "descriptive_stable_pending_manual_validation").sum())
        if not outcome_readiness.empty and "readiness_status" in outcome_readiness.columns
        else 0
    )
    text = "\n".join(
        [
            "# Weekly Identification Candidate Report",
            "",
            f"- Weekly panel rows: {len(panel)}",
            f"- Clean event candidates: {len(clean_events)}",
            f"- Excluded event candidates: {len(exclusions)}",
            f"- Estimated HAC LP coefficient rows: {estimated_lp_rows}",
            f"- Estimated bill-surprise HAC LP rows: {shock_lp_rows}",
            f"- Weekly event-study rows: {event_study_rows}",
            f"- Leave-one-event-out rows: {leave_one_out_rows}",
            f"- Block-bootstrap rows: {bootstrap_rows}",
            f"- Placebo-test rows: {placebo_rows}",
            f"- Stable outcome/tau cells passing all gates: {stable_rows}",
            f"- Small-sample filter variants with passing cells: {sensitivity_leads}",
            f"- Large-rebuild targeted review cells: {large_rebuild_cells}",
            f"- Large-rebuild calendar validation rows: {large_rebuild_context_rows}",
            f"- Outcome-level descriptive-stable channels: {outcome_ready_rows}",
            f"- Design readiness: {status}",
            "",
            "The weekly HAC local-projection, event-time, leave-one-out, bootstrap, and placebo diagnostics are now estimated.",
            "Current design status is governed by `weekly_design_readiness.csv`.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce")


def _weekly_outcome_coverage(panel: pd.DataFrame) -> dict[str, float]:
    outcomes = ["deposits", "on_rrp", "reserves", "mmf_treasury_holdings"]
    return {
        outcome: float(panel[outcome].notna().mean()) if outcome in panel.columns and len(panel) else 0.0
        for outcome in outcomes
    }


def _weekly_event_filter_variants(
    clean_events: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame, str]]:
    events = clean_events.copy()
    starts = pd.to_datetime(events["start_date"], errors="coerce")
    bill_share = pd.to_numeric(events["mean_bill_share"], errors="coerce")
    rebuild = pd.to_numeric(events["delta_tga_event"], errors="coerce")
    tags = events["manual_tags"].fillna("").astype(str)
    tax_like = starts.dt.month.isin([4, 6, 9, 12]) & starts.dt.day.between(10, 20)
    debt_limit = tags.str.contains("debt", case=False, na=False)
    variants = [
        ("all_clean", pd.Series(True, index=events.index), "current clean event set"),
        ("exclude_tax_like_weeks", ~tax_like, "exclude tax-window starts"),
        ("exclude_debt_limit_tags", ~debt_limit, "exclude debt-limit tagged events"),
        ("bill_share_ge_075", bill_share >= 0.75, "bill-share at least 75 percent"),
        ("bill_share_ge_080", bill_share >= 0.80, "bill-share at least 80 percent"),
        ("rebuild_ge_150b", rebuild >= 150_000.0, "TGA rebuild at least $150 billion"),
        ("rebuild_ge_200b", rebuild >= 200_000.0, "TGA rebuild at least $200 billion"),
        ("post_2021", starts >= pd.Timestamp("2021-01-01"), "post-2021 monetary plumbing regime"),
        (
            "post_2021_bill_share_ge_075",
            (starts >= pd.Timestamp("2021-01-01")) & (bill_share >= 0.75),
            "post-2021 and bill-share at least 75 percent",
        ),
    ]
    return [
        (name, events.loc[mask.fillna(False)].reset_index(drop=True), description)
        for name, mask, description in variants
    ]


def _targeted_event_span(event: object) -> str:
    if isinstance(event, pd.Series):
        start = event.get("start_date", "")
        end = event.get("end_date", "")
    else:
        start = ""
        end = ""
    start_text = _date_or_blank(pd.to_datetime(start, errors="coerce"))
    end_text = _date_or_blank(pd.to_datetime(end, errors="coerce"))
    if not start_text and not end_text:
        return ""
    return f"{start_text}:{end_text}"


def _load_weekly_large_rebuild_context(root: Path) -> pd.DataFrame:
    path = root / "data" / "manual" / "weekly_large_rebuild_calendar_context.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _large_rebuild_main_events(clean_events: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    if clean_events.empty or roster.empty:
        return pd.DataFrame()
    main_ids = set(roster.loc[roster["sample_role"] == "main", "event_id"].astype(str))
    events = clean_events.loc[clean_events["event_id"].astype(str).isin(main_ids)].copy()
    return events.sort_values("start_date").reset_index(drop=True)


def _large_rebuild_events_for_roster(clean_events: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    if clean_events.empty or roster.empty:
        return pd.DataFrame()
    event_ids = set(roster["event_id"].astype(str))
    events = clean_events.loc[clean_events["event_id"].astype(str).isin(event_ids)].copy()
    return events.sort_values("start_date").reset_index(drop=True)


def _roster_row(roster: pd.DataFrame, event_id: object) -> dict[str, object]:
    if roster.empty or "event_id" not in roster.columns:
        return {}
    rows = roster.loc[roster["event_id"].astype(str) == str(event_id)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _event_sign_claim_use(
    outcome: str,
    table_role: str,
    roster_row: dict[str, object],
) -> str:
    if roster_row.get("sample_role") != "main":
        return "appendix_context_only"
    if outcome == "reserves" and table_role == "headline_reserves_main":
        return "descriptive_targeted_plumbing_evidence"
    return "appendix_mechanism_diagnostic_only"


def _event_sign_reference_group(
    group: pd.DataFrame,
    roster: pd.DataFrame,
    table_role: str,
) -> pd.DataFrame:
    if table_role == "headline_reserves_main":
        main_ids = set(roster.loc[roster["sample_role"] == "main", "event_id"].astype(str))
        main = group.loc[group["event_id"].astype(str).isin(main_ids)]
        return main if not main.empty else group
    return group


def _event_in_reference_sample(table_role: str, sample_role: object) -> bool:
    if table_role == "headline_reserves_main":
        return str(sample_role) == "main"
    return True


def _weekly_large_rebuild_state_matched_placebo(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    actual_event_study: pd.DataFrame,
    *,
    outcome: str,
) -> pd.DataFrame:
    if events.empty or actual_event_study.empty:
        return pd.DataFrame()
    pseudo_df = _state_matched_pseudo_events(
        panel,
        events,
        set(pd.to_datetime(events["start_date"], errors="coerce").dropna()),
        outcome=outcome,
    )
    pseudo_rows = weekly_large_rebuild_event_study(panel, pseudo_df)
    actual_summary = weekly_event_study_summary(actual_event_study)
    rows: list[dict[str, object]] = []
    for actual in actual_summary.loc[
        (actual_summary["outcome"] == outcome)
        & (actual_summary["tau"].isin([*LARGE_REBUILD_MAIN_TAUS, *LARGE_REBUILD_SECONDARY_TAUS]))
    ].itertuples(index=False):
        pseudo = (
            pseudo_rows.loc[
                (pseudo_rows["outcome"] == actual.outcome) & (pseudo_rows["tau"] == actual.tau),
                "level_change_from_baseline",
            ]
            .dropna()
            .astype(float)
        )
        if pseudo.empty:
            rows.append(
                {
                    "outcome": actual.outcome,
                    "tau": int(actual.tau),
                    "status": "no_pseudo_events",
                    "actual_mean_change": actual.mean_change,
                    "pseudo_events_n": 0,
                    "matched_event_groups_n": 0,
                    "placebo_design": "state_matched_large_rebuild_pseudo",
                }
            )
            continue
        threshold = abs(float(actual.mean_change))
        rows.append(
            {
                "outcome": actual.outcome,
                "tau": int(actual.tau),
                "status": "estimated",
                "actual_mean_change": actual.mean_change,
                "pseudo_events_n": int(pseudo.size),
                "matched_event_groups_n": int(pseudo_df["matched_event_id"].nunique())
                if "matched_event_id" in pseudo_df.columns
                else 0,
                "pseudo_mean_change": float(pseudo.mean()),
                "pseudo_abs_p90": float(pseudo.abs().quantile(0.90)),
                "false_positive_share": float((pseudo.abs() >= threshold).mean())
                if threshold
                else 1.0,
                "worst_match_tier": _worst_match_tier(pseudo_df),
                "placebo_design": "state_matched_large_rebuild_pseudo",
            }
        )
    return pd.DataFrame(rows)


def _state_matched_pseudo_events(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    event_starts: set[pd.Timestamp],
    *,
    outcome: str,
    max_matches_per_event: int = 25,
    min_matches_per_event: int = 5,
) -> pd.DataFrame:
    pseudo_weeks = _pseudo_event_weeks(panel, event_starts)
    if not pseudo_weeks:
        return pd.DataFrame()
    panel = panel.sort_values("week").reset_index(drop=True)
    feature_panel = _weekly_match_features(panel)
    state = _pre_event_state_features(panel, outcome)
    feature_panel = feature_panel.merge(state, on="week", how="left")
    candidates = feature_panel.loc[feature_panel["week"].isin(pseudo_weeks)].copy()
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        start = pd.to_datetime(event.start_date, errors="coerce")
        if pd.isna(start):
            continue
        event_feature = feature_panel.loc[feature_panel["week"] == start]
        if event_feature.empty:
            continue
        matched = _select_state_matched_pseudo_weeks(
            candidates,
            event_feature.iloc[0],
            max_matches=max_matches_per_event,
            min_matches=min_matches_per_event,
        )
        for match_idx, row in enumerate(matched.itertuples(index=False)):
            rows.append(
                {
                    "event_id": f"pseudo_{event.event_id}_{match_idx:03d}",
                    "matched_event_id": event.event_id,
                    "start_date": row.week,
                    "baseline_date": row.week - pd.Timedelta(weeks=1),
                    "end_date": row.week,
                    "delta_tga_event": 0.0,
                    "mean_bill_share": np.nan,
                    "manual_tags": (
                        f"pseudo;match_tier={row.match_tier};"
                        f"state_distance={row.state_distance:.6f};"
                        f"week_distance={row.week_distance:.6f}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _match_tiers_for_event(pseudo_events: pd.DataFrame, event_id: object) -> str:
    event_rows = _pseudo_rows_for_event(pseudo_events, event_id)
    tiers: list[str] = []
    for tag in event_rows.get("manual_tags", pd.Series(dtype=str)).fillna("").astype(str):
        tier = _tag_value(tag, "match_tier")
        if tier and tier not in tiers:
            tiers.append(tier)
    return ";".join(tiers)


def _state_distance_for_event(pseudo_events: pd.DataFrame, event_id: object, stat: str) -> float:
    event_rows = _pseudo_rows_for_event(pseudo_events, event_id)
    values = [
        float(value)
        for value in (
            _tag_value(tag, "state_distance")
            for tag in event_rows.get("manual_tags", pd.Series(dtype=str)).fillna("").astype(str)
        )
        if value
    ]
    if not values:
        return np.nan
    series = pd.Series(values, dtype="float64")
    if stat == "min":
        return float(series.min())
    if stat == "median":
        return float(series.median())
    raise ValueError(f"unknown state-distance stat: {stat}")


def _tag_float_series(rows: pd.DataFrame, key: str) -> pd.Series:
    values = [
        float(value)
        for value in (
            _tag_value(tag, key)
            for tag in rows.get("manual_tags", pd.Series(dtype=str)).fillna("").astype(str)
        )
        if value
    ]
    return pd.Series(values, dtype="float64")


def _series_stat(values: pd.Series, stat: str) -> float:
    if values.empty:
        return np.nan
    if stat == "min":
        return float(values.min())
    if stat == "median":
        return float(values.median())
    if stat == "max":
        return float(values.max())
    raise ValueError(f"unknown series stat: {stat}")


def _pseudo_rows_for_event(pseudo_events: pd.DataFrame, event_id: object) -> pd.DataFrame:
    if pseudo_events.empty or "matched_event_id" not in pseudo_events.columns:
        return pd.DataFrame()
    return pseudo_events.loc[pseudo_events["matched_event_id"].astype(str) == str(event_id)]


def _tag_value(tag: str, key: str) -> str:
    prefix = f"{key}="
    for part in str(tag).split(";"):
        if part.startswith(prefix):
            return part.removeprefix(prefix)
    return ""


def _has_on_rrp_reversal(roster: pd.DataFrame) -> bool:
    if roster.empty or "on_rrp_tau8_sign_reversal" not in roster.columns:
        return False
    return bool(roster["on_rrp_tau8_sign_reversal"].fillna(False).astype(bool).any())


def _has_large_event_concentration(event_sign: pd.DataFrame) -> bool:
    if event_sign.empty or "contribution_share_of_abs_sum" not in event_sign.columns:
        return False
    headline = event_sign.loc[event_sign["table_role"] == "headline_reserves_main"]
    if headline.empty:
        return False
    return bool((headline["contribution_share_of_abs_sum"] > 0.50).any())


def _format_cell_list(cells: pd.DataFrame) -> str:
    if cells.empty:
        return ""
    return ";".join(
        f"{row.outcome}:tau{int(row.tau)}={row.status}"
        for row in cells.sort_values(["outcome", "tau"]).itertuples(index=False)
    )


def _write_empty_svg(path: Path, message: str) -> None:
    ensure_dir(path.parent)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" '
        'viewBox="0 0 900 520">'
        '<rect width="900" height="520" fill="#ffffff"/>'
        f'<text x="450" y="260" text-anchor="middle" font-family="Arial" '
        f'font-size="18" fill="#333333">{escape(message)}</text></svg>'
    )
    path.write_text(svg, encoding="utf-8")


def _write_event_time_svg(
    path: Path,
    rows: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    title: str,
    outcome: str,
) -> None:
    ensure_dir(path.parent)
    width = 980
    height = 620
    left = 78
    right = 235
    top = 72
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    taus = list(range(-4, 9))
    y_values = rows["level_change_from_baseline"].astype(float)
    y_min = min(float(y_values.min()), 0.0)
    y_max = max(float(y_values.max()), 0.0)
    pad = max((y_max - y_min) * 0.10, 1.0)
    y_min -= pad
    y_max += pad
    colors = ["#0f766e", "#b45309", "#2563eb", "#be123c", "#4d7c0f", "#7c3aed"]

    def x_pos(tau: int) -> float:
        return left + ((tau - min(taus)) / (max(taus) - min(taus))) * plot_w

    def y_pos(value: float) -> float:
        return top + ((y_max - value) / (y_max - y_min)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Arial" font-size="22" '
        f'font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{left}" y="56" font-family="Arial" font-size="13" '
        f'fill="#4b5563">Change from mean tau -4:-2 baseline, USD millions</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#6b7280" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#6b7280" stroke-width="1"/>',
    ]
    zero_y = y_pos(0.0)
    parts.append(
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}" '
        'stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 4"/>'
    )
    for tau in taus:
        x = x_pos(tau)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" '
                f'y2="{top + plot_h + 5}" stroke="#6b7280" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 23}" text-anchor="middle" '
                f'font-family="Arial" font-size="11" fill="#374151">{tau}</text>',
            ]
        )
    for value in _nice_ticks(y_min, y_max, count=5):
        y = y_pos(value)
        parts.extend(
            [
                f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" '
                'stroke="#6b7280" stroke-width="1"/>',
                f'<text x="{left - 9}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="Arial" font-size="11" fill="#374151">{_format_number(value, digits=0)}</text>',
            ]
        )
    event_ids = list(rows["event_id"].drop_duplicates())
    for idx, event_id in enumerate(event_ids):
        group = rows.loc[rows["event_id"] == event_id].sort_values("tau")
        points = [
            f"{x_pos(int(row.tau)):.2f},{y_pos(float(row.level_change_from_baseline)):.2f}"
            for row in group.itertuples(index=False)
            if int(row.tau) in taus
        ]
        roster_row = _roster_row(roster, event_id)
        sample_role = str(roster_row.get("sample_role", ""))
        stroke = colors[idx % len(colors)]
        dash = ' stroke-dasharray="5 4"' if sample_role != "main" else ""
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{stroke}" '
            f'stroke-width="2"{dash}/>'
        )
        label_y = top + 16 + idx * 22
        parts.extend(
            [
                f'<line x1="{left + plot_w + 24}" y1="{label_y - 4}" '
                f'x2="{left + plot_w + 48}" y2="{label_y - 4}" stroke="{stroke}" '
                f'stroke-width="2"{dash}/>',
                f'<text x="{left + plot_w + 56}" y="{label_y}" font-family="Arial" '
                f'font-size="12" fill="#111827">{escape(str(event_id))} '
                f'({escape(sample_role or "event")})</text>',
            ]
        )
    mean_points = _event_time_mean_points(rows, roster, outcome)
    if mean_points:
        points = " ".join(f"{x_pos(tau):.2f},{y_pos(value):.2f}" for tau, value in mean_points)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#111827" '
            'stroke-width="3.4"/>'
        )
        parts.append(
            f'<text x="{left + plot_w + 56}" y="{top + 16 + len(event_ids) * 22}" '
            'font-family="Arial" font-size="12" font-weight="700" fill="#111827">'
            f'{escape(_mean_label(outcome))}</text>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" '
            'font-family="Arial" font-size="13" fill="#374151">Event time tau, weeks</text>',
            f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20,{top + plot_h / 2})" '
            'text-anchor="middle" font-family="Arial" font-size="13" fill="#374151">'
            "Change from baseline</text>",
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def _event_time_mean_points(
    rows: pd.DataFrame,
    roster: pd.DataFrame,
    outcome: str,
) -> list[tuple[int, float]]:
    work = rows.copy()
    if outcome == "reserves":
        main_ids = set(roster.loc[roster["sample_role"] == "main", "event_id"].astype(str))
        work = work.loc[work["event_id"].astype(str).isin(main_ids)]
    points: list[tuple[int, float]] = []
    for tau, group in work.groupby("tau", observed=True):
        points.append((int(tau), float(group["level_change_from_baseline"].mean())))
    return sorted(points)


def _mean_label(outcome: str) -> str:
    return "main-sample mean" if outcome == "reserves" else "all-event mean"


def _nice_ticks(y_min: float, y_max: float, *, count: int) -> list[float]:
    if count <= 1:
        return [y_min]
    return [y_min + (y_max - y_min) * idx / (count - 1) for idx in range(count)]


def _format_number(value: object, *, digits: int = 1) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    return f"{float(number):,.{digits}f}"


def _pre_event_state_features(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    values = pd.to_numeric(panel[outcome], errors="coerce") if outcome in panel.columns else pd.Series(np.nan)
    rows: list[dict[str, object]] = []
    for idx, week in enumerate(pd.to_datetime(panel["week"], errors="coerce")):
        baseline_indices = [idx + tau for tau in (-4, -3, -2)]
        baseline_values = [values.iloc[pos] for pos in baseline_indices if 0 <= pos < len(values)]
        baseline = pd.Series(baseline_values, dtype="float64").dropna()
        tau_minus1 = values.iloc[idx - 1] if idx - 1 >= 0 and idx - 1 < len(values) else np.nan
        rows.append(
            {
                "week": week,
                "pre_state_level": float(baseline.mean()) if len(baseline) == 3 else np.nan,
                "pre_state_tau_minus1_gap": float(tau_minus1 - baseline.mean())
                if len(baseline) == 3 and pd.notna(tau_minus1)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _select_state_matched_pseudo_weeks(
    candidates: pd.DataFrame,
    event_feature: pd.Series,
    *,
    max_matches: int,
    min_matches: int,
) -> pd.DataFrame:
    scale_level = _matching_scale(candidates["pre_state_level"], event_feature.get("pre_state_level", np.nan))
    scale_gap = _matching_scale(
        candidates["pre_state_tau_minus1_gap"],
        event_feature.get("pre_state_tau_minus1_gap", np.nan),
    )
    tiers = [
        ("month_regime_calendar_state", ["calendar_month", "tax_week", "debt_limit_window", "on_rrp_regime_pre"]),
        ("quarter_regime_calendar_state", ["calendar_quarter", "tax_week", "debt_limit_window", "on_rrp_regime_pre"]),
        ("quarter_calendar_state", ["calendar_quarter", "tax_week", "debt_limit_window"]),
        ("quarter_state", ["calendar_quarter"]),
        ("all_eligible_state", []),
    ]
    for tier, columns in tiers:
        matched = candidates.copy()
        for column in columns:
            matched = matched.loc[matched[column] == event_feature[column]]
        matched = matched.dropna(subset=["pre_state_level", "pre_state_tau_minus1_gap"])
        if len(matched) >= min_matches or tier == "all_eligible_state":
            matched = matched.copy()
            matched["match_tier"] = tier
            matched["week_distance"] = (
                matched["week_of_year"].astype(float) - float(event_feature["week_of_year"])
            ).abs()
            matched["state_distance"] = (
                (matched["pre_state_level"] - float(event_feature["pre_state_level"])).abs()
                / scale_level
                + (
                    matched["pre_state_tau_minus1_gap"]
                    - float(event_feature["pre_state_tau_minus1_gap"])
                ).abs()
                / scale_gap
            )
            return matched.sort_values(["state_distance", "week_distance", "week"]).head(
                max_matches
            )
    raise AssertionError("unreachable state-matched placebo tier")


def _matching_scale(values: pd.Series, event_value: object) -> float:
    work = pd.to_numeric(values, errors="coerce").dropna()
    scale = float(work.std(ddof=0)) if not work.empty else np.nan
    if not pd.notna(scale) or scale == 0:
        scale = abs(float(event_value)) if pd.notna(event_value) and float(event_value) != 0 else 1.0
    return scale


def _large_rebuild_cell_row(
    event_study: pd.DataFrame,
    leave_one: pd.DataFrame,
    bootstrap: pd.DataFrame,
    placebo: pd.DataFrame,
    randomization: pd.DataFrame,
    match_quality: pd.DataFrame,
    *,
    outcome: str,
    tau: int,
    table_role: str,
    randomization_p_max: float,
) -> dict[str, object]:
    group = event_study.loc[
        (event_study["outcome"] == outcome) & (event_study["tau"] == tau)
    ].dropna(subset=["level_change_from_baseline"])
    event_ids = sorted(group["event_id"].dropna().astype(str).unique()) if not group.empty else []
    changes = group["level_change_from_baseline"].astype(float) if not group.empty else pd.Series(dtype=float)
    dominant_sign = _sign_label(float(changes.mean())) if not changes.empty else "zero_or_missing"
    same_sign_share = (
        float((_signs_match_series(changes, dominant_sign)).mean()) if not changes.empty else np.nan
    )
    loo_row = _first_matching_row(leave_one, outcome, tau)
    boot_row = _first_matching_row(bootstrap, outcome, tau)
    placebo_row = _first_matching_row(placebo, outcome, tau)
    randomization_row = _first_matching_row(randomization, outcome, tau)
    match_quality_status, match_quality_events = _match_quality_cell_status(match_quality, event_ids)
    blocker_flags = _large_rebuild_cell_blockers(
        outcome,
        table_role,
        loo_row,
        boot_row,
        placebo_row,
        randomization_row,
        match_quality_status=match_quality_status,
        randomization_p_max=randomization_p_max,
        event_count=len(event_ids),
    )
    passes = not blocker_flags and table_role == "headline_reserves_main"
    return {
        "targeted_design": LARGE_REBUILD_DESIGN,
        "table_role": table_role,
        "outcome": outcome,
        "tau": int(tau),
        "primary_cell": bool(outcome == "reserves" and tau in LARGE_REBUILD_MAIN_TAUS),
        "event_count": int(len(event_ids)),
        "event_ids": ";".join(event_ids),
        "mean_change": float(changes.mean()) if not changes.empty else np.nan,
        "median_change": float(changes.median()) if not changes.empty else np.nan,
        "same_sign_event_share": same_sign_share,
        "dominant_sign": dominant_sign,
        "leave_one_out_same_sign_share": loo_row.get("same_sign_leave_one_out_share", np.nan),
        "bootstrap_same_sign_share": boot_row.get("bootstrap_same_sign_share", np.nan),
        "bootstrap_ci_lower": boot_row.get("ci_lower", np.nan),
        "bootstrap_ci_upper": boot_row.get("ci_upper", np.nan),
        "placebo_false_positive_share": placebo_row.get("false_positive_share", np.nan),
        "pseudo_events_n": int(placebo_row.get("pseudo_events_n", 0) or 0),
        "matched_event_groups_n": int(placebo_row.get("matched_event_groups_n", 0) or 0),
        "pseudo_mean_change": placebo_row.get("pseudo_mean_change", np.nan),
        "pseudo_abs_p90": placebo_row.get("pseudo_abs_p90", np.nan),
        "worst_match_tier": placebo_row.get("worst_match_tier", ""),
        "match_quality_status": match_quality_status,
        "match_quality_blocker_events": ";".join(match_quality_events),
        "randomization_p_value": randomization_row.get("two_sided_p_value", np.nan),
        "randomization_p_max": float(randomization_p_max),
        "randomization_status": randomization_row.get("status", "not_applicable"),
        "claim_use": "descriptive_targeted_plumbing_evidence" if passes else "do_not_promote",
        "status": "passes_targeted_descriptive_gates" if passes else "blocked",
        "blocker_flags": ";".join(blocker_flags),
        "interpretation": _large_rebuild_cell_interpretation(outcome, table_role, passes),
    }


def _large_rebuild_cell_blockers(
    outcome: str,
    table_role: str,
    loo_row: dict[str, object],
    boot_row: dict[str, object],
    placebo_row: dict[str, object],
    randomization_row: dict[str, object],
    *,
    match_quality_status: str,
    randomization_p_max: float,
    event_count: int,
) -> list[str]:
    blockers: list[str] = []
    if table_role != "headline_reserves_main":
        blockers.append("appendix_or_secondary_not_headline_claim")
    if outcome != "reserves":
        blockers.append("non_reserves_outcome_not_primary")
    if event_count < 5:
        blockers.append("main_sample_below_5_events")
    loo_share = loo_row.get("same_sign_leave_one_out_share", np.nan)
    boot_share = boot_row.get("bootstrap_same_sign_share", np.nan)
    ci_lower = boot_row.get("ci_lower", np.nan)
    ci_upper = boot_row.get("ci_upper", np.nan)
    false_positive = placebo_row.get("false_positive_share", np.nan)
    matched_groups = placebo_row.get("matched_event_groups_n", 0)
    randomization_status = randomization_row.get("status", "")
    randomization_p = randomization_row.get("two_sided_p_value", np.nan)
    if pd.isna(loo_share) or float(loo_share) < 0.80:
        blockers.append("leave_one_out_unstable")
    if pd.isna(boot_share) or float(boot_share) < 0.80:
        blockers.append("bootstrap_unstable")
    if pd.isna(ci_lower) or pd.isna(ci_upper) or (float(ci_lower) <= 0 <= float(ci_upper)):
        blockers.append("bootstrap_ci_crosses_zero")
    if pd.isna(false_positive) or float(false_positive) >= 0.10:
        blockers.append("placebo_false_positive_high")
    if int(matched_groups or 0) < event_count:
        blockers.append("incomplete_pseudo_matching")
    if "all_eligible_state" in str(placebo_row.get("worst_match_tier", "")):
        blockers.append("uses_all_eligible_state_match_tier")
    if table_role == "headline_reserves_main" and match_quality_status != "ok":
        blockers.append(f"event_match_quality_{match_quality_status}")
    if table_role == "headline_reserves_main":
        if randomization_status != "estimated":
            blockers.append("randomization_not_estimated")
        elif pd.isna(randomization_p) or float(randomization_p) > float(randomization_p_max):
            blockers.append("randomization_p_value_high")
    return blockers


def _match_quality_cell_status(
    match_quality: pd.DataFrame,
    event_ids: list[str],
) -> tuple[str, list[str]]:
    if not event_ids:
        return "not_applicable", []
    if match_quality.empty or not {"event_id", "status"}.issubset(match_quality.columns):
        return "missing", event_ids
    rows = match_quality.loc[match_quality["event_id"].astype(str).isin(event_ids)]
    if len(rows) < len(event_ids):
        observed = set(rows["event_id"].astype(str))
        return "missing", [event_id for event_id in event_ids if event_id not in observed]
    blocked = rows.loc[rows["status"].astype(str) != "ok"]
    if blocked.empty:
        return "ok", []
    return "blocked", sorted(blocked["event_id"].astype(str).unique())


def _has_match_quality_blocker(match_quality: pd.DataFrame) -> bool:
    if match_quality.empty or "status" not in match_quality.columns:
        return True
    return bool((match_quality["status"].astype(str) != "ok").any())


def _large_rebuild_cell_interpretation(outcome: str, table_role: str, passes: bool) -> str:
    if outcome == "on_rrp":
        return "appendix mechanism diagnostic only; ON RRP sign reversals block headline use"
    if table_role != "headline_reserves_main":
        return "secondary reserves sensitivity; do not headline unless pre-specified"
    if passes:
        return "descriptive targeted reserves plumbing evidence; not a causal bill-issuance result"
    return "blocked under targeted large-rebuild gates"


def _first_matching_row(table: pd.DataFrame, outcome: str, tau: int) -> dict[str, object]:
    if table.empty or not {"outcome", "tau"}.issubset(table.columns):
        return {}
    rows = table.loc[(table["outcome"] == outcome) & (table["tau"] == tau)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _signs_match_series(values: pd.Series, sign: str) -> pd.Series:
    if sign == "positive":
        return values > 0
    if sign == "negative":
        return values < 0
    return pd.Series(False, index=values.index)


def _worst_match_tier(pseudo_df: pd.DataFrame) -> str:
    if pseudo_df.empty or "manual_tags" not in pseudo_df.columns:
        return ""
    tiers = [
        "month_regime_calendar_state",
        "quarter_regime_calendar_state",
        "quarter_calendar_state",
        "quarter_state",
        "all_eligible_state",
    ]
    found = set()
    for tag in pseudo_df["manual_tags"].fillna("").astype(str):
        for tier in tiers:
            if f"match_tier={tier}" in tag:
                found.add(tier)
    if not found:
        return ""
    return max(found, key=tiers.index)


def _event_cell_change(group: pd.DataFrame, outcome: str, tau: int) -> float:
    rows = group.loc[(group["outcome"] == outcome) & (group["tau"] == tau)]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0]["level_change_from_baseline"])


def _first_value(group: pd.DataFrame, column: str) -> object:
    if column not in group.columns or group.empty:
        return ""
    value = group.iloc[0][column]
    return "" if pd.isna(value) else value


def _large_rebuild_event_validation_decision(
    context_row: dict[str, object],
    on_rrp_tau8: float,
    manual_tags: str,
) -> str:
    if not context_row:
        return "missing_official_context"
    if "debt" in manual_tags.lower():
        return "documented_but_debt_limit_confounded"
    if pd.notna(on_rrp_tau8) and on_rrp_tau8 > 0:
        return "documented_but_on_rrp_sign_reversal"
    return "documented_context_manual_review_required"


def _format_stable_cells(stable: pd.DataFrame) -> str:
    if stable.empty:
        return ""
    return ";".join(
        f"{row.outcome}:tau{int(row.tau)}"
        for row in stable.sort_values(["outcome", "tau"]).itertuples(index=False)
    )


def _series_min(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table.columns:
        return np.nan
    values = table[column].dropna()
    return float(values.min()) if not values.empty else np.nan


def _series_max(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table.columns:
        return np.nan
    values = table[column].dropna()
    return float(values.max()) if not values.empty else np.nan


def _ci_cross_count(bootstrap: pd.DataFrame) -> int:
    if bootstrap.empty or not {"ci_lower", "ci_upper"}.issubset(bootstrap.columns):
        return 0
    return int(((bootstrap["ci_lower"] <= 0) & (bootstrap["ci_upper"] >= 0)).sum())


def _weekly_filter_variant_decision(event_count: int, stable_cells: int) -> str:
    if stable_cells == 0:
        return "no_passing_cells"
    if event_count < 30:
        return "small_sample_sensitivity_only"
    return "candidate_filter_pending_manual_validation"


def _weekly_filter_variant_next_step(event_count: int, stable_cells: int) -> str:
    if stable_cells == 0:
        return "do_not_promote_this_filter"
    if event_count < 30:
        return "use_only_as_targeted_event_design_lead"
    return "manual_validate_filter_before_claim_use"


def _pseudo_event_weeks(panel: pd.DataFrame, event_starts: set[pd.Timestamp]) -> list[pd.Timestamp]:
    event_indices = {
        idx
        for idx, week in enumerate(panel["week"])
        if pd.to_datetime(week, errors="coerce") in event_starts
    }
    excluded = {
        idx + offset
        for idx in event_indices
        for offset in range(-8, 9)
    }
    weeks = []
    for idx, week in enumerate(panel["week"]):
        if idx < 4 or idx + 8 >= len(panel) or idx in excluded:
            continue
        weeks.append(pd.to_datetime(week))
    return weeks


def _future_weekly_cumulative_change(series: pd.Series, horizon: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if horizon == 0:
        return values
    return values.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))


def _outcome_level(panel: pd.DataFrame, outcome: str) -> pd.Series:
    level = outcome.removeprefix("d_")
    if level in panel.columns:
        return pd.to_numeric(panel[level], errors="coerce")
    return pd.Series(np.nan, index=panel.index)


def _event_baseline_value(panel: pd.DataFrame, baseline_idx: int, outcome: str) -> float:
    if baseline_idx < 0 or baseline_idx >= len(panel):
        return np.nan
    return pd.to_numeric(pd.Series([panel.loc[baseline_idx, outcome]]), errors="coerce").iloc[0]


def _sign_label(value: float) -> str:
    if pd.isna(value) or value == 0:
        return "zero_or_missing"
    return "positive" if value > 0 else "negative"


def _same_nonzero_sign(left: float, right: float) -> bool:
    if pd.isna(left) or pd.isna(right) or left == 0 or right == 0:
        return False
    return bool(np.sign(left) == np.sign(right))


def _ols_hc1_hac(
    df: pd.DataFrame,
    *,
    outcome: str,
    predictors: list[str],
    hac_lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y = df[outcome].to_numpy(dtype=float)
    x = df[predictors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    nobs, k = x.shape
    hc1_scale = nobs / max(nobs - k, 1)
    meat_hc1 = x.T @ ((residuals**2)[:, None] * x)
    vcov_hc1 = hc1_scale * xtx_inv @ meat_hc1 @ xtx_inv
    meat_hac = meat_hc1.copy()
    for lag in range(1, min(hac_lags, nobs - 1) + 1):
        weight = 1.0 - lag / (hac_lags + 1.0)
        gamma = x[lag:].T @ ((residuals[lag:] * residuals[:-lag])[:, None] * x[:-lag])
        meat_hac += weight * (gamma + gamma.T)
    vcov_hac = xtx_inv @ meat_hac @ xtx_inv
    se_hc1 = np.sqrt(np.maximum(np.diag(vcov_hc1), 0.0))
    se_hac = np.sqrt(np.maximum(np.diag(vcov_hac), 0.0))
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residuals**2).sum()) / total if total else 0.0
    return beta, se_hc1, se_hac, r2


def _bill_surprise(panel: pd.DataFrame) -> pd.Series:
    bill = panel["gross_bill_settlement_100b"]
    return bill - bill.shift(1).rolling(13, min_periods=8).mean()


def _weekly_on_rrp_regime(on_rrp: pd.Series) -> pd.Series:
    return pd.Series(
        pd.cut(
            pd.to_numeric(on_rrp, errors="coerce"),
            bins=[float("-inf"), 250_000.0, 1_000_000.0, float("inf")],
            labels=["scarce", "transition", "abundant"],
        ),
        index=on_rrp.index,
        dtype="string",
    )


def _high_rate_placeholder(panel: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.NA, index=panel.index, dtype="string")


def _attach_event_flags(root: Path, panel: pd.DataFrame) -> pd.DataFrame:
    path = root / TGAREFILL_EXPORTS["event_candidates"]["path"]
    if not path.exists():
        return panel
    events = pd.read_csv(path)
    events["start_date"] = pd.to_datetime(events["start_date"], errors="coerce")
    start_weeks = set(events["start_date"].dropna())
    panel["rapid_tga_rebuild_event"] = panel["week"].isin(start_weeks)
    tags = events["manual_tags"] if "manual_tags" in events.columns else pd.Series("", index=events.index)
    tagged = events[tags.astype("string").str.contains("debt", case=False, na=False)]
    debt_weeks = set(tagged["start_date"].dropna())
    panel["debt_limit_window"] = panel["week"].isin(debt_weeks)
    return panel


def _max_internal_gap_weeks(weeks: pd.Series) -> int:
    if len(weeks) < 2:
        return 0
    sorted_weeks = pd.to_datetime(weeks).sort_values()
    gaps = [
        int((right - left).days // 7 - 1)
        for left, right in zip(sorted_weeks.iloc[:-1], sorted_weeks.iloc[1:])
    ]
    return max(gaps, default=0)


def _date_or_blank(value: object) -> str:
    date = pd.to_datetime(value, errors="coerce")
    return date.date().isoformat() if pd.notna(date) else ""
