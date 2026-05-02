from __future__ import annotations

import pandas as pd

from liqsub.analysis import (
    add_analysis_fields,
    bill_shock_events,
    candidate_event_month_table,
    candidate_event_review_table,
    candidate_lp_table,
    candidate_table,
    claim_readiness_table,
    classify_on_rrp_regime_with_thresholds,
    coverage_qa,
    design_readiness_table,
    event_study_by_event,
    evidence_gate_summary,
    external_calendar_evidence_table,
    filtered_candidate_event_review_table,
    filtered_candidate_lp_table,
    filtered_candidate_shortlist_table,
    first_pass_regressions,
    future_row_qa,
    estimation_panel,
    local_projection_diagnostics,
    on_rrp_manual_review_table,
    regime_classification_qa,
    regime_threshold_sensitivity,
    residual_shock_pretrend_diagnostics,
    sample_windows,
    strong_candidate_narrative_table,
    source_metadata_table,
    internal_evidence_gate_summary,
    write_evidence_gate_summary,
)


def test_add_analysis_fields_adds_changes_and_regimes() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "on_rrp": [100_000.0, 500_000.0, 1_500_000.0, 800_000.0],
            "deposits": [100.0, 110.0, 105.0, 120.0],
            "gross_bill_issuance": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = add_analysis_fields(panel)
    assert list(out["on_rrp_regime_fixed"]) == ["scarce", "transition", "abundant", "transition"]
    assert pd.isna(out.loc[0, "d_deposits"])
    assert list(out["d_deposits"].iloc[1:]) == [10.0, -5.0, 15.0]
    assert list(out["gross_bill_issuance_100b"]) == [0.0001, 0.0002, 0.0003, 0.0004]


def test_sample_windows_reports_coverage() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "gross_bill_issuance": [1.0, None],
        }
    )
    out = sample_windows(panel)
    row = out.loc[out["column"] == "gross_bill_issuance"].iloc[0]
    assert row["non_null"] == 1
    assert row["first_month"] == "2024-01-01"


def test_source_and_coverage_qa_report_units_and_gaps() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "gross_bill_issuance": [1.0, None, 3.0, None],
        }
    )
    metadata = source_metadata_table(panel)
    metadata_row = metadata.loc[metadata["column"] == "gross_bill_issuance"].iloc[0]
    assert metadata_row["status"] == "present"
    assert metadata_row["unit"] == "usd_millions"
    on_rrp_metadata = metadata.loc[metadata["column"] == "on_rrp"].iloc[0]
    assert on_rrp_metadata["source_family"] == "fred_on_rrp_operations"
    assert on_rrp_metadata["native_frequency"] == "daily"
    assert on_rrp_metadata["monthly_aggregation"] == "arithmetic mean over calendar month"
    iorb_metadata = metadata.loc[metadata["column"] == "iorb_rate"].iloc[0]
    assert iorb_metadata["monthly_aggregation"] == "arithmetic mean over calendar month"

    coverage = coverage_qa(panel)
    coverage_row = coverage.loc[coverage["column"] == "gross_bill_issuance"].iloc[0]
    assert coverage_row["non_null"] == 2
    assert coverage_row["max_internal_gap_months"] == 1
    assert coverage_row["trailing_missing_months"] == 1


def test_future_row_qa_flags_future_months() -> None:
    future_month = pd.Timestamp("2024-03-01")
    panel = pd.DataFrame({"month": [future_month], "gross_bill_issuance": [1.0]})
    out = future_row_qa(panel, as_of_date="2024-01-01")
    assert len(out) == 1
    assert out.loc[0, "future_or_scheduled"]


def test_terminal_incomplete_months_are_flagged_and_excluded() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "gross_bill_issuance": [1.0, 2.0, 3.0, 4.0],
            "deposits": [10.0, 11.0, 12.0, None],
            "on_rrp": [5.0, 6.0, 7.0, None],
        }
    )
    out = add_analysis_fields(panel, as_of_date="2024-04-01")
    assert out.loc[3, "terminal_completeness_status"] == "terminal_incomplete"
    assert out.loc[3, "baseline_estimation_use"] == "exclude_from_causal_baseline"
    assert len(estimation_panel(out)) == 3

    qa = future_row_qa(out, as_of_date="2024-04-01")
    assert list(qa["month"]) == ["2024-04-01"]
    assert qa.loc[0, "terminal_incomplete"]


def test_custom_on_rrp_threshold_classifier() -> None:
    out = classify_on_rrp_regime_with_thresholds(
        pd.Series([50.0, 200.0, 800.0]),
        scarce_cutoff=100.0,
        abundant_cutoff=500.0,
    )
    assert list(out) == ["scarce", "transition", "abundant"]


def test_first_pass_regressions_skips_small_samples() -> None:
    panel = pd.DataFrame(
        {
            "d_deposits": [1.0, 2.0],
            "gross_bill_issuance_100b": [0.5, 0.7],
            "bill_share": [0.8, 0.9],
            "iorb_rate": [5.0, 5.0],
        }
    )
    out = first_pass_regressions(panel, min_nobs=3)
    assert "skipped_too_few_observations" in set(out["status"])


def test_bill_shock_events_are_isolated() -> None:
    months = pd.date_range("2020-01-01", periods=48, freq="MS")
    panel = pd.DataFrame(
        {
            "month": months,
            "gross_bill_issuance": [100.0 + (500.0 if i in {20, 21, 30} else 0.0) for i in range(48)],
            "bill_share": [0.8] * 48,
            "deposits": [1000.0 + i for i in range(48)],
        }
    )
    out = add_analysis_fields(panel)
    events = bill_shock_events(out)
    assert len(events) >= 1
    assert out["isolated_large_positive_bill_shock"].sum() <= out["large_positive_bill_shock"].sum()
    event_rows = event_study_by_event(out)
    assert set(event_rows["window"]).issubset({"pre3", "impact", "post3", "post6"})


def test_local_projection_diagnostics_estimates_when_sample_is_large_enough() -> None:
    months = pd.date_range("2010-01-01", periods=80, freq="MS")
    panel = pd.DataFrame(
        {
            "month": months,
            "gross_bill_issuance": [100.0 + (i % 12) * 10.0 for i in range(80)],
            "bill_share": [0.75 + ((i % 3) * 0.02) for i in range(80)],
            "bill_yield": [2.0 + (i % 5) * 0.1 for i in range(80)],
            "deposits": [1000.0 + i * 2.0 for i in range(80)],
        }
    )
    out = add_analysis_fields(panel)
    diagnostics = local_projection_diagnostics(out, min_nobs=24)
    assert "estimated" in set(diagnostics["status"])
    assert "bill_supply_shock_resid_100b" in set(diagnostics["predictor"].dropna())


def test_residual_shock_pretrend_diagnostics_estimates_when_sample_is_large_enough() -> None:
    months = pd.date_range("2010-01-01", periods=80, freq="MS")
    panel = pd.DataFrame(
        {
            "month": months,
            "gross_bill_issuance": [100.0 + (i % 12) * 10.0 for i in range(80)],
            "bill_share": [0.75 + ((i % 3) * 0.02) for i in range(80)],
            "deposits": [1000.0 + i * 2.0 for i in range(80)],
        }
    )
    out = add_analysis_fields(panel)
    diagnostics = residual_shock_pretrend_diagnostics(out, min_nobs=24)
    assert "estimated" in set(diagnostics["status"])
    assert "shock_t_hc1" in diagnostics.columns


def test_regime_threshold_sensitivity_counts_events() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "on_rrp": [10.0, 50.0, 200_000.0, 800_000.0, 1_200_000.0, 2_000_000.0],
            "gross_bill_issuance": [100.0] * 6,
            "bill_share": [0.8] * 6,
        }
    )
    out = add_analysis_fields(panel)
    out["isolated_large_positive_bill_shock"] = [False, True, False, False, True, False]
    sensitivity = regime_threshold_sensitivity(out)
    assert set(sensitivity["threshold_spec"]) >= {"fixed_250b_1000b"}
    assert sensitivity["n_isolated_bill_shock_events"].sum() >= 2


def test_regime_classification_qa_counts_missing_and_events() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "on_rrp": [50.0, 500_000.0, 2_000_000.0, None],
            "gross_bill_issuance": [100.0] * 4,
            "bill_share": [0.8] * 4,
        }
    )
    out = add_analysis_fields(panel)
    out["isolated_large_positive_bill_shock"] = [True, False, True, False]
    qa = regime_classification_qa(out)
    fixed = qa.loc[qa["regime_variable"] == "on_rrp_regime_fixed"]
    assert {"scarce", "transition", "abundant"}.issubset(set(fixed["regime"].dropna()))
    assert fixed["n_isolated_bill_shock_events"].sum() == 2


def test_design_readiness_marks_sparse_samples() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "gross_bill_issuance": [100.0] * 12,
            "bill_share": [0.8] * 12,
            "deposits": [1000.0 + i for i in range(12)],
            "domestic_deposits": [900.0 + i for i in range(12)],
            "bill_yield": [1.0] * 12,
        }
    )
    out = add_analysis_fields(panel)
    readiness = design_readiness_table(out)
    row = readiness.loc[
        (readiness["sample"] == "full_treasury_deposits")
        & (readiness["outcome"] == "d_deposits")
    ].iloc[0]
    assert row["status"] == "too_sparse"
    assert row["recommended_next_step"] == "use_broader_sample_or_lower_frequency_design"


def test_candidate_lp_table_returns_dataframe_for_small_sample() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "gross_bill_issuance": [100.0] * 12,
            "bill_share": [0.8] * 12,
            "deposits": [1000.0 + i for i in range(12)],
            "domestic_deposits": [900.0 + i for i in range(12)],
            "bill_yield": [1.0] * 12,
        }
    )
    out = add_analysis_fields(panel)
    candidates = candidate_lp_table(out)
    assert isinstance(candidates, pd.DataFrame)


def test_candidate_event_review_table_returns_dataframe_for_small_sample() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "gross_bill_issuance": [100.0] * 12,
            "bill_share": [0.8] * 12,
            "deposits": [1000.0 + i for i in range(12)],
            "domestic_deposits": [900.0 + i for i in range(12)],
            "bill_yield": [1.0] * 12,
        }
    )
    out = add_analysis_fields(panel)
    review = candidate_event_review_table(out)
    assert isinstance(review, pd.DataFrame)


def test_candidate_event_month_table_returns_dataframe_for_small_sample() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "gross_bill_issuance": [100.0] * 12,
            "bill_share": [0.8] * 12,
            "deposits": [1000.0 + i for i in range(12)],
            "domestic_deposits": [900.0 + i for i in range(12)],
            "bill_yield": [1.0] * 12,
        }
    )
    out = add_analysis_fields(panel)
    months = candidate_event_month_table(out)
    assert isinstance(months, pd.DataFrame)


def test_filtered_candidate_tables_return_dataframes_for_small_sample() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "gross_bill_issuance": [100.0] * 12,
            "bill_share": [0.8] * 12,
            "deposits": [1000.0 + i for i in range(12)],
            "domestic_deposits": [900.0 + i for i in range(12)],
            "bill_yield": [1.0] * 12,
        }
    )
    out = add_analysis_fields(panel)
    assert isinstance(filtered_candidate_lp_table(out), pd.DataFrame)
    assert isinstance(filtered_candidate_event_review_table(out), pd.DataFrame)
    assert isinstance(filtered_candidate_shortlist_table(out), pd.DataFrame)
    assert isinstance(on_rrp_manual_review_table(out), pd.DataFrame)
    assert isinstance(strong_candidate_narrative_table(out), pd.DataFrame)
    assert isinstance(external_calendar_evidence_table(out), pd.DataFrame)
    assert isinstance(candidate_table(out), pd.DataFrame)
    assert isinstance(claim_readiness_table(out), pd.DataFrame)


def test_evidence_gate_summary_combines_monthly_and_weekly(tmp_path) -> None:
    table_dir = tmp_path / "output" / "tables"
    table_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "readiness_status": ["blocked"],
            "outcome": ["d_mmf_treasury_holdings"],
            "blocker_flags": ["material_pre_event_movement"],
        }
    ).to_csv(table_dir / "monthly_readiness_summary.csv", index=False)
    pd.DataFrame(
        {
            "promotion_decision": ["do_not_promote_descriptive_context_only"],
            "max_abs_lp_t": [1.1],
        }
    ).to_csv(table_dir / "monthly_on_rrp_manual_review.csv", index=False)
    pd.DataFrame(
        {
            "outcome": ["d_mmf_treasury_holdings"],
            "event_status": ["event_pattern_reviewable"],
        }
    ).to_csv(table_dir / "monthly_filtered_candidate_shortlist.csv", index=False)
    pd.DataFrame(
        {
            "status": ["blocked"],
            "blocker_flags": ["placebo_false_positive_high"],
        }
    ).to_csv(table_dir / "weekly_stability_candidates.csv", index=False)
    pd.DataFrame({"blocker_flags": ["outcome_coverage_below_90pct"]}).to_csv(
        table_dir / "weekly_design_readiness.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "status": ["descriptive_reserves_plumbing_ready_for_writeup"],
            "claim_use": ["descriptive_targeted_plumbing_evidence"],
            "headline_cells": ["reserves:tau0=passes_targeted_descriptive_gates"],
            "binding_limitations": ["five_event_main_sample"],
        }
    ).to_csv(table_dir / "weekly_large_rebuild_final_review.csv", index=False)
    pd.DataFrame(
        {
            "table_role": ["headline_reserves_main"],
            "tau": [0],
            "mean_change": [-100.0],
            "status": ["passes_targeted_descriptive_gates"],
        }
    ).to_csv(table_dir / "weekly_large_rebuild_cell_summary.csv", index=False)

    out = evidence_gate_summary(tmp_path)
    assert set(out["design_path"]) >= {
        "monthly_broad_bill_supply_designs",
        "weekly_large_rebuild_targeted_reserves",
        "cross_design_diagnostic_summary",
    }
    targeted = out.loc[out["design_path"] == "weekly_large_rebuild_targeted_reserves"].iloc[0]
    assert targeted["claim_use"] == "descriptive_targeted_plumbing_evidence"
    alias_out = internal_evidence_gate_summary(tmp_path)
    assert list(alias_out["design_path"]) == list(out["design_path"])

    paths = write_evidence_gate_summary(tmp_path)
    assert (tmp_path / "output" / "tables" / "evidence_gate_summary.csv").exists()
    assert not (tmp_path / "output" / "tables" / "internal_claim_artifact_map.csv").exists()
    report_path = tmp_path / "output" / "reports" / "evidence_gate_summary.md"
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Backend State" in report
    assert "Backend Next Work" in report
    assert "internal_evidence_gate_summary" not in report.lower()
    assert "evidence_gate_summary" in paths
    assert paths["evidence_gate_summary"] == "output/tables/evidence_gate_summary.csv"
    assert not paths["evidence_gate_report"].startswith("/")
