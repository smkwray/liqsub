from __future__ import annotations

import pandas as pd

from liqsub.tgarefill import (
    _isolate_weekly_events,
    add_weekly_terminal_completeness_fields,
    load_weekly_tgarefill_panel,
    validate_tgarefill_exports,
    weekly_block_bootstrap,
    weekly_claim_readiness,
    weekly_event_study_by_event,
    weekly_event_study_summary,
    weekly_event_filter_sensitivity,
    weekly_large_rebuild_abnormal_changes,
    weekly_large_rebuild_blocker_summary,
    weekly_large_rebuild_cell_summary,
    weekly_large_rebuild_event_sign_stability,
    weekly_large_rebuild_final_review,
    weekly_large_rebuild_match_quality,
    weekly_large_rebuild_randomization_inference,
    weekly_leave_one_event_out,
    weekly_large_rebuild_calendar_validation,
    weekly_large_rebuild_event_roster,
    weekly_large_rebuild_event_study,
    weekly_large_rebuild_event_time_svg,
    weekly_large_rebuild_diagnostic_writeup,
    weekly_large_rebuild_targeted_review,
    weekly_lp_coefficients,
    weekly_outcome_claim_readiness,
    weekly_placebo_tests,
    weekly_stable_claim_candidates,
    weekly_terminal_period_qa,
    tgarefill_promotion_reconciliation,
    write_tgarefill_promotion_reconciliation_report,
)


def test_validate_tgarefill_exports_reports_missing(tmp_path) -> None:
    report = validate_tgarefill_exports(tmp_path)
    assert report.status == "failed"
    assert report.errors


def test_weekly_terminal_incomplete_periods_are_flagged() -> None:
    panel = pd.DataFrame(
        {
            "week": pd.to_datetime(["2024-01-03", "2024-01-10", "2024-01-17"]),
            "tga": [1.0, 2.0, 3.0],
            "reserves": [1.0, 2.0, None],
            "on_rrp": [1.0, 2.0, None],
            "deposits": [1.0, 2.0, None],
        }
    )
    out = add_weekly_terminal_completeness_fields(panel)
    assert out.loc[2, "terminal_completeness_status"] == "terminal_incomplete"
    assert out.loc[2, "baseline_estimation_use"] == "exclude_from_causal_baseline"
    qa = weekly_terminal_period_qa(out)
    assert list(qa["week"]) == ["2024-01-17"]


def _write_minimal_tgarefill_exports(raw) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "master_weekly_panel.csv").write_text(
        "date,tga_weekly_average,reserve_balances_weekly_wednesday,"
        "reverse_repos_weekly_wednesday,commercial_bank_deposits_weekly_sa,bill,coupon,"
        "bill_share,mmf_treasury_holdings,mmf_repo_total\n"
        "2024-01-03,10,20,30,40,100000000000,50000000000,0.67,1000,200\n"
        "2024-01-10,15,21,28,41,120000000000,40000000000,0.75,1010,190\n",
        encoding="utf-8",
    )
    (raw / "event_candidates.csv").write_text(
        "event_id,baseline_date,start_date,end_date,duration_weeks,delta_tga_event,"
        "max_weekly_delta_tga,mean_bill_share,issuance_mix,manual_tags,manual_notes\n"
        "event_001,2024-01-03,2024-01-10,2024-01-10,1,5,5,0.75,bill,,\n",
        encoding="utf-8",
    )
    (raw / "auction_shock_lp.csv").write_text(
        "response_var,horizon,beta,se,se_nw,t_stat,t_stat_nw\n"
        "deposits,0,1,1,1,1,1\n",
        encoding="utf-8",
    )
    (raw / "canonical_bill_surprise_shocks.csv").write_text(
        "date,bill_size_surprise,bill_size_surprise_announcement_week,tax_receipt_surprise,"
        "rapid_rebuild_flag,regime,rmp_regime,sample,canonical_sample_end\n"
        "2024-01-03,0,0,0,0,on_rrp_scarce,pre_rmp,canonical_pre_rmp_through_2025_11,2025-11-30\n"
        "2024-01-10,10,12,0,1,on_rrp_scarce,pre_rmp,canonical_pre_rmp_through_2025_11,2025-11-30\n",
        encoding="utf-8",
    )
    (raw / "promotion_robustness_summary.csv").write_text(
        "spec,response_var,response_label,shock_sd_bn,placebo_significant_count,"
        "all_channel_placebo_significant_count,h4_effect_bn,h4_t_stat_nw,h4_significant_5pct,note\n"
        "canonical_issue_week,mmf_treasury_holdings,MMF Treasury Holdings,18.8,0,1,44.1,3.8,True,main\n"
        "canonical_issue_week,on_rrp_daily_total,ON RRP,18.8,1,1,-30.4,-5.0,True,main\n"
        "canonical_issue_week,commercial_bank_deposits_weekly_nsa,Bank Deposits,18.8,0,1,6.9,0.6,False,boundary\n"
        "canonical_issue_week,reserve_balances_weekly_wednesday,Reserves,18.8,0,1,-1.8,-0.2,False,boundary\n"
        "same_week_announcement_timing,mmf_treasury_holdings,MMF Treasury Holdings,18.9,0,3,35.8,3.9,True,timing\n"
        "same_week_announcement_timing,on_rrp_daily_total,ON RRP,18.9,0,3,-35.6,-6.1,True,timing\n"
        "same_week_announcement_timing,commercial_bank_deposits_weekly_nsa,Bank Deposits,18.9,0,3,11.1,1.0,False,timing\n"
        "same_week_announcement_timing,reserve_balances_weekly_wednesday,Reserves,18.9,1,3,1.8,0.1,False,timing\n",
        encoding="utf-8",
    )
    (raw / "mmfalloc_downstream_summary.csv").write_text(
        "metric,value,unit,note\n"
        "source_gates_passed,True,boolean,All gates pass\n"
        "sample_start,2022-12-31,date,First month\n"
        "sample_end,2025-11-30,date,Last month\n"
        "event_count,4,events,Events represented\n"
        "mean_event_delta_treasury_total,229261.771,USD millions,Mean delta\n"
        "mean_event_delta_fed_onrrp,-141051.2669,USD millions,Mean delta\n"
        "mean_event_delta_repo_ex_fed,-10898.5529,USD millions,Mean delta\n"
        "claim_boundary,focused_tga_refill_mmf_allocation_support_only,label,Boundary\n",
        encoding="utf-8",
    )
    (raw / "mmfalloc_source_gates.csv").write_text(
        "gate,passed,value,threshold,note\n"
        "coverage,True,36,36,Contiguous months\n"
        "required_fields,True,True,true,Required fields\n"
        "mapping,True,1.0,0.95,Mapped share\n"
        "reconciliation,True,0.0421,<=0.75,Reconciliation\n",
        encoding="utf-8",
    )
    (raw / "mmfalloc_baseline.csv").write_text(
        "event_month,horizon,category,delta_millions,event_bill_surprise_millions,delta_per_bill_surprise\n"
        "2023-06-30,0_vs_minus1,treasury_bills,100.0,190000.0,0.0005\n"
        "2023-06-30,0_vs_minus1,fed_onrrp,-80.0,190000.0,-0.0004\n",
        encoding="utf-8",
    )


def test_validate_tgarefill_exports_catches_duplicate_week(tmp_path) -> None:
    raw = tmp_path / "data" / "raw" / "tgarefill"
    _write_minimal_tgarefill_exports(raw)
    (raw / "master_weekly_panel.csv").write_text(
        "date,tga_weekly_average,reserve_balances_weekly_wednesday,"
        "reverse_repos_weekly_wednesday,commercial_bank_deposits_weekly_sa,bill,coupon,bill_share\n"
        "2024-01-03,10,20,30,40,100,50,0.67\n"
        "2024-01-03,15,21,28,41,120,40,0.75\n",
        encoding="utf-8",
    )
    report = validate_tgarefill_exports(tmp_path)
    assert report.status == "failed"
    assert any("duplicate values in date" in error for error in report.errors)


def test_validate_tgarefill_exports_catches_event_date_order(tmp_path) -> None:
    raw = tmp_path / "data" / "raw" / "tgarefill"
    _write_minimal_tgarefill_exports(raw)
    (raw / "event_candidates.csv").write_text(
        "event_id,baseline_date,start_date,end_date,delta_tga_event,manual_tags\n"
        "event_001,2024-01-10,2024-01-03,2024-01-10,5,\n",
        encoding="utf-8",
    )
    report = validate_tgarefill_exports(tmp_path)
    assert report.status == "failed"
    assert any("start_date precedes baseline_date" in error for error in report.errors)


def test_validate_tgarefill_exports_catches_invalid_bill_share(tmp_path) -> None:
    raw = tmp_path / "data" / "raw" / "tgarefill"
    _write_minimal_tgarefill_exports(raw)
    (raw / "master_weekly_panel.csv").write_text(
        "date,tga_weekly_average,reserve_balances_weekly_wednesday,"
        "reverse_repos_weekly_wednesday,commercial_bank_deposits_weekly_sa,bill,coupon,bill_share\n"
        "2024-01-03,10,20,30,40,100,50,1.5\n",
        encoding="utf-8",
    )
    report = validate_tgarefill_exports(tmp_path)
    assert report.status == "failed"
    assert any("bill_share contains values outside [0, 1]" in error for error in report.errors)


def test_weekly_tgarefill_panel_builds_from_csv_exports(tmp_path) -> None:
    raw = tmp_path / "data" / "raw" / "tgarefill"
    _write_minimal_tgarefill_exports(raw)
    report = validate_tgarefill_exports(tmp_path)
    panel = load_weekly_tgarefill_panel(tmp_path)
    lp = weekly_lp_coefficients(panel, min_nobs=2)
    events = pd.DataFrame(
        {
            "event_id": ["event_001"],
            "baseline_date": ["2024-01-03"],
            "start_date": ["2024-01-10"],
            "delta_tga_event": [5.0],
            "mean_bill_share": [0.75],
            "manual_tags": [""],
        }
    )
    event_rows = weekly_event_study_by_event(panel, events)
    loo = weekly_leave_one_event_out(event_rows)
    bootstrap = weekly_block_bootstrap(event_rows, n_bootstrap=20)
    placebo = weekly_placebo_tests(panel, events, event_rows)
    stable = weekly_stable_claim_candidates(loo, bootstrap, placebo)
    sensitivity = weekly_event_filter_sensitivity(panel, events)
    targeted = weekly_large_rebuild_targeted_review(panel, events)
    calendar = weekly_large_rebuild_calendar_validation(tmp_path, targeted)
    roster = weekly_large_rebuild_event_roster(tmp_path, events)
    cell_summary = weekly_large_rebuild_cell_summary(tmp_path, panel, events)
    event_sign = weekly_large_rebuild_event_sign_stability(tmp_path, panel, events)
    abnormal = weekly_large_rebuild_abnormal_changes(tmp_path, panel, events)
    final_review = weekly_large_rebuild_final_review(roster, cell_summary, event_sign, abnormal)
    readiness = weekly_claim_readiness(panel, events, lp, event_rows, loo, bootstrap, placebo)
    assert report.status == "ok"
    assert list(panel["gross_bill_settlement_100b"]) == [1.0, 1.2]
    assert panel.loc[1, "rapid_tga_rebuild_event"]
    assert isinstance(lp, pd.DataFrame)
    assert isinstance(event_rows, pd.DataFrame)
    assert isinstance(loo, pd.DataFrame)
    assert isinstance(bootstrap, pd.DataFrame)
    assert isinstance(placebo, pd.DataFrame)
    assert isinstance(stable, pd.DataFrame)
    assert isinstance(sensitivity, pd.DataFrame)
    assert isinstance(targeted, pd.DataFrame)
    assert isinstance(calendar, pd.DataFrame)
    assert isinstance(roster, pd.DataFrame)
    assert isinstance(cell_summary, pd.DataFrame)
    assert isinstance(event_sign, pd.DataFrame)
    assert isinstance(abnormal, pd.DataFrame)
    assert isinstance(final_review, pd.DataFrame)
    assert readiness.loc[0, "readiness_status"] == "blocked"


def test_tgarefill_promotion_reconciliation_promotes_focused_claim(tmp_path) -> None:
    raw = tmp_path / "data" / "raw" / "tgarefill"
    _write_minimal_tgarefill_exports(raw)

    out = tgarefill_promotion_reconciliation(tmp_path)
    report_path = tmp_path / "output" / "reports" / "tgarefill_promotion_reconciliation.md"
    write_tgarefill_promotion_reconciliation_report(out, path=report_path)

    assert set(out["status"]) >= {"supported_focused_claim", "not_supported_as_channel"}
    promoted = out.loc[out["status"] == "supported_focused_claim", "channel"]
    assert set(promoted) == {"MMF Treasury Holdings", "ON RRP"}
    validation = out.loc[out["status"] == "imported_descriptive_allocation"].iloc[0]
    assert validation["claim_label"] == "focused_tga_refill_mmf_allocation_support_only"
    assert "Broad `liqsub` substitution remains blocked" in report_path.read_text(encoding="utf-8")
    assert "Fund-Level MMF Validation" in report_path.read_text(encoding="utf-8")


def test_weekly_large_rebuild_randomization_inference_is_event_matched(tmp_path) -> None:
    weeks = pd.date_range("2024-01-03", periods=90, freq="W-WED")
    panel = pd.DataFrame(
        {
            "week": weeks,
            "deposits": [2000.0] * 90,
            "on_rrp": [500.0] * 90,
            "reserves": [1000.0 - (idx % 13) * 2 for idx in range(90)],
            "mmf_treasury_holdings": [50.0] * 90,
            "tax_week": [False] * 90,
            "debt_limit_window": [False] * 90,
            "covid_crisis_window": [False] * 90,
            "bank_stress_window": [False] * 90,
            "on_rrp_regime_pre": ["scarce"] * 90,
        }
    )
    clean = pd.DataFrame(
        {
            "event_id": ["event_039", "event_041"],
            "start_date": [weeks[20], weeks[50]],
            "end_date": [weeks[20], weeks[50]],
            "delta_tga_event": [250_000.0, 260_000.0],
            "mean_bill_share": [0.85, 0.8],
            "manual_tags": ["", ""],
        }
    )

    out = weekly_large_rebuild_randomization_inference(
        tmp_path, panel, clean, draws=200, seed=7
    )

    assert set(out["tau"]) == {0, 2, 4}
    assert set(out["status"]) == {"estimated"}
    assert set(out["matched_event_groups_n"]) == {2}
    assert out["two_sided_p_value"].between(0, 1).all()
    assert set(out["placebo_design"]) == {"state_matched_event_level_randomization"}


def test_weekly_lp_coefficients_estimates_hac_rows() -> None:
    weeks = pd.date_range("2020-01-01", periods=140, freq="W-WED")
    panel = pd.DataFrame(
        {
            "week": weeks,
            "deposits": [1000 + i for i in range(140)],
            "d_deposits": [1.0] * 140,
            "bill_size_surprise_100b": [0.1 + (i % 7) * 0.01 for i in range(140)],
            "bill_share": [0.7] * 140,
            "coupon_settlement_100b": [0.3] * 140,
            "tax_week": [False] * 140,
            "covid_crisis_window": [False] * 140,
            "bank_stress_window": [False] * 140,
            "debt_limit_window": [False] * 140,
        }
    )
    out = weekly_lp_coefficients(panel, min_nobs=40)
    shock_rows = out.loc[out["predictor"] == "bill_size_surprise_100b"]
    assert "estimated" in set(out["status"])
    assert set(shock_rows["horizon_weeks"]) >= {0, 1, 2, 4, 8}
    assert "se_hac" in out.columns


def test_weekly_event_study_by_event_and_summary() -> None:
    weeks = pd.date_range("2024-01-03", periods=16, freq="W-WED")
    panel = pd.DataFrame(
        {
            "week": weeks,
            "deposits": [1000.0 + i for i in range(16)],
            "on_rrp": [500.0 - i for i in range(16)],
            "reserves": [200.0 + i * 2 for i in range(16)],
            "mmf_treasury_holdings": [50.0 + i for i in range(16)],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["event_001"],
            "baseline_date": [weeks[4]],
            "start_date": [weeks[5]],
            "delta_tga_event": [100.0],
            "mean_bill_share": [0.8],
            "manual_tags": [""],
        }
    )
    by_event = weekly_event_study_by_event(panel, events)
    summary = weekly_event_study_summary(by_event)
    loo = weekly_leave_one_event_out(by_event)
    assert set(by_event["tau"]) == set(range(-4, 9))
    assert {"deposits", "on_rrp", "reserves", "mmf_treasury_holdings"}.issubset(set(summary["outcome"]))
    row = by_event.loc[(by_event["outcome"] == "deposits") & (by_event["tau"] == 0)].iloc[0]
    assert row["level_change_from_baseline"] == 1.0
    assert isinstance(loo, pd.DataFrame)


def test_weekly_large_rebuild_event_study_uses_pre_window_baseline() -> None:
    weeks = pd.date_range("2024-01-03", periods=12, freq="W-WED")
    panel = pd.DataFrame(
        {
            "week": weeks,
            "deposits": [1000.0 + i for i in range(12)],
            "on_rrp": [500.0 + i for i in range(12)],
            "reserves": [200.0 + i * 10 for i in range(12)],
            "mmf_treasury_holdings": [50.0 + i for i in range(12)],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["event_039"],
            "start_date": [weeks[5]],
            "end_date": [weeks[5]],
            "delta_tga_event": [250_000.0],
            "mean_bill_share": [0.8],
            "manual_tags": [""],
        }
    )
    out = weekly_large_rebuild_event_study(panel, events)
    tau_minus1 = out.loc[(out["outcome"] == "reserves") & (out["tau"] == -1)].iloc[0]
    tau0 = out.loc[(out["outcome"] == "reserves") & (out["tau"] == 0)].iloc[0]
    assert tau_minus1["baseline_level"] == 220.0
    assert tau_minus1["level_change_from_baseline"] == 20.0
    assert tau0["level_change_from_baseline"] == 30.0
    assert tau_minus1["baseline_method"] == "mean_tau_minus4_to_minus2"


def test_weekly_large_rebuild_roster_marks_debt_limit_event_appendix(tmp_path) -> None:
    clean = pd.DataFrame(
        {
            "event_id": ["event_038", "event_039", "event_041", "event_045"],
            "start_date": ["2023-06-21", "2023-09-20", "2024-04-17", "2025-04-16"],
            "end_date": ["2023-06-21", "2023-09-20", "2024-04-17", "2025-04-16"],
            "delta_tga_event": [300_000.0, 250_000.0, 257_000.0, 323_000.0],
            "mean_bill_share": [0.8, 0.85, 0.78, 0.76],
            "manual_tags": ["", "", "", ""],
        }
    )
    roster = weekly_large_rebuild_event_roster(tmp_path, clean)
    roles = dict(zip(roster["event_id"], roster["sample_role"]))
    reasons = dict(zip(roster["event_id"], roster["exclusion_reason"]))
    on_rrp_flags = dict(zip(roster["event_id"], roster["on_rrp_tau8_sign_reversal"]))
    assert roles["event_038"] == "appendix"
    assert reasons["event_038"] == "debt_limit_confounded"
    assert roles["event_039"] == "main"
    assert on_rrp_flags["event_041"]
    assert on_rrp_flags["event_045"]


def test_weekly_large_rebuild_empty_outputs_are_parseable(tmp_path) -> None:
    panel = pd.DataFrame({"week": pd.date_range("2024-01-03", periods=8, freq="W-WED")})
    clean = pd.DataFrame(
        {
            "event_id": ["small"],
            "start_date": ["2024-01-10"],
            "end_date": ["2024-01-10"],
            "delta_tga_event": [10.0],
            "mean_bill_share": [0.8],
            "manual_tags": [""],
        }
    )
    roster = weekly_large_rebuild_event_roster(tmp_path, clean)
    cell_summary = weekly_large_rebuild_cell_summary(tmp_path, panel, clean)
    event_sign = weekly_large_rebuild_event_sign_stability(tmp_path, panel, clean)
    abnormal = weekly_large_rebuild_abnormal_changes(tmp_path, panel, clean)
    final_review = weekly_large_rebuild_final_review(roster, cell_summary, event_sign, abnormal)
    roster_path = tmp_path / "roster.csv"
    summary_path = tmp_path / "summary.csv"
    event_sign_path = tmp_path / "event_sign.csv"
    abnormal_path = tmp_path / "abnormal.csv"
    final_path = tmp_path / "final.csv"
    roster.to_csv(roster_path, index=False)
    cell_summary.to_csv(summary_path, index=False)
    event_sign.to_csv(event_sign_path, index=False)
    abnormal.to_csv(abnormal_path, index=False)
    final_review.to_csv(final_path, index=False)
    assert list(pd.read_csv(roster_path).columns)
    assert list(pd.read_csv(summary_path).columns)
    assert list(pd.read_csv(event_sign_path).columns)
    assert list(pd.read_csv(abnormal_path).columns)
    assert list(pd.read_csv(final_path).columns)


def test_weekly_large_rebuild_event_sign_and_abnormal_outputs(tmp_path) -> None:
    weeks = pd.date_range("2024-01-03", periods=80, freq="W-WED")
    reserves = [1000.0 for _ in range(80)]
    on_rrp = [500.0 for _ in range(80)]
    reserves[20] = 900.0
    reserves[22] = 850.0
    reserves[24] = 820.0
    reserves[40] = 940.0
    reserves[42] = 870.0
    reserves[44] = 830.0
    on_rrp[28] = 450.0
    on_rrp[48] = 460.0
    panel = pd.DataFrame(
        {
            "week": weeks,
            "deposits": [2000.0] * 80,
            "on_rrp": on_rrp,
            "reserves": reserves,
            "mmf_treasury_holdings": [50.0] * 80,
            "tax_week": [False] * 80,
            "debt_limit_window": [False] * 80,
            "covid_crisis_window": [False] * 80,
            "bank_stress_window": [False] * 80,
            "on_rrp_regime_pre": ["scarce"] * 80,
        }
    )
    clean = pd.DataFrame(
        {
            "event_id": ["event_039", "event_041"],
            "start_date": [weeks[20], weeks[40]],
            "end_date": [weeks[20], weeks[40]],
            "delta_tga_event": [250_000.0, 260_000.0],
            "mean_bill_share": [0.85, 0.8],
            "manual_tags": ["", ""],
        }
    )
    event_sign = weekly_large_rebuild_event_sign_stability(tmp_path, panel, clean)
    cell_summary = weekly_large_rebuild_cell_summary(tmp_path, panel, clean)
    abnormal = weekly_large_rebuild_abnormal_changes(tmp_path, panel, clean)
    match_quality = weekly_large_rebuild_match_quality(tmp_path, panel, clean)
    final_review = weekly_large_rebuild_final_review(
        weekly_large_rebuild_event_roster(tmp_path, clean),
        cell_summary,
        event_sign,
        abnormal,
    )
    assert {"event_id", "outcome", "tau", "contribution_share_of_abs_sum"}.issubset(
        event_sign.columns
    )
    assert {"actual_change", "matched_pseudo_mean_change", "abnormal_change"}.issubset(
        abnormal.columns
    )
    assert {"matched_pseudo_n", "match_tiers", "median_state_distance"}.issubset(
        match_quality.columns
    )
    assert set(match_quality["event_id"]) == {"event_039", "event_041"}
    assert match_quality["matched_pseudo_n"].ge(5).all()
    assert {"randomization_p_value", "randomization_status"}.issubset(cell_summary.columns)
    assert {"match_quality_status", "match_quality_blocker_events"}.issubset(cell_summary.columns)
    reserves_main = cell_summary.loc[cell_summary["table_role"] == "headline_reserves_main"]
    assert reserves_main["randomization_p_value"].between(0, 1).all()
    assert set(reserves_main["randomization_p_max"]) == {0.10}
    assert set(reserves_main["randomization_status"]) == {"estimated"}
    high_randomization_p = (
        reserves_main["randomization_p_value"] > reserves_main["randomization_p_max"]
    )
    if high_randomization_p.any():
        assert reserves_main.loc[high_randomization_p, "blocker_flags"].str.contains(
            "randomization_p_value_high"
        ).all()
    assert reserves_main["match_quality_status"].notna().all()
    if set(reserves_main["match_quality_status"]) != {"ok"}:
        assert reserves_main["blocker_flags"].str.contains("event_match_quality").all()
    assert set(event_sign.loc[event_sign["outcome"] == "reserves", "tau"]) == {0, 2, 4}
    assert isinstance(final_review, pd.DataFrame)
    blocker_summary = weekly_large_rebuild_blocker_summary(cell_summary)
    assert {"primary_blocker", "next_step", "blocker_count"}.issubset(blocker_summary.columns)
    assert set(blocker_summary["tau"]) >= {0, 2, 4}
    bad_quality = match_quality.copy()
    bad_quality.loc[bad_quality.index[0], "status"] = "uses_fallback_match_tier"
    bad_review = weekly_large_rebuild_final_review(
        weekly_large_rebuild_event_roster(tmp_path, clean),
        cell_summary,
        event_sign,
        abnormal,
        bad_quality,
    )
    assert "event_match_quality_review_required" in bad_review.loc[0, "binding_limitations"]


def test_weekly_large_rebuild_writeup_and_svg_outputs(tmp_path) -> None:
    weeks = pd.date_range("2024-01-03", periods=80, freq="W-WED")
    panel = pd.DataFrame(
        {
            "week": weeks,
            "deposits": [2000.0] * 80,
            "on_rrp": [500.0 - (idx % 9) for idx in range(80)],
            "reserves": [1000.0 - (idx % 11) * 5 for idx in range(80)],
            "mmf_treasury_holdings": [50.0] * 80,
            "tax_week": [False] * 80,
            "debt_limit_window": [False] * 80,
            "covid_crisis_window": [False] * 80,
            "bank_stress_window": [False] * 80,
            "on_rrp_regime_pre": ["scarce"] * 80,
        }
    )
    clean = pd.DataFrame(
        {
            "event_id": ["event_039", "event_041"],
            "start_date": [weeks[20], weeks[40]],
            "end_date": [weeks[20], weeks[40]],
            "delta_tga_event": [250_000.0, 260_000.0],
            "mean_bill_share": [0.85, 0.8],
            "manual_tags": ["", ""],
        }
    )
    roster = weekly_large_rebuild_event_roster(tmp_path, clean)
    cell_summary = weekly_large_rebuild_cell_summary(tmp_path, panel, clean)
    event_sign = weekly_large_rebuild_event_sign_stability(tmp_path, panel, clean)
    abnormal = weekly_large_rebuild_abnormal_changes(tmp_path, panel, clean)
    final_review = weekly_large_rebuild_final_review(roster, cell_summary, event_sign, abnormal)
    svg_path = tmp_path / "reserves.svg"
    report_path = tmp_path / "writeup.md"
    weekly_large_rebuild_event_time_svg(
        tmp_path,
        panel,
        clean,
        outcome="reserves",
        path=svg_path,
    )
    weekly_large_rebuild_diagnostic_writeup(
        roster,
        cell_summary,
        event_sign,
        abnormal,
        final_review,
        path=report_path,
    )
    assert svg_path.read_text(encoding="utf-8").startswith("<svg")
    report = report_path.read_text(encoding="utf-8")
    assert "Weekly Large-Rebuild Diagnostic" in report
    assert "randomization p" in report
    assert "matched-placebo, bootstrap, match-quality, or randomization gates" in report


def test_weekly_leave_one_event_out_estimates_stability() -> None:
    rows = []
    for event_id in ["a", "b", "c"]:
        for tau in [0, 1, 2, 4, 8]:
            rows.append(
                {
                    "event_id": event_id,
                    "outcome": "deposits",
                    "tau": tau,
                    "level_change_from_baseline": 10.0 + tau,
                }
            )
    out = weekly_leave_one_event_out(pd.DataFrame(rows))
    assert set(out["tau"]) == {0, 1, 2, 4, 8}
    assert out["same_sign_leave_one_out_share"].min() == 1.0


def test_weekly_block_bootstrap_and_placebo_tables(tmp_path) -> None:
    weeks = pd.date_range("2024-01-03", periods=40, freq="W-WED")
    panel = pd.DataFrame(
        {
            "week": weeks,
            "deposits": [1000.0 + i for i in range(40)],
            "on_rrp": [500.0 - i for i in range(40)],
            "reserves": [200.0 + i * 2 for i in range(40)],
            "mmf_treasury_holdings": [50.0 + i for i in range(40)],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["event_001", "event_002"],
            "baseline_date": [weeks[9], weeks[19]],
            "start_date": [weeks[10], weeks[20]],
            "delta_tga_event": [100.0, 120.0],
            "mean_bill_share": [0.8, 0.85],
            "manual_tags": ["", ""],
        }
    )
    event_rows = weekly_event_study_by_event(panel, events)
    bootstrap = weekly_block_bootstrap(event_rows, n_bootstrap=20)
    placebo = weekly_placebo_tests(panel, events, event_rows)
    stable = weekly_stable_claim_candidates(
        weekly_leave_one_event_out(event_rows),
        bootstrap,
        placebo,
    )
    sensitivity = weekly_event_filter_sensitivity(panel, events)
    targeted = weekly_large_rebuild_targeted_review(panel, events)
    calendar = weekly_large_rebuild_calendar_validation(tmp_path, targeted)
    assert {"bootstrap_same_sign_share", "ci_lower", "ci_upper"}.issubset(bootstrap.columns)
    assert "false_positive_share" in placebo.columns
    assert set(placebo["placebo_design"]) == {"matched_event_pseudo"}
    assert "matched_event_groups_n" in placebo.columns
    assert "decision" in sensitivity.columns
    assert isinstance(targeted, pd.DataFrame)
    assert isinstance(calendar, pd.DataFrame)
    assert isinstance(stable, pd.DataFrame)


def test_weekly_stable_claim_candidates_passes_clean_cells() -> None:
    loo = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "tau": [2],
            "full_sample_mean": [10.0],
            "same_sign_leave_one_out_share": [1.0],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "tau": [2],
            "bootstrap_same_sign_share": [1.0],
            "ci_lower": [1.0],
            "ci_upper": [20.0],
        }
    )
    placebo = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "tau": [2],
            "false_positive_share": [0.05],
        }
    )
    out = weekly_stable_claim_candidates(loo, bootstrap, placebo)
    assert out.loc[0, "status"] == "passes_all_gates"


def test_weekly_outcome_claim_readiness_is_outcome_specific() -> None:
    panel = pd.DataFrame(
        {
            "deposits": [1.0] * 40,
            "on_rrp": [1.0] * 40,
            "reserves": [1.0] * 40,
            "mmf_treasury_holdings": [pd.NA] * 20 + [1.0] * 20,
        }
    )
    events = pd.DataFrame({"event_id": [f"event_{idx:03d}" for idx in range(30)]})
    stable = pd.DataFrame(
        {
            "outcome": ["reserves"],
            "tau": [0],
            "status": ["passes_all_gates"],
        }
    )
    out = weekly_outcome_claim_readiness(panel, events, stable)
    reserves = out.loc[out["outcome"] == "reserves"].iloc[0]
    mmf = out.loc[out["outcome"] == "mmf_treasury_holdings"].iloc[0]
    assert reserves["readiness_status"] == "descriptive_stable_pending_manual_validation"
    assert "outcome_coverage_below_90pct" in mmf["blocker_flags"]


def test_weekly_event_isolation_keeps_largest_rebuild_in_cluster() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["small", "large", "later"],
            "baseline_date": ["2024-01-03", "2024-01-17", "2024-04-03"],
            "start_date": ["2024-01-10", "2024-01-24", "2024-04-10"],
            "end_date": ["2024-01-10", "2024-01-24", "2024-04-10"],
            "delta_tga_event": [10.0, 50.0, 20.0],
            "mean_bill_share": [0.6, 0.7, 0.8],
            "manual_tags": ["", "", ""],
            "status": ["clean_candidate", "clean_candidate", "clean_candidate"],
            "exclusion_reason": ["", "", ""],
        }
    )
    clean, exclusions = _isolate_weekly_events(events, min_gap_weeks=8)
    assert list(clean["event_id"]) == ["large", "later"]
    assert list(exclusions["event_id"]) == ["small"]
    assert "clustered_within_8_weeks" in exclusions.loc[0, "exclusion_reason"]


def test_promotion_unknown_boolean_and_missing_placebo_do_not_promote(tmp_path):
    raw = tmp_path / "data/raw/tgarefill"
    _write_minimal_tgarefill_exports(raw)
    path = raw / "promotion_robustness_summary.csv"
    frame = pd.read_csv(path)
    frame["h4_significant_5pct"] = frame["h4_significant_5pct"].astype(object)
    frame.loc[0, "h4_significant_5pct"] = "unknown"
    frame.loc[1, "h4_significant_5pct"] = "False"
    frame.to_csv(path, index=False)
    out = tgarefill_promotion_reconciliation(tmp_path)
    assert not out.status.eq("supported_focused_claim").any()
    assert out.loc[out.channel.eq("MMF Treasury Holdings"), "status"].iloc[0] == "blocked_incomplete_evidence"
    frame = frame.loc[frame.spec.eq("canonical_issue_week")]
    frame["h4_significant_5pct"] = True
    frame.to_csv(path, index=False)
    out = tgarefill_promotion_reconciliation(tmp_path)
    assert not out.status.eq("supported_focused_claim").any()
    assert out.loc[out.channel.eq("ON RRP"), "pretrend_status"].iloc[0] == "missing_same_week_placebo_evidence"


def test_mmf_summary_cannot_override_missing_or_failed_source_gates(tmp_path):
    raw = tmp_path / "data/raw/tgarefill"
    _write_minimal_tgarefill_exports(raw)
    path = raw / "mmfalloc_source_gates.csv"
    path.write_text(path.read_text().replace("mapping,True", "mapping,False"))
    out = tgarefill_promotion_reconciliation(tmp_path)
    mmf = out.loc[out.channel.eq("Fund-level MMF allocation")].iloc[0]
    assert mmf.status == "blocked_mmfalloc_gates"
    assert mmf.permitted_language == ""
    path.unlink()
    out = tgarefill_promotion_reconciliation(tmp_path)
    assert out.loc[out.channel.eq("Fund-level MMF allocation"), "claim_use"].iloc[0] == "not_available"
