from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MONTHLY_PANEL_REQUIRED_FIELDS = [
    "month",
    "gross_bill_issuance",
    "bill_share",
    "coupon_issuance",
    "deposits",
    "retail_mmf_assets",
    "institutional_mmf_assets",
    "total_mmf_assets",
    "mmf_treasury_holdings",
    "on_rrp",
    "reserves",
    "tga",
    "on_rrp_regime",
]

UPSTREAM_BUYCURVE_REQUIRED_FIELDS = [
    "month",
    "security_type",
    "maturity_bucket",
    "auction_count",
    "accepted_amount_sum",
    "offering_amount_sum",
    "weighted_maturity_years",
    "bill_share_by_accepted_amount",
]

REGIME_SPLIT_FIELDS = ["on_rrp_regime", "qe_qt_regime", "high_rate_regime"]


PUBLIC_OUTPUT_REQUIRED_COLUMNS = {
    "output/tables/backend_input_inventory.csv": [
        "input_path",
        "source_family",
        "required",
        "exists",
        "status",
        "bytes",
        "rows",
        "mtime_utc",
        "age_days",
        "freshness_days",
        "freshness_status",
        "date_column",
        "first_date",
        "last_date",
        "valid_observation_rows",
        "first_valid_observation",
        "last_valid_observation",
        "observation_age_days",
        "observation_freshness_days",
        "observation_freshness_status",
        "sha256",
    ],
    "output/tables/source_cache_manifest.csv": [
        "input_path",
        "source_family",
        "source_id",
        "source_name",
        "acquisition_mode",
        "source_url",
        "required",
        "cache_status",
        "cache_sha256",
        "cache_rows",
        "valid_observation_rows",
        "first_valid_observation",
        "last_valid_observation",
        "mtime_freshness_status",
        "observation_freshness_status",
    ],
    "output/tables/source_refresh_status.csv": [
        "input_path",
        "source_family",
        "required",
        "exists",
        "cache_status",
        "mtime_freshness_status",
        "observation_freshness_status",
        "valid_observation_rows",
        "last_valid_observation",
        "latest_fetch_status",
        "latest_fetch_rows",
        "latest_fetch_last_observation",
        "refresh_status",
        "required_action",
    ],
    "output/tables/evidence_gate_summary.csv": [
        "design_path",
        "status",
        "claim_use",
        "evidence_basis",
        "primary_artifacts",
        "binding_blockers",
        "next_design_step",
    ],
    "output/tables/monthly_bill_shock_events.csv": [
        "month",
        "gross_bill_issuance",
        "bill_supply_shock_resid_100b",
        "bill_share",
    ],
    "output/tables/monthly_candidate_event_months.csv": [
        "sample",
        "outcome",
        "event_month",
        "impact_sum_change",
        "manual_review_priority",
    ],
    "output/tables/monthly_candidate_event_review.csv": [
        "sample",
        "outcome",
        "status",
        "n_events",
        "recommended_next_step",
    ],
    "output/tables/monthly_candidate_lp_coefficients.csv": [
        "sample",
        "outcome",
        "horizon_months",
        "nobs",
        "shock_beta",
        "shock_t_hc1",
    ],
    "output/tables/monthly_candidate_table.csv": [
        "sample",
        "outcome",
        "event_month",
        "candidate_status",
        "priority",
        "direction_pattern",
        "external_source_count",
        "required_next_step",
    ],
    "output/tables/monthly_claim_readiness.csv": [
        "sample",
        "outcome",
        "readiness_status",
        "blocker_flags",
        "required_next_step",
    ],
    "output/tables/monthly_correlations.csv": ["left", "right", "correlation"],
    "output/tables/monthly_coverage_qa.csv": [
        "column",
        "status",
        "source_family",
        "unit",
        "coverage_share",
    ],
    "output/tables/monthly_descriptive_summary.csv": [
        "column",
        "nobs",
        "mean",
        "std",
        "min",
        "max",
    ],
    "output/tables/monthly_design_readiness.csv": [
        "sample",
        "outcome",
        "status",
        "nobs",
        "recommended_next_step",
    ],
    "output/tables/monthly_event_study_by_event.csv": [
        "event_month",
        "outcome",
        "window",
        "sum_change",
        "mean_change",
    ],
    "output/tables/monthly_event_study_summary.csv": [
        "outcome",
        "window",
        "n_events",
        "mean_sum_change",
    ],
    "output/tables/monthly_external_calendar_evidence.csv": [
        "event_month",
        "evidence_status",
        "evidence_topic",
        "official_source",
        "source_url",
        "calendar_note",
    ],
    "output/tables/monthly_filtered_candidate_event_review.csv": [
        "sample",
        "outcome",
        "status",
        "excluded_event_months",
        "remaining_isolated_bill_shock_events",
        "recommended_next_step",
    ],
    "output/tables/monthly_filtered_candidate_lp_coefficients.csv": [
        "sample",
        "outcome",
        "status",
        "horizon_months",
        "shock_beta",
        "shock_t_hc1",
    ],
    "output/tables/monthly_filtered_candidate_shortlist.csv": [
        "sample",
        "outcome",
        "priority",
        "event_status",
        "recommended_next_step",
    ],
    "output/tables/monthly_first_pass_regressions.csv": [
        "outcome",
        "sample",
        "spec",
        "status",
        "predictor",
        "nobs",
    ],
    "output/tables/monthly_future_row_qa.csv": [
        "month",
        "future_or_scheduled",
        "baseline_estimation_use",
    ],
    "output/tables/monthly_local_projection_diagnostics.csv": [
        "outcome",
        "horizon_months",
        "spec",
        "status",
        "predictor",
        "nobs",
    ],
    "output/tables/monthly_on_rrp_manual_review.csv": [
        "sample",
        "outcome",
        "decision",
        "event_months",
        "decision_flags",
        "required_next_step",
    ],
    "output/tables/monthly_pretrend_diagnostics.csv": [
        "outcome",
        "pretrend_measure",
        "status",
        "nobs",
        "gross_bill_t_hc1",
    ],
    "output/tables/public_output_aliases.csv": [
        "canonical_path",
        "compatibility_path",
        "relationship",
        "status",
        "canonical_exists",
        "compatibility_exists",
        "removal_policy",
    ],
    "output/tables/monthly_regime_classification_qa.csv": [
        "regime_variable",
        "status",
        "regime",
        "n_months",
    ],
    "output/tables/monthly_regime_interaction_diagnostics.csv": [
        "outcome",
        "status",
        "predictor",
        "nobs",
    ],
    "output/tables/monthly_regime_summary.csv": [
        "regime_variable",
        "regime",
        "n_months",
        "mean_gross_bill_issuance",
    ],
    "output/tables/monthly_regime_threshold_sensitivity.csv": [
        "threshold_spec",
        "regime",
        "n_months",
        "n_isolated_bill_shock_events",
    ],
    "output/tables/monthly_residual_shock_pretrend_diagnostics.csv": [
        "outcome",
        "pretrend_measure",
        "status",
        "nobs",
        "shock_t_hc1",
    ],
    "output/tables/monthly_sample_recommendations.csv": [
        "sample",
        "required_columns",
        "n_months",
        "recommended_use",
    ],
    "output/tables/monthly_sample_windows.csv": [
        "column",
        "status",
        "non_null",
        "first_month",
        "last_month",
    ],
    "output/tables/monthly_source_metadata.csv": [
        "column",
        "status",
        "source_family",
        "unit",
    ],
    "output/tables/monthly_strong_candidate_narratives.csv": [
        "sample",
        "outcome",
        "event_month",
        "priority",
        "narrative_status",
    ],
    "output/tables/weekly_block_bootstrap.csv": [
        "outcome",
        "tau",
        "status",
        "n_events",
        "bootstrap_same_sign_share",
    ],
    "output/tables/weekly_claim_readiness.csv": [
        "design",
        "readiness_status",
        "clean_positive_events_n",
        "blocker_flags",
        "required_next_step",
    ],
    "output/tables/weekly_event_candidates_clean.csv": [
        "event_id",
        "baseline_date",
        "start_date",
        "end_date",
        "status",
        "exclusion_reason",
    ],
    "output/tables/weekly_event_exclusion_log.csv": [
        "event_id",
        "baseline_date",
        "start_date",
        "end_date",
        "status",
        "exclusion_reason",
    ],
    "output/tables/weekly_event_filter_sensitivity.csv": [
        "variant",
        "description",
        "clean_events_n",
        "decision",
        "required_next_step",
    ],
    "output/tables/weekly_event_study_by_event.csv": [
        "event_id",
        "event_start",
        "event_week",
        "tau",
        "outcome",
        "level_change_from_baseline",
    ],
    "output/tables/weekly_event_study_summary.csv": [
        "outcome",
        "tau",
        "n_events",
        "mean_change",
    ],
    "output/tables/weekly_large_rebuild_abnormal_changes.csv": [
        "targeted_design",
        "event_id",
        "event_start",
        "outcome",
        "tau",
        "abnormal_change",
    ],
    "output/tables/weekly_large_rebuild_calendar_validation.csv": [
        "targeted_design",
        "event_id",
        "event_start",
        "evidence_status",
        "official_source",
        "validation_decision",
    ],
    "output/tables/weekly_large_rebuild_cell_summary.csv": [
        "targeted_design",
        "table_role",
        "outcome",
        "tau",
        "event_count",
        "mean_change",
        "match_quality_status",
        "randomization_p_value",
        "randomization_p_max",
        "randomization_status",
        "status",
        "blocker_flags",
    ],
    "output/tables/weekly_large_rebuild_blocker_summary.csv": [
        "targeted_design",
        "table_role",
        "outcome",
        "tau",
        "status",
        "blocker_count",
        "primary_blocker",
        "next_step",
    ],
    "output/tables/weekly_large_rebuild_event_roster.csv": [
        "targeted_design",
        "event_id",
        "event_start",
        "event_end",
        "sample_role",
        "exclusion_reason",
    ],
    "output/tables/weekly_large_rebuild_event_sign_stability.csv": [
        "targeted_design",
        "event_id",
        "event_start",
        "sample_role",
        "outcome",
        "tau",
        "sign",
    ],
    "output/tables/weekly_large_rebuild_final_review.csv": [
        "targeted_design",
        "review_scope",
        "status",
        "claim_use",
        "main_event_count",
        "binding_limitations",
        "required_next_step",
    ],
    "output/tables/weekly_large_rebuild_match_quality.csv": [
        "targeted_design",
        "event_id",
        "event_start",
        "outcome",
        "matched_pseudo_n",
        "match_tiers",
        "worst_match_tier",
        "median_state_distance",
        "median_week_distance",
        "status",
    ],
    "output/tables/weekly_large_rebuild_randomization_inference.csv": [
        "targeted_design",
        "outcome",
        "tau",
        "event_count",
        "matched_event_groups_n",
        "draws",
        "actual_mean_change",
        "two_sided_p_value",
        "placebo_design",
        "status",
    ],
    "output/tables/weekly_large_rebuild_targeted_review.csv": [
        "targeted_design",
        "decision",
        "outcome",
        "tau",
        "event_id",
        "required_next_step",
    ],
    "output/tables/weekly_leave_one_event_out.csv": [
        "outcome",
        "tau",
        "status",
        "n_events",
        "same_sign_leave_one_out_share",
    ],
    "output/tables/weekly_lp_coefficients.csv": [
        "outcome",
        "horizon_weeks",
        "status",
        "predictor",
        "nobs",
    ],
    "output/tables/weekly_outcome_claim_readiness.csv": [
        "outcome",
        "readiness_status",
        "clean_positive_events_n",
        "outcome_coverage_share",
        "claim_use",
    ],
    "output/tables/weekly_placebo_tests.csv": [
        "outcome",
        "tau",
        "status",
        "actual_mean_change",
        "false_positive_share",
        "placebo_design",
    ],
    "output/tables/weekly_pretrend_balance.csv": [
        "event_id",
        "event_start",
        "outcome",
        "pre4_sum",
        "status",
    ],
    "output/tables/weekly_stable_claim_candidates.csv": [
        "outcome",
        "tau",
        "status",
        "event_mean_change",
        "blocker_flags",
        "claim_use",
    ],
    "output/tables/weekly_timing_alignment_qa.csv": [
        "column",
        "non_null",
        "coverage_share",
        "first_week",
        "last_week",
    ],
    "output/tables/weekly_terminal_period_qa.csv": [
        "week",
        "terminal_incomplete",
        "terminal_non_null_share",
        "baseline_estimation_use",
    ],
    "output/tables/weekly_upstream_tgarefill_qa.csv": [
        "export",
        "status",
        "rows",
    ],
    "output/tables/tgarefill_promotion_reconciliation.csv": [
        "claim_id",
        "status",
        "claim_use",
        "evidence_basis",
        "channel",
        "h4_effect_bn",
        "h4_t_stat_nw",
        "significant_5pct", "pretrend_status", "permitted_language",
        "response_var", "sample", "timing", "shock_spec", "shock_sd_bn", "n_obs",
        "same_week_placebo_count", "aggregate_complete",
        "broad_substitution_reconciliation",
        "forbidden_upgrade",
    ],
}

PUBLIC_OUTPUT_DATE_COLUMNS = {
    "output/tables/backend_input_inventory.csv": [
        "mtime_utc",
        "first_date",
        "last_date",
        "first_valid_observation",
        "last_valid_observation",
    ],
    "output/tables/source_cache_manifest.csv": ["first_valid_observation", "last_valid_observation"],
    "output/tables/source_refresh_status.csv": [
        "last_valid_observation",
        "latest_fetch_last_observation",
    ],
    "output/tables/monthly_bill_shock_events.csv": ["month"],
    "output/tables/monthly_candidate_event_months.csv": ["event_month"],
    "output/tables/monthly_candidate_table.csv": ["event_month"],
    "output/tables/monthly_event_study_by_event.csv": ["event_month"],
    "output/tables/monthly_external_calendar_evidence.csv": ["event_month"],
    "output/tables/monthly_future_row_qa.csv": ["month"],
    "output/tables/monthly_sample_windows.csv": ["first_month", "last_month"],
    "output/tables/weekly_event_candidates_clean.csv": [
        "baseline_date",
        "start_date",
        "end_date",
    ],
    "output/tables/weekly_event_exclusion_log.csv": ["baseline_date", "start_date", "end_date"],
    "output/tables/weekly_event_study_by_event.csv": ["event_start", "event_week"],
    "output/tables/weekly_large_rebuild_abnormal_changes.csv": ["event_start"],
    "output/tables/weekly_large_rebuild_calendar_validation.csv": ["event_start"],
    "output/tables/weekly_large_rebuild_event_roster.csv": ["event_start", "event_end"],
    "output/tables/weekly_large_rebuild_event_sign_stability.csv": ["event_start"],
    "output/tables/weekly_large_rebuild_match_quality.csv": ["event_start"],
    "output/tables/weekly_large_rebuild_targeted_review.csv": ["event_start", "event_week"],
    "output/tables/weekly_pretrend_balance.csv": ["event_start"],
    "output/tables/weekly_terminal_period_qa.csv": ["week"],
    "output/tables/weekly_timing_alignment_qa.csv": ["first_week", "last_week"],
}

PUBLIC_OUTPUT_NONNEGATIVE_COLUMNS = {
    "output/tables/backend_input_inventory.csv": [
        "bytes",
        "rows",
        "age_days",
        "freshness_days",
        "valid_observation_rows",
        "observation_age_days",
        "observation_freshness_days",
    ],
    "output/tables/source_cache_manifest.csv": ["cache_rows", "valid_observation_rows"],
    "output/tables/source_refresh_status.csv": ["valid_observation_rows", "latest_fetch_rows"],
    "output/tables/monthly_future_row_qa.csv": ["non_null_columns"],
    "output/tables/monthly_candidate_event_review.csv": [
        "n_events",
        "pre3_n_events",
        "impact_n_events",
        "post3_n_events",
        "post6_n_events",
    ],
    "output/tables/monthly_candidate_lp_coefficients.csv": [
        "horizon_months",
        "nobs",
        "isolated_bill_shock_events",
    ],
    "output/tables/monthly_candidate_table.csv": ["external_source_count"],
    "output/tables/monthly_claim_readiness.csv": [
        "n_event_months",
        "n_external_source_urls",
        "remaining_isolated_bill_shock_events",
    ],
    "output/tables/monthly_coverage_qa.csv": [
        "non_null",
        "missing",
        "max_internal_gap_months",
        "leading_missing_months",
        "trailing_missing_months",
    ],
    "output/tables/monthly_descriptive_summary.csv": ["nobs"],
    "output/tables/monthly_design_readiness.csv": [
        "nobs",
        "sample_months",
        "isolated_bill_shock_events",
    ],
    "output/tables/monthly_event_study_by_event.csv": ["n_obs_in_window"],
    "output/tables/monthly_event_study_summary.csv": ["n_events"],
    "output/tables/monthly_filtered_candidate_event_review.csv": [
        "remaining_isolated_bill_shock_events",
        "n_events",
        "pre3_n_events",
        "impact_n_events",
        "post3_n_events",
        "post6_n_events",
    ],
    "output/tables/monthly_filtered_candidate_lp_coefficients.csv": [
        "horizon_months",
        "nobs",
        "remaining_isolated_bill_shock_events",
    ],
    "output/tables/monthly_filtered_candidate_shortlist.csv": [
        "remaining_isolated_bill_shock_events",
    ],
    "output/tables/monthly_first_pass_regressions.csv": ["nobs", "min_nobs"],
    "output/tables/monthly_local_projection_diagnostics.csv": ["horizon_months", "nobs"],
    "output/tables/monthly_on_rrp_manual_review.csv": ["remaining_isolated_bill_shock_events"],
    "output/tables/monthly_pretrend_diagnostics.csv": ["nobs"],
    "output/tables/monthly_regime_classification_qa.csv": [
        "n_months",
        "n_isolated_bill_shock_events",
        "missing_months_for_variable",
    ],
    "output/tables/monthly_regime_interaction_diagnostics.csv": ["nobs"],
    "output/tables/monthly_regime_summary.csv": ["n_months"],
    "output/tables/monthly_regime_threshold_sensitivity.csv": [
        "n_months",
        "n_isolated_bill_shock_events",
    ],
    "output/tables/monthly_residual_shock_pretrend_diagnostics.csv": ["nobs"],
    "output/tables/monthly_sample_recommendations.csv": ["n_months"],
    "output/tables/monthly_sample_windows.csv": ["non_null"],
    "output/tables/weekly_block_bootstrap.csv": ["tau", "n_events", "n_bootstrap"],
    "output/tables/weekly_claim_readiness.csv": [
        "clean_positive_events_n",
        "estimated_lp_rows",
        "event_study_rows",
        "leave_one_event_out_rows",
        "block_bootstrap_rows",
        "placebo_rows",
    ],
    "output/tables/weekly_event_filter_sensitivity.csv": [
        "clean_events_n",
        "stable_cells_n",
        "bootstrap_ci_crossing_cells",
    ],
    "output/tables/weekly_event_study_summary.csv": ["n_events"],
    "output/tables/weekly_large_rebuild_abnormal_changes.csv": [
        "tau",
        "matched_pseudo_n",
    ],
    "output/tables/weekly_large_rebuild_blocker_summary.csv": [
        "tau",
        "blocker_count",
    ],
    "output/tables/weekly_large_rebuild_cell_summary.csv": [
        "tau",
        "event_count",
        "pseudo_events_n",
        "matched_event_groups_n",
    ],
    "output/tables/weekly_large_rebuild_event_sign_stability.csv": ["tau"],
    "output/tables/weekly_large_rebuild_final_review.csv": [
        "main_event_count",
        "headline_passing_cells",
        "headline_blocked_cells",
    ],
    "output/tables/weekly_large_rebuild_match_quality.csv": [
        "matched_pseudo_n",
        "min_state_distance",
        "median_state_distance",
        "max_state_distance",
        "min_week_distance",
        "median_week_distance",
        "max_week_distance",
    ],
    "output/tables/weekly_large_rebuild_randomization_inference.csv": [
        "tau",
        "event_count",
        "matched_event_groups_n",
        "draws",
    ],
    "output/tables/weekly_large_rebuild_targeted_review.csv": ["tau", "event_count"],
    "output/tables/weekly_leave_one_event_out.csv": ["tau", "n_events"],
    "output/tables/weekly_lp_coefficients.csv": ["horizon_weeks", "nobs", "hac_lags"],
    "output/tables/weekly_outcome_claim_readiness.csv": [
        "clean_positive_events_n",
        "stable_cells_n",
        "blocked_cells_n",
    ],
    "output/tables/weekly_placebo_tests.csv": [
        "tau",
        "pseudo_events_n",
        "matched_event_groups_n",
    ],
    "output/tables/weekly_pretrend_balance.csv": ["pre4_non_null"],
    "output/tables/weekly_stable_claim_candidates.csv": ["tau"],
    "output/tables/weekly_terminal_period_qa.csv": ["non_null_columns"],
    "output/tables/weekly_timing_alignment_qa.csv": ["non_null", "max_internal_gap_weeks"],
    "output/tables/weekly_upstream_tgarefill_qa.csv": ["rows"],
}

PUBLIC_OUTPUT_SHARE_COLUMNS = {
    "output/tables/monthly_bill_shock_events.csv": ["bill_share"],
    "output/tables/monthly_coverage_qa.csv": ["coverage_share"],
    "output/tables/monthly_design_readiness.csv": [
        "outcome_coverage_share_within_sample",
    ],
    "output/tables/monthly_regime_classification_qa.csv": ["share_of_panel"],
    "output/tables/weekly_block_bootstrap.csv": ["bootstrap_same_sign_share"],
    "output/tables/weekly_claim_readiness.csv": [
        "min_outcome_coverage_share",
        "min_leave_one_out_same_sign_share",
        "min_bootstrap_same_sign_share",
        "max_placebo_false_positive_share",
    ],
    "output/tables/weekly_event_filter_sensitivity.csv": [
        "min_placebo_false_positive_share",
        "max_placebo_false_positive_share",
        "min_bootstrap_same_sign_share",
    ],
    "output/tables/weekly_large_rebuild_cell_summary.csv": [
        "same_sign_event_share",
        "leave_one_out_same_sign_share",
        "bootstrap_same_sign_share",
        "placebo_false_positive_share",
        "randomization_p_value",
        "randomization_p_max",
    ],
    "output/tables/weekly_large_rebuild_event_sign_stability.csv": [
        "contribution_share_of_abs_sum",
    ],
    "output/tables/weekly_large_rebuild_randomization_inference.csv": [
        "two_sided_p_value",
    ],
    "output/tables/weekly_large_rebuild_targeted_review.csv": [
        "mean_bill_share",
        "leave_one_out_same_sign_share",
        "bootstrap_same_sign_share",
        "placebo_false_positive_share",
    ],
    "output/tables/monthly_future_row_qa.csv": ["terminal_non_null_share"],
    "output/tables/weekly_leave_one_event_out.csv": ["same_sign_leave_one_out_share"],
    "output/tables/weekly_outcome_claim_readiness.csv": ["outcome_coverage_share"],
    "output/tables/weekly_placebo_tests.csv": ["false_positive_share"],
    "output/tables/weekly_stable_claim_candidates.csv": [
        "leave_one_out_same_sign_share",
        "bootstrap_same_sign_share",
        "placebo_false_positive_share",
    ],
    "output/tables/weekly_terminal_period_qa.csv": ["terminal_non_null_share"],
    "output/tables/weekly_timing_alignment_qa.csv": ["coverage_share"],
}

PUBLIC_OUTPUT_REQUIRED_TEXT_COLUMNS = {
    "output/tables/backend_input_inventory.csv": [
        "input_path",
        "source_family",
        "status",
        "freshness_status",
        "observation_freshness_status",
    ],
    "output/tables/source_cache_manifest.csv": [
        "input_path",
        "source_family",
        "source_id",
        "source_name",
        "acquisition_mode",
        "cache_status",
        "mtime_freshness_status",
        "observation_freshness_status",
    ],
    "output/tables/source_refresh_status.csv": [
        "input_path",
        "source_family",
        "cache_status",
        "mtime_freshness_status",
        "observation_freshness_status",
        "refresh_status",
        "required_action",
    ],
    "output/tables/evidence_gate_summary.csv": ["design_path", "status"],
    "output/tables/monthly_candidate_event_review.csv": ["sample", "outcome", "status"],
    "output/tables/monthly_claim_readiness.csv": ["sample", "outcome", "readiness_status"],
    "output/tables/monthly_design_readiness.csv": ["sample", "outcome", "status"],
    "output/tables/monthly_filtered_candidate_shortlist.csv": [
        "sample",
        "outcome",
        "priority",
    ],
    "output/tables/monthly_on_rrp_manual_review.csv": ["sample", "outcome", "decision"],
    "output/tables/weekly_claim_readiness.csv": ["design", "readiness_status"],
    "output/tables/weekly_large_rebuild_cell_summary.csv": [
        "targeted_design",
        "table_role",
        "outcome",
        "match_quality_status",
        "randomization_status",
        "status",
    ],
    "output/tables/weekly_large_rebuild_blocker_summary.csv": [
        "targeted_design",
        "table_role",
        "outcome",
        "status",
        "next_step",
    ],
    "output/tables/weekly_large_rebuild_event_roster.csv": [
        "targeted_design",
        "event_id",
        "sample_role",
    ],
    "output/tables/weekly_large_rebuild_final_review.csv": [
        "targeted_design",
        "review_scope",
        "status",
    ],
    "output/tables/weekly_large_rebuild_match_quality.csv": [
        "targeted_design",
        "event_id",
        "outcome",
        "match_tiers",
        "worst_match_tier",
        "status",
    ],
    "output/tables/weekly_large_rebuild_randomization_inference.csv": [
        "targeted_design",
        "outcome",
        "placebo_design",
        "status",
    ],
    "output/tables/public_output_aliases.csv": [
        "canonical_path",
        "compatibility_path",
        "relationship",
        "status",
    ],
    "output/tables/weekly_stable_claim_candidates.csv": ["outcome", "status"],
    "output/tables/tgarefill_promotion_reconciliation.csv": [
        "claim_id",
        "status",
        "claim_use",
        "evidence_basis",
        "channel",
    ],
}

PUBLIC_OUTPUT_ALIASES = {
    "output/tables/monthly_readiness_summary.csv": "output/tables/monthly_claim_readiness.csv",
    "output/tables/weekly_design_readiness.csv": "output/tables/weekly_claim_readiness.csv",
    "output/tables/weekly_outcome_readiness.csv": "output/tables/weekly_outcome_claim_readiness.csv",
    "output/tables/weekly_stability_candidates.csv": "output/tables/weekly_stable_claim_candidates.csv",
}


def _install_public_output_aliases() -> None:
    contract_maps = [
        PUBLIC_OUTPUT_REQUIRED_COLUMNS,
        PUBLIC_OUTPUT_DATE_COLUMNS,
        PUBLIC_OUTPUT_NONNEGATIVE_COLUMNS,
        PUBLIC_OUTPUT_SHARE_COLUMNS,
        PUBLIC_OUTPUT_REQUIRED_TEXT_COLUMNS,
    ]
    for alias, source in PUBLIC_OUTPUT_ALIASES.items():
        for contract in contract_maps:
            if source in contract:
                contract[alias] = list(contract[source])


_install_public_output_aliases()


@dataclass(frozen=True)
class OutputSchemaReport:
    status: str
    checked_files: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_public_output_schemas(root: Path) -> OutputSchemaReport:
    errors: list[str] = []
    warnings: list[str] = []
    output_dir = root / "output"
    checked = 0

    table_dir = output_dir / "tables"
    for path in sorted(table_dir.glob("*.csv")) if table_dir.exists() else []:
        rel = path.relative_to(root).as_posix()
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        checked += 1
        if rel not in PUBLIC_OUTPUT_REQUIRED_COLUMNS:
            errors.append(f"{rel}: missing public output schema contract")
        header = _csv_header(path)
        if not header:
            errors.append(f"{rel}: missing header row")
        if any(str(column).startswith("Unnamed:") for column in header):
            errors.append(f"{rel}: contains unnamed columns")

    for rel, required_columns in PUBLIC_OUTPUT_REQUIRED_COLUMNS.items():
        path = root / rel
        if not path.exists():
            warnings.append(f"{rel}: expected public output not found")
            continue
        header = _csv_header(path)
        missing = [column for column in required_columns if column not in header]
        if missing:
            errors.append(f"{rel}: missing required columns {','.join(missing)}")
            continue
        errors.extend(_validate_public_output_values(path, rel))

    return OutputSchemaReport(
        status="failed" if errors else "ok",
        checked_files=checked,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _csv_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except pd.errors.EmptyDataError:
        return []


def _validate_public_output_values(path: Path, rel: str) -> list[str]:
    errors: list[str] = []
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return [f"{rel}: missing header row"]
    if table.empty:
        return errors

    for column in PUBLIC_OUTPUT_DATE_COLUMNS.get(rel, []):
        if column not in table.columns:
            continue
        values = _nonempty_series(table[column])
        if values.empty:
            continue
        parsed = pd.to_datetime(values, errors="coerce")
        if parsed.isna().any():
            errors.append(f"{rel}: column `{column}` contains unparsable dates")

    for column in PUBLIC_OUTPUT_NONNEGATIVE_COLUMNS.get(rel, []):
        if column not in table.columns:
            continue
        values = _numeric_series(table[column])
        if values.isna().any():
            errors.append(f"{rel}: column `{column}` contains non-numeric values")
            continue
        if (values < 0).any():
            errors.append(f"{rel}: column `{column}` contains negative values")

    for column in PUBLIC_OUTPUT_SHARE_COLUMNS.get(rel, []):
        if column not in table.columns:
            continue
        values = _numeric_series(table[column])
        if values.isna().any():
            errors.append(f"{rel}: column `{column}` contains non-numeric values")
            continue
        if ((values < 0) | (values > 1)).any():
            errors.append(f"{rel}: column `{column}` contains values outside [0, 1]")

    if rel == "output/tables/monthly_correlations.csv" and "correlation" in table.columns:
        values = _numeric_series(table["correlation"])
        if values.isna().any():
            errors.append(f"{rel}: column `correlation` contains non-numeric values")
        elif ((values < -1) | (values > 1)).any():
            errors.append(f"{rel}: column `correlation` contains values outside [-1, 1]")

    for column in PUBLIC_OUTPUT_REQUIRED_TEXT_COLUMNS.get(rel, []):
        if column not in table.columns:
            continue
        values = _nonempty_series(table[column])
        if len(values) != len(table):
            errors.append(f"{rel}: column `{column}` contains blank values")

    return errors


def _nonempty_series(series: pd.Series) -> pd.Series:
    text = series.dropna().astype(str).str.strip()
    return text.loc[text != ""]


def _numeric_series(series: pd.Series) -> pd.Series:
    values = series.dropna()
    if values.empty:
        return pd.Series(dtype="float64")
    text = values.astype(str).str.strip()
    text = text.loc[text != ""]
    return pd.to_numeric(text, errors="coerce")
