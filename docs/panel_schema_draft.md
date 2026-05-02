# Panel Schema Draft

## Monthly panel first

Grain: one row per calendar month.

Core blocks:

- Treasury supply: gross bill issuance, coupon issuance, bill share, weighted maturity, bill-share shock.
- MMFs: total assets, government MMF assets, Treasury holdings, repo holdings, agency holdings, ON RRP exposure.
- Fed plumbing: ON RRP, reserves, TGA, Fed Treasury holdings, IORB, ON RRP rate, fed funds, SOFR.
- Banks: deposits, cash assets, Treasury/agency securities, loans, bank group splits if available.
- Rates/stress: bill yields, deposit rates, repo spreads, VIX/MOVE when added later.
- Regimes: ON RRP abundant/transition/scarce, QE/QT, high-rate/low-rate.

## Weekly panel second

Weekly analysis should come after the monthly MVP. H.8 deposits are noisy at weekly frequency, and MMF holdings may be monthly. Weekly local projections should use carefully documented timing conventions and bill issuance/surprise measures.

## Broad liquidity aggregate

The broad liquidity aggregate is a measured/proxied construct, not a primitive fact. Draft components:

```text
deposits + government MMF assets or shares + direct bill-holding proxy + ON RRP
```

Document direct versus proxied components before running regressions.

## Current analysis outputs

`python -B -m liqsub.cli analyze-monthly` writes:

- `data/clean/monthly_liquidity_substitution_panel_analysis.csv`
- `output/tables/monthly_source_metadata.csv`
- `output/tables/monthly_coverage_qa.csv`
- `output/tables/monthly_future_row_qa.csv`
- `output/tables/monthly_regime_classification_qa.csv`
- `output/tables/monthly_sample_windows.csv`
- `output/tables/monthly_descriptive_summary.csv`
- `output/tables/monthly_regime_summary.csv`
- `output/tables/monthly_regime_threshold_sensitivity.csv`
- `output/tables/monthly_regime_interaction_diagnostics.csv`
- `output/tables/monthly_sample_recommendations.csv`
- `output/tables/monthly_design_readiness.csv`
- `output/tables/monthly_candidate_lp_coefficients.csv`
- `output/tables/monthly_candidate_event_review.csv`
- `output/tables/monthly_candidate_event_months.csv`
- `output/tables/monthly_filtered_candidate_lp_coefficients.csv`
- `output/tables/monthly_filtered_candidate_event_review.csv`
- `output/tables/monthly_filtered_candidate_shortlist.csv`
- `output/tables/monthly_strong_candidate_narratives.csv`
- `output/tables/monthly_external_calendar_evidence.csv`
- `output/tables/monthly_candidate_table.csv`
- `output/tables/monthly_readiness_summary.csv`
- `output/tables/monthly_claim_readiness.csv`
- `output/tables/monthly_correlations.csv`
- `output/tables/monthly_bill_shock_events.csv`
- `output/tables/monthly_event_study_by_event.csv`
- `output/tables/monthly_event_study_summary.csv`
- `output/tables/monthly_first_pass_regressions.csv`
- `output/tables/monthly_local_projection_diagnostics.csv`
- `output/tables/monthly_pretrend_diagnostics.csv`
- `output/tables/monthly_residual_shock_pretrend_diagnostics.csv`
- `output/reports/monthly_mvp_report.md`
- `output/reports/monthly_candidate_review_report.md`

Weekly `tgarefill` reuse outputs:

- `data/clean/weekly_liquidity_substitution_panel.csv`
- `output/tables/weekly_upstream_tgarefill_qa.csv`
- `output/tables/weekly_timing_alignment_qa.csv`
- `output/tables/weekly_event_candidates_clean.csv`
- `output/tables/weekly_event_exclusion_log.csv`
- `output/tables/weekly_pretrend_balance.csv`
- `output/tables/weekly_lp_coefficients.csv`
- `output/tables/weekly_event_study_by_event.csv`
- `output/tables/weekly_event_study_summary.csv`
- `output/tables/weekly_leave_one_event_out.csv`
- `output/tables/weekly_block_bootstrap.csv`
- `output/tables/weekly_placebo_tests.csv`
- `output/tables/weekly_stability_candidates.csv`
- `output/tables/weekly_stable_claim_candidates.csv`
- `output/tables/weekly_outcome_readiness.csv`
- `output/tables/weekly_outcome_claim_readiness.csv`
- `output/tables/weekly_design_readiness.csv`
- `output/tables/weekly_claim_readiness.csv`
- `output/tables/weekly_event_filter_sensitivity.csv`
- `output/tables/weekly_large_rebuild_event_roster.csv`
- `output/tables/weekly_large_rebuild_cell_summary.csv`
- `output/tables/weekly_large_rebuild_event_sign_stability.csv`
- `output/tables/weekly_large_rebuild_abnormal_changes.csv`
- `output/tables/weekly_large_rebuild_final_review.csv`
- `output/tables/evidence_gate_summary.csv`
- `output/reports/weekly_identification_candidate_report.md`
- `output/reports/evidence_gate_summary.md`

The source metadata and coverage QA tables are the starting point for any sample decision: they document units, native frequency, monthly aggregation, missingness, first/last coverage, and internal gaps. The regime classification QA table documents regime counts, missingness, and isolated bill-shock event counts by regime definition. The design-readiness table turns those diagnostics into outcome/sample-level statuses: `candidate_design`, `diagnostic_only`, `too_sparse`, or `missing_required_columns`. The candidate LP table filters the continuous residual-shock LP diagnostics down to readiness-approved outcome/sample rows for manual review. The candidate event-review and event-month tables summarize pre, impact, and post windows for the same readiness-approved rows. The filtered candidate tables exclude high-pre-event-movement shock months from candidate LP and event-review diagnostics. The filtered shortlist joins filtered event status with h0/h3/h6 LP coefficients and assigns a manual-review priority. The strong-candidate narrative table expands the strongest shortlist rows with event-month context and explicit calendar-check prompts. The external-calendar evidence table attaches official-source context for the strong event months. The candidate table, readiness table, and candidate review report collapse those rows into a compact descriptive summary for backend triage.

The first-pass regressions, residual bill-supply event windows, local-projection diagnostics, and regime-interaction diagnostics are descriptive diagnostics, not causal estimates. The current broad `supply_only` specification can estimate the full available sample and the ON RRP scarce split. Abundant and transition split regressions are intentionally skipped when the sample has too few observations. Raw bill-issuance pretrends, residual-shock pretrends, and pre-event movement diagnostics should be checked before any coefficient is used for public causal interpretation.

The weekly path should reserve the word `shock` for ex ante or upstream surprise measures from `tgarefill`. The monthly residualized bill-supply objects are descriptive large-move diagnostics.
