# liqsub

`liqsub` builds a historical liquidity-plumbing diagnostics backend around Treasury supply,
bank deposits, money-market funds, reserves, the Treasury General Account, and ON RRP.

The repository is designed as a reproducible historical snapshot, not a live data service. It
ships source contracts, clean monthly and weekly panels, public diagnostic tables, release
manifests, bundle checks, and conservative evidence gates. The generated outputs are meant for
backend diagnostics and descriptive analysis; they do not establish a headline causal channel.

## Setup

Use a project environment under `~/venvs`:

```bash
python -m venv ~/venvs/liqsub
source ~/venvs/liqsub/bin/activate
python -m pip install -e '.[dev]'
python -B -m pytest -q
```

## Source Snapshot

The local rebuild depends on project-local source caches and imported upstream exports:

```text
data/raw/fred/
data/raw/fiscaldata/
data/raw/ofr/
data/raw/buycurve/monthly_issuance_maturity_panel.csv
data/raw/tgarefill/master_weekly_panel.csv
data/raw/tgarefill/event_candidates.csv
data/raw/tgarefill/auction_shock_lp.csv
data/raw/tgarefill/canonical_bill_surprise_shocks.csv
data/raw/tgarefill/promotion_robustness_summary.csv
data/manual/event_calendar_context.csv
data/manual/weekly_large_rebuild_calendar_context.csv
```

The public release bundle excludes `data/raw/`; preserve a full source snapshot separately when
source-level rebuild reproducibility is required. Public source-cache and refresh metadata are
written to:

```text
output/tables/backend_input_inventory.csv
output/tables/source_cache_manifest.csv
output/tables/source_refresh_status.csv
```

Known source notes:

- FRED `bill_yield_1mo` can use a valid local cache if a live refresh times out.
- FRED `institutional_mmf_assets` is the discontinued historical `WIMFNS` series ending in 2021-02.
- Current MMF assets and portfolio holdings come from the OFR MMF source.
- Continued live refresh is not part of the default workflow unless the historical snapshot is
  intentionally updated.

## Main Commands

Run the full non-mutating readiness check:

```bash
python -B -m liqsub.cli rebuild-public --check-only
python -B -m liqsub.cli release-public --check-only
```

Run the full rebuild and public release sequence:

```bash
python -B -m liqsub.cli rebuild-public
python -B -m liqsub.cli release-public
```

Useful individual checks:

```bash
python -B -m liqsub.cli validate-config
python -B -m liqsub.cli validate-inputs
python -B -m liqsub.cli check-buycurve
python -B -m liqsub.cli check-tgarefill
python -B -m liqsub.cli validate-output-schemas
python -B -m liqsub.cli validate-public-boundary
python -B -m liqsub.cli validate-rebuild-lineage
python -B -m liqsub.cli status
```

Public bundles are written under `output/bundles/` and exclude private planning folders, local
environment files, raw/cache/interim data, internal generated outputs, hidden system files, and
prior bundles.

## Key Outputs

Stable clean panels:

```text
data/clean/monthly_liquidity_substitution_panel.csv
data/clean/monthly_liquidity_substitution_panel_analysis.csv
data/clean/weekly_liquidity_substitution_panel.csv
```

Public diagnostics:

```text
output/tables/monthly_source_metadata.csv
output/tables/monthly_coverage_qa.csv
output/tables/monthly_sample_windows.csv
output/tables/monthly_readiness_summary.csv
output/tables/evidence_gate_summary.csv
output/tables/weekly_design_readiness.csv
output/tables/weekly_stability_candidates.csv
output/tables/weekly_large_rebuild_cell_summary.csv
output/tables/weekly_large_rebuild_final_review.csv
output/tables/tgarefill_promotion_reconciliation.csv
output/reports/monthly_mvp_report.md
output/reports/weekly_identification_candidate_report.md
output/reports/weekly_large_rebuild_diagnostic_report.md
output/reports/tgarefill_promotion_reconciliation.md
output/reports/evidence_gate_summary.md
```

Release/provenance outputs:

```text
output/manifests/latest.json
output/manifests/latest_successful_rebuild.json
output/manifests/output_tiers.csv
output/manifests/output_tiers.json
output/reports/RELEASE_NOTES.md
```

See `docs/public_artifact_contract.md` for the stable-backend versus diagnostic-output boundary.

## Interpretation Boundary

The backend is useful for:

- building monthly and weekly historical liquidity-plumbing panels;
- auditing source coverage, cache state, and generated artifacts;
- checking whether broad monthly or weekly designs pass conservative readiness gates;
- preserving narrow descriptive diagnostics around large TGA rebuild episodes.
- reconciling the promoted `tgarefill` bill-surprise result with the blocked broad-substitution
  evidence gate.

The backend should not be used to claim:

- a causal bill-supply liquidity-substitution effect;
- a headline deposit, MMF, reserve, or ON RRP mechanism result;
- a uniformly covered 1973-2026 mechanism panel;
- live-current source coverage after the frozen snapshot date.

The promoted `tgarefill` import is deliberately narrower than the blocked broad substitution
design: bill surprises during the selected TGA rebuild sample are associated with MMF Treasury
holdings and ON RRP balances. This is aggregate association evidence, not a causal funding-route
or funding-share estimate. Fund-level MMF allocation imports remain separate descriptive context.

Partially observed terminal periods are preserved in clean panels but marked out of estimation with
`baseline_estimation_use=exclude_from_causal_baseline`; see:

```text
output/tables/monthly_future_row_qa.csv
output/tables/weekly_terminal_period_qa.csv
```

## Public/Private Boundary

The following paths are local-only and must remain ignored or excluded from public bundles:

```text
do/
.env
data/raw/
data/interim/
data/cache/
output/internal/
output/bundles/
```

Optional internal report generation is explicit and writes only under ignored `output/internal/`:

```bash
python -B -m liqsub.cli write-internal-reports
```

Run `validate-public-boundary` before sharing a bundle.

## Scope Boundaries

This repository does not duplicate Treasury auction normalization from the upstream issuance
project, does not regenerate the upstream weekly TGA-refill event source from first principles,
and does not provide a live dashboard. Its purpose is to preserve a clean historical backend
snapshot with transparent inputs, generated diagnostics, and public release checks.
