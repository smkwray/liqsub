# Public Artifact Contract

This document separates stable backend artifacts from diagnostic outputs. Generated files under
`output/` and `data/clean/` remain reproducible outputs, not source files. Private prose and
planning artifacts must be generated only with `liqsub.cli write-internal-reports`; those files are
written under ignored `output/internal/` and are excluded from the public manifest tier index.

## Stable Backend Outputs

These artifacts are the public backend surface for downstream code:

- `data/clean/monthly_liquidity_substitution_panel.csv`
- `data/clean/monthly_liquidity_substitution_panel_analysis.csv`
- `data/clean/weekly_liquidity_substitution_panel.csv`
- `output/tables/backend_input_inventory.csv`
- `output/tables/source_cache_manifest.csv`
- `output/tables/source_refresh_status.csv`
- `output/tables/monthly_source_metadata.csv`
- `output/tables/monthly_coverage_qa.csv`
- `output/tables/monthly_future_row_qa.csv`
- `output/tables/monthly_sample_windows.csv`
- `output/tables/monthly_candidate_table.csv`
- `output/tables/monthly_readiness_summary.csv`
- `output/tables/public_output_aliases.csv`
- `output/tables/evidence_gate_summary.csv`
- `output/tables/weekly_upstream_tgarefill_qa.csv`
- `output/tables/weekly_terminal_period_qa.csv`
- `output/tables/weekly_timing_alignment_qa.csv`
- `output/tables/weekly_event_candidates_clean.csv`
- `output/tables/weekly_event_exclusion_log.csv`
- `output/tables/weekly_design_readiness.csv`
- `output/tables/weekly_outcome_readiness.csv`
- `output/tables/weekly_stability_candidates.csv`
- `output/tables/weekly_large_rebuild_event_roster.csv`
- `output/tables/weekly_large_rebuild_cell_summary.csv`
- `output/tables/weekly_large_rebuild_blocker_summary.csv`
- `output/tables/weekly_large_rebuild_event_sign_stability.csv`
- `output/tables/weekly_large_rebuild_abnormal_changes.csv`
- `output/tables/weekly_large_rebuild_match_quality.csv`
- `output/tables/weekly_large_rebuild_randomization_inference.csv`
- `output/tables/weekly_large_rebuild_final_review.csv`
- `output/reports/monthly_candidate_review_report.md`
- `output/reports/evidence_gate_summary.md`
- `output/reports/weekly_large_rebuild_diagnostic_report.md`
- `output/manifests/output_tiers.csv`
- `output/manifests/output_tiers.json`
- `output/manifests/latest.json`

Legacy compatibility aliases are still emitted for now:

- `output/tables/monthly_claim_readiness.csv`
- `output/tables/weekly_claim_readiness.csv`
- `output/tables/weekly_outcome_claim_readiness.csv`
- `output/tables/weekly_stable_claim_candidates.csv`
- `output/reports/monthly_candidate_narrative_report.md`
- `output/reports/weekly_large_rebuild_diagnostic_writeup.md`

Schema compatibility for public CSV outputs is checked by:

```bash
python -B -m liqsub.cli validate-output-schemas
```

The validator fails if a public CSV has no header, has unnamed columns, misses required columns,
appears in `output/tables/` without a schema contract, or violates basic value contracts such as
unparseable dates, negative count fields, out-of-range share fields, invalid correlations, or blank
required status fields. The release path also enforces a stable artifact contract across the clean
monthly/weekly panels, stable public tables, and stable public reports. Missing stable artifacts,
duplicate panel dates, unsorted panel grains, too-few panel rows, alias/canonical CSV drift, and
missing panel contract fields block `release-public`.

Terminal-period completeness is explicit rather than silent. Stable clean panels may preserve
partially observed terminal month/week rows for provenance, but generated analysis fields mark them
with `terminal_completeness_status=terminal_incomplete` and
`baseline_estimation_use=exclude_from_causal_baseline`. Monthly terminal exclusions are reported in
`output/tables/monthly_future_row_qa.csv`; weekly terminal exclusions are reported in
`output/tables/weekly_terminal_period_qa.csv`.

Backend source-input readiness is checked by:

```bash
python -B -m liqsub.cli validate-inputs
python -B -m liqsub.cli write-input-inventory
python -B -m liqsub.cli write-source-cache-manifest
python -B -m liqsub.cli write-source-refresh-status
```

The inventory records required and optional raw/manual backend inputs with existence, row counts,
raw date ranges when available, valid-observation ranges when profileable, byte sizes, modification
time, cache age, observation age, freshness thresholds, freshness status, and hashes. Freshness
thresholds are configured in `config/project.yaml` under `source_freshness`, with separate mtime
and observation-recency windows by source family. Required missing inputs and required inputs with
zero valid observations block `release-public`; stale mtime or source-observation recency is
surfaced as a release warning. The source-cache manifest is a sanitized public provenance table over
the same cache set; it includes source family, source id, acquisition mode, official source URL where
configured, cache hash, cache status, and valid-observation range without local absolute paths.
The source-refresh status table summarizes each cache's latest fetch status, freshness state, and
required action; it distinguishes clean caches from live-refresh failures where an existing cache
remains valid.
The upstream `buycurve` and weekly `tgarefill` gates also enforce source-specific contracts before
rebuilds: required fields, parseable/sorted dates, duplicate-grain checks, numeric/nonnegative
amount checks, share bounds, and event date ordering where applicable.

The public/private boundary is checked by:

```bash
python -B -m liqsub.cli validate-public-boundary
python -B -m liqsub.cli validate-rebuild-lineage
```

That command fails if ignored private paths are missing from `.gitignore`, if `output/internal/`,
hidden files, or `internal_*` internals appear in the public tier index, or if `internal_*` files are
written directly into public output directories. The lineage check fails if
`output/manifests/latest_successful_rebuild.json` is missing, if the current backend input hashes
do not match the last successful rebuild, if config/source-contract hashes have changed since that
rebuild, or if stable generated outputs have changed since that rebuild.

Public backend bundles are written by:

```bash
python -B -m liqsub.cli write-public-bundle
```

The bundle includes `README.md`, `pyproject.toml`, `.gitignore`, `config/`, `docs/`, `src/`,
`tests/`, `data/clean/`, `data/manual/`, and public `output/` files. It excludes `do/`, `.env`,
`data/raw/`, `data/interim/`, `data/cache/`, `output/internal/`, `output/bundles/`, hidden system
files, test/build caches, historical `run_manifest_*.json` files, and public-output `internal_*`
files. Each zip contains
`PUBLIC_BUNDLE_MANIFEST.json` with file hashes and boundary-check metadata.

The full release sequence is:

```bash
python -B -m liqsub.cli release-public --check-only
python -B -m liqsub.cli release-public
python -B -m liqsub.cli rebuild-public --check-only
python -B -m liqsub.cli rebuild-public
```

`--check-only` validates release readiness without creating a zip or mutating release artifacts. The
`rebuild-public --check-only` command also runs the buycurve and tgarefill upstream export validators
before the release readiness check. `release-public` requires a successful rebuild lineage stamp, so
run `rebuild-public` after changing raw/manual inputs or generated stable outputs. The full command validates config, validates required backend inputs, writes
`output/tables/backend_input_inventory.csv`, writes source cache and refresh-status manifests,
validates public output schemas, validates the public
artifact contract, validates the public boundary, writes `output/reports/RELEASE_NOTES.md`,
refreshes `output_tiers`, refreshes the run manifest, verifies manifest integrity, writes a public
bundle, and smoke-tests the resulting zip by extracting it into a temporary directory. The smoke test
verifies zip member safety, bundle manifest file hashes, bundled public table schemas, and the
public/private boundary. A bundle can be checked independently with:

```bash
python -B -m liqsub.cli smoke-public-bundle output/bundles/<bundle>.zip
```

`rebuild-public` is the local full-pipeline wrapper. It does not fetch live network sources; it
validates local source caches, rebuilds monthly outputs, reruns monthly analysis, validates the
weekly `tgarefill` cache, rebuilds weekly outputs, reruns weekly analysis, and then calls
`release-public`.

## Diagnostic Outputs

These artifacts are public diagnostics. They are useful for backend review but are not stable API
unless promoted into the stable list above:

- monthly descriptive, regime, correlation, regression, pretrend, event-study, and local-projection
  tables;
- monthly filtered-candidate and manual-review tables;
- weekly LP, event-study, leave-one-event-out, bootstrap, placebo, filter-sensitivity, and
  large-rebuild calendar-validation tables;
- generated SVG event-time figures and narrative diagnostic reports.

Downstream code should read diagnostic outputs only with version checks or explicit schema checks.

## Internal Outputs

Internal outputs are opt-in:

```bash
python -B -m liqsub.cli write-internal-reports
```

They are written under `output/internal/` and must not be treated as public package artifacts.
