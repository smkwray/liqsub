from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from liqsub.config import load_yaml, source_contract_paths, source_freshness_policy, validate_project_config
from liqsub.fred import FRED_SERIES
from liqsub.paths import ensure_dir, relative_to_root
from liqsub.schemas import (
    MONTHLY_PANEL_REQUIRED_FIELDS,
    PUBLIC_OUTPUT_ALIASES,
    PUBLIC_OUTPUT_REQUIRED_COLUMNS,
    validate_public_output_schemas,
)


INPUT_PATHS = [
    "data/raw/buycurve/monthly_issuance_maturity_panel.csv",
    "data/raw/tgarefill/master_weekly_panel.csv",
    "data/raw/tgarefill/event_candidates.csv",
    "data/raw/tgarefill/auction_shock_lp.csv",
    "data/raw/tgarefill/canonical_bill_surprise_shocks.csv",
    "data/raw/tgarefill/promotion_robustness_summary.csv",
    "data/raw/tgarefill/mmfalloc_downstream_summary.csv",
    "data/raw/tgarefill/mmfalloc_source_gates.csv",
    "data/raw/tgarefill/mmfalloc_baseline.csv",
    "data/raw/fiscaldata/dts_operating_cash_balance.csv",
    "data/raw/ofr/mmf.json",
    "data/manual/event_calendar_context.csv",
    "data/manual/weekly_large_rebuild_calendar_context.csv",
]

KEY_OUTPUTS = [
    "data/clean/monthly_liquidity_substitution_panel.csv",
    "data/clean/monthly_liquidity_substitution_panel_analysis.csv",
    "data/clean/weekly_liquidity_substitution_panel.csv",
    "output/tables/monthly_future_row_qa.csv",
    "output/tables/monthly_readiness_summary.csv",
    "output/tables/weekly_design_readiness.csv",
    "output/tables/weekly_stability_candidates.csv",
    "output/tables/tgarefill_promotion_reconciliation.csv",
    "output/tables/public_output_aliases.csv",
    "output/tables/backend_input_inventory.csv",
    "output/tables/evidence_gate_summary.csv",
    "output/reports/monthly_mvp_report.md",
    "output/reports/monthly_candidate_review_report.md",
    "output/reports/weekly_identification_candidate_report.md",
    "output/reports/weekly_large_rebuild_diagnostic_report.md",
    "output/reports/evidence_gate_summary.md",
    "output/reports/tgarefill_promotion_reconciliation.md",
]

STABLE_PUBLIC_TABLES = [
    "output/tables/backend_input_inventory.csv",
    "output/tables/source_cache_manifest.csv",
    "output/tables/source_refresh_status.csv",
    "output/tables/monthly_source_metadata.csv",
    "output/tables/monthly_coverage_qa.csv",
    "output/tables/monthly_future_row_qa.csv",
    "output/tables/monthly_sample_windows.csv",
    "output/tables/monthly_candidate_table.csv",
    "output/tables/monthly_readiness_summary.csv",
    "output/tables/public_output_aliases.csv",
    "output/tables/evidence_gate_summary.csv",
    "output/tables/weekly_upstream_tgarefill_qa.csv",
    "output/tables/weekly_terminal_period_qa.csv",
    "output/tables/weekly_timing_alignment_qa.csv",
    "output/tables/weekly_event_candidates_clean.csv",
    "output/tables/weekly_event_exclusion_log.csv",
    "output/tables/weekly_design_readiness.csv",
    "output/tables/weekly_outcome_readiness.csv",
    "output/tables/weekly_stability_candidates.csv",
    "output/tables/weekly_large_rebuild_event_roster.csv",
    "output/tables/weekly_large_rebuild_cell_summary.csv",
    "output/tables/weekly_large_rebuild_blocker_summary.csv",
    "output/tables/weekly_large_rebuild_event_sign_stability.csv",
    "output/tables/weekly_large_rebuild_abnormal_changes.csv",
    "output/tables/weekly_large_rebuild_match_quality.csv",
    "output/tables/weekly_large_rebuild_randomization_inference.csv",
    "output/tables/weekly_large_rebuild_final_review.csv",
    "output/tables/tgarefill_promotion_reconciliation.csv",
]

STABLE_PUBLIC_REPORTS = [
    "output/reports/monthly_candidate_review_report.md",
    "output/reports/evidence_gate_summary.md",
    "output/reports/weekly_large_rebuild_diagnostic_report.md",
    "output/reports/tgarefill_promotion_reconciliation.md",
]

REBUILD_LINEAGE_PATH = "output/manifests/latest_successful_rebuild.json"

STABLE_CLEAN_PANELS = {
    "data/clean/monthly_liquidity_substitution_panel.csv": {
        "date_column": "month",
        "min_rows": 120,
        "required_columns": MONTHLY_PANEL_REQUIRED_FIELDS,
    },
    "data/clean/monthly_liquidity_substitution_panel_analysis.csv": {
        "date_column": "month",
        "min_rows": 120,
        "required_columns": [
            *MONTHLY_PANEL_REQUIRED_FIELDS,
            "d_deposits",
            "d_on_rrp",
            "d_reserves",
            "d_tga",
            "isolated_large_positive_bill_shock",
        ],
    },
    "data/clean/weekly_liquidity_substitution_panel.csv": {
        "date_column": "week",
        "min_rows": 250,
        "required_columns": [
            "week",
            "tga",
            "reserves",
            "on_rrp",
            "deposits",
            "gross_bill_settlement_100b",
            "coupon_settlement_100b",
            "bill_share",
            "d_tga",
            "d_reserves",
            "d_on_rrp",
            "d_deposits",
            "debt_limit_window",
            "tax_week",
            "on_rrp_regime_pre",
            "terminal_completeness_status",
            "baseline_estimation_use",
        ],
    },
}


OUTPUT_TIER_RULES = [
    ("qa", ["qa", "metadata", "sample_windows", "sample_recommendations", "aliases"]),
    ("candidate_review", ["claim", "stable_claim"]),
    ("review", ["readiness", "stability_candidates"]),
    ("review", ["shortlist", "narrative", "external_calendar", "calendar_context"]),
    ("diagnostics", ["regression", "correlation", "diagnostic", "lp_", "event_study", "pretrend", "bootstrap", "placebo", "leave_one"]),
    ("reports", ["report"]),
    ("manifests", ["manifest"]),
]


PUBLIC_BUNDLE_ROOTS = [
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "config",
    "docs",
    "src",
    "tests",
    "data/clean",
    "data/manual",
    "output",
]

PUBLIC_BUNDLE_EXCLUDED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
    ".venv",
    "venv",
    "env",
}

INPUT_INVENTORY_COLUMNS = [
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
]


def artifact_status(root: Path) -> dict[str, Any]:
    monthly = _csv_profile(root / "data/clean/monthly_liquidity_substitution_panel.csv", date_col="month")
    weekly = _csv_profile(root / "data/clean/weekly_liquidity_substitution_panel.csv", date_col="week")
    monthly_readiness = _csv_counts(root / "output/tables/monthly_readiness_summary.csv", "readiness_status")
    weekly_readiness = _csv_counts(root / "output/tables/weekly_design_readiness.csv", "readiness_status")
    stable_weekly = _csv_counts(root / "output/tables/weekly_stability_candidates.csv", "status")
    return {
        "status": "ok",
        "monthly_panel": monthly,
        "weekly_panel": weekly,
        "monthly_readiness_summary": monthly_readiness,
        "weekly_design_readiness": weekly_readiness,
        "weekly_stability_candidates": stable_weekly,
        "monthly_claim_readiness": _legacy_counts(monthly_readiness, root / "output/tables/monthly_claim_readiness.csv"),
        "weekly_claim_readiness": _legacy_counts(weekly_readiness, root / "output/tables/weekly_claim_readiness.csv"),
        "weekly_stable_claim_candidates": _legacy_counts(
            stable_weekly,
            root / "output/tables/weekly_stable_claim_candidates.csv",
            column="status",
        ),
        "latest_manifest": _latest_manifest(root),
        "public_output_aliases": _csv_profile(root / "output/tables/public_output_aliases.csv"),
        "output_tiers": _output_tier_counts(root),
    }


def release_public_backend(
    root: Path,
    *,
    destination: Path | None = None,
    check_only: bool = False,
    require_lineage: bool = True,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []

    config_report = validate_project_config(root)
    steps.append({"step": "validate_config", "status": config_report.status, "errors": config_report.errors})
    if config_report.errors:
        return _blocked_release(steps, config_report.errors)

    input_report = validate_backend_inputs(root)
    steps.append(
        {
            "step": "validate_backend_inputs",
            "status": input_report["status"],
            "errors": input_report["errors"],
            "warnings": input_report["warnings"],
            "checked_files": input_report["checked_files"],
            "missing_required": input_report["missing_required"],
            "stale_inputs": input_report["stale_inputs"],
        }
    )
    if input_report["errors"]:
        return _blocked_release(steps, input_report["errors"])

    if not check_only:
        inventory_result = write_backend_input_inventory(root)
        steps.append({"step": "write_backend_input_inventory", **inventory_result})
        cache_manifest_result = write_source_cache_manifest(root)
        steps.append({"step": "write_source_cache_manifest", **cache_manifest_result})
        refresh_status_result = write_source_refresh_status(root)
        steps.append({"step": "write_source_refresh_status", **refresh_status_result})
        write_public_output_aliases(root)

    schema_report = validate_public_output_schemas(root)
    steps.append(
        {
            "step": "validate_output_schemas",
            "status": schema_report.status,
            "checked_files": schema_report.checked_files,
            "errors": schema_report.errors,
            "warnings": schema_report.warnings,
        }
    )
    if schema_report.errors:
        return _blocked_release(steps, schema_report.errors)

    contract_report = validate_public_artifact_contract(root)
    steps.append({"step": "validate_public_artifact_contract", **contract_report})
    if contract_report["errors"]:
        return _blocked_release(steps, contract_report["errors"])

    if require_lineage:
        lineage_report = validate_rebuild_lineage(root)
        steps.append({"step": "validate_rebuild_lineage", **lineage_report})
        if lineage_report["errors"]:
            return _blocked_release(steps, lineage_report["errors"])

    boundary_report = validate_public_boundary(root)
    steps.append({"step": "validate_public_boundary", **boundary_report})
    if boundary_report["errors"]:
        return _blocked_release(steps, boundary_report["errors"])

    if check_only:
        files = public_bundle_file_list(root)
        steps.append(
            {
                "step": "plan_public_bundle",
                "status": "ok",
                "file_count": len(files),
            }
        )
        return {
            "status": "ok",
            "check_only": True,
            "file_count": len(files),
            "steps": steps,
            "warnings": _release_warnings(steps),
        }

    notes_result = write_release_notes(root)
    steps.append({"step": "write_release_notes", **notes_result})

    tier_result = write_output_tiers(root)
    steps.append({"step": "write_output_tiers", **tier_result})

    manifest_result = write_run_manifest(root)
    steps.append({"step": "write_manifest", **manifest_result})

    manifest_check = validate_manifest_integrity(root)
    steps.append({"step": "validate_manifest_integrity", **manifest_check})
    if manifest_check["errors"]:
        return _blocked_release(steps, manifest_check["errors"])

    bundle_result = write_public_bundle(root, destination=destination)
    steps.append({"step": "write_public_bundle", **bundle_result})
    if bundle_result["status"] != "ok":
        return _blocked_release(steps, bundle_result.get("errors", []))

    bundle_path = Path(str(bundle_result["bundle"]))
    if not bundle_path.is_absolute():
        bundle_path = root / bundle_path
    smoke_result = smoke_test_public_bundle(bundle_path)
    steps.append({"step": "smoke_test_public_bundle", **smoke_result})
    if smoke_result["status"] != "ok":
        return _blocked_release(steps, smoke_result.get("errors", []))

    return {
        "status": "ok",
        "check_only": False,
        "bundle": bundle_result["bundle"],
        "bytes": bundle_result["bytes"],
        "file_count": bundle_result["file_count"],
        "release_notes": notes_result["path"],
        "steps": steps,
        "warnings": _release_warnings(steps),
    }


def _blocked_release(steps: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    return {
        "status": "blocked",
        "errors": errors,
        "steps": steps,
        "warnings": _release_warnings(steps),
    }


def _release_warnings(steps: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for step in steps:
        for warning in step.get("warnings", []) or []:
            warnings.append(str(warning))
    return sorted(dict.fromkeys(warnings))


def validate_public_artifact_contract(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checked = 0

    for rel, contract in STABLE_CLEAN_PANELS.items():
        checked += 1
        errors.extend(_validate_clean_panel(root, rel, contract))

    for rel in STABLE_PUBLIC_TABLES:
        checked += 1
        path = root / rel
        if not path.exists():
            errors.append(f"{rel}: missing stable public table")
            continue
        if rel not in PUBLIC_OUTPUT_REQUIRED_COLUMNS:
            errors.append(f"{rel}: stable public table has no schema contract")
            continue
        profile = _csv_profile(path)
        if profile.get("rows", 0) == 0 and rel not in _allowed_empty_stable_tables():
            errors.append(f"{rel}: stable public table is empty")

    for rel in STABLE_PUBLIC_REPORTS:
        checked += 1
        path = root / rel
        if not path.exists():
            errors.append(f"{rel}: missing stable public report")
            continue
        if path.stat().st_size == 0:
            errors.append(f"{rel}: stable public report is empty")

    errors.extend(_validate_alias_equivalence(root))
    return {
        "status": "ok" if not errors else "failed",
        "errors": sorted(dict.fromkeys(errors)),
        "warnings": warnings,
        "checked_files": checked,
    }


def _allowed_empty_stable_tables() -> set[str]:
    return {
        "output/tables/monthly_candidate_table.csv",
        "output/tables/monthly_future_row_qa.csv",
    }


def _validate_clean_panel(root: Path, rel: str, contract: dict[str, Any]) -> list[str]:
    path = root / rel
    errors: list[str] = []
    if not path.exists():
        return [f"{rel}: missing stable clean panel"]
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return [f"{rel}: stable clean panel is empty"]
    required_columns = list(contract["required_columns"])
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        errors.append(f"{rel}: missing required columns {','.join(missing)}")
    min_rows = int(contract.get("min_rows", 1))
    if len(table) < min_rows:
        errors.append(f"{rel}: expected at least {min_rows} rows, found {len(table)}")
    date_column = str(contract["date_column"])
    if date_column not in table.columns:
        errors.append(f"{rel}: missing date column `{date_column}`")
        return errors
    dates = pd.to_datetime(table[date_column], errors="coerce")
    if dates.isna().any():
        errors.append(f"{rel}: column `{date_column}` contains unparsable dates")
    if dates.duplicated().any():
        errors.append(f"{rel}: column `{date_column}` contains duplicate dates")
    if not dates.is_monotonic_increasing:
        errors.append(f"{rel}: column `{date_column}` is not sorted ascending")
    text_columns = {
        "on_rrp_regime",
        "on_rrp_regime_pre",
        "terminal_completeness_status",
        "baseline_estimation_use",
    }
    numeric_candidates = [
        column for column in required_columns if column != date_column and column not in text_columns
    ]
    for column in numeric_candidates:
        if column not in table.columns:
            continue
        values = table[column].dropna()
        if not values.empty and pd.to_numeric(values, errors="coerce").isna().any():
            errors.append(f"{rel}: column `{column}` contains non-numeric values")
    return errors


def _validate_alias_equivalence(root: Path) -> list[str]:
    errors: list[str] = []
    for canonical, compatibility in sorted(PUBLIC_OUTPUT_ALIASES.items()):
        canonical_path = root / canonical
        compatibility_path = root / compatibility
        if not canonical_path.exists() or not compatibility_path.exists():
            continue
        if _sha256(canonical_path) != _sha256(compatibility_path):
            errors.append(f"{compatibility}: compatibility alias differs from canonical {canonical}")
    return errors


def write_rebuild_lineage(root: Path) -> dict[str, Any]:
    manifest_dir = ensure_dir(root / "output" / "manifests")
    payload = {
        "created_at_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "configuration": [_file_record(root, rel) for rel in _lineage_configuration_paths(root)],
        "inputs": backend_input_inventory(root).to_dict(orient="records"),
        "outputs": [_file_record(root, rel) for rel in _rebuild_lineage_output_paths()],
        "configuration_fingerprint": _fingerprint_records(
            [_file_record(root, rel) for rel in _lineage_configuration_paths(root)]
        ),
        "input_fingerprint": _fingerprint_records(backend_input_inventory(root).to_dict(orient="records")),
        "output_fingerprint": _fingerprint_records(
            [_file_record(root, rel) for rel in _rebuild_lineage_output_paths()]
        ),
    }
    path = manifest_dir / Path(REBUILD_LINEAGE_PATH).name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": "ok",
        "path": relative_to_root(root, path),
        "configuration": int(len(payload["configuration"])),
        "inputs": int(len(payload["inputs"])),
        "outputs": int(len(payload["outputs"])),
    }


def validate_rebuild_lineage(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    path = root / REBUILD_LINEAGE_PATH
    if not path.exists():
        return {
            "status": "failed",
            "errors": [f"{REBUILD_LINEAGE_PATH}: missing rebuild lineage; run rebuild-public before release-public"],
            "warnings": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "errors": [f"{REBUILD_LINEAGE_PATH}: invalid JSON: {exc}"],
            "warnings": [],
        }

    recorded_inputs = _records_by_path(payload.get("inputs"))
    current_inputs = _records_by_path(backend_input_inventory(root).to_dict(orient="records"))
    errors.extend(_compare_lineage_records(recorded_inputs, current_inputs, label="input"))

    recorded_config = _records_by_path(payload.get("configuration"))
    current_config = _records_by_path([_file_record(root, rel) for rel in _lineage_configuration_paths(root)])
    errors.extend(_compare_lineage_records(recorded_config, current_config, label="configuration"))

    recorded_outputs = _records_by_path(payload.get("outputs"))
    current_outputs = _records_by_path([_file_record(root, rel) for rel in _rebuild_lineage_output_paths()])
    errors.extend(_compare_lineage_records(recorded_outputs, current_outputs, label="output"))

    recorded_config_fingerprint = str(payload.get("configuration_fingerprint", ""))
    current_config_fingerprint = _fingerprint_records(list(current_config.values()))
    if recorded_config_fingerprint and recorded_config_fingerprint != current_config_fingerprint:
        errors.append(f"{REBUILD_LINEAGE_PATH}: configuration fingerprint mismatch")

    recorded_input_fingerprint = str(payload.get("input_fingerprint", ""))
    current_input_fingerprint = _fingerprint_records(list(current_inputs.values()))
    if recorded_input_fingerprint and recorded_input_fingerprint != current_input_fingerprint:
        errors.append(f"{REBUILD_LINEAGE_PATH}: input fingerprint mismatch")

    recorded_output_fingerprint = str(payload.get("output_fingerprint", ""))
    current_output_fingerprint = _fingerprint_records(list(current_outputs.values()))
    if recorded_output_fingerprint and recorded_output_fingerprint != current_output_fingerprint:
        errors.append(f"{REBUILD_LINEAGE_PATH}: output fingerprint mismatch")

    return {
        "status": "ok" if not errors else "failed",
        "errors": sorted(dict.fromkeys(errors)),
        "warnings": [],
        "configuration": len(current_config),
        "inputs": len(current_inputs),
        "outputs": len(current_outputs),
    }


def _records_by_path(records: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("path") or record.get("input_path"), str):
            key = str(record.get("path") or record.get("input_path"))
            out[key] = record
    return out


def _compare_lineage_records(
    recorded: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    for rel in sorted(set(current) - set(recorded)):
        errors.append(f"{REBUILD_LINEAGE_PATH}: missing recorded {label} lineage for {rel}")
    for rel in sorted(set(recorded) - set(current)):
        errors.append(f"{REBUILD_LINEAGE_PATH}: recorded {label} no longer exists in contract: {rel}")
    for rel in sorted(set(recorded) & set(current)):
        old = recorded[rel]
        new = current[rel]
        if bool(old.get("exists", False)) != bool(new.get("exists", False)):
            errors.append(f"{REBUILD_LINEAGE_PATH}: {label} existence changed for {rel}")
            continue
        old_hash = str(old.get("sha256", ""))
        new_hash = str(new.get("sha256", ""))
        if old_hash != new_hash:
            errors.append(f"{REBUILD_LINEAGE_PATH}: {label} hash changed for {rel}")
    return errors


def _fingerprint_records(records: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "path": str(record.get("path") or record.get("input_path")),
            "exists": bool(record.get("exists", False)),
            "sha256": str(record.get("sha256", "")),
        }
        for record in records
    ]
    normalized.sort(key=lambda record: record["path"])
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rebuild_lineage_output_paths() -> list[str]:
    time_derived_release_tables = {
        "output/tables/backend_input_inventory.csv",
        "output/tables/source_cache_manifest.csv",
        "output/tables/source_refresh_status.csv",
    }
    return sorted(
        dict.fromkeys(
            [
                *STABLE_CLEAN_PANELS.keys(),
                *(path for path in STABLE_PUBLIC_TABLES if path not in time_derived_release_tables),
                *STABLE_PUBLIC_REPORTS,
            ]
        )
    )


def _lineage_configuration_paths(root: Path) -> list[str]:
    paths = [
        "config/project.yaml",
        "config/panel_schema.yaml",
        "config/upstream_buycurve_outputs.yaml",
        "config/upstream_tgarefill_outputs.yaml",
    ]
    source_contract_dir = root / "config" / "source_contracts"
    if source_contract_dir.exists():
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted(source_contract_dir.glob("*.yaml"))
            if path.is_file()
        )
    return sorted(dict.fromkeys(paths))


def write_public_bundle(root: Path, *, destination: Path | None = None) -> dict[str, Any]:
    boundary_report = validate_public_boundary(root)
    if boundary_report["status"] != "ok":
        return {
            "status": "blocked",
            "errors": boundary_report["errors"],
            "warnings": boundary_report["warnings"],
        }

    files = public_bundle_file_list(root)
    bundle_dir = ensure_dir(destination or root / "output" / "bundles")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bundle_path = bundle_dir / f"liqsub_public_backend_{timestamp}.zip"
    manifest = {
        "created_at_utc": timestamp,
        "file_count": len(files),
        "boundary": boundary_report,
        "files": [_file_record(root, rel) for rel in files],
    }
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in files:
            archive.write(root / rel, arcname=rel)
        archive.writestr("PUBLIC_BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))

    stat = bundle_path.stat()
    return {
        "status": "ok",
        "bundle": relative_to_root(root, bundle_path) if bundle_path.is_relative_to(root) else str(bundle_path),
        "bytes": int(stat.st_size),
        "file_count": int(len(files)),
        "warnings": boundary_report["warnings"],
    }


def validate_backend_inputs(root: Path) -> dict[str, Any]:
    inventory = backend_input_inventory(root)
    missing_required = inventory.loc[
        inventory["required"].astype(bool) & ~inventory["exists"].astype(bool),
        "input_path",
    ].astype(str).tolist()
    errors = [f"missing required backend input: {path}" for path in missing_required]
    empty_required = inventory.loc[
        inventory["required"].astype(bool)
        & inventory["exists"].astype(bool)
        & (pd.to_numeric(inventory["valid_observation_rows"], errors="coerce").fillna(1) == 0),
        "input_path",
    ].astype(str).tolist()
    errors.extend(f"required backend input has zero valid observations: {path}" for path in empty_required)
    warnings = inventory.loc[
        ~inventory["required"].astype(bool) & ~inventory["exists"].astype(bool),
        "input_path",
    ].astype(str).tolist()
    warnings = [f"missing optional backend input: {path}" for path in warnings]
    observation_stale_mask = inventory["observation_freshness_status"].isin(["stale", "stale_missing_observation"])
    observation_policy_mask = inventory["source_family"].isin(["fred", "fiscaldata", "ofr"])
    stale_inputs = inventory.loc[
        inventory["freshness_status"].isin(["stale", "stale_missing_mtime"])
        | (observation_policy_mask & observation_stale_mask),
        "input_path",
    ].astype(str).tolist()
    warnings.extend(f"stale backend input cache: {path}" for path in stale_inputs)
    return {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "checked_files": int(len(inventory)),
        "missing_required": int(len(missing_required)),
        "stale_inputs": int(len(stale_inputs)),
    }


def write_backend_input_inventory(root: Path) -> dict[str, Any]:
    table_dir = ensure_dir(root / "output" / "tables")
    path = table_dir / "backend_input_inventory.csv"
    inventory = backend_input_inventory(root)
    inventory.to_csv(path, index=False)
    return {
        "status": "ok",
        "rows": int(len(inventory)),
        "path": relative_to_root(root, path),
    }


def write_source_cache_manifest(root: Path) -> dict[str, Any]:
    table_dir = ensure_dir(root / "output" / "tables")
    path = table_dir / "source_cache_manifest.csv"
    table = source_cache_manifest(root)
    table.to_csv(path, index=False)
    return {
        "status": "ok",
        "rows": int(len(table)),
        "path": relative_to_root(root, path),
    }


def write_source_refresh_status(root: Path) -> dict[str, Any]:
    table_dir = ensure_dir(root / "output" / "tables")
    path = table_dir / "source_refresh_status.csv"
    table = source_refresh_status(root)
    table.to_csv(path, index=False)
    return {
        "status": "ok",
        "rows": int(len(table)),
        "path": relative_to_root(root, path),
    }


def source_refresh_status(root: Path) -> pd.DataFrame:
    inventory = backend_input_inventory(root)
    latest_fetch = _raw_refresh_status_by_input(root)
    rows: list[dict[str, Any]] = []
    for record in inventory.to_dict(orient="records"):
        input_path = str(record.get("input_path", ""))
        fetch = latest_fetch.get(input_path, {})
        refresh_status, required_action = _source_refresh_decision(record, fetch)
        rows.append(
            {
                "input_path": input_path,
                "source_family": record.get("source_family", ""),
                "required": record.get("required", ""),
                "exists": record.get("exists", ""),
                "cache_status": record.get("status", ""),
                "mtime_freshness_status": record.get("freshness_status", ""),
                "observation_freshness_status": record.get("observation_freshness_status", ""),
                "valid_observation_rows": record.get("valid_observation_rows", ""),
                "last_valid_observation": record.get("last_valid_observation", ""),
                "latest_fetch_status": fetch.get("latest_fetch_status", ""),
                "latest_fetch_rows": fetch.get("latest_fetch_rows", ""),
                "latest_fetch_last_observation": fetch.get("latest_fetch_last_observation", ""),
                "refresh_status": refresh_status,
                "required_action": required_action,
            }
        )
    columns = [
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
    ]
    return pd.DataFrame(rows, columns=columns)


def source_cache_manifest(root: Path) -> pd.DataFrame:
    inventory = backend_input_inventory(root)
    contracts = _source_contract_index(root)
    rows: list[dict[str, Any]] = []
    for record in inventory.to_dict(orient="records"):
        source_family = str(record.get("source_family", ""))
        contract = _contract_for_inventory_record(contracts, record)
        access = contract.get("access", {}) if isinstance(contract, dict) else {}
        rows.append(
            {
                "input_path": record.get("input_path", ""),
                "source_family": source_family,
                "source_id": contract.get("source_id", source_family) if isinstance(contract, dict) else source_family,
                "source_name": contract.get("source_name", source_family) if isinstance(contract, dict) else source_family,
                "acquisition_mode": access.get("mode", "local_or_manual") if isinstance(access, dict) else "local_or_manual",
                "source_url": access.get("source_url", "") if isinstance(access, dict) else "",
                "required": record.get("required", ""),
                "cache_status": record.get("status", ""),
                "cache_sha256": record.get("sha256", ""),
                "cache_rows": record.get("rows", ""),
                "valid_observation_rows": record.get("valid_observation_rows", ""),
                "first_valid_observation": record.get("first_valid_observation", ""),
                "last_valid_observation": record.get("last_valid_observation", ""),
                "mtime_freshness_status": record.get("freshness_status", ""),
                "observation_freshness_status": record.get("observation_freshness_status", ""),
            }
        )
    columns = [
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
    ]
    return pd.DataFrame(rows, columns=columns)


def backend_input_inventory(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rel_path in _backend_input_specs(root):
        record = _file_record(root, str(rel_path["input_path"]))
        freshness = _freshness_record(root / str(rel_path["input_path"]), int(rel_path["freshness_days"]))
        observation = _observation_freshness_record(
            root / str(rel_path["input_path"]),
            rel_path["observation_freshness_days"],
        )
        rows.append(
            {
                "input_path": rel_path["input_path"],
                "source_family": rel_path["source_family"],
                "required": bool(rel_path["required"]),
                "exists": bool(record.get("exists", False)),
                "status": "ok"
                if record.get("exists", False)
                else ("missing_required" if rel_path["required"] else "missing_optional"),
                "bytes": record.get("bytes", ""),
                "rows": record.get("rows", ""),
                "mtime_utc": freshness["mtime_utc"],
                "age_days": freshness["age_days"],
                "freshness_days": freshness["freshness_days"],
                "freshness_status": freshness["freshness_status"],
                "date_column": record.get("date_column", ""),
                "first_date": record.get("first_date", ""),
                "last_date": record.get("last_date", ""),
                "valid_observation_rows": observation["valid_observation_rows"],
                "first_valid_observation": observation["first_valid_observation"],
                "last_valid_observation": observation["last_valid_observation"],
                "observation_age_days": observation["observation_age_days"],
                "observation_freshness_days": observation["observation_freshness_days"],
                "observation_freshness_status": observation["observation_freshness_status"],
                "sha256": record.get("sha256", ""),
            }
        )
    return pd.DataFrame(rows, columns=INPUT_INVENTORY_COLUMNS)


def _backend_input_specs(root: Path | None = None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for path in INPUT_PATHS:
        source_family = _source_family_for_input(path)
        policy = source_freshness_policy(root, source_family)
        specs.append(
            {
                "input_path": path,
                "source_family": source_family,
                "required": True,
                "freshness_days": int(policy["mtime_days"] or 90),
                "observation_freshness_days": policy["observation_days"],
            }
        )
    for name, meta in sorted(FRED_SERIES.items()):
        series_id = str(meta["id"])
        policy = source_freshness_policy(root, "fred")
        specs.append(
            {
                "input_path": f"data/raw/fred/{name}__{series_id}.csv",
                "source_family": "fred",
                "required": bool(meta.get("required", False)),
                "freshness_days": int(policy["mtime_days"] or 30),
                "observation_freshness_days": None
                if meta.get("discontinued_after")
                else policy["observation_days"],
            }
        )
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for spec in specs:
        path = str(spec["input_path"])
        if path in seen:
            continue
        seen.add(path)
        unique.append(spec)
    return unique


def _source_family_for_input(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 3 and parts[0] == "data":
        return parts[2] if parts[1] == "raw" else parts[1]
    return "project"


def _raw_refresh_status_by_input(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    manifest_specs = [
        ("fred", root / "data/raw/fred/manifest.csv"),
        ("fiscaldata", root / "data/raw/fiscaldata/manifest.csv"),
        ("ofr", root / "data/raw/ofr/manifest.csv"),
    ]
    for source_family, path in manifest_specs:
        if not path.exists():
            continue
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        for record in table.to_dict(orient="records"):
            rel_path = _normalize_raw_manifest_path(root, source_family, str(record.get("path", "")))
            if not rel_path:
                continue
            rows[rel_path] = {
                "latest_fetch_status": record.get("status", ""),
                "latest_fetch_rows": record.get("rows", ""),
                "latest_fetch_last_observation": record.get("last_observation", "")
                or record.get("cache_last_observation", ""),
            }
    return rows


def _normalize_raw_manifest_path(root: Path, source_family: str, raw_path: str) -> str:
    if not raw_path:
        return ""
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            parts = path.parts
            if "data" in parts:
                index = parts.index("data")
                return str(Path(*parts[index:]))
            return ""
    if raw_path.startswith("data/raw/"):
        return raw_path
    if raw_path.startswith(f"{source_family}/"):
        return str(Path("data/raw") / raw_path)
    return str(Path("data/raw") / source_family / raw_path)


def _source_refresh_decision(record: dict[str, Any], fetch: dict[str, Any]) -> tuple[str, str]:
    required = bool(record.get("required", False))
    exists = bool(record.get("exists", False))
    input_path = str(record.get("input_path", ""))
    if not exists:
        return (
            "missing_required" if required else "missing_optional",
            "populate required source cache" if required else "optional source cache is absent",
        )
    if str(record.get("status", "")) != "ok":
        return "cache_not_ok", "inspect source cache"
    if str(record.get("freshness_status", "")) in {"stale", "stale_missing_mtime"}:
        return "stale_mtime", "refresh source cache"
    if str(record.get("observation_freshness_status", "")) in {"stale", "stale_missing_observation"}:
        return "stale_observations", "refresh source cache or update source policy"
    latest_fetch_status = str(fetch.get("latest_fetch_status", ""))
    if latest_fetch_status.startswith("kept_existing_cache_after_failed_refresh"):
        return "ok_live_refresh_failed_cache_valid", "retry live refresh later if latest upstream observations are needed"
    if latest_fetch_status.startswith("failed"):
        return (
            "refresh_failed_required_cache_valid" if required else "refresh_failed_optional_cache_valid",
            "retry live refresh; current cache remains usable",
        )
    if input_path.endswith("institutional_mmf_assets__WIMFNS.csv"):
        return "ok_discontinued_upstream_series", "none; source is a documented historical FRED series"
    return "ok", "none"


def _source_contract_index(root: Path) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for path in source_contract_paths(root):
        try:
            contract = load_yaml(path)
        except Exception:
            continue
        source_id = contract.get("source_id")
        if isinstance(source_id, str):
            contracts[source_id] = contract
    return contracts


def _contract_for_inventory_record(contracts: dict[str, dict[str, Any]], record: dict[str, Any]) -> dict[str, Any]:
    input_path = str(record.get("input_path", ""))
    source_family = str(record.get("source_family", ""))
    if source_family == "buycurve":
        return contracts.get("bill_issuance_from_buycurve", {})
    if source_family == "fiscaldata":
        return contracts.get("dts", {})
    if source_family == "ofr":
        return contracts.get("ofr_sec_mmf", {})
    if source_family == "tgarefill":
        return contracts.get("tgarefill", {})
    if source_family == "fred":
        return _fred_contract_for_path(contracts, input_path)
    return {}


def _fred_contract_for_path(contracts: dict[str, dict[str, Any]], input_path: str) -> dict[str, Any]:
    name = Path(input_path).name.split("__", maxsplit=1)[0]
    if name in {"reserves", "tga_wednesday", "tga_week_average", "fed_treasury_holdings"}:
        return contracts.get("h41", {})
    if name in {
        "deposits",
        "deposits_nsa",
        "domestic_deposits",
        "cash_assets",
        "bank_treasury_agency_securities",
        "total_bank_credit",
    }:
        return contracts.get("h8", {})
    if name == "on_rrp":
        return contracts.get("on_rrp_operations", {})
    if name in {"iorb_rate", "fed_funds", "sofr", "bill_yield_1mo", "bill_yield_3mo", "bill_yield_6mo"}:
        return contracts.get("rates", {})
    return {"source_id": "fred", "source_name": "Federal Reserve Economic Data", "access": {"mode": "fred_csv", "source_url": "https://fred.stlouisfed.org/graph/fredgraph.csv"}}


def _freshness_record(path: Path, freshness_days: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "mtime_utc": "",
            "age_days": "",
            "freshness_days": freshness_days,
            "freshness_status": "missing",
        }
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        return {
            "mtime_utc": "",
            "age_days": "",
            "freshness_days": freshness_days,
            "freshness_status": "stale_missing_mtime",
        }
    age_seconds = max(0.0, (datetime.now(UTC) - mtime).total_seconds())
    age_days = round(age_seconds / 86_400, 3)
    return {
        "mtime_utc": mtime.isoformat(timespec="seconds"),
        "age_days": age_days,
        "freshness_days": freshness_days,
        "freshness_status": "fresh" if age_days <= freshness_days else "stale",
    }


def _observation_freshness_record(path: Path, freshness_days: int | None) -> dict[str, Any]:
    empty = {
        "valid_observation_rows": "",
        "first_valid_observation": "",
        "last_valid_observation": "",
        "observation_age_days": "",
        "observation_freshness_days": freshness_days if freshness_days is not None else "",
        "observation_freshness_status": "missing" if not path.exists() else "not_profiled",
    }
    if not path.exists():
        return empty
    if path.suffix == ".csv":
        profile = _csv_observation_profile(path)
    elif path.suffix == ".json":
        profile = _json_observation_profile(path)
    else:
        return empty
    if not profile:
        return empty
    if profile.get("observation_freshness_status") == "not_profiled":
        return {
            **empty,
            **profile,
        }
    try:
        valid_rows = int(profile.get("valid_observation_rows", 0))
    except (TypeError, ValueError):
        return {
            **empty,
            **profile,
            "observation_freshness_status": "not_profiled",
        }
    last = str(profile.get("last_valid_observation", ""))
    if valid_rows == 0:
        return {
            **profile,
            "observation_age_days": "",
            "observation_freshness_days": freshness_days if freshness_days is not None else "",
            "observation_freshness_status": "empty_observations",
        }
    last_date = pd.to_datetime(last, errors="coerce")
    if pd.isna(last_date):
        return {
            **profile,
            "observation_age_days": "",
            "observation_freshness_days": freshness_days if freshness_days is not None else "",
            "observation_freshness_status": "stale_missing_observation",
        }
    age_seconds = max(0.0, (datetime.now(UTC) - last_date.to_pydatetime().replace(tzinfo=UTC)).total_seconds())
    age_days = round(age_seconds / 86_400, 3)
    return {
        **profile,
        "observation_age_days": age_days,
        "observation_freshness_days": freshness_days if freshness_days is not None else "",
        "observation_freshness_status": "fresh"
        if freshness_days is None or age_days <= freshness_days
        else "stale",
    }


def _csv_observation_profile(path: Path) -> dict[str, Any]:
    try:
        table = pd.read_csv(path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return {
            "valid_observation_rows": 0,
            "first_valid_observation": "",
            "last_valid_observation": "",
        }
    date_col = _first_present(
        table,
        [
            "observation_date",
            "date",
            "month",
            "week",
            "record_date",
            "event_date",
            "baseline_date",
            "start_date",
            "end_date",
        ],
    )
    if not date_col:
        return {
            "valid_observation_rows": "",
            "first_valid_observation": "",
            "last_valid_observation": "",
            "observation_freshness_status": "not_profiled",
        }
    dates = pd.to_datetime(table[date_col], errors="coerce")
    if {"close_today_bal", "open_today_bal"}.issubset(table.columns):
        values = pd.to_numeric(table["close_today_bal"], errors="coerce").fillna(
            pd.to_numeric(table["open_today_bal"], errors="coerce")
        )
        valid = dates.notna() & values.notna()
    else:
        value_col = _first_present(table, ["value", "close_today_bal", "accepted_amount_sum"])
        if value_col:
            values = pd.to_numeric(table[value_col], errors="coerce")
            valid = dates.notna() & values.notna()
        else:
            numeric_cols = [
                col
                for col in table.columns
                if col != date_col and pd.to_numeric(table[col], errors="coerce").notna().any()
            ]
            valid = dates.notna()
            if numeric_cols:
                numeric = table[numeric_cols].apply(pd.to_numeric, errors="coerce")
                valid &= numeric.notna().any(axis=1)
    valid_dates = dates.loc[valid].dropna()
    return {
        "valid_observation_rows": int(len(valid_dates)),
        "first_valid_observation": valid_dates.min().date().isoformat() if not valid_dates.empty else "",
        "last_valid_observation": valid_dates.max().date().isoformat() if not valid_dates.empty else "",
    }


def _json_observation_profile(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {
            "valid_observation_rows": 0,
            "first_valid_observation": "",
            "last_valid_observation": "",
        }
    observations: list[tuple[Any, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("timeseries"), dict):
        for series in payload["timeseries"].values():
            if not isinstance(series, dict):
                continue
            timeseries = series.get("timeseries")
            if not isinstance(timeseries, dict):
                continue
            for values in timeseries.values():
                if isinstance(values, list):
                    observations.extend(
                        (row[0], row[1])
                        for row in values
                        if isinstance(row, list | tuple) and len(row) >= 2
                    )
    if not observations:
        return {
            "valid_observation_rows": "",
            "first_valid_observation": "",
            "last_valid_observation": "",
            "observation_freshness_status": "not_profiled",
        }
    dates = pd.to_datetime([row[0] for row in observations], errors="coerce")
    values = pd.to_numeric(pd.Series([row[1] for row in observations]), errors="coerce")
    valid_dates = pd.Series(dates).loc[pd.Series(dates).notna() & values.notna()]
    return {
        "valid_observation_rows": int(len(valid_dates)),
        "first_valid_observation": valid_dates.min().date().isoformat() if not valid_dates.empty else "",
        "last_valid_observation": valid_dates.max().date().isoformat() if not valid_dates.empty else "",
    }


def write_release_notes(root: Path) -> dict[str, Any]:
    report_dir = ensure_dir(root / "output" / "reports")
    path = report_dir / "RELEASE_NOTES.md"
    status = artifact_status(root)
    latest_manifest = _latest_manifest(root)
    tier_counts = status.get("output_tiers", {})
    input_report = validate_backend_inputs(root)
    monthly = status.get("monthly_panel", {})
    weekly = status.get("weekly_panel", {})
    lines = [
        "# liqsub Public Backend Release Notes",
        "",
        f"Created at UTC: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "## Backend Stage",
        "",
        "This release is a reproducible backend diagnostics packet. It includes source code, config, docs, manual context inputs, clean panels, public outputs, manifests, and validation artifacts.",
        "",
        "## Data Panels",
        "",
        (
            "- Monthly panel: "
            f"{monthly.get('rows', 0)} rows, "
            f"{monthly.get('first_date', '')} to {monthly.get('last_date', '')}."
        ),
        (
        "- Weekly panel: "
            f"{weekly.get('rows', 0)} rows, "
            f"{weekly.get('first_date', '')} to {weekly.get('last_date', '')}."
        ),
        "",
        "## Source Inputs",
        "",
        f"- Backend input files checked: {input_report['checked_files']}",
        f"- Missing required backend inputs: {input_report['missing_required']}",
        f"- Stale backend input caches: {input_report['stale_inputs']}",
        "",
        "## Release Checks",
        "",
        "- Config validation is required before release.",
        "- Public CSV schema validation is required before release.",
        "- Public/private boundary validation is required before release.",
        "- The public bundle is smoke-tested after extraction.",
        "",
        "## Public Artifact Index",
        "",
    ]
    if tier_counts:
        for tier, count in sorted(tier_counts.items()):
            lines.append(f"- {tier}: {count}")
    else:
        lines.append("- No public output tiers were indexed before these notes were written.")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Latest manifest: `{latest_manifest}`" if latest_manifest else "- Latest manifest: not yet written.",
            "- Bundle manifest: `PUBLIC_BUNDLE_MANIFEST.json` inside the zip.",
            "",
            "## Public/Private Boundary",
            "",
            "The bundle excludes `do/`, `.env`, raw/interim/cache data, internal outputs, prior bundles, local test/build caches, and hidden system files.",
            "",
            "## Backend Limitations",
            "",
            "- Generated outputs are diagnostics unless a downstream consumer explicitly depends on a stable artifact listed in `docs/public_artifact_contract.md`.",
            "- Raw source caches are excluded from the public bundle; rebuilds need source access or equivalent raw inputs.",
            "- Legacy compatibility aliases are still emitted for some old output names while downstream readers migrate to neutral names.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"status": "ok", "path": relative_to_root(root, path)}


def smoke_test_public_bundle(bundle_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not bundle_path.exists():
        return {"status": "failed", "errors": [f"bundle does not exist: {bundle_path}"]}

    with tempfile.TemporaryDirectory(prefix="liqsub_bundle_smoke_") as tmp:
        extract_root = Path(tmp) / "bundle"
        extract_root.mkdir()
        try:
            with zipfile.ZipFile(bundle_path) as archive:
                names = archive.namelist()
                errors.extend(_validate_bundle_member_names(names))
                if not errors:
                    for member in archive.infolist():
                        target = (extract_root / member.filename).resolve()
                        if not target.is_relative_to(extract_root.resolve()):
                            errors.append(f"bundle member escapes extract root: {member.filename}")
                            continue
                        if member.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with archive.open(member) as source, target.open("wb") as dest:
                                dest.write(source.read())
        except zipfile.BadZipFile:
            errors.append(f"invalid zip archive: {bundle_path}")

        if not errors:
            errors.extend(_validate_bundle_manifest(extract_root))
            schema_report = validate_public_output_schemas(extract_root)
            errors.extend(schema_report.errors)
            warnings.extend(schema_report.warnings)
            boundary = validate_public_boundary(extract_root)
            errors.extend(boundary["errors"])
            checked = {
                "archive_entries": len(names),
                "public_output_rows": boundary["checked"]["public_output_rows"],
                "public_private_files": boundary["checked"]["public_private_files"],
                "schema_files": schema_report.checked_files,
            }
            warnings.extend(boundary["warnings"])
        else:
            checked = {"archive_entries": 0}

    return {
        "status": "ok" if not errors else "failed",
        "errors": sorted(dict.fromkeys(errors)),
        "warnings": sorted(dict.fromkeys(warnings)),
        "checked": checked,
    }


def _validate_bundle_manifest(extract_root: Path) -> list[str]:
    manifest_path = extract_root / "PUBLIC_BUNDLE_MANIFEST.json"
    if not manifest_path.exists():
        return ["bundle is missing PUBLIC_BUNDLE_MANIFEST.json"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"PUBLIC_BUNDLE_MANIFEST.json is not valid JSON: {exc}"]
    errors: list[str] = []
    files = payload.get("files")
    if not isinstance(files, list):
        return ["PUBLIC_BUNDLE_MANIFEST.json is missing files list"]
    manifest_members = {str(record.get("path", "")): record for record in files if isinstance(record, dict)}
    actual_members = sorted(
        path.relative_to(extract_root).as_posix()
        for path in extract_root.rglob("*")
        if path.is_file() and path.name != "PUBLIC_BUNDLE_MANIFEST.json"
    )
    missing_from_manifest = sorted(set(actual_members) - set(manifest_members))
    extra_in_manifest = sorted(set(manifest_members) - set(actual_members))
    for rel in missing_from_manifest:
        errors.append(f"PUBLIC_BUNDLE_MANIFEST.json omits bundled file: {rel}")
    for rel in extra_in_manifest:
        errors.append(f"PUBLIC_BUNDLE_MANIFEST.json lists missing file: {rel}")
    for rel, record in manifest_members.items():
        path = extract_root / rel
        if not path.exists():
            continue
        expected_hash = str(record.get("sha256", ""))
        if expected_hash and _sha256(path) != expected_hash:
            errors.append(f"PUBLIC_BUNDLE_MANIFEST.json hash mismatch for {rel}")
        expected_bytes = record.get("bytes")
        if expected_bytes not in {"", None} and int(expected_bytes) != path.stat().st_size:
            errors.append(f"PUBLIC_BUNDLE_MANIFEST.json byte-size mismatch for {rel}")
    return errors


def _validate_bundle_member_names(names: list[str]) -> list[str]:
    errors: list[str] = []
    forbidden_needles = [
        "do/",
        ".env",
        "data/raw/",
        "data/interim/",
        "data/cache/",
        "output/internal/",
        "output/bundles/",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".DS_Store",
    ]
    if "PUBLIC_BUNDLE_MANIFEST.json" not in names:
        errors.append("bundle is missing PUBLIC_BUNDLE_MANIFEST.json")
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            errors.append(f"bundle member has unsafe path: {name}")
        for needle in forbidden_needles:
            if name == needle.strip("/") or needle in name:
                errors.append(f"bundle member includes forbidden path: {name}")
        if name.startswith("output/") and Path(name).name.startswith(("internal_", "private_")):
            errors.append(f"bundle member includes private artifact name: {name}")
    return sorted(set(errors))


def public_bundle_file_list(root: Path) -> list[str]:
    files: list[str] = []
    for rel_root in PUBLIC_BUNDLE_ROOTS:
        path = root / rel_root
        if path.is_file():
            if _is_public_bundle_file(root, path):
                files.append(path.relative_to(root).as_posix())
            continue
        if not path.exists():
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and _is_public_bundle_file(root, child):
                files.append(child.relative_to(root).as_posix())
    return sorted(dict.fromkeys(files))


def _is_public_bundle_file(root: Path, path: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    rel = path.relative_to(root).as_posix()
    if path.name == ".DS_Store":
        return False
    if any(part in PUBLIC_BUNDLE_EXCLUDED_DIRS for part in rel_parts):
        return False
    if any(part.endswith(".egg-info") for part in rel_parts):
        return False
    if any(part.startswith(".") for part in rel_parts) and rel != ".gitignore":
        return False
    if rel_parts[0] == "do":
        return False
    if rel in {".env", "plan.md"} or rel.endswith(".plan.md"):
        return False
    if rel_parts[0] == "data" and len(rel_parts) > 1 and rel_parts[1] in {"raw", "interim", "cache"}:
        return False
    if rel_parts[0] == "output":
        if "internal" in rel_parts or "bundles" in rel_parts:
            return False
        if path.name.startswith(("internal_", "private_")):
            return False
        if rel_parts[:2] == ("output", "manifests") and path.name.startswith("run_manifest_"):
            return False
    return True


def validate_public_boundary(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    ignored_patterns = _gitignore_patterns(root)
    required_ignored = {
        "do/": "private planning directory",
        ".env": "local environment file",
        "output/**": "generated output tree",
        "data/raw/**": "raw source cache",
        "data/clean/**": "generated clean panels",
        "data/interim/**": "generated interim data",
    }
    for pattern, description in required_ignored.items():
        if pattern not in ignored_patterns:
            errors.append(f".gitignore does not ignore {description}: {pattern}")

    tier_table = output_tier_table(root)
    if not tier_table.empty:
        paths = tier_table["path"].astype(str)
        hidden_paths = sorted(
            path
            for path in paths.tolist()
            if any(part.startswith(".") for part in Path(path).parts)
        )
        private_name_paths = sorted(
            path
            for path in paths.tolist()
            if Path(path).name.startswith(("internal_", "private_"))
        )
        for path in private_name_paths:
            errors.append(f"public output tier index includes private artifact: {path}")
        for path in hidden_paths:
            errors.append(f"public output tier index includes hidden/system artifact: {path}")

    public_private_files = sorted(
        relative_to_root(root, path)
        for directory in [root / "output" / "tables", root / "output" / "reports"]
        if directory.exists()
        for pattern in ("internal_*", "private_*")
        for path in directory.glob(pattern)
        if path.is_file()
    )
    for path in public_private_files:
        errors.append(f"private artifact is in public output directory: {path}")

    if (root / "do").exists() and "do/" in ignored_patterns:
        warnings.append("do/ exists locally and is ignored; keep it out of public artifacts")
    if (root / "output" / "internal").exists():
        warnings.append("output/internal exists locally and is excluded from public indexes")

    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "checked": {
            "gitignore_patterns": len(ignored_patterns),
            "public_output_rows": int(len(tier_table)),
            "public_private_files": len(public_private_files),
        },
    }


def write_public_output_aliases(root: Path) -> dict[str, Any]:
    table_dir = ensure_dir(root / "output" / "tables")
    path = table_dir / "public_output_aliases.csv"
    table = public_output_alias_table(root)
    table.to_csv(path, index=False)
    return {"status": "ok", "rows": int(len(table)), "path": relative_to_root(root, path)}


def public_output_alias_table(root: Path) -> pd.DataFrame:
    columns = [
        "canonical_path",
        "compatibility_path",
        "relationship",
        "status",
        "canonical_exists",
        "compatibility_exists",
        "removal_policy",
    ]
    rows = []
    for canonical, compatibility in sorted(PUBLIC_OUTPUT_ALIASES.items()):
        rows.append(
            {
                "canonical_path": canonical,
                "compatibility_path": compatibility,
                "relationship": "compatibility_alias",
                "status": "active_legacy_alias",
                "canonical_exists": bool((root / canonical).exists()),
                "compatibility_exists": bool((root / compatibility).exists()),
                "removal_policy": "keep_until_downstream_consumers_migrate",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_output_tiers(root: Path) -> dict[str, Any]:
    manifest_dir = ensure_dir(root / "output" / "manifests")
    write_public_output_aliases(root)
    table = output_tier_table(root)
    csv_path = manifest_dir / "output_tiers.csv"
    json_path = manifest_dir / "output_tiers.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(table.to_json(orient="records", indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "rows": int(len(table)),
        "csv": relative_to_root(root, csv_path),
        "json": relative_to_root(root, json_path),
    }


def output_tier_table(root: Path) -> pd.DataFrame:
    columns = ["path", "tier", "bytes", "rows", "first_date", "last_date", "sha256"]
    files = sorted((root / "output").glob("**/*"))
    rows: list[dict[str, Any]] = []
    for path in files:
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts) or "internal" in rel_parts or "bundles" in rel_parts:
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith("output/manifests/"):
            continue
        tier = classify_output_tier(rel)
        record = _file_record(root, rel)
        rows.append(
            {
                "path": rel,
                "tier": tier,
                "bytes": record.get("bytes", 0),
                "rows": record.get("rows", ""),
                "first_date": record.get("first_date", ""),
                "last_date": record.get("last_date", ""),
                "sha256": record.get("sha256", ""),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["tier", "path"]).reset_index(drop=True)


def classify_output_tier(rel_path: str) -> str:
    lowered = rel_path.lower()
    if "/reports/" in lowered:
        return "reports"
    if "/manifests/" in lowered:
        return "manifests"
    for tier, needles in OUTPUT_TIER_RULES:
        if any(needle in lowered for needle in needles):
            return tier
    return "diagnostics"


def write_run_manifest(root: Path) -> dict[str, Any]:
    manifest_dir = ensure_dir(root / "output" / "manifests")
    write_public_output_aliases(root)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_tiers = output_tier_table(root).to_dict(orient="records")
    payload = {
        "created_at_utc": timestamp,
        "git": _git_status(root),
        "inputs": backend_input_inventory(root).to_dict(orient="records"),
        "outputs": [_file_record(root, rel) for rel in _stable_public_artifact_paths()],
        "output_tiers": output_tiers,
        "status": artifact_status(root),
    }
    path = manifest_dir / f"run_manifest_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    latest = manifest_dir / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": "ok",
        "manifest": relative_to_root(root, path),
        "latest": relative_to_root(root, latest),
    }


def validate_manifest_integrity(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    latest = root / "output" / "manifests" / "latest.json"
    if not latest.exists():
        return {"status": "failed", "errors": ["output/manifests/latest.json: missing manifest"], "warnings": []}
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "errors": [f"output/manifests/latest.json: invalid JSON: {exc}"],
            "warnings": [],
        }
    inputs = payload.get("inputs", [])
    if not isinstance(inputs, list):
        errors.append("output/manifests/latest.json: inputs is not a list")
    elif len(inputs) != len(_backend_input_specs(root)):
        errors.append(
            f"output/manifests/latest.json: expected {len(_backend_input_specs(root))} backend inputs, found {len(inputs)}"
        )
    for rel in _stable_public_artifact_paths():
        record = _find_manifest_record(payload.get("outputs", []), rel)
        if record is None:
            errors.append(f"output/manifests/latest.json: missing output record for {rel}")
            continue
        actual = _file_record(root, rel)
        if bool(record.get("exists", False)) != bool(actual.get("exists", False)):
            errors.append(f"output/manifests/latest.json: existence mismatch for {rel}")
            continue
        if actual.get("exists") and record.get("sha256") != actual.get("sha256"):
            errors.append(f"output/manifests/latest.json: hash mismatch for {rel}")

    tier_table = output_tier_table(root)
    for _, row in tier_table.iterrows():
        rel = str(row["path"])
        actual = _file_record(root, rel)
        if str(row.get("sha256", "")) != str(actual.get("sha256", "")):
            errors.append(f"output_tiers: hash mismatch for {rel}")

    return {"status": "ok" if not errors else "failed", "errors": errors, "warnings": []}


def _find_manifest_record(records: Any, rel: str) -> dict[str, Any] | None:
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and record.get("path") == rel:
            return record
    return None


def _stable_public_artifact_paths() -> list[str]:
    paths = [
        *STABLE_CLEAN_PANELS.keys(),
        *STABLE_PUBLIC_TABLES,
        *STABLE_PUBLIC_REPORTS,
        REBUILD_LINEAGE_PATH,
        "output/manifests/output_tiers.csv",
        "output/manifests/output_tiers.json",
    ]
    return sorted(dict.fromkeys(paths))


def _output_tier_counts(root: Path) -> dict[str, int]:
    table = output_tier_table(root)
    if table.empty:
        return {}
    return {str(key): int(value) for key, value in table["tier"].value_counts().to_dict().items()}


def _file_record(root: Path, rel_path: str) -> dict[str, Any]:
    path = root / rel_path
    record: dict[str, Any] = {
        "path": rel_path,
        "exists": path.exists(),
    }
    if not path.exists():
        return record
    stat = path.stat()
    record.update(
        {
            "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            "sha256": _sha256(path),
        }
    )
    if path.suffix == ".csv":
        record.update(_csv_profile(path))
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_profile(path: Path, *, date_col: str | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {"exists": True, "rows": 0, "columns": []}
    out: dict[str, Any] = {
        "exists": True,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    selected_date_col = date_col or _first_present(df, ["month", "week", "date"])
    if selected_date_col:
        dates = pd.to_datetime(df[selected_date_col], errors="coerce").dropna()
        out["date_column"] = selected_date_col
        out["first_date"] = dates.min().date().isoformat() if not dates.empty else ""
        out["last_date"] = dates.max().date().isoformat() if not dates.empty else ""
    return out


def _csv_counts(path: Path, column: str) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {"exists": True, "rows": 0, "counts": {}}
    counts = df[column].value_counts(dropna=False).to_dict() if column in df.columns else {}
    return {
        "exists": True,
        "rows": int(len(df)),
        "counts": {str(key): int(value) for key, value in counts.items()},
    }


def _legacy_counts(preferred: dict[str, Any], path: Path, *, column: str = "readiness_status") -> dict[str, Any]:
    if preferred.get("exists"):
        return preferred
    return _csv_counts(path, column)


def _first_present(df: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in df.columns:
            return column
    return None


def _latest_manifest(root: Path) -> str:
    path = root / "output" / "manifests" / "latest.json"
    return relative_to_root(root, path) if path.exists() else ""


def _gitignore_patterns(root: Path) -> set[str]:
    path = root / ".gitignore"
    if not path.exists():
        return set()
    patterns: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.add(line)
    return patterns


def _git_status(root: Path) -> dict[str, Any]:
    return {
        "head": _git_command(root, ["rev-parse", "--short", "HEAD"]),
        "branch": _git_command(root, ["branch", "--show-current"]),
        "status_short": _git_command(root, ["status", "--short"]),
    }


def _git_command(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""
    return result.stdout.strip()
