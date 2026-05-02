from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from liqsub.artifacts import (
    artifact_status,
    validate_backend_inputs,
    release_public_backend,
    smoke_test_public_bundle,
    validate_public_boundary,
    validate_rebuild_lineage,
    write_backend_input_inventory,
    write_public_bundle,
    write_public_output_aliases,
    write_rebuild_lineage,
    write_release_notes,
    write_source_cache_manifest,
    write_source_refresh_status,
    write_output_tiers,
    write_run_manifest,
)
from liqsub.buycurve import validate_buycurve_panel
from liqsub.analysis import write_evidence_gate_summary, write_monthly_analysis
from liqsub.config import validate_project_config
from liqsub.fiscaldata import download_dts_operating_cash_balance
from liqsub.fred import download_fred_series
from liqsub.ofr import download_ofr_mmf
from liqsub.panel import write_monthly_outputs
from liqsub.paths import find_project_root, relative_to_root
from liqsub.schemas import validate_public_output_schemas
from liqsub.tgarefill import validate_tgarefill_exports, write_weekly_analysis, write_weekly_outputs


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="liqsub")
    parser.add_argument("--root", type=Path, default=find_project_root())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config")
    sub.add_parser("check-buycurve")
    sub.add_parser("check-tgarefill")
    fred = sub.add_parser("fetch-fred")
    fred.add_argument("--required-only", action="store_true")
    fred.add_argument("--series", action="append", help="Fetch one FRED series key; repeat to fetch several.")
    fred.add_argument("--timeout", type=int, default=45, help="Per-series network timeout in seconds.")
    fred.add_argument("--retries", type=int, default=1, help="Retries per FRED series after the first attempt.")
    fred.add_argument("--progress", action="store_true", help="Print per-series progress to stderr.")
    sub.add_parser("fetch-dts")
    sub.add_parser("fetch-ofr")
    sub.add_parser("build-monthly")
    sub.add_parser("analyze-monthly")
    sub.add_parser("build-weekly")
    sub.add_parser("analyze-weekly")
    rebuild = sub.add_parser("rebuild-public")
    rebuild.add_argument("--destination", type=Path)
    rebuild.add_argument("--check-only", action="store_true")
    sub.add_parser("status")
    sub.add_parser("validate-inputs")
    sub.add_parser("write-input-inventory")
    sub.add_parser("write-source-cache-manifest")
    sub.add_parser("write-source-refresh-status")
    sub.add_parser("write-output-tiers")
    sub.add_parser("write-manifest")
    bundle = sub.add_parser("write-public-bundle")
    bundle.add_argument("--destination", type=Path)
    release = sub.add_parser("release-public")
    release.add_argument("--destination", type=Path)
    release.add_argument("--check-only", action="store_true")
    sub.add_parser("write-release-notes")
    smoke = sub.add_parser("smoke-public-bundle")
    smoke.add_argument("bundle", type=Path)
    sub.add_parser("write-internal-reports")
    sub.add_parser("validate-public-boundary")
    sub.add_parser("validate-rebuild-lineage")
    sub.add_parser("validate-output-schemas")
    sub.add_parser("fetch-all")

    args = parser.parse_args(argv)
    root = args.root.resolve()
    raw_dir = root / "data" / "raw"

    if args.command == "validate-config":
        report = validate_project_config(root)
        _print_json({"status": report.status, "errors": report.errors, "warnings": report.warnings})
        return 1 if report.errors else 0
    if args.command == "check-buycurve":
        path = raw_dir / "buycurve" / "monthly_issuance_maturity_panel.csv"
        report = validate_buycurve_panel(path)
        _print_json(
            {
                "status": report.status,
                "path": relative_to_root(root, report.path),
                "errors": report.errors,
                "rows": report.rows,
            }
        )
        return 0 if report.status in {"ok", "blocked"} else 1
    if args.command == "check-tgarefill":
        report = validate_tgarefill_exports(root)
        _print_json(
            {
                "status": report.status,
                "errors": report.errors,
                "rows_by_export": report.rows_by_export,
            }
        )
        return 0 if report.status == "ok" else 1
    if args.command == "fetch-fred":
        manifest = download_fred_series(
            raw_dir,
            required_only=args.required_only,
            series_keys=args.series,
            timeout=args.timeout,
            retries=args.retries,
            progress=(lambda message: print(message, file=sys.stderr, flush=True))
            if args.progress
            else None,
        )
        _print_json(
            {
                "status": "ok",
                "rows": int(len(manifest)),
                "manifest": relative_to_root(root, raw_dir / "fred" / "manifest.csv"),
            }
        )
        return 0
    if args.command == "fetch-dts":
        path = download_dts_operating_cash_balance(raw_dir)
        _print_json({"status": "ok", "path": relative_to_root(root, path)})
        return 0
    if args.command == "fetch-ofr":
        path = download_ofr_mmf(raw_dir)
        _print_json({"status": "ok", "path": relative_to_root(root, path)})
        return 0
    if args.command == "fetch-all":
        fred_manifest = download_fred_series(raw_dir, progress=lambda message: print(message, file=sys.stderr, flush=True))
        dts_path = download_dts_operating_cash_balance(raw_dir)
        ofr_path = download_ofr_mmf(raw_dir)
        _print_json(
            {
                "status": "ok",
                "fred_series": int(len(fred_manifest)),
                "dts": relative_to_root(root, dts_path),
                "ofr": relative_to_root(root, ofr_path),
            }
        )
        return 0
    if args.command == "build-monthly":
        _print_json(write_monthly_outputs(root))
        return 0
    if args.command == "analyze-monthly":
        _print_json({"status": "ok", "outputs": write_monthly_analysis(root)})
        return 0
    if args.command == "build-weekly":
        _print_json(write_weekly_outputs(root))
        return 0
    if args.command == "analyze-weekly":
        _print_json({"status": "ok", "outputs": write_weekly_analysis(root)})
        return 0
    if args.command == "rebuild-public":
        result = _rebuild_public(root, destination=args.destination, check_only=args.check_only)
        _print_json(result)
        return 0 if result["status"] == "ok" else 1
    if args.command == "status":
        _print_json(artifact_status(root))
        return 0
    if args.command == "validate-inputs":
        result = validate_backend_inputs(root)
        _print_json(result)
        return 0 if result["status"] == "ok" else 1
    if args.command == "write-input-inventory":
        result = write_backend_input_inventory(root)
        _print_json(result)
        return 0 if result["status"] == "ok" else 1
    if args.command == "write-source-cache-manifest":
        result = write_source_cache_manifest(root)
        _print_json(result)
        return 0 if result["status"] == "ok" else 1
    if args.command == "write-source-refresh-status":
        result = write_source_refresh_status(root)
        _print_json(result)
        return 0 if result["status"] == "ok" else 1
    if args.command == "write-output-tiers":
        _print_json(write_output_tiers(root))
        return 0
    if args.command == "write-manifest":
        _print_json(write_run_manifest(root))
        return 0
    if args.command == "write-public-bundle":
        result = write_public_bundle(root, destination=args.destination)
        _print_json(result)
        return 0 if result["status"] == "ok" else 1
    if args.command == "release-public":
        result = release_public_backend(root, destination=args.destination, check_only=args.check_only)
        _print_json(result)
        return 0 if result["status"] == "ok" else 1
    if args.command == "write-release-notes":
        result = write_release_notes(root)
        _print_json(result)
        return 0 if result["status"] == "ok" else 1
    if args.command == "smoke-public-bundle":
        result = smoke_test_public_bundle(args.bundle)
        _print_json(result)
        return 0 if result["status"] == "ok" else 1
    if args.command == "write-internal-reports":
        _print_json({"status": "ok", "outputs": write_evidence_gate_summary(root, include_internal=True)})
        return 0
    if args.command == "validate-public-boundary":
        report = validate_public_boundary(root)
        _print_json(report)
        return 0 if report["status"] == "ok" else 1
    if args.command == "validate-rebuild-lineage":
        report = validate_rebuild_lineage(root)
        _print_json(report)
        return 0 if report["status"] == "ok" else 1
    if args.command == "validate-output-schemas":
        report = validate_public_output_schemas(root)
        _print_json(
            {
                "status": report.status,
                "checked_files": report.checked_files,
                "errors": report.errors,
                "warnings": report.warnings,
            }
        )
        return 0 if report.status == "ok" else 1
    raise AssertionError(args.command)


def _rebuild_public(root: Path, *, destination: Path | None = None, check_only: bool = False) -> dict[str, Any]:
    raw_dir = root / "data" / "raw"
    steps: list[dict[str, Any]] = []

    config = validate_project_config(root)
    steps.append({"step": "validate_config", "status": config.status, "errors": config.errors, "warnings": config.warnings})
    if config.errors:
        return _blocked_rebuild(steps, config.errors)

    input_report = validate_backend_inputs(root)
    steps.append({"step": "validate_inputs", **input_report})
    if input_report["errors"]:
        return _blocked_rebuild(steps, input_report["errors"])

    buycurve = validate_buycurve_panel(raw_dir / "buycurve" / "monthly_issuance_maturity_panel.csv")
    steps.append(
        {
            "step": "check_buycurve",
            "status": buycurve.status,
            "errors": buycurve.errors,
            "rows": buycurve.rows,
        }
    )
    if buycurve.status != "ok":
        return _blocked_rebuild(steps, list(buycurve.errors))

    if check_only:
        tgarefill = validate_tgarefill_exports(root)
        steps.append(
            {
                "step": "check_tgarefill",
                "status": tgarefill.status,
                "errors": tgarefill.errors,
                "rows_by_export": tgarefill.rows_by_export,
            }
        )
        if tgarefill.status != "ok":
            return _blocked_rebuild(steps, list(tgarefill.errors))
        release_check = release_public_backend(root, destination=destination, check_only=True)
        steps.append({"step": "release_public_check", **release_check})
        if release_check["status"] != "ok":
            return _blocked_rebuild(steps, release_check.get("errors", []))
        return {"status": "ok", "check_only": True, "steps": steps, "warnings": _step_warnings(steps)}

    monthly_build = write_monthly_outputs(root)
    steps.append({"step": "build_monthly", **monthly_build})

    monthly_analysis = write_monthly_analysis(root)
    steps.append({"step": "analyze_monthly", "status": "ok", "outputs": monthly_analysis})

    tgarefill = validate_tgarefill_exports(root)
    steps.append(
        {
            "step": "check_tgarefill",
            "status": tgarefill.status,
            "errors": tgarefill.errors,
            "rows_by_export": tgarefill.rows_by_export,
        }
    )
    if tgarefill.status != "ok":
        return _blocked_rebuild(steps, list(tgarefill.errors))

    weekly_build = write_weekly_outputs(root)
    steps.append({"step": "build_weekly", **weekly_build})

    weekly_analysis = write_weekly_analysis(root)
    steps.append({"step": "analyze_weekly", "status": "ok", "outputs": weekly_analysis})

    inventory = write_backend_input_inventory(root)
    steps.append({"step": "write_backend_input_inventory", **inventory})

    cache_manifest = write_source_cache_manifest(root)
    steps.append({"step": "write_source_cache_manifest", **cache_manifest})

    refresh_status = write_source_refresh_status(root)
    steps.append({"step": "write_source_refresh_status", **refresh_status})

    aliases = write_public_output_aliases(root)
    steps.append({"step": "write_public_output_aliases", **aliases})

    lineage = write_rebuild_lineage(root)
    steps.append({"step": "write_rebuild_lineage", **lineage})

    release = release_public_backend(root, destination=destination)
    steps.append({"step": "release_public", **release})
    if release["status"] != "ok":
        return _blocked_rebuild(steps, release.get("errors", []))

    return {
        "status": "ok",
        "check_only": False,
        "bundle": release["bundle"],
        "release_notes": release["release_notes"],
        "steps": steps,
        "warnings": _step_warnings(steps),
    }


def _blocked_rebuild(steps: list[dict[str, Any]], errors: list[str] | tuple[str, ...]) -> dict[str, Any]:
    return {
        "status": "blocked",
        "errors": list(errors),
        "steps": steps,
        "warnings": _step_warnings(steps),
    }


def _step_warnings(steps: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for step in steps:
        for warning in step.get("warnings", []) or []:
            warnings.append(str(warning))
    return sorted(dict.fromkeys(warnings))


if __name__ == "__main__":
    raise SystemExit(main())
