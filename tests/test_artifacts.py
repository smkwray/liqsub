from __future__ import annotations

import json
import zipfile

from liqsub.artifacts import (
    artifact_status,
    backend_input_inventory,
    classify_output_tier,
    output_tier_table,
    public_bundle_file_list,
    public_output_alias_table,
    release_public_backend,
    smoke_test_public_bundle,
    source_refresh_status,
    validate_manifest_integrity,
    validate_backend_inputs,
    validate_public_artifact_contract,
    validate_public_boundary,
    validate_rebuild_lineage,
    write_backend_input_inventory,
    write_public_bundle,
    write_rebuild_lineage,
    write_release_notes,
    write_source_cache_manifest,
    write_source_refresh_status,
    write_output_tiers,
    write_run_manifest,
)


def test_artifact_status_profiles_existing_outputs(tmp_path) -> None:
    clean = tmp_path / "data" / "clean"
    tables = tmp_path / "output" / "tables"
    clean.mkdir(parents=True)
    tables.mkdir(parents=True)
    (clean / "monthly_liquidity_substitution_panel.csv").write_text(
        "month,value\n2024-01-01,1\n",
        encoding="utf-8",
    )
    (clean / "weekly_liquidity_substitution_panel.csv").write_text(
        "week,value\n2024-01-03,1\n",
        encoding="utf-8",
    )
    (tables / "weekly_claim_readiness.csv").write_text(
        "readiness_status\nblocked\n",
        encoding="utf-8",
    )
    status = artifact_status(tmp_path)
    assert status["monthly_panel"]["rows"] == 1
    assert status["weekly_panel"]["first_date"] == "2024-01-03"
    assert status["weekly_claim_readiness"]["counts"]["blocked"] == 1
    assert status["latest_manifest"] == ""


def test_write_run_manifest_writes_latest_json(tmp_path) -> None:
    clean = tmp_path / "data" / "clean"
    clean.mkdir(parents=True)
    (clean / "monthly_liquidity_substitution_panel.csv").write_text(
        "month,value\n2024-01-01,1\n",
        encoding="utf-8",
    )
    result = write_run_manifest(tmp_path)
    latest = tmp_path / "output" / "manifests" / "latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["manifest"].startswith("output/manifests/run_manifest_")
    assert result["latest"] == "output/manifests/latest.json"
    assert not result["manifest"].startswith("/")
    assert latest.exists()
    assert payload["outputs"]
    assert "output_tiers" in payload
    assert payload["status"]["latest_manifest"] == ""

    status = artifact_status(tmp_path)
    assert status["latest_manifest"] == "output/manifests/latest.json"


def test_classify_output_tier_uses_guardrail_categories() -> None:
    assert classify_output_tier("output/tables/monthly_future_row_qa.csv") == "qa"
    assert classify_output_tier("output/tables/weekly_stable_claim_candidates.csv") == "candidate_review"
    assert classify_output_tier("output/tables/monthly_residual_event_window_summary.csv") == "diagnostics"
    assert classify_output_tier("output/reports/weekly_identification_candidate_report.md") == "reports"
    assert classify_output_tier("output/manifests/latest.json") == "manifests"


def test_write_output_tiers_indexes_output_files(tmp_path) -> None:
    tables = tmp_path / "output" / "tables"
    reports = tmp_path / "output" / "reports"
    internal = tmp_path / "output" / "internal" / "reports"
    tables.mkdir(parents=True)
    reports.mkdir(parents=True)
    internal.mkdir(parents=True)
    (tables / "weekly_claim_readiness.csv").write_text(
        "readiness_status\nblocked\n",
        encoding="utf-8",
    )
    (tables / "weekly_design_readiness.csv").write_text(
        "readiness_status\nblocked\n",
        encoding="utf-8",
    )
    (tables / "weekly_placebo_tests.csv").write_text(
        "week,value\n2024-01-03,1\n",
        encoding="utf-8",
    )
    (reports / "weekly_identification_candidate_report.md").write_text("report\n", encoding="utf-8")
    (internal / "private.md").write_text("internal\n", encoding="utf-8")
    (tmp_path / "output" / ".DS_Store").write_text("hidden\n", encoding="utf-8")

    result = write_output_tiers(tmp_path)
    table = output_tier_table(tmp_path)
    csv_path = tmp_path / "output" / "manifests" / "output_tiers.csv"
    json_path = tmp_path / "output" / "manifests" / "output_tiers.json"

    assert result["status"] == "ok"
    assert result["csv"] == "output/manifests/output_tiers.csv"
    assert result["json"] == "output/manifests/output_tiers.json"
    assert csv_path.exists()
    assert json_path.exists()
    assert "candidate_review" in set(table["tier"])
    assert "diagnostics" in set(table["tier"])
    assert "reports" in set(table["tier"])
    assert (tables / "public_output_aliases.csv").exists()
    aliases = public_output_alias_table(tmp_path)
    row = aliases.loc[
        aliases["canonical_path"] == "output/tables/weekly_design_readiness.csv"
    ].iloc[0]
    assert row["compatibility_path"] == "output/tables/weekly_claim_readiness.csv"
    assert row["canonical_exists"]
    assert row["compatibility_exists"]
    assert not table["path"].str.contains("internal").any()
    assert not table["path"].str.contains(".DS_Store", regex=False).any()


def test_validate_public_boundary_blocks_public_internal_artifacts(tmp_path) -> None:
    (tmp_path / ".gitignore").write_text(
        "do/\n.env\noutput/**\ndata/raw/**\ndata/clean/**\ndata/interim/**\n",
        encoding="utf-8",
    )
    tables = tmp_path / "output" / "tables"
    internal = tmp_path / "output" / "internal" / "tables"
    tables.mkdir(parents=True)
    internal.mkdir(parents=True)
    (tables / "internal_private.csv").write_text("x\n1\n", encoding="utf-8")
    (internal / "private.csv").write_text("x\n1\n", encoding="utf-8")

    report = validate_public_boundary(tmp_path)

    assert report["status"] == "failed"
    assert any("private artifact is in public output directory" in error for error in report["errors"])
    assert any("output/internal exists locally" in warning for warning in report["warnings"])


def test_validate_public_boundary_accepts_ignored_internal_tree(tmp_path) -> None:
    (tmp_path / ".gitignore").write_text(
        "do/\n.env\noutput/**\ndata/raw/**\ndata/clean/**\ndata/interim/**\n",
        encoding="utf-8",
    )
    reports = tmp_path / "output" / "reports"
    internal = tmp_path / "output" / "internal" / "reports"
    reports.mkdir(parents=True)
    internal.mkdir(parents=True)
    (reports / "public.md").write_text("public\n", encoding="utf-8")
    (internal / "internal_private.md").write_text("private\n", encoding="utf-8")

    report = validate_public_boundary(tmp_path)

    assert report["status"] == "ok"
    assert report["errors"] == []


def test_public_bundle_excludes_private_and_raw_paths(tmp_path) -> None:
    (tmp_path / ".gitignore").write_text(
        "do/\n.env\noutput/**\ndata/raw/**\ndata/clean/**\ndata/interim/**\n",
        encoding="utf-8",
    )
    for directory in [
        "src/liqsub",
        "tests",
        "config",
        "docs",
        "do",
        "data/raw/fred",
        "data/clean",
        "data/manual",
        "output/tables",
        "output/manifests",
        "output/internal/reports",
        "output/bundles",
    ]:
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='liqsub'\n", encoding="utf-8")
    (tmp_path / ".env").write_text("PLACEHOLDER=1\n", encoding="utf-8")
    (tmp_path / "src" / "liqsub" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_public.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (tmp_path / "do" / "private.md").write_text("private\n", encoding="utf-8")
    (tmp_path / "data" / "raw" / "fred" / "series.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "data" / "clean" / "panel.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "data" / "manual" / "context.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "output" / "tables" / "monthly_candidate_table.csv").write_text(
        "sample,outcome,event_month,candidate_status,priority,direction_pattern,"
        "external_source_count,required_next_step\n",
        encoding="utf-8",
    )
    (tmp_path / "output" / "manifests" / "latest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "output" / "manifests" / "run_manifest_20260101T000000Z.json").write_text("{}", encoding="utf-8")
    (tmp_path / "output" / "internal" / "reports" / "private.md").write_text("private\n", encoding="utf-8")
    (tmp_path / "output" / "bundles" / "old.zip").write_text("old\n", encoding="utf-8")

    files = public_bundle_file_list(tmp_path)

    assert "README.md" in files
    assert "src/liqsub/__init__.py" in files
    assert "tests/test_public.py" in files
    assert "data/clean/panel.csv" in files
    assert "data/manual/context.csv" in files
    assert "output/tables/monthly_candidate_table.csv" in files
    assert "output/manifests/latest.json" in files
    assert ".env" not in files
    assert "do/private.md" not in files
    assert "data/raw/fred/series.csv" not in files
    assert "output/internal/reports/private.md" not in files
    assert "output/bundles/old.zip" not in files
    assert "output/manifests/run_manifest_20260101T000000Z.json" not in files


def test_write_public_bundle_creates_zip_with_manifest(tmp_path) -> None:
    (tmp_path / ".gitignore").write_text(
        "do/\n.env\noutput/**\ndata/raw/**\ndata/clean/**\ndata/interim/**\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "liqsub").mkdir(parents=True)
    (tmp_path / "output" / "tables").mkdir(parents=True)
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")
    (tmp_path / "src" / "liqsub" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "output" / "tables" / "monthly_candidate_table.csv").write_text(
        "sample,outcome,event_month,candidate_status,priority,direction_pattern,"
        "external_source_count,required_next_step\n",
        encoding="utf-8",
    )

    result = write_public_bundle(tmp_path)

    assert result["status"] == "ok"
    bundle = tmp_path / result["bundle"]
    assert bundle.exists()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "README.md" in names
    assert "output/tables/monthly_candidate_table.csv" in names
    assert "PUBLIC_BUNDLE_MANIFEST.json" in names
    assert not any(name.startswith("output/internal/") for name in names)

    smoke = smoke_test_public_bundle(bundle)
    assert smoke["status"] == "ok"
    assert smoke["errors"] == []


def test_smoke_test_public_bundle_rejects_forbidden_member(tmp_path) -> None:
    bundle = tmp_path / "bad.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("PUBLIC_BUNDLE_MANIFEST.json", "{}")
        archive.writestr("do/private.md", "private")

    report = smoke_test_public_bundle(bundle)

    assert report["status"] == "failed"
    assert any("forbidden path" in error for error in report["errors"])


def test_smoke_test_public_bundle_rejects_manifest_hash_mismatch(tmp_path) -> None:
    bundle = tmp_path / "bad_hash.zip"
    manifest = {
        "created_at_utc": "20260101T000000Z",
        "file_count": 1,
        "files": [
            {
                "path": "README.md",
                "exists": True,
                "bytes": 7,
                "sha256": "0" * 64,
            }
        ],
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("README.md", "readme\n")
        archive.writestr("PUBLIC_BUNDLE_MANIFEST.json", json.dumps(manifest))

    report = smoke_test_public_bundle(bundle)

    assert report["status"] == "failed"
    assert any("hash mismatch" in error for error in report["errors"])


def test_release_public_backend_blocks_on_config_errors(tmp_path) -> None:
    result = release_public_backend(tmp_path)

    assert result["status"] == "blocked"
    assert any("missing project config" in error for error in result["errors"])


def test_release_public_backend_runs_full_public_sequence(tmp_path, monkeypatch) -> None:
    class ConfigReport:
        status = "ok"
        errors: tuple[str, ...] = ()
        warnings: tuple[str, ...] = ()

    class SchemaReport:
        status = "ok"
        checked_files = 1
        errors: tuple[str, ...] = ()
        warnings: tuple[str, ...] = ()

    (tmp_path / ".gitignore").write_text(
        "do/\n.env\noutput/**\ndata/raw/**\ndata/clean/**\ndata/interim/**\n",
        encoding="utf-8",
    )
    (tmp_path / "output" / "tables").mkdir(parents=True)
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")
    (tmp_path / "output" / "tables" / "public_output_aliases.csv").write_text(
        "canonical_path,compatibility_path,relationship,status,canonical_exists,compatibility_exists,removal_policy\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("liqsub.artifacts.validate_project_config", lambda root: ConfigReport())
    monkeypatch.setattr("liqsub.artifacts.validate_public_output_schemas", lambda root: SchemaReport())
    monkeypatch.setattr(
        "liqsub.artifacts.validate_public_artifact_contract",
        lambda root: {"status": "ok", "errors": [], "warnings": [], "checked_files": 1},
    )
    monkeypatch.setattr(
        "liqsub.artifacts.validate_rebuild_lineage",
        lambda root: {"status": "ok", "errors": [], "warnings": [], "inputs": 1, "outputs": 1},
    )
    monkeypatch.setattr(
        "liqsub.artifacts.validate_manifest_integrity",
        lambda root: {"status": "ok", "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        "liqsub.artifacts.validate_backend_inputs",
        lambda root: {
            "status": "ok",
            "errors": [],
            "warnings": [],
            "checked_files": 1,
            "missing_required": 0,
            "stale_inputs": 0,
        },
    )

    result = release_public_backend(tmp_path)

    assert result["status"] == "ok"
    assert result["bundle"].startswith("output/bundles/liqsub_public_backend_")
    assert result["release_notes"] == "output/reports/RELEASE_NOTES.md"
    assert [step["step"] for step in result["steps"]] == [
        "validate_config",
        "validate_backend_inputs",
        "write_backend_input_inventory",
        "write_source_cache_manifest",
        "write_source_refresh_status",
        "validate_output_schemas",
        "validate_public_artifact_contract",
        "validate_rebuild_lineage",
        "validate_public_boundary",
        "write_release_notes",
        "write_output_tiers",
        "write_manifest",
        "validate_manifest_integrity",
        "write_public_bundle",
        "smoke_test_public_bundle",
    ]


def test_release_public_backend_check_only_does_not_write_bundle(tmp_path, monkeypatch) -> None:
    class ConfigReport:
        status = "ok"
        errors: tuple[str, ...] = ()
        warnings: tuple[str, ...] = ()

    class SchemaReport:
        status = "ok"
        checked_files = 1
        errors: tuple[str, ...] = ()
        warnings: tuple[str, ...] = ()

    (tmp_path / ".gitignore").write_text(
        "do/\n.env\noutput/**\ndata/raw/**\ndata/clean/**\ndata/interim/**\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")
    monkeypatch.setattr("liqsub.artifacts.validate_project_config", lambda root: ConfigReport())
    monkeypatch.setattr("liqsub.artifacts.validate_public_output_schemas", lambda root: SchemaReport())
    monkeypatch.setattr(
        "liqsub.artifacts.validate_public_artifact_contract",
        lambda root: {"status": "ok", "errors": [], "warnings": [], "checked_files": 1},
    )
    monkeypatch.setattr(
        "liqsub.artifacts.validate_rebuild_lineage",
        lambda root: {"status": "ok", "errors": [], "warnings": [], "inputs": 1, "outputs": 1},
    )
    monkeypatch.setattr(
        "liqsub.artifacts.validate_backend_inputs",
        lambda root: {
            "status": "ok",
            "errors": [],
            "warnings": [],
            "checked_files": 1,
            "missing_required": 0,
            "stale_inputs": 0,
        },
    )

    result = release_public_backend(tmp_path, check_only=True)

    assert result["status"] == "ok"
    assert result["check_only"] is True
    assert "bundle" not in result
    assert not (tmp_path / "output" / "bundles").exists()
    assert not (tmp_path / "output" / "tables" / "public_output_aliases.csv").exists()
    assert [step["step"] for step in result["steps"]] == [
        "validate_config",
        "validate_backend_inputs",
        "validate_output_schemas",
        "validate_public_artifact_contract",
        "validate_rebuild_lineage",
        "validate_public_boundary",
        "plan_public_bundle",
    ]


def test_write_release_notes_summarizes_backend_status(tmp_path) -> None:
    clean = tmp_path / "data" / "clean"
    manifests = tmp_path / "output" / "manifests"
    clean.mkdir(parents=True)
    manifests.mkdir(parents=True)
    (clean / "monthly_liquidity_substitution_panel.csv").write_text(
        "month,value\n2024-01-01,1\n",
        encoding="utf-8",
    )
    (clean / "weekly_liquidity_substitution_panel.csv").write_text(
        "week,value\n2024-01-03,1\n",
        encoding="utf-8",
    )
    (manifests / "latest.json").write_text("{}", encoding="utf-8")

    result = write_release_notes(tmp_path)
    text = (tmp_path / result["path"]).read_text(encoding="utf-8")

    assert result["status"] == "ok"
    assert "reproducible backend diagnostics packet" in text
    assert "Monthly panel: 1 rows" in text
    assert "Weekly panel: 1 rows" in text


def test_backend_input_inventory_reports_missing_required_inputs(tmp_path) -> None:
    inventory = backend_input_inventory(tmp_path)
    report = validate_backend_inputs(tmp_path)

    assert "input_path" in inventory.columns
    assert "status" in inventory.columns
    assert "freshness_status" in inventory.columns
    assert report["status"] == "failed"
    assert report["missing_required"] > 0
    assert any("missing required backend input" in error for error in report["errors"])


def test_write_backend_input_inventory_writes_public_table(tmp_path) -> None:
    result = write_backend_input_inventory(tmp_path)
    path = tmp_path / result["path"]

    assert result["status"] == "ok"
    assert path.exists()
    assert result["rows"] > 0
    assert "input_path" in path.read_text(encoding="utf-8").splitlines()[0]


def test_write_source_cache_manifest_writes_sanitized_public_table(tmp_path) -> None:
    result = write_source_cache_manifest(tmp_path)
    path = tmp_path / result["path"]

    assert result["status"] == "ok"
    assert path.exists()
    assert result["rows"] > 0
    text = path.read_text(encoding="utf-8")
    assert "input_path,source_family,source_id" in text.splitlines()[0]
    assert "/Users" + "/" not in text


def test_write_source_refresh_status_reports_fetch_fallback(tmp_path) -> None:
    fred_dir = tmp_path / "data" / "raw" / "fred"
    fred_dir.mkdir(parents=True)
    (fred_dir / "bill_yield_1mo__DGS1MO.csv").write_text(
        "observation_date,DGS1MO\n2026-04-29,3.87\n",
        encoding="utf-8",
    )
    (fred_dir / "manifest.csv").write_text(
        "source,series_key,series_id,path,status,rows,last_observation,cache_rows,cache_last_observation\n"
        "FRED,bill_yield_1mo,DGS1MO,fred/bill_yield_1mo__DGS1MO.csv,"
        "kept_existing_cache_after_failed_refresh: timeout,6188,2026-04-29,6188,2026-04-29\n",
        encoding="utf-8",
    )

    result = write_source_refresh_status(tmp_path)
    table = source_refresh_status(tmp_path)
    row = table.loc[table["input_path"] == "data/raw/fred/bill_yield_1mo__DGS1MO.csv"].iloc[0]

    assert result["status"] == "ok"
    assert result["path"] == "output/tables/source_refresh_status.csv"
    assert row["refresh_status"] == "ok_live_refresh_failed_cache_valid"
    assert "retry live refresh" in row["required_action"]


def test_backend_input_inventory_reports_fresh_existing_file(tmp_path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "project.yaml").write_text(
        "source_freshness:\n  fred:\n    mtime_days: 30\n    observation_days: 90\n",
        encoding="utf-8",
    )
    path = tmp_path / "data" / "raw" / "fred"
    path.mkdir(parents=True)
    (path / "reserves__WRESBAL.csv").write_text("date,value\n2024-01-01,1\n", encoding="utf-8")

    inventory = backend_input_inventory(tmp_path)
    row = inventory.loc[inventory["input_path"] == "data/raw/fred/reserves__WRESBAL.csv"].iloc[0]

    assert row["exists"]
    assert row["freshness_status"] == "fresh"
    assert float(row["freshness_days"]) == 30
    assert int(row["valid_observation_rows"]) == 1
    assert row["last_valid_observation"] == "2024-01-01"
    assert float(row["observation_freshness_days"]) == 90


def test_backend_input_inventory_profiles_fred_observation_date(tmp_path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "project.yaml").write_text(
        "source_freshness:\n  fred:\n    mtime_days: 30\n    observation_days: 90\n",
        encoding="utf-8",
    )
    path = tmp_path / "data" / "raw" / "fred"
    path.mkdir(parents=True)
    (path / "bill_yield_1mo__DGS1MO.csv").write_text(
        "observation_date,DGS1MO\n2026-04-29,3.87\n",
        encoding="utf-8",
    )

    inventory = backend_input_inventory(tmp_path)
    row = inventory.loc[inventory["input_path"] == "data/raw/fred/bill_yield_1mo__DGS1MO.csv"].iloc[0]

    assert int(row["valid_observation_rows"]) == 1
    assert row["last_valid_observation"] == "2026-04-29"


def test_backend_input_inventory_does_not_mark_discontinued_fred_series_stale(tmp_path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "project.yaml").write_text(
        "source_freshness:\n  fred:\n    mtime_days: 30\n    observation_days: 90\n",
        encoding="utf-8",
    )
    path = tmp_path / "data" / "raw" / "fred"
    path.mkdir(parents=True)
    (path / "institutional_mmf_assets__WIMFNS.csv").write_text(
        "observation_date,WIMFNS\n2021-02-01,2882.8\n",
        encoding="utf-8",
    )

    inventory = backend_input_inventory(tmp_path)
    row = inventory.loc[
        inventory["input_path"] == "data/raw/fred/institutional_mmf_assets__WIMFNS.csv"
    ].iloc[0]

    assert row["last_valid_observation"] == "2021-02-01"
    assert row["observation_freshness_status"] == "fresh"


def test_backend_input_inventory_profiles_dts_open_today_fallback(tmp_path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "project.yaml").write_text(
        "source_freshness:\n  fiscaldata:\n    mtime_days: 30\n    observation_days: 45\n",
        encoding="utf-8",
    )
    path = tmp_path / "data" / "raw" / "fiscaldata"
    path.mkdir(parents=True)
    (path / "dts_operating_cash_balance.csv").write_text(
        "record_date,account_type,close_today_bal,open_today_bal,open_month_bal\n"
        "2026-04-29,Treasury General Account (TGA) Closing Balance,,988102,988102\n",
        encoding="utf-8",
    )

    inventory = backend_input_inventory(tmp_path)
    row = inventory.loc[
        inventory["input_path"] == "data/raw/fiscaldata/dts_operating_cash_balance.csv"
    ].iloc[0]

    assert int(row["valid_observation_rows"]) == 1
    assert row["last_valid_observation"] == "2026-04-29"


def test_validate_backend_inputs_blocks_required_file_with_no_valid_observations(tmp_path) -> None:
    path = tmp_path / "data" / "raw" / "fred"
    path.mkdir(parents=True)
    (path / "reserves__WRESBAL.csv").write_text("date,value\n2024-01-01,\n", encoding="utf-8")

    report = validate_backend_inputs(tmp_path)

    assert report["status"] == "failed"
    assert any("zero valid observations" in error for error in report["errors"])


def test_validate_public_artifact_contract_blocks_missing_clean_panel(tmp_path) -> None:
    report = validate_public_artifact_contract(tmp_path)

    assert report["status"] == "failed"
    assert any("missing stable clean panel" in error for error in report["errors"])


def test_validate_public_artifact_contract_blocks_duplicate_week(tmp_path) -> None:
    clean = tmp_path / "data" / "clean"
    tables = tmp_path / "output" / "tables"
    reports = tmp_path / "output" / "reports"
    clean.mkdir(parents=True)
    tables.mkdir(parents=True)
    reports.mkdir(parents=True)
    (clean / "monthly_liquidity_substitution_panel.csv").write_text(
        "month,gross_bill_issuance,bill_share,coupon_issuance,deposits,total_mmf_assets,"
        "retail_mmf_assets,institutional_mmf_assets,mmf_treasury_holdings,on_rrp,reserves,tga,on_rrp_regime\n"
        + "\n".join(f"2020-{month:02d}-01,1,0.5,1,1,1,1,1,1,1,1,1,scarce" for month in range(1, 13))
        + "\n",
        encoding="utf-8",
    )
    (clean / "monthly_liquidity_substitution_panel_analysis.csv").write_text(
        "month,gross_bill_issuance,bill_share,coupon_issuance,deposits,total_mmf_assets,"
        "retail_mmf_assets,institutional_mmf_assets,mmf_treasury_holdings,on_rrp,reserves,tga,on_rrp_regime,"
        "d_deposits,d_on_rrp,d_reserves,d_tga,isolated_large_positive_bill_shock\n"
        + "\n".join(f"2020-{month:02d}-01,1,0.5,1,1,1,1,1,1,1,1,1,scarce,0,0,0,0,0" for month in range(1, 13))
        + "\n",
        encoding="utf-8",
    )
    weekly_header = (
        "week,tga,reserves,on_rrp,deposits,gross_bill_settlement_100b,coupon_settlement_100b,"
        "bill_share,d_tga,d_reserves,d_on_rrp,d_deposits,debt_limit_window,tax_week,on_rrp_regime_pre\n"
    )
    (clean / "weekly_liquidity_substitution_panel.csv").write_text(
        weekly_header
        + "2024-01-03,1,1,1,1,1,1,0.5,0,0,0,0,0,0,scarce\n"
        + "2024-01-03,1,1,1,1,1,1,0.5,0,0,0,0,0,0,scarce\n",
        encoding="utf-8",
    )

    report = validate_public_artifact_contract(tmp_path)

    assert report["status"] == "failed"
    assert any("duplicate dates" in error for error in report["errors"])


def test_validate_manifest_integrity_records_full_backend_input_set(tmp_path) -> None:
    (tmp_path / "output" / "manifests").mkdir(parents=True)
    payload = {"inputs": [], "outputs": [], "output_tiers": []}
    (tmp_path / "output" / "manifests" / "latest.json").write_text(json.dumps(payload), encoding="utf-8")

    report = validate_manifest_integrity(tmp_path)

    assert report["status"] == "failed"
    assert any("backend inputs" in error for error in report["errors"])


def test_rebuild_lineage_blocks_changed_input_hash(tmp_path) -> None:
    raw = tmp_path / "data" / "raw" / "buycurve"
    raw.mkdir(parents=True)
    source = raw / "monthly_issuance_maturity_panel.csv"
    source.write_text("month,value\n2024-01-01,1\n", encoding="utf-8")

    result = write_rebuild_lineage(tmp_path)
    source.write_text("month,value\n2024-01-01,2\n", encoding="utf-8")
    report = validate_rebuild_lineage(tmp_path)

    assert result["status"] == "ok"
    assert report["status"] == "failed"
    assert any("input hash changed" in error for error in report["errors"])


def test_rebuild_lineage_blocks_changed_config_hash(tmp_path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    project = config / "project.yaml"
    project.write_text("project: liqsub\n", encoding="utf-8")

    result = write_rebuild_lineage(tmp_path)
    project.write_text("project: liqsub\nversion: changed\n", encoding="utf-8")
    report = validate_rebuild_lineage(tmp_path)

    assert result["status"] == "ok"
    assert report["status"] == "failed"
    assert any("configuration hash changed" in error for error in report["errors"])
