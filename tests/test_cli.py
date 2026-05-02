from __future__ import annotations

import json
import zipfile
from pathlib import Path

from liqsub.cli import main


def test_write_internal_reports_is_explicit_cli_command(tmp_path, capsys) -> None:
    rc = main(["--root", str(tmp_path), "write-internal-reports"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert "internal_evidence_gate_summary" in payload["outputs"]
    assert payload["outputs"]["internal_evidence_gate_summary"].startswith("output/internal/")
    assert not payload["outputs"]["internal_evidence_gate_summary"].startswith("/")
    assert (tmp_path / "output" / "internal" / "tables" / "internal_evidence_gate_summary.csv").exists()
    assert not (tmp_path / "output" / "tables" / "internal_evidence_gate_summary.csv").exists()


def test_validate_output_schemas_reports_headerless_csv(tmp_path, capsys) -> None:
    tables = tmp_path / "output" / "tables"
    tables.mkdir(parents=True)
    (tables / "bad.csv").write_text("", encoding="utf-8")

    rc = main(["--root", str(tmp_path), "validate-output-schemas"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 1
    assert not (tables / "public_output_aliases.csv").exists()
    assert payload["status"] == "failed"
    assert any("missing header row" in error for error in payload["errors"])
    assert any("missing public output schema contract" in error for error in payload["errors"])


def test_validate_public_boundary_cli_reports_status(tmp_path, capsys) -> None:
    (tmp_path / ".gitignore").write_text(
        "do/\n.env\noutput/**\ndata/raw/**\ndata/clean/**\ndata/interim/**\n",
        encoding="utf-8",
    )

    rc = main(["--root", str(tmp_path), "validate-public-boundary"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["errors"] == []


def test_validate_rebuild_lineage_cli_blocks_when_missing(tmp_path, capsys) -> None:
    rc = main(["--root", str(tmp_path), "validate-rebuild-lineage"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 1
    assert payload["status"] == "failed"
    assert any("missing rebuild lineage" in error for error in payload["errors"])


def test_write_public_bundle_cli_reports_archive(tmp_path, capsys) -> None:
    (tmp_path / ".gitignore").write_text(
        "do/\n.env\noutput/**\ndata/raw/**\ndata/clean/**\ndata/interim/**\n",
        encoding="utf-8",
    )
    (tmp_path / "output" / "tables").mkdir(parents=True)
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")
    (tmp_path / "output" / "tables" / "public.csv").write_text("x\n1\n", encoding="utf-8")

    rc = main(["--root", str(tmp_path), "write-public-bundle"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["bundle"].startswith("output/bundles/liqsub_public_backend_")
    assert (tmp_path / payload["bundle"]).exists()


def test_release_public_cli_blocks_when_config_missing(tmp_path, capsys) -> None:
    rc = main(["--root", str(tmp_path), "release-public"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 1
    assert payload["status"] == "blocked"
    assert any("missing project config" in error for error in payload["errors"])


def test_release_public_check_only_cli_blocks_when_config_missing(tmp_path, capsys) -> None:
    rc = main(["--root", str(tmp_path), "release-public", "--check-only"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 1
    assert payload["status"] == "blocked"
    assert any("missing project config" in error for error in payload["errors"])


def test_write_release_notes_cli_reports_path(tmp_path, capsys) -> None:
    (tmp_path / "data" / "clean").mkdir(parents=True)
    (tmp_path / "data" / "clean" / "monthly_liquidity_substitution_panel.csv").write_text(
        "month,value\n2024-01-01,1\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "clean" / "weekly_liquidity_substitution_panel.csv").write_text(
        "week,value\n2024-01-03,1\n",
        encoding="utf-8",
    )

    rc = main(["--root", str(tmp_path), "write-release-notes"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["path"] == "output/reports/RELEASE_NOTES.md"
    assert (tmp_path / payload["path"]).exists()


def test_validate_inputs_cli_reports_missing_required(tmp_path, capsys) -> None:
    rc = main(["--root", str(tmp_path), "validate-inputs"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 1
    assert payload["status"] == "failed"
    assert payload["missing_required"] > 0


def test_write_input_inventory_cli_reports_path(tmp_path, capsys) -> None:
    rc = main(["--root", str(tmp_path), "write-input-inventory"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["path"] == "output/tables/backend_input_inventory.csv"
    assert (tmp_path / payload["path"]).exists()


def test_write_source_cache_manifest_cli_reports_path(tmp_path, capsys) -> None:
    rc = main(["--root", str(tmp_path), "write-source-cache-manifest"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["path"] == "output/tables/source_cache_manifest.csv"
    assert (tmp_path / payload["path"]).exists()


def test_write_source_refresh_status_cli_reports_path(tmp_path, capsys) -> None:
    rc = main(["--root", str(tmp_path), "write-source-refresh-status"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["path"] == "output/tables/source_refresh_status.csv"
    assert (tmp_path / payload["path"]).exists()


def test_smoke_public_bundle_cli_reports_bad_archive(tmp_path, capsys) -> None:
    bundle = tmp_path / "bad.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("PUBLIC_BUNDLE_MANIFEST.json", "{}")
        archive.writestr("output/internal/private.md", "private")

    rc = main(["--root", str(tmp_path), "smoke-public-bundle", str(bundle)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 1
    assert payload["status"] == "failed"
    assert any("forbidden path" in error for error in payload["errors"])


def test_rebuild_public_cli_blocks_when_config_missing(tmp_path, capsys) -> None:
    rc = main(["--root", str(tmp_path), "rebuild-public"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 1
    assert payload["status"] == "blocked"
    assert payload["steps"][0]["step"] == "validate_config"
    assert any("missing project config" in error for error in payload["errors"])


def test_rebuild_public_check_only_cli_uses_release_check(tmp_path, capsys, monkeypatch) -> None:
    class ConfigReport:
        status = "ok"
        errors: tuple[str, ...] = ()
        warnings: tuple[str, ...] = ()

    class BuycurveReport:
        status = "ok"
        errors: tuple[str, ...] = ()
        rows = 12
        path = Path("data/raw/buycurve/monthly_issuance_maturity_panel.csv")

    class TgarefillReport:
        status = "ok"
        errors: tuple[str, ...] = ()
        rows_by_export = {"master_weekly_panel": 10}

    monkeypatch.setattr("liqsub.cli.validate_project_config", lambda root: ConfigReport())
    monkeypatch.setattr(
        "liqsub.cli.validate_backend_inputs",
        lambda root: {
            "status": "ok",
            "errors": [],
            "warnings": [],
            "checked_files": 1,
            "missing_required": 0,
            "stale_inputs": 0,
        },
    )
    monkeypatch.setattr("liqsub.cli.validate_buycurve_panel", lambda path: BuycurveReport())
    monkeypatch.setattr("liqsub.cli.validate_tgarefill_exports", lambda root: TgarefillReport())
    monkeypatch.setattr(
        "liqsub.cli.release_public_backend",
        lambda root, destination=None, check_only=False: {
            "status": "ok",
            "check_only": check_only,
            "file_count": 1,
            "steps": [],
            "warnings": [],
        },
    )

    rc = main(["--root", str(tmp_path), "rebuild-public", "--check-only"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["check_only"] is True
    assert [step["step"] for step in payload["steps"]] == [
        "validate_config",
        "validate_inputs",
        "check_buycurve",
        "check_tgarefill",
        "release_public_check",
    ]


def test_rebuild_public_cli_runs_local_pipeline(tmp_path, capsys, monkeypatch) -> None:
    class ConfigReport:
        status = "ok"
        errors: tuple[str, ...] = ()
        warnings: tuple[str, ...] = ()

    class BuycurveReport:
        status = "ok"
        errors: tuple[str, ...] = ()
        rows = 12
        path = Path("data/raw/buycurve/monthly_issuance_maturity_panel.csv")

    class TgarefillReport:
        status = "ok"
        errors: tuple[str, ...] = ()
        rows_by_export = {"master_weekly_panel": 10}

    monkeypatch.setattr("liqsub.cli.validate_project_config", lambda root: ConfigReport())
    monkeypatch.setattr(
        "liqsub.cli.validate_backend_inputs",
        lambda root: {
            "status": "ok",
            "errors": [],
            "warnings": [],
            "checked_files": 1,
            "missing_required": 0,
            "stale_inputs": 0,
        },
    )
    monkeypatch.setattr("liqsub.cli.validate_buycurve_panel", lambda path: BuycurveReport())
    monkeypatch.setattr("liqsub.cli.write_monthly_outputs", lambda root: {"status": "ok", "outputs": {"panel": "x"}})
    monkeypatch.setattr("liqsub.cli.write_monthly_analysis", lambda root: {"monthly": "ok"})
    monkeypatch.setattr("liqsub.cli.validate_tgarefill_exports", lambda root: TgarefillReport())
    monkeypatch.setattr("liqsub.cli.write_weekly_outputs", lambda root: {"status": "ok", "outputs": {"panel": "y"}})
    monkeypatch.setattr("liqsub.cli.write_weekly_analysis", lambda root: {"weekly": "ok"})
    monkeypatch.setattr("liqsub.cli.write_backend_input_inventory", lambda root: {"status": "ok", "path": "output/tables/backend_input_inventory.csv", "rows": 1})
    monkeypatch.setattr("liqsub.cli.write_source_cache_manifest", lambda root: {"status": "ok", "path": "output/tables/source_cache_manifest.csv", "rows": 1})
    monkeypatch.setattr("liqsub.cli.write_source_refresh_status", lambda root: {"status": "ok", "path": "output/tables/source_refresh_status.csv", "rows": 1})
    monkeypatch.setattr("liqsub.cli.write_public_output_aliases", lambda root: {"status": "ok", "path": "output/tables/public_output_aliases.csv", "rows": 1})
    monkeypatch.setattr("liqsub.cli.write_rebuild_lineage", lambda root: {"status": "ok", "path": "output/manifests/latest_successful_rebuild.json", "inputs": 1, "outputs": 1})
    monkeypatch.setattr(
        "liqsub.cli.release_public_backend",
        lambda root, destination=None: {
            "status": "ok",
            "bundle": "output/bundles/test.zip",
            "release_notes": "output/reports/RELEASE_NOTES.md",
            "steps": [],
            "warnings": [],
        },
    )

    rc = main(["--root", str(tmp_path), "rebuild-public"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["bundle"] == "output/bundles/test.zip"
    assert [step["step"] for step in payload["steps"]] == [
        "validate_config",
        "validate_inputs",
        "check_buycurve",
        "build_monthly",
        "analyze_monthly",
        "check_tgarefill",
        "build_weekly",
        "analyze_weekly",
        "write_backend_input_inventory",
        "write_source_cache_manifest",
        "write_source_refresh_status",
        "write_public_output_aliases",
        "write_rebuild_lineage",
        "release_public",
    ]
