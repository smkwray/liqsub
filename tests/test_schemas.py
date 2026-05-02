from __future__ import annotations

from liqsub.schemas import validate_public_output_schemas


def test_validate_public_output_schemas_accepts_headered_zero_row_outputs(tmp_path) -> None:
    tables = tmp_path / "output" / "tables"
    tables.mkdir(parents=True)
    (tables / "monthly_candidate_table.csv").write_text(
        "sample,outcome,event_month,candidate_status,priority,direction_pattern,"
        "external_source_count,required_next_step\n",
        encoding="utf-8",
    )
    (tables / "monthly_external_calendar_evidence.csv").write_text(
        "event_month,evidence_status,evidence_topic,official_source,source_url,calendar_note\n",
        encoding="utf-8",
    )
    (tables / "monthly_strong_candidate_narratives.csv").write_text(
        "sample,outcome,event_month,priority,narrative_status\n",
        encoding="utf-8",
    )

    report = validate_public_output_schemas(tmp_path)

    assert report.status == "ok"
    assert report.checked_files == 3


def test_validate_public_output_schemas_rejects_missing_required_columns(tmp_path) -> None:
    tables = tmp_path / "output" / "tables"
    tables.mkdir(parents=True)
    (tables / "evidence_gate_summary.csv").write_text("design_path,status\n", encoding="utf-8")

    report = validate_public_output_schemas(tmp_path)

    assert report.status == "failed"
    assert any("evidence_gate_summary.csv" in error for error in report.errors)


def test_validate_public_output_schemas_rejects_uncontracted_public_csv(tmp_path) -> None:
    tables = tmp_path / "output" / "tables"
    tables.mkdir(parents=True)
    (tables / "new_public_table.csv").write_text("column\nvalue\n", encoding="utf-8")

    report = validate_public_output_schemas(tmp_path)

    assert report.status == "failed"
    assert any("missing public output schema contract" in error for error in report.errors)


def test_validate_public_output_schemas_rejects_bad_dates(tmp_path) -> None:
    tables = tmp_path / "output" / "tables"
    tables.mkdir(parents=True)
    (tables / "monthly_bill_shock_events.csv").write_text(
        "month,gross_bill_issuance,bill_supply_shock_resid_100b,bill_share\n"
        "not-a-date,100,1,0.5\n",
        encoding="utf-8",
    )

    report = validate_public_output_schemas(tmp_path)

    assert report.status == "failed"
    assert any("unparsable dates" in error for error in report.errors)


def test_validate_public_output_schemas_rejects_invalid_shares(tmp_path) -> None:
    tables = tmp_path / "output" / "tables"
    tables.mkdir(parents=True)
    (tables / "monthly_coverage_qa.csv").write_text(
        "column,status,source_family,unit,coverage_share\nvalue,present,test,unit,1.5\n",
        encoding="utf-8",
    )

    report = validate_public_output_schemas(tmp_path)

    assert report.status == "failed"
    assert any("outside [0, 1]" in error for error in report.errors)


def test_validate_public_output_schemas_rejects_negative_counts(tmp_path) -> None:
    tables = tmp_path / "output" / "tables"
    tables.mkdir(parents=True)
    (tables / "monthly_descriptive_summary.csv").write_text(
        "column,nobs,mean,std,min,max\nvalue,-1,0,0,0,0\n",
        encoding="utf-8",
    )

    report = validate_public_output_schemas(tmp_path)

    assert report.status == "failed"
    assert any("negative values" in error for error in report.errors)


def test_validate_public_output_schemas_rejects_blank_required_text(tmp_path) -> None:
    tables = tmp_path / "output" / "tables"
    tables.mkdir(parents=True)
    (tables / "evidence_gate_summary.csv").write_text(
        "design_path,status,claim_use,evidence_basis,primary_artifacts,binding_blockers,next_design_step\n"
        ",blocked,use,basis,artifact,blocker,next\n",
        encoding="utf-8",
    )

    report = validate_public_output_schemas(tmp_path)

    assert report.status == "failed"
    assert any("blank values" in error for error in report.errors)
