from __future__ import annotations

from pathlib import Path

from liqsub.config import project_as_of_date, validate_project_config, weekly_large_rebuild_randomization_p_max


def test_project_config_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    report = validate_project_config(root)
    assert report.status == "ok", report.errors
    assert project_as_of_date(root).isoformat() == "2026-05-01"
    assert weekly_large_rebuild_randomization_p_max(root) == 0.10
