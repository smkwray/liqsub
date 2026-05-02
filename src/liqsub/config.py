from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from liqsub.paths import find_project_root

DEFAULT_WEEKLY_LARGE_REBUILD_RANDOMIZATION_P_MAX = 0.10
DEFAULT_SOURCE_FRESHNESS_POLICY = {
    "mtime_days": 90,
    "observation_days": None,
}


class ConfigError(ValueError):
    """Raised when project configuration is internally inconsistent."""


@dataclass(frozen=True)
class ValidationReport:
    status: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ConfigError("\n".join(self.errors))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected YAML mapping: {path}")
    return data


def source_contract_paths(root: Path | None = None) -> list[Path]:
    project_root = root or find_project_root()
    return sorted((project_root / "config" / "source_contracts").glob("*.yaml"))


def load_project_config(root: Path | None = None) -> dict[str, Any]:
    project_root = root or find_project_root()
    path = project_root / "config" / "project.yaml"
    if not path.exists():
        return {}
    return load_yaml(path)


def weekly_large_rebuild_randomization_p_max(root: Path | None = None) -> float:
    config = load_project_config(root)
    raw_value = (
        config.get("analysis", {})
        .get("weekly_large_rebuild", {})
        .get("randomization_p_max", DEFAULT_WEEKLY_LARGE_REBUILD_RANDOMIZATION_P_MAX)
    )
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("analysis.weekly_large_rebuild.randomization_p_max must be numeric") from exc
    if not 0 <= value <= 1:
        raise ConfigError("analysis.weekly_large_rebuild.randomization_p_max must be between 0 and 1")
    return value


def project_as_of_date(root: Path | None = None) -> date:
    config = load_project_config(root)
    raw_value = config.get("as_of_date")
    if raw_value is None:
        return date.today()
    if isinstance(raw_value, date):
        return raw_value
    if isinstance(raw_value, str):
        try:
            return date.fromisoformat(raw_value)
        except ValueError as exc:
            raise ConfigError("as_of_date must use YYYY-MM-DD format") from exc
    raise ConfigError("as_of_date must use YYYY-MM-DD format")


def source_freshness_policy(root: Path | None, source_family: str) -> dict[str, int | None]:
    config = load_project_config(root)
    policies = config.get("source_freshness", {})
    if not isinstance(policies, dict):
        return dict(DEFAULT_SOURCE_FRESHNESS_POLICY)
    default = policies.get("default", DEFAULT_SOURCE_FRESHNESS_POLICY)
    if not isinstance(default, dict):
        default = DEFAULT_SOURCE_FRESHNESS_POLICY
    family_policy = policies.get(source_family, {})
    if not isinstance(family_policy, dict):
        family_policy = {}
    merged = {**DEFAULT_SOURCE_FRESHNESS_POLICY, **default, **family_policy}
    return {
        "mtime_days": _optional_positive_int(merged.get("mtime_days"), default=90),
        "observation_days": _optional_positive_int(merged.get("observation_days"), default=None),
    }


def _optional_positive_int(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _source_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for path in source_contract_paths(root):
        contract = load_yaml(path)
        source_id = contract.get("source_id")
        if isinstance(source_id, str):
            ids.add(source_id)
    ids.update({"derived", "DTS_or_Treasury_context", "fdic_or_other_rate_source"})
    return ids


def validate_source_contract(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    contract = load_yaml(path)
    required_keys = ["source_id", "source_name", "role", "status", "frequency", "raw_storage", "access"]
    for key in required_keys:
        if key not in contract:
            errors.append(f"{path}: missing required key `{key}`")

    expected_fields = contract.get("expected_fields")
    if not isinstance(expected_fields, list) or not expected_fields:
        errors.append(f"{path}: expected_fields must be a non-empty list")
    elif len(expected_fields) != len(set(expected_fields)):
        errors.append(f"{path}: expected_fields contains duplicates")

    access = contract.get("access")
    if not isinstance(access, dict):
        errors.append(f"{path}: access must be a mapping")
    else:
        source_url = str(access.get("source_url", ""))
        if "TODO_INSPECT_OFFICIAL_SOURCE" in source_url:
            errors.append(f"{path}: unresolved source_url placeholder")
        if access.get("mode") in {"fred_csv", "fiscaldata_api", "ofr_api"} and not source_url:
            errors.append(f"{path}: access.source_url is required for {access.get('mode')}")

    monthly = contract.get("monthly_aggregation")
    if monthly is not None and not isinstance(monthly, dict):
        errors.append(f"{path}: monthly_aggregation must be a mapping when present")
    if monthly is None and contract.get("frequency") in {"daily", "weekly", "daily_weekly_monthly"}:
        warnings.append(f"{path}: no monthly_aggregation documented")

    return errors, warnings


def validate_panel_schema(path: Path, source_ids: set[str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    schema = load_yaml(path)
    panels = schema.get("panels")
    if not isinstance(panels, dict) or not panels:
        return [f"{path}: panels must be a non-empty mapping"], warnings

    valid_grain_terms = ("calendar month", "week ending date")
    for panel_name, panel in panels.items():
        if not isinstance(panel, dict):
            errors.append(f"{path}: panel `{panel_name}` must be a mapping")
            continue
        grain = str(panel.get("grain", ""))
        if not any(term in grain for term in valid_grain_terms):
            errors.append(f"{path}: panel `{panel_name}` has unsupported grain `{grain}`")
        fields = panel.get("fields")
        if not isinstance(fields, dict) or not fields:
            errors.append(f"{path}: panel `{panel_name}` fields must be a non-empty mapping")
            continue
        if len(fields) != len(set(fields)):
            errors.append(f"{path}: panel `{panel_name}` fields contain duplicates")
        for field_name, meta in fields.items():
            if not isinstance(meta, dict):
                errors.append(f"{path}: field `{panel_name}.{field_name}` metadata must be a mapping")
                continue
            if "dtype" not in meta:
                errors.append(f"{path}: field `{panel_name}.{field_name}` missing dtype")
            source = meta.get("source")
            if isinstance(source, str) and source not in source_ids:
                warnings.append(f"{path}: field `{panel_name}.{field_name}` uses unresolved source `{source}`")

    return errors, warnings


def validate_project_settings(root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    path = root / "config" / "project.yaml"
    if not path.exists():
        errors.append(f"{path}: missing project config")
        return errors, warnings
    try:
        weekly_large_rebuild_randomization_p_max(root)
    except ConfigError as exc:
        errors.append(f"{path}: {exc}")
    try:
        project_as_of_date(root)
    except ConfigError as exc:
        errors.append(f"{path}: {exc}")
    config = load_project_config(root)
    freshness = config.get("source_freshness")
    if freshness is not None:
        if not isinstance(freshness, dict):
            errors.append(f"{path}: source_freshness must be a mapping")
        else:
            for source_family, policy in freshness.items():
                if not isinstance(policy, dict):
                    errors.append(f"{path}: source_freshness.{source_family} must be a mapping")
                    continue
                for field in ["mtime_days", "observation_days"]:
                    value = policy.get(field)
                    if value is None:
                        continue
                    try:
                        parsed = int(value)
                    except (TypeError, ValueError):
                        errors.append(f"{path}: source_freshness.{source_family}.{field} must be a positive integer or null")
                        continue
                    if parsed <= 0:
                        errors.append(f"{path}: source_freshness.{source_family}.{field} must be positive")
    return errors, warnings


def validate_project_config(root: Path | None = None) -> ValidationReport:
    project_root = root or find_project_root()
    errors: list[str] = []
    warnings: list[str] = []

    settings_errors, settings_warnings = validate_project_settings(project_root)
    errors.extend(settings_errors)
    warnings.extend(settings_warnings)

    contract_paths = source_contract_paths(project_root)
    if not contract_paths:
        errors.append(f"{project_root}: no source contracts found")
    seen_ids: set[str] = set()
    for path in contract_paths:
        contract = load_yaml(path)
        source_id = contract.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"{path}: source_id must be a non-empty string")
        elif source_id in seen_ids:
            errors.append(f"{path}: duplicate source_id `{source_id}`")
        else:
            seen_ids.add(source_id)
        contract_errors, contract_warnings = validate_source_contract(path)
        errors.extend(contract_errors)
        warnings.extend(contract_warnings)

    panel_path = project_root / "config" / "panel_schema.yaml"
    if not panel_path.exists():
        errors.append(f"{panel_path}: missing panel schema")
    else:
        panel_errors, panel_warnings = validate_panel_schema(panel_path, _source_ids(project_root))
        errors.extend(panel_errors)
        warnings.extend(panel_warnings)

    return ValidationReport(
        status="failed" if errors else "ok",
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
