from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from liqsub.http import read_url
from liqsub.paths import ensure_dir, relative_to_root


OFR_BASE_URL = "https://data.financialresearch.gov/v1"
OFR_DATASETS = ("mmf",)
MMF_SERIES_MAP = {
    "MMF-MMF_TOT-M": "total_mmf_assets",
    "MMF-MMF_T_TOT-M": "mmf_treasury_holdings",
    "MMF-MMF_RP_TOT-M": "mmf_repo_holdings",
    "MMF-MMF_RP_T_TOT-M": "mmf_treasury_repo_holdings",
    "MMF-MMF_RP_wFR-M": "mmf_on_rrp_exposure",
    "MMF-MMF_AG_TOT-M": "mmf_agency_holdings",
}


def _get_json(url: str, timeout: int = 120) -> dict[str, Any]:
    payload = json.loads(read_url(url, accept="application/json", timeout=timeout).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object from {url}")
    return payload


def fetch_dataset_payload(dataset: str, *, start_date: str = "2000-01-01") -> dict[str, Any]:
    params = urlencode({"dataset": dataset, "start_date": start_date})
    return _get_json(f"{OFR_BASE_URL}/series/dataset?{params}")


def download_ofr_mmf(raw_dir: Path) -> Path:
    destination_dir = ensure_dir(raw_dir / "ofr")
    payload = fetch_dataset_payload("mmf")
    destination = destination_dir / "mmf.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "source": "OFR",
                "dataset": "mmf",
                "path": relative_to_root(raw_dir, destination),
                "status": "downloaded",
            }
        ]
    ).to_csv(destination_dir / "manifest.csv", index=False)
    return destination


def _extract_observations(inner_ts: dict[str, Any]) -> list[Any]:
    for key in ("aggregation", "observations", "data", "values"):
        observations = inner_ts.get(key)
        if isinstance(observations, list):
            return observations
    for value in inner_ts.values():
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, list):
                    return nested
    return []


def dataset_payload_to_long_frame(payload: dict[str, Any]) -> pd.DataFrame:
    timeseries = payload.get("timeseries")
    if not isinstance(timeseries, dict):
        return pd.DataFrame(columns=["date", "value", "series_key", "name", "frequency", "unit"])
    rows: list[dict[str, Any]] = []
    for series_key, series_data in timeseries.items():
        if not isinstance(series_data, dict):
            continue
        inner_ts = series_data.get("timeseries", {})
        if not isinstance(inner_ts, dict):
            continue
        meta = series_data.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        desc = meta.get("description", {})
        unit = meta.get("unit", {})
        schedule = meta.get("schedule", {})
        for obs in _extract_observations(inner_ts):
            if isinstance(obs, list) and len(obs) >= 2:
                rows.append(
                    {
                        "date": obs[0],
                        "value": obs[1],
                        "series_key": series_key,
                        "name": desc.get("name", "") if isinstance(desc, dict) else "",
                        "frequency": schedule.get("observation_frequency", "")
                        if isinstance(schedule, dict)
                        else "",
                        "unit": unit.get("name", "") if isinstance(unit, dict) else "",
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["date", "value", "series_key", "name", "frequency", "unit"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date"])


def build_monthly_ofr_panel(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "ofr" / "mmf.json"
    if not path.exists():
        return pd.DataFrame(columns=["month"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    long = dataset_payload_to_long_frame(payload)
    frames: list[pd.DataFrame] = []
    for series_key, column in MMF_SERIES_MAP.items():
        series = long.loc[long["series_key"] == series_key, ["date", "value"]].dropna().copy()
        if series.empty:
            continue
        series[column] = series["value"] / 1_000_000.0
        monthly = (
            series.set_index("date")[column]
            .sort_index()
            .resample("ME")
            .last()
            .to_frame()
        )
        frames.append(monthly)
    if not frames:
        return pd.DataFrame(columns=["month"])
    out = pd.concat(frames, axis=1, sort=True).reset_index().rename(columns={"date": "month"})
    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M").dt.to_timestamp()
    return out
