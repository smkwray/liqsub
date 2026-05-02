from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import pandas as pd

from liqsub.http import read_url
from liqsub.paths import ensure_dir, relative_to_root


FRED_SERIES: dict[str, dict[str, object]] = {
    "tga_wednesday": {
        "id": "WDTGAL",
        "frequency": "weekly",
        "unit": "USD_millions",
        "monthly": "last",
        "required": False,
    },
    "tga_week_average": {
        "id": "WTREGEN",
        "frequency": "weekly",
        "unit": "USD_millions",
        "monthly": "last",
        "required": False,
    },
    "reserves": {
        "id": "WRESBAL",
        "frequency": "weekly",
        "unit": "USD_millions",
        "monthly": "last",
        "required": True,
    },
    "fed_treasury_holdings": {
        "id": "TREAST",
        "frequency": "weekly",
        "unit": "USD_millions",
        "monthly": "last",
        "required": False,
    },
    "on_rrp": {
        "id": "RRPONTSYD",
        "frequency": "daily",
        "unit": "USD_billions",
        "monthly": "mean",
        "required": True,
    },
    "deposits": {
        "id": "DPSACBW027SBOG",
        "frequency": "weekly",
        "unit": "USD_billions",
        "monthly": "last",
        "required": True,
    },
    "deposits_nsa": {
        "id": "DPSACBW027NBOG",
        "frequency": "weekly",
        "unit": "USD_billions",
        "monthly": "last",
        "required": False,
    },
    "domestic_deposits": {
        "id": "DPSDCBW027SBOG",
        "frequency": "weekly",
        "unit": "USD_billions",
        "monthly": "last",
        "required": False,
    },
    "cash_assets": {
        "id": "CASACBW027SBOG",
        "frequency": "weekly",
        "unit": "USD_billions",
        "monthly": "last",
        "required": False,
    },
    "bank_treasury_agency_securities": {
        "id": "TASACBW027SBOG",
        "frequency": "weekly",
        "unit": "USD_billions",
        "monthly": "last",
        "required": True,
    },
    "total_bank_credit": {
        "id": "TOTBKCR",
        "frequency": "weekly",
        "unit": "USD_billions",
        "monthly": "last",
        "required": False,
    },
    "retail_mmf_assets": {
        "id": "WRMFNS",
        "frequency": "weekly",
        "unit": "USD_billions",
        "monthly": "last",
        "required": False,
    },
    "institutional_mmf_assets": {
        "id": "WIMFNS",
        "frequency": "weekly",
        "unit": "USD_billions",
        "monthly": "last",
        "required": False,
        "discontinued_after": "2021-02-01",
    },
    "iorb_rate": {
        "id": "IORB",
        "frequency": "daily",
        "unit": "percent",
        "monthly": "mean",
        "required": True,
    },
    "fed_funds": {
        "id": "EFFR",
        "frequency": "daily",
        "unit": "percent",
        "monthly": "mean",
        "required": False,
    },
    "sofr": {
        "id": "SOFR",
        "frequency": "daily",
        "unit": "percent",
        "monthly": "mean",
        "required": True,
    },
    "bill_yield_1mo": {
        "id": "DGS1MO",
        "frequency": "daily",
        "unit": "percent",
        "monthly": "mean",
        "required": False,
    },
    "bill_yield_3mo": {
        "id": "DGS3MO",
        "frequency": "daily",
        "unit": "percent",
        "monthly": "mean",
        "required": True,
    },
    "bill_yield_6mo": {
        "id": "DGS6MO",
        "frequency": "daily",
        "unit": "percent",
        "monthly": "mean",
        "required": False,
    },
}


def fred_graph_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={quote(series_id)}"


def fetch_fred_csv_text(series_id: str, timeout: int = 45) -> str:
    try:
        return read_url(fred_graph_csv_url(series_id), accept="text/csv,*/*", timeout=timeout).decode(
            "utf-8"
        )
    except Exception as exc:
        raise RuntimeError(f"failed to download FRED series {series_id}: {exc}") from exc


def parse_fred_csv(text: str, *, series_id: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(text))
    if "observation_date" in df.columns and series_id in df.columns:
        out = df.rename(columns={"observation_date": "date", series_id: "value"}).copy()
    elif {"date", "value"}.issubset(df.columns):
        out = df.rename(columns={"date": "date", "value": "value"}).copy()
    else:
        raise ValueError(f"unexpected FRED CSV schema for {series_id}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    values = out["value"].astype("string").mask(out["value"].astype("string") == ".")
    out["value"] = pd.to_numeric(values, errors="coerce")
    return out[["date", "value"]].dropna(subset=["date"])


def _cached_fred_profile(path: Path, *, series_id: str) -> dict[str, object]:
    if not path.exists():
        return {"cache_valid": False, "cache_rows": 0, "cache_last_observation": ""}
    try:
        parsed = parse_fred_csv(path.read_text(encoding="utf-8"), series_id=series_id)
    except Exception:
        return {"cache_valid": False, "cache_rows": 0, "cache_last_observation": ""}
    valid = parsed.dropna(subset=["value"])
    return {
        "cache_valid": not valid.empty,
        "cache_rows": int(len(valid)),
        "cache_last_observation": valid["date"].max().date().isoformat() if not valid.empty else "",
    }


def _fetch_fred_with_retries(series_id: str, *, timeout: int, retries: int) -> str:
    last_error: Exception | None = None
    for _attempt in range(max(1, retries + 1)):
        try:
            return fetch_fred_csv_text(series_id, timeout=timeout)
        except Exception as exc:  # pragma: no cover - exercised through download_fred_series tests
            last_error = exc
    raise RuntimeError(str(last_error) if last_error else f"failed to download FRED series {series_id}")


def download_fred_series(
    raw_dir: Path,
    *,
    required_only: bool = False,
    series_keys: list[str] | None = None,
    timeout: int = 45,
    retries: int = 1,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    destination_dir = ensure_dir(raw_dir / "fred")
    action_rows: list[dict[str, object]] = []
    required_failures: list[str] = []
    statuses_by_key: dict[str, dict[str, object]] = {}
    wanted = set(series_keys or [])
    unknown = sorted(wanted - set(FRED_SERIES))
    if unknown:
        raise ValueError(f"unknown FRED series key(s): {', '.join(unknown)}")

    for key, meta in FRED_SERIES.items():
        if required_only and not bool(meta["required"]):
            continue
        if wanted and key not in wanted:
            continue
        series_id = str(meta["id"])
        filename = f"{key}__{series_id}.csv"
        destination = destination_dir / filename
        cache_profile = _cached_fred_profile(destination, series_id=series_id)
        row = {
            "source": "FRED",
            "series_key": key,
            "series_id": series_id,
            "frequency": meta["frequency"],
            "unit": meta["unit"],
            "monthly": meta["monthly"],
            "required": bool(meta["required"]),
            "path": relative_to_root(raw_dir, destination),
            "status": "downloaded",
            "cache_rows": cache_profile["cache_rows"],
            "cache_last_observation": cache_profile["cache_last_observation"],
            "timeout_seconds": int(timeout),
            "retries": int(retries),
        }
        if progress:
            progress(f"fetching {key} ({series_id})")
        try:
            text = _fetch_fred_with_retries(series_id, timeout=timeout, retries=retries)
            # Validate before writing.
            parsed = parse_fred_csv(text, series_id=series_id)
            destination.write_text(text, encoding="utf-8")
            valid = parsed.dropna(subset=["value"])
            row["rows"] = int(len(valid))
            row["last_observation"] = valid["date"].max().date().isoformat() if not valid.empty else ""
        except Exception as exc:
            if cache_profile["cache_valid"]:
                row["status"] = f"kept_existing_cache_after_failed_refresh: {exc}"
            else:
                row["status"] = f"failed: {exc}"
            if meta["required"] and not cache_profile["cache_valid"]:
                required_failures.append(key)
        action_rows.append(row)
        statuses_by_key[key] = row

    manifest = []
    for key, meta in FRED_SERIES.items():
        series_id = str(meta["id"])
        path = destination_dir / f"{key}__{series_id}.csv"
        profile = _cached_fred_profile(path, series_id=series_id)
        action = statuses_by_key.get(key, {})
        manifest.append(
            {
                "source": "FRED",
                "series_key": key,
                "series_id": series_id,
                "frequency": meta["frequency"],
                "unit": meta["unit"],
                "monthly": meta["monthly"],
                "required": bool(meta["required"]),
                "path": relative_to_root(raw_dir, path),
                "status": action.get(
                    "status",
                    "existing_cache" if profile["cache_valid"] else "missing_cache",
                ),
                "rows": action.get("rows", profile["cache_rows"]),
                "last_observation": action.get("last_observation", profile["cache_last_observation"]),
                "cache_rows": profile["cache_rows"],
                "cache_last_observation": profile["cache_last_observation"],
                "timeout_seconds": action.get("timeout_seconds", ""),
                "retries": action.get("retries", ""),
                "discontinued_after": meta.get("discontinued_after", ""),
            }
        )
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(destination_dir / "manifest.csv", index=False)
    if required_failures:
        raise RuntimeError(f"required FRED downloads failed: {', '.join(required_failures)}")
    return manifest_df


def load_fred_raw(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key, meta in FRED_SERIES.items():
        path = raw_dir / "fred" / f"{key}__{meta['id']}.csv"
        if not path.exists():
            continue
        parsed = parse_fred_csv(path.read_text(encoding="utf-8"), series_id=str(meta["id"]))
        parsed["series_key"] = key
        parsed["series_id"] = meta["id"]
        parsed["unit"] = meta["unit"]
        parsed["monthly"] = meta["monthly"]
        frames.append(parsed)
    if not frames:
        return pd.DataFrame(columns=["date", "value", "series_key", "series_id", "unit", "monthly"])
    return pd.concat(frames, ignore_index=True)


def _to_usd_millions(series: pd.Series, unit: str) -> pd.Series:
    if unit == "USD_billions":
        return series * 1000.0
    return series


def build_monthly_fred_panel(raw_dir: Path) -> pd.DataFrame:
    long = load_fred_raw(raw_dir)
    if long.empty:
        return pd.DataFrame(columns=["month"])
    frames: list[pd.DataFrame] = []
    for key, meta in FRED_SERIES.items():
        series_df = long.loc[long["series_key"] == key, ["date", "value"]].dropna().copy()
        if series_df.empty:
            continue
        series_df["value"] = _to_usd_millions(series_df["value"], str(meta["unit"]))
        monthly_method = str(meta["monthly"])
        monthly = series_df.set_index("date").sort_index()["value"].resample("ME")
        values = monthly.mean() if monthly_method == "mean" else monthly.last()
        frames.append(values.rename(key).to_frame())
    if not frames:
        return pd.DataFrame(columns=["month"])
    out = pd.concat(frames, axis=1, sort=True).reset_index().rename(columns={"date": "month"})
    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M").dt.to_timestamp()
    if "retail_mmf_assets" in out.columns or "institutional_mmf_assets" in out.columns:
        retail = out.get("retail_mmf_assets", 0.0)
        institutional = out.get("institutional_mmf_assets", 0.0)
        out["total_mmf_assets"] = retail + institutional
    if "bill_yield_3mo" in out.columns:
        out["bill_yield"] = out["bill_yield_3mo"]
    return out
