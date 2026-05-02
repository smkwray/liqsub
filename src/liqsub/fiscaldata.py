from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from liqsub.http import read_url
from liqsub.paths import ensure_dir, relative_to_root


FISCALDATA_BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
DTS_ENDPOINT = "/v1/accounting/dts/operating_cash_balance"


def _get_json(url: str, timeout: int = 120) -> dict[str, object]:
    return json.loads(read_url(url, accept="application/json", timeout=timeout).decode("utf-8"))


def fetch_fiscaldata_endpoint(
    endpoint: str,
    *,
    fields: list[str] | None = None,
    filters: str | None = None,
    page_size: int = 10_000,
    max_pages: int = 100,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    page_number = 1
    while page_number <= max_pages:
        params: dict[str, object] = {"page[size]": page_size, "page[number]": page_number}
        if fields:
            params["fields"] = ",".join(fields)
        if filters:
            params["filter"] = filters
        url = f"{FISCALDATA_BASE_URL}{endpoint}?{urlencode(params)}"
        payload = _get_json(url)
        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            break
        rows.extend(row for row in data if isinstance(row, dict))
        meta = payload.get("meta", {})
        if isinstance(meta, dict):
            total_count = meta.get("total-count")
            total_pages = meta.get("total-pages")
            if isinstance(total_count, int) and len(rows) >= total_count:
                break
            if isinstance(total_pages, int) and page_number >= total_pages:
                break
        if len(data) < page_size:
            break
        page_number += 1
    return rows


def download_dts_operating_cash_balance(raw_dir: Path) -> Path:
    destination_dir = ensure_dir(raw_dir / "fiscaldata")
    fields = ["record_date", "account_type", "close_today_bal", "open_today_bal", "open_month_bal"]
    rows = fetch_fiscaldata_endpoint(DTS_ENDPOINT, fields=fields)
    destination = destination_dir / "dts_operating_cash_balance.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    manifest = pd.DataFrame(
        [
            {
                "source": "FiscalData",
                "endpoint": DTS_ENDPOINT,
                "rows": len(rows),
                "path": relative_to_root(raw_dir, destination),
                "status": "downloaded",
            }
        ]
    )
    manifest.to_csv(destination_dir / "manifest.csv", index=False)
    return destination


def build_monthly_dts_tga(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "fiscaldata" / "dts_operating_cash_balance.csv"
    if not path.exists():
        return pd.DataFrame(columns=["month", "tga_dts"])
    df = pd.read_csv(path)
    if df.empty or "record_date" not in df.columns:
        return pd.DataFrame(columns=["month", "tga_dts"])
    df["date"] = pd.to_datetime(df["record_date"], errors="coerce")
    account = df.get("account_type", pd.Series("", index=df.index)).astype("string")
    mask = account.str.contains("Federal Reserve|Treasury General Account", case=False, na=False)
    exclude = account.str.contains("Opening Balance|Deposits|Withdrawals", case=False, na=False)
    tga = df.loc[mask & ~exclude].copy()
    if tga.empty:
        return pd.DataFrame(columns=["month", "tga_dts"])
    close_vals = pd.to_numeric(tga.get("close_today_bal"), errors="coerce")
    open_vals = pd.to_numeric(tga.get("open_today_bal"), errors="coerce")
    tga["tga_dts"] = close_vals.fillna(open_vals)
    tga = tga.dropna(subset=["date", "tga_dts"]).sort_values("date")
    monthly = tga.set_index("date")["tga_dts"].resample("ME").last().to_frame()
    monthly = monthly.reset_index().rename(columns={"date": "month"})
    monthly["month"] = pd.to_datetime(monthly["month"]).dt.to_period("M").dt.to_timestamp()
    return monthly
