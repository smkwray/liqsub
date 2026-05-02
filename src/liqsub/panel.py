from __future__ import annotations

from pathlib import Path

import pandas as pd

from liqsub.buycurve import derive_monthly_treasury_supply, validate_buycurve_panel
from liqsub.config import project_as_of_date
from liqsub.fiscaldata import build_monthly_dts_tga
from liqsub.fred import build_monthly_fred_panel
from liqsub.ofr import build_monthly_ofr_panel
from liqsub.paths import ensure_dir, relative_to_root


def classify_on_rrp_regime(on_rrp_usd_millions: pd.Series) -> pd.Series:
    values = pd.to_numeric(on_rrp_usd_millions, errors="coerce")
    return pd.Series(
        pd.cut(
            values,
            bins=[float("-inf"), 250_000.0, 1_000_000.0, float("inf")],
            labels=["scarce", "transition", "abundant"],
        ),
        index=values.index,
        dtype="string",
    )


def build_monthly_plumbing_panel(raw_dir: Path) -> pd.DataFrame:
    frames = [
        build_monthly_fred_panel(raw_dir),
        build_monthly_dts_tga(raw_dir),
        build_monthly_ofr_panel(raw_dir),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["month"])
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="month", how="outer")
    for column in list(panel.columns):
        if not column.endswith("_x"):
            continue
        base = column[:-2]
        y_col = f"{base}_y"
        if y_col in panel.columns:
            panel[base] = panel[y_col].combine_first(panel[column])
            panel = panel.drop(columns=[column, y_col])
    panel = panel.sort_values("month").reset_index(drop=True)
    if "tga_dts" in panel.columns and "tga_wednesday" in panel.columns:
        panel["tga"] = panel["tga_dts"].combine_first(panel["tga_wednesday"])
    elif "tga_dts" in panel.columns:
        panel["tga"] = panel["tga_dts"]
    elif "tga_wednesday" in panel.columns:
        panel["tga"] = panel["tga_wednesday"]
    if "on_rrp" in panel.columns:
        panel["on_rrp_regime"] = classify_on_rrp_regime(panel["on_rrp"])
    return panel


def build_monthly_panel(raw_dir: Path, buycurve_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    plumbing = build_monthly_plumbing_panel(raw_dir)
    validation = validate_buycurve_panel(buycurve_path)
    status: dict[str, object] = {
        "buycurve_status": validation.status,
        "buycurve_errors": list(validation.errors),
    }
    if validation.status != "ok":
        status["panel_status"] = "partial_without_treasury_supply"
        return plumbing, status
    treasury = derive_monthly_treasury_supply(buycurve_path)
    panel = treasury.merge(plumbing, on="month", how="outer").sort_values("month").reset_index(drop=True)
    status["panel_status"] = "ok"
    return panel, status


def write_monthly_outputs(root: Path) -> dict[str, object]:
    raw_dir = root / "data" / "raw"
    clean_dir = ensure_dir(root / "data" / "clean")
    buycurve_path = raw_dir / "buycurve" / "monthly_issuance_maturity_panel.csv"
    panel, status = build_monthly_panel(raw_dir, buycurve_path)
    output_name = (
        "monthly_liquidity_substitution_panel.csv"
        if status["panel_status"] == "ok"
        else "monthly_plumbing_panel_partial.csv"
    )
    output_path = clean_dir / output_name
    panel.to_csv(output_path, index=False)
    qa_path = clean_dir / "monthly_panel_qa.csv"
    qa_rows = []
    today = pd.Timestamp(project_as_of_date(root))
    future_mask = pd.to_datetime(panel["month"], errors="coerce") > today if "month" in panel else pd.Series(False)
    for column in panel.columns:
        qa_rows.append(
            {
                "column": column,
                "non_null": int(panel[column].notna().sum()),
                "missing": int(panel[column].isna().sum()),
                "future_or_scheduled_non_null": int(panel.loc[future_mask, column].notna().sum())
                if len(panel)
                else 0,
            }
        )
    pd.DataFrame(qa_rows).to_csv(qa_path, index=False)
    return {
        **status,
        "rows": int(len(panel)),
        "columns": list(panel.columns),
        "output": relative_to_root(root, output_path),
        "qa": relative_to_root(root, qa_path),
    }
