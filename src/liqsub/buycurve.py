from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from liqsub.schemas import UPSTREAM_BUYCURVE_REQUIRED_FIELDS


class UpstreamBlockedError(FileNotFoundError):
    """Raised when the required buycurve export has not been copied locally."""


@dataclass(frozen=True)
class BuycurveValidation:
    status: str
    path: Path
    errors: tuple[str, ...]
    rows: int = 0

    def raise_if_failed(self) -> None:
        if self.errors:
            if self.status == "blocked":
                raise UpstreamBlockedError(self.errors[0])
            raise ValueError("\n".join(self.errors))


def validate_buycurve_panel(path: Path) -> BuycurveValidation:
    if not path.exists():
        return BuycurveValidation(
            status="blocked",
            path=path,
            errors=(f"required upstream buycurve export is missing: {path}",),
        )

    errors: list[str] = []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return BuycurveValidation(
            status="failed",
            path=path,
            errors=(f"{path}: file is empty",),
        )
    missing = [field for field in UPSTREAM_BUYCURVE_REQUIRED_FIELDS if field not in df.columns]
    if missing:
        errors.append(f"{path}: missing required columns: {', '.join(missing)}")
    if df.empty:
        errors.append(f"{path}: no rows")
    if "month" in df.columns:
        months = pd.to_datetime(df["month"], errors="coerce")
        if months.isna().any():
            errors.append(f"{path}: month contains unparsable values")
        if not months.is_monotonic_increasing:
            errors.append(f"{path}: month is not sorted ascending")
    grain = ["month", "security_type", "maturity_bucket"]
    if all(col in df.columns for col in grain) and df.duplicated(subset=grain).any():
        errors.append(f"{path}: duplicate rows at month x security_type x maturity_bucket grain")
    if "security_type" in df.columns:
        known = {"bill", "note", "bond", "tips", "frn", "coupon"}
        observed = set(df["security_type"].astype("string").str.lower().dropna().unique())
        unknown = sorted(value for value in observed if not any(k in value for k in known))
        if unknown:
            errors.append(f"{path}: unknown security_type values: {', '.join(unknown[:10])}")
    numeric_nonnegative = [
        "auction_count",
        "accepted_amount_sum",
        "offering_amount_sum",
        "weighted_maturity_years",
        "bill_share_by_accepted_amount",
    ]
    for column in numeric_nonnegative:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            errors.append(f"{path}: {column} contains non-numeric values")
            continue
        if (values < 0).any():
            errors.append(f"{path}: {column} contains negative values")
    if "auction_count" in df.columns:
        counts = pd.to_numeric(df["auction_count"], errors="coerce")
        if counts.notna().all() and (counts % 1 != 0).any():
            errors.append(f"{path}: auction_count contains non-integer values")
    if "bill_share_by_accepted_amount" in df.columns:
        shares = pd.to_numeric(df["bill_share_by_accepted_amount"], errors="coerce")
        if shares.notna().all() and ((shares < 0) | (shares > 1)).any():
            errors.append(f"{path}: bill_share_by_accepted_amount contains values outside [0, 1]")
    if "weighted_maturity_years" in df.columns:
        maturity = pd.to_numeric(df["weighted_maturity_years"], errors="coerce")
        if maturity.notna().all() and (maturity > 100).any():
            errors.append(f"{path}: weighted_maturity_years contains implausible values above 100")
    return BuycurveValidation(
        status="failed" if errors else "ok",
        path=path,
        errors=tuple(errors),
        rows=int(len(df)),
    )


def derive_monthly_treasury_supply(path: Path) -> pd.DataFrame:
    validation = validate_buycurve_panel(path)
    validation.raise_if_failed()
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"], errors="raise").dt.to_period("M").dt.to_timestamp()
    df["accepted_amount_sum"] = pd.to_numeric(df["accepted_amount_sum"], errors="coerce")
    df["weighted_maturity_years"] = pd.to_numeric(df["weighted_maturity_years"], errors="coerce")
    security = df["security_type"].astype("string").str.lower()
    df["is_bill"] = security.str.contains("bill", na=False)
    df["is_coupon"] = ~df["is_bill"]

    weighted = df["weighted_maturity_years"] * df["accepted_amount_sum"]
    df["accepted_weighted_maturity"] = weighted.where(df["weighted_maturity_years"].notna())
    grouped = df.groupby("month", as_index=False).agg(
        gross_bill_issuance_usd=("accepted_amount_sum", lambda s: s[df.loc[s.index, "is_bill"]].sum()),
        coupon_issuance_usd=("accepted_amount_sum", lambda s: s[df.loc[s.index, "is_coupon"]].sum()),
        accepted_amount_total=("accepted_amount_sum", "sum"),
        accepted_weighted_maturity=("accepted_weighted_maturity", "sum"),
    )
    grouped["gross_bill_issuance"] = grouped["gross_bill_issuance_usd"] / 1_000_000.0
    grouped["coupon_issuance"] = grouped["coupon_issuance_usd"] / 1_000_000.0
    grouped["weighted_maturity_years"] = (
        grouped["accepted_weighted_maturity"] / grouped["accepted_amount_total"]
    )
    total = grouped["gross_bill_issuance"] + grouped["coupon_issuance"]
    grouped["bill_share"] = grouped["gross_bill_issuance"] / total.where(total != 0)
    return grouped[
        ["month", "gross_bill_issuance", "coupon_issuance", "bill_share", "weighted_maturity_years"]
    ]
