# Dependency on buycurve

`liqsub` should wait for the shared Treasury issuance/maturity panel from `buycurve`.

Required upstream export:

```text
buycurve/data/clean/monthly_issuance_maturity_panel.csv
```

Project-local import path:

```text
liqsub/data/raw/buycurve/monthly_issuance_maturity_panel.csv
```

Expected grain: month x security type x maturity bucket.

Expected columns:

- `month`
- `security_type`
- `maturity_bucket`
- `auction_count`
- `accepted_amount_sum`
- `offering_amount_sum`
- `weighted_maturity_years`
- `bill_share_by_accepted_amount`

Use this file to derive gross bill issuance, coupon issuance, bill share, and maturity-mix controls. Do not reimplement Treasury auction maturity bucketing in `liqsub` unless `buycurve` cannot produce the shared panel.

As of 2026-04-29, this export has been copied into `liqsub/data/raw/buycurve/monthly_issuance_maturity_panel.csv`, and `liqsub` can build `data/clean/monthly_liquidity_substitution_panel.csv`.
