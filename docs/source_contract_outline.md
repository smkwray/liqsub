# Source Contract Outline

Contracts under `config/source_contracts/` define the required source blocks and now include executable access metadata for the non-buycurve MVP sources.

## Required first

- `bill_issuance_from_buycurve.yaml`: local copy of the `buycurve` monthly issuance/maturity panel.
- `h41.yaml`: reserves, TGA, ON RRP, Fed Treasury holdings through FRED/Fed DDP.
- `h8.yaml`: deposits, bank cash assets, bank securities through FRED/Fed DDP.
- `ofr_sec_mmf.yaml`: MMF assets, Treasury holdings, repo holdings, ON RRP exposure through the OFR STFM API.

## Next

- `rates.yaml`: IORB, ON RRP rate, fed funds, SOFR, bill yields, deposit rates.
- `dts.yaml`: TGA and cash-flow context, net issuance support.
- `on_rrp_operations.yaml`: daily ON RRP usage and counterparty/rate fields.

## Current executable outputs

- `data/raw/fiscaldata/dts_operating_cash_balance.csv`: downloaded directly from FiscalData.
- `data/raw/ofr/mmf.json`: downloaded directly from OFR STFM.
- `data/raw/fred/*.csv`: project-local normalized copies from sibling public FRED caches because the live FRED endpoint timed out locally.
- `data/raw/buycurve/monthly_issuance_maturity_panel.csv`: local copy of the buycurve Treasury issuance/maturity export.
- `data/clean/monthly_liquidity_substitution_panel.csv`: full monthly MVP panel with buycurve Treasury supply and liquidity-plumbing sources.
- `data/clean/monthly_plumbing_panel_partial.csv`: monthly deposits, MMF, TGA, reserves, ON RRP, and rates panel without buycurve Treasury supply.
- `data/clean/monthly_panel_qa.csv`: consolidated non-null/missingness QA for the monthly build.

## Regime requirement

ON RRP abundant versus scarce periods are not optional. The threshold definition should be documented and sensitivity-tested before interpreting bill-supply effects.
