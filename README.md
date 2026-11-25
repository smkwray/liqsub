# liqsub

`liqsub` builds monthly and weekly historical panels linking US Treasury issuance to commercial-bank deposits, money-market funds (MMFs), bank reserves, the Treasury General Account (TGA), and the overnight reverse repurchase agreement facility (ON RRP). It reads local caches of public Federal Reserve, Treasury, and Office of Financial Research (OFR) series plus issuance and weekly TGA-rebuild exports from the sibling repositories [buycurve](https://github.com/smkwray/buycurve) and [tgarefill](https://github.com/smkwray/tgarefill), and writes clean panels, coverage checks, event-window tables, Markdown reports, manifests, and release ZIP archives. A rebuild needs those sibling exports; the test suite runs offline on synthetic fixtures.

## Data

Public macro and market caches live under `data/raw/` and require no API keys:

- FRED (Federal Reserve Economic Data, St. Louis Fed; via `fredgraph.csv`): weekly and daily series from the Federal Reserve's H.4.1 balance-sheet release (TGA, reserves, Federal Reserve Treasury holdings), ON RRP balances, commercial-bank deposits, cash assets, Treasury and agency securities, and bank credit from the H.8 release on bank balance sheets, historical MMF assets (the institutional FRED series ended in February 2021), and short-term rates and Treasury bill yields. If a live refresh fails, any series with a valid local cache keeps its cached file.
- US Treasury FiscalData Daily Treasury Statement: daily operating cash balances.
- OFR: monthly MMF total assets, Treasury holdings, repo, ON RRP exposure, and agency holdings.

Required sibling and manual CSV inputs:

- The monthly Treasury issuance panel from `buycurve`, copied to `data/raw/buycurve/` (see `config/upstream_buycurve_outputs.yaml`).
- Eight weekly TGA-rebuild and money-fund exports from `tgarefill`, copied to `data/raw/tgarefill/` (listed in `config/upstream_tgarefill_outputs.yaml`).
- Two calendar context CSVs under `data/manual/`, included in the repository.

## Install

Python 3.11 or later is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Usage

A fresh clone has no data. First copy `data/clean/monthly_issuance_maturity_panel.csv` from a `buycurve` checkout into `data/raw/buycurve/`, and the eight files listed in `config/upstream_tgarefill_outputs.yaml` from `tgarefill`'s `outputs/tables/` into `data/raw/tgarefill/`. Then check them and fetch the public FRED, Daily Treasury Statement, and OFR caches:

```bash
liqsub check-buycurve
liqsub check-tgarefill
liqsub fetch-all
```

Check configuration and local inputs without modifying outputs:

```bash
liqsub validate-config
liqsub rebuild-public --check-only
liqsub release-public --check-only
```

Rebuild the monthly and weekly panels, diagnostic tables, reports, manifests, and public release archive from local caches:

```bash
liqsub rebuild-public
liqsub release-public
```

Run `liqsub fetch-all` again before a rebuild to refresh the public caches. Individual fetch, build, and check subcommands are also available (`fetch-fred`, `fetch-dts`, `fetch-ofr`, `validate-inputs`, `build-monthly`, `analyze-monthly`, `build-weekly`, `analyze-weekly`, `status`). `python -B -m liqsub.cli <command>` runs the same interface; run `liqsub --help` for all options.

## Output

Local raw inputs, clean panels, tables, reports, manifests, and release archives live under `data/` and `output/` and are ignored by Git.

| Path | Description |
| --- | --- |
| `data/clean/monthly_liquidity_substitution_panel.csv` | Monthly Treasury issuance, TGA, reserves, ON RRP, H.8 bank, and MMF panel (levels and changes in USD millions, rates in percent) |
| `data/clean/weekly_liquidity_substitution_panel.csv` | Weekly TGA, reserves, ON RRP, H.8 bank, and rate panel (levels and changes in USD millions) |
| `output/tables/` and `output/reports/` | Input coverage checks, terminal-period flags, large-TGA-rebuild event-window tables, and Markdown summaries |
| `output/bundles/` | Public release ZIP archives excluding raw, intermediate, cache, and internal files |

See [the public artifact contract](docs/public_artifact_contract.md) for the complete output manifest.

## Limits

- `liqsub` builds a reproducible historical snapshot: `rebuild-public` uses the local files in `data/raw/` and does not refresh external APIs unless you run a fetch command.
- Panels and event windows are descriptive and do not identify causal liquidity substitution across deposits, MMFs, reserves, or ON RRP.
- Partially observed months and weeks at the end of the sample stay in the clean panels for traceability, marked with an exclusion flag so baseline summaries omit them.
- Source coverage varies by series and date range (for example, the FRED institutional MMF series ended in February 2021 while OFR provides ongoing monthly MMF holdings), so the combined panel is not uniformly covered across its full span.

## Tests

The test suite uses synthetic fixtures and runs offline:

```bash
python -B -m pytest -q
ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
