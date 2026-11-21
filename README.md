# liqsub

`liqsub` builds monthly and weekly historical panels linking US Treasury issuance to commercial-bank deposits, money-market funds (MMFs), bank reserves, the Treasury General Account (TGA), and the overnight reverse repurchase agreement facility (ON RRP). It reads local caches of public Federal Reserve, Treasury, and OFR series plus upstream issuance and weekly TGA-rebuild exports, and writes clean panels, coverage checks, event-window diagnostic tables, Markdown reports, manifests, and release ZIP archives.

## Data

Public macro and market caches live under `data/raw/` and require no API keys:

- FRED (via `fredgraph.csv`): weekly and daily Federal Reserve H.4.1 series (TGA, reserves, Fed Treasury holdings), ON RRP balances, H.8 commercial-bank deposits, cash assets, Treasury and agency securities, and bank credit, historical MMF assets (the institutional FRED series ended in February 2021), and short-term rates and Treasury bill yields. If a live refresh fails, any series with a valid local cache keeps its cached file.
- US Treasury FiscalData Daily Treasury Statement: daily operating cash balances.
- Office of Financial Research (OFR): monthly MMF total assets, Treasury holdings, repo, ON RRP exposure, and agency holdings.

Required local upstream and manual CSV inputs:

- Monthly Treasury issuance panel under `data/raw/buycurve/` (see `config/upstream_buycurve_outputs.yaml`).
- Eight weekly TGA-rebuild and money-fund CSV exports under `data/raw/tgarefill/` (listed in `config/upstream_tgarefill_outputs.yaml`).
- Two calendar context CSVs under `data/manual/`.

## Install

Python 3.11 or later is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Usage

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

To refresh the public FRED, Daily Treasury Statement, and OFR caches before rebuilding:

```bash
liqsub fetch-all
```

Individual fetch, build, and check subcommands are also available (`fetch-fred`, `fetch-dts`, `fetch-ofr`, `validate-inputs`, `build-monthly`, `analyze-monthly`, `build-weekly`, `analyze-weekly`, `status`). `python -B -m liqsub.cli <command>` runs the same interface; run `liqsub --help` for all options.

## Output

Local raw inputs, clean panels, tables, reports, manifests, and release archives live under `data/` and `output/` and are ignored by Git.

| Path | Description |
| --- | --- |
| `data/clean/monthly_liquidity_substitution_panel.csv` | Monthly Treasury issuance, TGA, reserves, ON RRP, H.8 bank, and MMF panel (levels and changes in USD millions, rates in percent) |
| `data/clean/weekly_liquidity_substitution_panel.csv` | Weekly TGA, reserves, ON RRP, H.8 bank, and rate panel (levels and changes in USD millions) |
| `output/tables/` and `output/reports/` | Input coverage checks, terminal-period flags, large-TGA-rebuild event-window tables, and Markdown summaries |
| `output/bundles/` | Public release ZIP archives excluding raw, intermediate, cache, and internal files |

See `docs/public_artifact_contract.md` for the complete output manifest.

## Limits

- `liqsub` builds a reproducible historical snapshot: `rebuild-public` uses the local files in `data/raw/` and does not refresh external APIs unless you run a fetch command.
- The panels and event-window outputs are descriptive diagnostics and do not establish a causal liquidity-substitution effect across deposits, MMFs, reserves, or ON RRP.
- Partially observed months and weeks at the end of the sample stay in the clean panels for traceability, marked with an exclusion flag so baseline summaries omit them.
- Source coverage varies by series and date range (for example, the FRED institutional MMF series ended in February 2021 while OFR provides ongoing monthly MMF holdings), so the combined panel is not uniformly covered across its full span.

## Tests

The test suite uses synthetic fixtures and runs offline:

```bash
python -B -m pytest -q
ruff check .
```
