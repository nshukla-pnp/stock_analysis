# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management (Python 3.14).

```bash
# Install dependencies
uv sync

# Run the analysis
uv run python main.py

# Run with a specific Python
uv run --python 3.14 python main.py
```

## Architecture

Single-file pipeline in [main.py](main.py) with four distinct stages:

1. **Fetch metadata** (`get_sp500_meta`) — scrapes S&P 500 constituents from Wikipedia via `requests` + `pandas.read_html`, converts to a Polars DataFrame. Ticker dots are normalised to hyphens (e.g. `BRK.B` → `BRK-B`) to match yfinance's format.

2. **Download prices** (`get_week_prices`) — fetches one week of adjusted-close prices for all ~500 tickers via `yfinance.download` (multithreaded). Wide pandas MultiIndex output is unpivoted into a long-form Polars DataFrame with columns `[date, ticker, close]`.

3. **Aggregations** (`weekly_returns`, `sector_avg`) — DuckDB SQL queries run directly against Polars DataFrames via Apache Arrow zero-copy (no intermediate serialisation). Results are returned as Polars DataFrames via `.pl()`.

4. **Charts** — six matplotlib/seaborn plots saved as numbered PNGs (`01_` through `06_`) in the working directory (`OUT = "."`):
   - `01_weekly_returns_bar.png` — all tickers sorted by return
   - `02_sector_boxplot.png` — returns by GICS sector
   - `03_top20_price_lines.png` — normalised price lines for top-`TOP_N` performers
   - `04_return_distribution.png` — histogram + KDE of all returns
   - `05_correlation_heatmap.png` — return correlation for top-`TOP_N` most-complete tickers
   - `06_sector_avg_return.png` — mean return per sector with stock counts

## Key Constants

| Constant | Default | Purpose |
|----------|---------|---------|
| `TOP_N`  | `20`    | Number of stocks in multi-line and correlation charts |
| `OUT`    | `"."`   | Output directory for PNG files |

## Data Flow

```
Wikipedia HTML
    └─→ get_sp500_meta() → meta (Polars)
                                │
yfinance API                    │
    └─→ get_week_prices(tickers) → prices (Polars, long-form)
                                        │
                           DuckDB SQL ──┤
                                        ├─→ returns (Polars)
                                        └─→ sectors (Polars)
                                                │
                                         6× matplotlib/seaborn plots → PNG files
```

## Dependencies

- **polars** — primary in-memory DataFrame library
- **duckdb** — SQL aggregations over Polars frames via Arrow (zero-copy)
- **yfinance** — price data download
- **pandas** — used only as an intermediate (Wikipedia scraping, yfinance output); converted to Polars immediately
- **pyarrow** — enables Arrow zero-copy bridge between Polars and DuckDB
- **matplotlib / seaborn** — charting
- **lxml** — HTML parser backend for `pandas.read_html`
