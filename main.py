"""
stock_analysis/main.py

1. Fetch S&P 500 constituents from Wikipedia  → Polars DataFrame
2. Download one week of price data via yfinance → Polars DataFrame
3. Use DuckDB for aggregations / joins (zero-copy via Arrow)
4. Plot six charts with matplotlib / seaborn
"""

import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import yfinance as yf

warnings.filterwarnings("ignore")

TOP_N = 20      # stocks shown in multi-line / correlation charts
OUT   = "."     # directory for saved PNGs


# ── 1. S&P 500 constituents ───────────────────────────────────────────────────

def get_sp500_meta() -> pl.DataFrame:
    """Scrape the S&P 500 table from Wikipedia and return a Polars DataFrame."""
    import io
    import pandas as pd
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print("Fetching S&P 500 constituents from Wikipedia …")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; stock-analysis-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    pdf = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
    pdf.columns = ["ticker", "company", "sector", "sub_industry"]
    # Wikipedia uses dots (BRK.B); yfinance expects hyphens (BRK-B)
    pdf["ticker"] = pdf["ticker"].str.replace(".", "-", regex=False)

    meta = pl.from_pandas(pdf)
    print(f"  → {len(meta)} constituents loaded.\n")
    return meta


# ── 2. One week of prices ─────────────────────────────────────────────────────

def get_week_prices(tickers: list[str]) -> pl.DataFrame:
    """Download adjusted-close prices and return a long-form Polars DataFrame."""
    import pandas as pd

    end   = datetime.date.today()
    start = end - datetime.timedelta(days=7)
    print(f"Downloading prices ({start} → {end}) for {len(tickers)} tickers …")

    raw = yf.download(
        tickers,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance multi-ticker → MultiIndex columns; extract "Close" level
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Wide pandas → long Polars
    close = close.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
    prices = (
        pl.from_pandas(close)
        .unpivot(index="date", variable_name="ticker", value_name="close")
        .drop_nulls("close")
    )
    days    = prices["date"].n_unique()
    symbols = prices["ticker"].n_unique()
    print(f"  → {symbols} tickers, {days} trading days.\n")
    return prices


# ── 3. Fundamental data for Buffett-style ratios ─────────────────────────────

def get_fundamental_data(tickers: list[str], max_workers: int = 30) -> pl.DataFrame:
    """
    Fetch fundamental metrics for every ticker via yfinance (threaded).

    yfinance field notes:
      roe / roa            – decimals  (0.15 = 15 %)
      debt_to_equity       – %-scaled  (50 ≈ D/E of 0.5×)
      net_margin / op_margin – decimals
      current_ratio, pe_ratio, pb_ratio – raw ratios
      fcf, market_cap      – absolute USD
    NULL debt_to_equity means no debt was reported → treated as zero-debt later.
    """
    def fetch_one(ticker: str) -> dict:
        row: dict = {
            "ticker": ticker, "roe": None, "roa": None,
            "debt_to_equity": None, "current_ratio": None,
            "net_margin": None, "operating_margin": None,
            "pe_ratio": None, "pb_ratio": None,
            "fcf": None, "market_cap": None,
            "revenue_growth": None, "earnings_growth": None,
        }
        try:
            info = yf.Ticker(ticker).info
            row.update({
                "roe":              info.get("returnOnEquity"),
                "roa":              info.get("returnOnAssets"),
                "debt_to_equity":   info.get("debtToEquity"),
                "current_ratio":    info.get("currentRatio"),
                "net_margin":       info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "pe_ratio":         info.get("trailingPE"),
                "pb_ratio":         info.get("priceToBook"),
                "fcf":              info.get("freeCashflow"),
                "market_cap":       info.get("marketCap"),
                "revenue_growth":   info.get("revenueGrowth"),
                "earnings_growth":  info.get("earningsGrowth"),
            })
        except Exception:
            pass
        return row

    print(f"Fetching fundamental data for {len(tickers)} tickers "
          f"({max_workers} threads) …")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 100 == 0:
                print(f"  {i}/{len(tickers)} …")

    df = pl.DataFrame(results)
    num_cols = [c for c in df.columns if c != "ticker"]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num_cols])
    valid = df.drop_nulls(subset=["roe", "net_margin"])
    print(f"  → {len(valid)}/{len(tickers)} tickers with full fundamental data.\n")
    return df


# ── 4. DuckDB aggregations (Arrow zero-copy) ──────────────────────────────────

def weekly_returns(prices: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    """Compute 1-week return per ticker using DuckDB, joined with sector info."""
    return duckdb.sql("""
        WITH first_last AS (
            SELECT
                ticker,
                FIRST(close ORDER BY date) AS open_price,
                LAST (close ORDER BY date) AS close_price
            FROM prices
            GROUP BY ticker
        )
        SELECT
            fl.ticker,
            m.company,
            m.sector,
            ROUND((fl.close_price / fl.open_price - 1) * 100, 3) AS return_pct
        FROM first_last fl
        JOIN meta m USING (ticker)
        ORDER BY return_pct
    """).pl()


def sector_avg(returns: pl.DataFrame) -> pl.DataFrame:
    return duckdb.sql("""
        SELECT sector,
               ROUND(AVG(return_pct), 3) AS avg_return,
               COUNT(*)                  AS n_stocks
        FROM returns
        GROUP BY sector
        ORDER BY avg_return
    """).pl()


# ── 5. Buffett composite score ────────────────────────────────────────────────

def buffett_scores(fundamentals: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    """
    Composite Buffett-inspired score (0–100) per stock via DuckDB.

    Five criteria, 20 pts each:
      ROE           > 15 %  → 20  |  10–15 % → 10
      Debt/Equity   < 50    → 20  |  50–100  → 10   (yfinance: 50 ≈ 0.5×)
      Net Margin    > 20 %  → 20  |  10–20 % → 10
      Current Ratio > 2.0   → 20  |  1.5–2.0 → 10
      ROA           > 10 %  → 20  |  5–10 %  → 10

    NULL debt_to_equity is treated as zero-debt (best score, 20 pts).
    Stocks missing ROE or Net Margin are excluded.
    """
    return duckdb.sql("""
        WITH scored AS (
            SELECT
                f.ticker,
                m.company,
                m.sector,
                ROUND(f.roe              * 100, 2) AS roe_pct,
                ROUND(f.roa              * 100, 2) AS roa_pct,
                ROUND(f.net_margin       * 100, 2) AS net_margin_pct,
                ROUND(f.operating_margin * 100, 2) AS op_margin_pct,
                f.debt_to_equity,
                f.current_ratio,
                f.pe_ratio,
                f.pb_ratio,
                ROUND(f.revenue_growth  * 100, 2) AS revenue_growth_pct,
                ROUND(f.earnings_growth * 100, 2) AS earnings_growth_pct,

                CASE
                    WHEN f.roe > 0.15 THEN 20
                    WHEN f.roe > 0.10 THEN 10
                    ELSE 0
                END AS score_roe,

                -- NULL ≈ no debt on record → full marks
                CASE
                    WHEN f.debt_to_equity IS NULL
                      OR f.debt_to_equity < 50  THEN 20
                    WHEN f.debt_to_equity < 100 THEN 10
                    ELSE 0
                END AS score_debt,

                CASE
                    WHEN f.net_margin > 0.20 THEN 20
                    WHEN f.net_margin > 0.10 THEN 10
                    ELSE 0
                END AS score_margin,

                CASE
                    WHEN f.current_ratio > 2.0 THEN 20
                    WHEN f.current_ratio > 1.5 THEN 10
                    ELSE 0
                END AS score_liquidity,

                CASE
                    WHEN f.roa > 0.10 THEN 20
                    WHEN f.roa > 0.05 THEN 10
                    ELSE 0
                END AS score_roa

            FROM fundamentals f
            JOIN meta m USING (ticker)
            WHERE f.roe        IS NOT NULL
              AND f.net_margin IS NOT NULL
        )
        SELECT *,
               score_roe + score_debt + score_margin
               + score_liquidity + score_roa AS buffett_score
        FROM scored
        ORDER BY buffett_score DESC, roe_pct DESC
    """).pl()


# ── 6. Charts (price & returns) ───────────────────────────────────────────────

def plot_weekly_returns_bar(returns: pl.DataFrame) -> None:
    """Horizontal bar showing every ticker sorted by 1-week return."""
    df     = returns.sort("return_pct").to_pandas()
    colors = df["return_pct"].apply(lambda x: "#d73027" if x < 0 else "#1a9850")

    fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.16)))
    ax.barh(df["ticker"], df["return_pct"], color=colors, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("1-Week Return (%)")
    ax.set_title("S&P 500 – 1-Week Returns (all constituents with data)")
    ax.tick_params(axis="y", labelsize=5)
    plt.tight_layout()
    path = f"{OUT}/01_weekly_returns_bar.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_sector_boxplot(returns: pl.DataFrame) -> None:
    """Box-plot of returns grouped by GICS sector."""
    df    = returns.to_pandas()
    order = df.groupby("sector")["return_pct"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="return_pct", y="sector", order=order,
                palette="RdYlGn", ax=ax)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("1-Week Return (%)")
    ax.set_title("S&P 500 – 1-Week Returns by Sector")
    plt.tight_layout()
    path = f"{OUT}/02_sector_boxplot.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_normalised_lines(prices: pl.DataFrame, returns: pl.DataFrame) -> None:
    """Normalised price lines for the TOP_N best-performing tickers."""
    top = (
        returns.sort("return_pct", descending=True)
        .head(TOP_N)["ticker"]
        .to_list()
    )
    wide = (
        prices.filter(pl.col("ticker").is_in(top))
        .pivot(on="ticker", index="date", values="close")
        .sort("date")
    )
    pdf  = wide.to_pandas().set_index("date")
    norm = pdf / pdf.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in norm.columns:
        ax.plot(norm.index, norm[col], linewidth=1.2, label=col)
    ax.axhline(100, color="black", linewidth=0.6, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d"))
    ax.set_ylabel("Normalised Price (start = 100)")
    ax.set_title(f"Top-{TOP_N} Best Performers – Normalised Price (1 week)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    plt.tight_layout()
    path = f"{OUT}/03_top{TOP_N}_price_lines.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_return_distribution(returns: pl.DataFrame) -> None:
    """Histogram + KDE of 1-week returns across the whole index."""
    vals = returns["return_pct"].to_pandas()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(vals, bins=40, kde=True, color="steelblue", ax=ax)
    ax.axvline(vals.mean(),   color="red",    linestyle="--",
               label=f"Mean   {vals.mean():.2f}%")
    ax.axvline(vals.median(), color="orange", linestyle="--",
               label=f"Median {vals.median():.2f}%")
    ax.set_xlabel("1-Week Return (%)")
    ax.set_title("Distribution of 1-Week Returns – S&P 500 Constituents")
    ax.legend()
    plt.tight_layout()
    path = f"{OUT}/04_return_distribution.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_correlation_heatmap(prices: pl.DataFrame) -> None:
    """Return-correlation heat-map for the TOP_N most-complete tickers."""
    top = (
        prices.group_by("ticker")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(TOP_N)["ticker"]
        .to_list()
    )
    wide = (
        prices.filter(pl.col("ticker").is_in(top))
        .pivot(on="ticker", index="date", values="close")
        .sort("date")
    )
    pdf  = wide.to_pandas().set_index("date").pct_change().dropna()
    corr = pdf.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.3, annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"Return Correlation – Top-{TOP_N} S&P 500 Stocks (1 week)")
    plt.tight_layout()
    path = f"{OUT}/05_correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_sector_avg_return(sector_df: pl.DataFrame) -> None:
    """Horizontal bar chart of mean return per sector with stock counts."""
    df     = sector_df.to_pandas()
    colors = df["avg_return"].apply(lambda x: "#d73027" if x < 0 else "#1a9850")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["sector"], df["avg_return"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    for i, (v, n) in enumerate(zip(df["avg_return"], df["n_stocks"])):
        offset = 0.05 if v >= 0 else -0.05
        ax.text(v + offset, i, f"n={n}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("Average 1-Week Return (%)")
    ax.set_title("S&P 500 – Average Return by Sector (1 week)")
    plt.tight_layout()
    path = f"{OUT}/06_sector_avg_return.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


# ── 7. Buffett-ratio charts ───────────────────────────────────────────────────

def plot_buffett_top20(scores: pl.DataFrame) -> None:
    """Horizontal bar – top 20 stocks by Buffett composite score, coloured by sector."""
    top     = scores.head(20).sort("buffett_score").to_pandas()
    sectors = top["sector"].unique()
    palette = dict(zip(sectors, sns.color_palette("tab10", len(sectors))))
    colors  = top["sector"].map(palette)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(top["ticker"], top["buffett_score"], color=colors, edgecolor="none")
    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{row['company'][:28]}  ROE {row['roe_pct']:.0f}%",
                va="center", fontsize=7)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Buffett Score  (0 = worst · 100 = best)")
    ax.set_title(
        "Top 20 S&P 500 Stocks – Buffett Composite Score\n"
        "Criteria: ROE ≥15%  |  Debt/Equity <0.5×  |  Net Margin ≥20%  "
        "|  Current Ratio ≥2  |  ROA ≥10%"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[s]) for s in sectors]
    ax.legend(handles, list(sectors), fontsize=7, loc="lower right")
    plt.tight_layout()
    path = f"{OUT}/07_buffett_top20.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_roe_vs_debt(scores: pl.DataFrame) -> None:
    """Scatter – ROE % vs Debt/Equity per sector. Buffett's sweet spot: top-left."""
    df = (
        scores
        .filter(
            pl.col("roe_pct").is_not_null()
            & pl.col("debt_to_equity").is_not_null()
            & (pl.col("roe_pct").abs() < 200)      # drop extreme outliers
            & (pl.col("debt_to_equity") < 600)
        )
        .to_pandas()
    )
    fig, ax = plt.subplots(figsize=(13, 8))
    for sector, grp in df.groupby("sector"):
        ax.scatter(grp["debt_to_equity"], grp["roe_pct"],
                   label=sector, alpha=0.7, s=45, linewidths=0)
    ax.axhline(15, color="green", linestyle="--", linewidth=1.2,
               label="ROE = 15 % (Buffett threshold)")
    ax.axvline(50, color="red",   linestyle="--", linewidth=1.2,
               label="D/E = 0.5× (Buffett threshold)")
    ax.set_xlabel("Debt / Equity  (yfinance units: 100 ≈ 1.0×)")
    ax.set_ylabel("Return on Equity (%)")
    ax.set_title(
        "S&P 500 – ROE vs. Debt/Equity\n"
        "Buffett sweet spot: top-left quadrant (high ROE, low debt)"
    )
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    path = f"{OUT}/08_roe_vs_debt.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_buffett_scorecard(scores: pl.DataFrame) -> None:
    """Heatmap – per-criterion scores (0 / 10 / 20) for the top 20 stocks."""
    top  = scores.head(20).to_pandas().set_index("ticker")
    cols = ["score_roe", "score_debt", "score_margin", "score_liquidity", "score_roa"]
    lbls = ["ROE\n≥15%", "Low Debt\n<0.5×", "Net Margin\n≥20%",
            "Liquidity\nCR≥2.0", "ROA\n≥10%"]
    heat = top[cols].rename(columns=dict(zip(cols, lbls)))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=0, vmax=20, linewidths=0.5, ax=ax, annot_kws={"size": 10})
    ax.set_xlim(0, 6.5)   # make room for total-score annotations
    ax.text(5.15, -0.35, "Total", fontsize=9, fontweight="bold")
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(5.15, i + 0.5, f"{int(row['buffett_score'])}/100",
                va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set_title(
        "Buffett Scorecard – Top 20 S&P 500 Stocks\n"
        "(0 = criterion not met · 10 = partial · 20 = fully met · 100 pts max)"
    )
    ax.set_ylabel("")
    plt.tight_layout()
    path = f"{OUT}/09_buffett_scorecard.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    meta    = get_sp500_meta()
    prices  = get_week_prices(meta["ticker"].to_list())
    returns = weekly_returns(prices, meta)
    sectors = sector_avg(returns)

    print("Generating price / return charts …")
    plot_weekly_returns_bar(returns)
    plot_sector_boxplot(returns)
    plot_normalised_lines(prices, returns)
    plot_return_distribution(returns)
    plot_correlation_heatmap(prices)
    plot_sector_avg_return(sectors)

    fundamentals = get_fundamental_data(meta["ticker"].to_list())
    scores       = buffett_scores(fundamentals, meta)

    print("Generating Buffett ratio charts …")
    plot_buffett_top20(scores)
    plot_roe_vs_debt(scores)
    plot_buffett_scorecard(scores)

    print("\nDone! Nine PNG files written to the current directory.")


if __name__ == "__main__":
    main()
