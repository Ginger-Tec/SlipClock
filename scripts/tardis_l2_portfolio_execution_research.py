#!/usr/bin/env python3
"""Portfolio-execution research tables from the current L2 feature lake.

The experiment is deliberately stylized around a Binance USDⓈ-M long/short
portfolio rebalance:

* choose a tradable symbol universe using ex-post L2 liquidity diagnostics,
* assume equal notional per symbol for a rebalance,
* estimate expected crossing cost and intraday schedule sensitivity from the
  already-materialized 1m order-book slippage features.

It is not a signal backtest.  It answers the practical question: if a strategy
must rebalance many symbols, what universe and execution clock are reasonable
before alpha is even considered?
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / "data" / "tardis" / "features" / "rust_l2_engine_streaming_sweep"
PRELIM_ROOT = PROJECT_ROOT / "data" / "tardis" / "research" / "prelim_l2_198d"
OUT_ROOT = PROJECT_ROOT / "data" / "tardis" / "research" / "portfolio_execution_current"
DOC_PATH = PROJECT_ROOT / "docs" / "l2_portfolio_execution_experiment_current.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    p.add_argument("--prelim-root", type=Path, default=PRELIM_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument("--markdown-out", type=Path, default=DOC_PATH)
    p.add_argument("--exchange", default="binance-futures")
    p.add_argument("--grid", default="1m")
    p.add_argument("--universe-sizes", default="25,50,100,150,200,all")
    p.add_argument("--notionals", default="10000,100000")
    p.add_argument("--start-date", help="inclusive YYYY-MM-DD date filter over hive date partition")
    p.add_argument("--end-date", help="inclusive YYYY-MM-DD date filter over hive date partition")
    p.add_argument("--min-dates", type=int, default=30)
    p.add_argument("--min-complete-rate", type=float, default=0.99)
    return p.parse_args()


def parse_sizes(raw: str, eligible_count: int) -> list[tuple[str, int]]:
    sizes: list[tuple[str, int]] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "all":
            sizes.append(("all", eligible_count))
        else:
            n = int(token)
            sizes.append((f"top{n}", min(n, eligible_count)))
    seen: set[str] = set()
    out: list[tuple[str, int]] = []
    for label, n in sizes:
        if label not in seen:
            seen.add(label)
            out.append((label, n))
    return out


def parse_notionals(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def collect_lazy(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def write_df(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    df.write_csv(path.with_suffix(".csv"))


def format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.6g}"
    return str(value)


def markdown_table(df: pl.DataFrame, cols: Iterable[str], n: int = 20) -> list[str]:
    selected = [c for c in cols if c in df.columns]
    rows = df.select(selected).head(n).to_dicts()
    if not rows:
        return ["_no rows_"]
    lines = ["| " + " | ".join(selected) + " |", "| " + " | ".join("---" for _ in selected) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(c)) for c in selected) + " |")
    return lines


def cost_mean_col(notional: int) -> str:
    return f"avg_{notional}_mid_mean_bps"


def cost_p90_col(notional: int) -> str:
    return f"avg_{notional}_mid_p90_bps"


def row_cost_expr(notional: int) -> pl.Expr:
    buy = f"buy_{notional}_mid_cost_bps"
    sell = f"sell_{notional}_mid_cost_bps"
    buy_complete = f"buy_{notional}_mid_complete"
    sell_complete = f"sell_{notional}_mid_complete"
    buy_safe = pl.when(pl.col(buy_complete) & pl.col(buy).is_finite()).then(pl.col(buy)).otherwise(None)
    sell_safe = pl.when(pl.col(sell_complete) & pl.col(sell).is_finite()).then(pl.col(sell)).otherwise(None)
    return ((buy_safe + sell_safe) / 2).alias(f"avg_{notional}_mid_cost_bps")


def load_ranked_symbols(args: argparse.Namespace, notionals: list[int]) -> pl.DataFrame:
    path = args.prelim_root / "symbol_summary.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/tardis_l2_prelim_research.py first")
    df = pl.read_parquet(path)
    stable_quote = pl.col("symbol").str.ends_with("USDT") | pl.col("symbol").str.ends_with("USDC")
    eligible = (
        df.filter(
            (pl.col("dates_seen") >= args.min_dates)
            & (pl.col("buy_100000_mid_complete_rate") >= args.min_complete_rate)
            & pl.col("avg_100000_mid_p90_bps").is_not_null()
            & stable_quote
        )
        .sort(["avg_100000_mid_p90_bps", "spread_p90_bps", "symbol"])
        .with_row_index("liquidity_rank", offset=1)
    )
    cols = [
        "liquidity_rank",
        "symbol",
        "dates_seen",
        "rows",
        "spread_mean_bps",
        "spread_p90_bps",
        "buy_100000_mid_complete_rate",
    ]
    for n in notionals:
        for c in [cost_mean_col(n), cost_p90_col(n)]:
            if c in eligible.columns:
                cols.append(c)
    return eligible.select(cols)


def build_universe_capacity(ranked: pl.DataFrame, sizes: list[tuple[str, int]], notionals: list[int]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for label, n_symbols in sizes:
        uni = ranked.head(n_symbols)
        for notional in notionals:
            mean_col = cost_mean_col(notional)
            p90_col = cost_p90_col(notional)
            cost_mean = uni[mean_col].mean()
            symbol_cost_p90 = uni[mean_col].quantile(0.90)
            p90_mean = uni[p90_col].mean() if p90_col in uni.columns else None
            gross = int(n_symbols * notional)
            rows.append(
                {
                    "universe": label,
                    "symbols": n_symbols,
                    "notional_per_symbol_usd": notional,
                    "gross_rebalance_usd": gross,
                    "equal_weight_mean_cost_bps": cost_mean,
                    "symbol_mean_cost_p90_bps": symbol_cost_p90,
                    "mean_of_symbol_p90_cost_bps": p90_mean,
                    "expected_cost_usd": gross * float(cost_mean or 0.0) / 10_000,
                    "spread_p90_mean_bps": uni["spread_p90_bps"].mean(),
                    "complete_rate_min": uni["buy_100000_mid_complete_rate"].min(),
                    "worst_symbol_by_mean_cost": uni.sort(mean_col, descending=True)["symbol"][0],
                }
            )
    return pl.DataFrame(rows)


def scan_symbol_minute(args: argparse.Namespace, symbols: list[str], notionals: list[int]) -> pl.DataFrame:
    pattern = (
        args.feature_root
        / f"exchange={args.exchange}"
        / "symbol=*"
        / "date=*"
        / f"grid={args.grid}"
        / "slippage.parquet"
    )
    minute = ((pl.col("sample_local_timestamp_us") // 60_000_000) % 1_440).cast(pl.Int16)
    lf = pl.scan_parquet(str(pattern), hive_partitioning=True).filter(pl.col("symbol").is_in(symbols))
    if args.start_date:
        lf = lf.filter(pl.col("date") >= pl.lit(args.start_date).str.strptime(pl.Date))
    if args.end_date:
        lf = lf.filter(pl.col("date") <= pl.lit(args.end_date).str.strptime(pl.Date))
    lf = (
        lf.with_columns(
            [
                minute.alias("minute_of_day"),
                (minute // 60).cast(pl.Int8).alias("hour_utc"),
                (minute % 60).cast(pl.Int8).alias("minute_utc"),
                pl.min_horizontal(minute, 1_440 - minute).cast(pl.Int16).alias("utc0_abs_min"),
                *[row_cost_expr(n) for n in notionals],
            ]
        )
        .group_by(["symbol", "minute_of_day", "hour_utc", "minute_utc"])
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("date").n_unique().alias("dates_seen"),
                pl.col("spread_bps").mean().alias("spread_mean_bps"),
                *[pl.col(f"avg_{n}_mid_cost_bps").mean().alias(f"avg_{n}_mid_cost_mean_bps") for n in notionals],
                *[pl.col(f"avg_{n}_mid_cost_bps").quantile(0.90).alias(f"avg_{n}_mid_cost_p90_bps") for n in notionals],
            ]
        )
    )
    return collect_lazy(lf)


def build_time_profiles(
    symbol_minute: pl.DataFrame,
    ranked: pl.DataFrame,
    sizes: list[tuple[str, int]],
    notionals: list[int],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    profile_rows: list[pl.DataFrame] = []
    utc0_rows: list[dict[str, object]] = []
    for label, n_symbols in sizes:
        symbols = ranked.head(n_symbols)["symbol"].to_list()
        subset = symbol_minute.filter(pl.col("symbol").is_in(symbols))
        for notional in notionals:
            cost = f"avg_{notional}_mid_cost_mean_bps"
            p90 = f"avg_{notional}_mid_cost_p90_bps"
            per_minute = (
                subset.group_by(["minute_of_day", "hour_utc", "minute_utc"])
                .agg(
                    [
                        pl.col("symbol").n_unique().alias("symbols"),
                        pl.col(cost).mean().alias("equal_weight_cost_mean_bps"),
                        pl.col(p90).mean().alias("mean_symbol_p90_cost_bps"),
                        pl.col("spread_mean_bps").mean().alias("equal_weight_spread_mean_bps"),
                    ]
                )
                .with_columns(
                    [
                        pl.lit(label).alias("universe"),
                        pl.lit(n_symbols).alias("universe_symbols"),
                        pl.lit(notional).alias("notional_per_symbol_usd"),
                    ]
                )
            )
            profile_rows.append(per_minute)

            with_bucket = per_minute.with_columns(
                (pl.min_horizontal(pl.col("minute_of_day"), 1_440 - pl.col("minute_of_day")) <= 120).alias("is_utc0_2h")
            )
            rest = with_bucket.filter(~pl.col("is_utc0_2h"))["equal_weight_cost_mean_bps"].mean()
            utc0 = with_bucket.filter(pl.col("is_utc0_2h"))["equal_weight_cost_mean_bps"].mean()
            best = with_bucket.sort("equal_weight_cost_mean_bps").head(1).row(0, named=True)
            worst = with_bucket.sort("equal_weight_cost_mean_bps", descending=True).head(1).row(0, named=True)
            utc0_rows.append(
                {
                    "universe": label,
                    "symbols": n_symbols,
                    "notional_per_symbol_usd": notional,
                    "rest_mean_cost_bps": rest,
                    "utc0_2h_mean_cost_bps": utc0,
                    "utc0_minus_rest_bps": (utc0 or 0.0) - (rest or 0.0),
                    "utc0_rel_delta": ((utc0 or 0.0) / rest - 1) if rest else None,
                    "best_minute_utc": f"{int(best['hour_utc']):02d}:{int(best['minute_utc']):02d}",
                    "best_minute_cost_bps": best["equal_weight_cost_mean_bps"],
                    "worst_minute_utc": f"{int(worst['hour_utc']):02d}:{int(worst['minute_utc']):02d}",
                    "worst_minute_cost_bps": worst["equal_weight_cost_mean_bps"],
                    "worst_minus_best_bps": worst["equal_weight_cost_mean_bps"] - best["equal_weight_cost_mean_bps"],
                }
            )
    minute_profile = pl.concat(profile_rows).select(
        [
            "universe",
            "universe_symbols",
            "notional_per_symbol_usd",
            "minute_of_day",
            "hour_utc",
            "minute_utc",
            "symbols",
            "equal_weight_cost_mean_bps",
            "mean_symbol_p90_cost_bps",
            "equal_weight_spread_mean_bps",
        ]
    )
    return minute_profile, pl.DataFrame(utc0_rows)


def write_markdown(
    args: argparse.Namespace,
    ranked: pl.DataFrame,
    capacity: pl.DataFrame,
    minute_profile: pl.DataFrame,
    utc0: pl.DataFrame,
    outputs: dict[str, Path],
) -> None:
    coverage_scope = "- Current data: materialized feature lake passed via `--feature-root`."
    coverage_path = args.prelim_root / "coverage_summary.parquet"
    if coverage_path.exists():
        try:
            cov = pl.read_parquet(coverage_path).row(0, named=True)
            coverage_scope = (
                f"- Current data: `{cov['dates']}` UTC dates, "
                f"`{cov['first_date']}`..`{cov['last_date']}`, "
                f"`{cov['symbols']}` stable-quote symbols in the prelim coverage table."
            )
        except Exception:
            coverage_scope = (
                "- Current data: materialized feature lake passed via `--feature-root`; "
                "coverage summary could not be read from `--prelim-root`."
            )
    best_100k = capacity.filter(pl.col("notional_per_symbol_usd") == 100_000).sort("symbols")
    worst_symbols = ranked.sort("avg_100000_mid_mean_bps", descending=True).head(20)
    best_symbols = ranked.sort("avg_100000_mid_mean_bps").head(20)
    all_100k_schedule = utc0.filter((pl.col("universe") == "all") & (pl.col("notional_per_symbol_usd") == 100_000))
    top100_100k_schedule = utc0.filter((pl.col("universe") == "top100") & (pl.col("notional_per_symbol_usd") == 100_000))
    worst_minutes_all = (
        minute_profile.filter((pl.col("universe") == "all") & (pl.col("notional_per_symbol_usd") == 100_000))
        .sort("equal_weight_cost_mean_bps", descending=True)
        .head(20)
    )
    best_minutes_all = (
        minute_profile.filter((pl.col("universe") == "all") & (pl.col("notional_per_symbol_usd") == 100_000))
        .sort("equal_weight_cost_mean_bps")
        .head(20)
    )
    lines = [
        "# Portfolio execution experiment from current L2 feature lake",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Research question",
        "",
        "When a Binance USDⓈ-M long/short strategy must rebalance many symbols, naive backtest fills at mid/close ignore a first-order cost surface: spread, depth, completion, and clock-time congestion.  This experiment turns the current L2 feature lake into a universe and schedule diagnostic before any alpha signal is evaluated.",
        "",
        "## Current-data scope",
        "",
        coverage_scope,
        f"- Eligibility: stable quote symbols, at least `{args.min_dates}` dates, 100k buy completion rate >= `{args.min_complete_rate}`.",
        f"- Eligible symbols after filter: `{ranked.height}`.",
        "- Costs are fill-local L2 simulated market-order costs using fixed per-symbol notionals, averaged across buy and sell directions.",
        "",
        "## Universe capacity curve, 100k USD per symbol",
        "",
        *markdown_table(
            best_100k,
            [
                "universe",
                "symbols",
                "gross_rebalance_usd",
                "equal_weight_mean_cost_bps",
                "symbol_mean_cost_p90_bps",
                "mean_of_symbol_p90_cost_bps",
                "expected_cost_usd",
                "spread_p90_mean_bps",
                "complete_rate_min",
                "worst_symbol_by_mean_cost",
            ],
            n=20,
        ),
        "",
        "## UTC0 vs rest and best/worst minute, 100k USD per symbol",
        "",
        *markdown_table(
            utc0.filter(pl.col("notional_per_symbol_usd") == 100_000),
            [
                "universe",
                "symbols",
                "rest_mean_cost_bps",
                "utc0_2h_mean_cost_bps",
                "utc0_minus_rest_bps",
                "utc0_rel_delta",
                "best_minute_utc",
                "best_minute_cost_bps",
                "worst_minute_utc",
                "worst_minute_cost_bps",
                "worst_minus_best_bps",
            ],
            n=20,
        ),
        "",
        "## Most liquid symbols by 100k mean cost",
        "",
        *markdown_table(
            best_symbols,
            [
                "liquidity_rank",
                "symbol",
                "dates_seen",
                "spread_p90_bps",
                "avg_100000_mid_mean_bps",
                "avg_100000_mid_p90_bps",
                "buy_100000_mid_complete_rate",
            ],
            n=20,
        ),
        "",
        "## Most expensive eligible symbols by 100k mean cost",
        "",
        *markdown_table(
            worst_symbols,
            [
                "liquidity_rank",
                "symbol",
                "dates_seen",
                "spread_p90_bps",
                "avg_100000_mid_mean_bps",
                "avg_100000_mid_p90_bps",
                "buy_100000_mid_complete_rate",
            ],
            n=20,
        ),
        "",
        "## Worst all-universe minutes, 100k USD per symbol",
        "",
        *markdown_table(
            worst_minutes_all,
            [
                "minute_of_day",
                "hour_utc",
                "minute_utc",
                "equal_weight_cost_mean_bps",
                "mean_symbol_p90_cost_bps",
                "equal_weight_spread_mean_bps",
            ],
            n=20,
        ),
        "",
        "## Best all-universe minutes, 100k USD per symbol",
        "",
        *markdown_table(
            best_minutes_all,
            [
                "minute_of_day",
                "hour_utc",
                "minute_utc",
                "equal_weight_cost_mean_bps",
                "mean_symbol_p90_cost_bps",
                "equal_weight_spread_mean_bps",
            ],
            n=20,
        ),
        "",
        "## Directional interpretation",
        "",
        "- The economically relevant question is not whether 00:00 UTC is universally toxic; it is how much execution budget is consumed by the chosen tradable universe and rebalance schedule.",
        "- A liquidity-ranked universe gives an immediate capacity curve: top-N universes are much cheaper and more reliable than the full tail of listed contracts.",
        "- Clock-time effects exist; in this materialized scope, the difference between the best and worst minute for the same universe can be larger than the UTC0-vs-rest contrast.",
        "- This supports an ICAIF framing around execution-aware universe construction and schedule diagnostics for AI/quant crypto portfolios.",
        "",
        "## Output artifacts",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- `{name}`: `{path.resolve().relative_to(PROJECT_ROOT)}`")
    lines.append("")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    notionals = parse_notionals(args.notionals)
    ranked = load_ranked_symbols(args, notionals)
    sizes = parse_sizes(args.universe_sizes, ranked.height)
    capacity = build_universe_capacity(ranked, sizes, notionals)

    union_symbols = ranked.head(max(n for _, n in sizes))["symbol"].to_list()
    symbol_minute = scan_symbol_minute(args, union_symbols, notionals)
    minute_profile, utc0 = build_time_profiles(symbol_minute, ranked, sizes, notionals)

    outputs = {
        "symbol_liquidity_rank": args.out_dir / "symbol_liquidity_rank.parquet",
        "universe_capacity": args.out_dir / "universe_capacity_curve.parquet",
        "symbol_minute": args.out_dir / "symbol_minute_profile.parquet",
        "universe_minute": args.out_dir / "universe_minute_profile.parquet",
        "universe_utc0": args.out_dir / "universe_utc0_comparison.parquet",
    }
    write_df(ranked, outputs["symbol_liquidity_rank"])
    write_df(capacity, outputs["universe_capacity"])
    write_df(symbol_minute, outputs["symbol_minute"])
    write_df(minute_profile, outputs["universe_minute"])
    write_df(utc0, outputs["universe_utc0"])
    manifest = {
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "exchange": args.exchange,
        "grid": args.grid,
        "startDate": args.start_date,
        "endDate": args.end_date,
        "minDates": args.min_dates,
        "minCompleteRate": args.min_complete_rate,
        "eligibleSymbols": ranked.height,
        "notionals": notionals,
        "universes": [{"label": label, "symbols": n} for label, n in sizes],
        "outputs": {k: str(v) for k, v in outputs.items()},
        "markdown": str(args.markdown_out),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args, ranked, capacity, minute_profile, utc0, outputs)
    print(json.dumps({"status": "ok", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
