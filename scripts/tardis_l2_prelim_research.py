#!/usr/bin/env python3
"""Preliminary research summaries from the Rust L2 1m feature lake.

This intentionally uses already-materialized fill-local L2 features:

* BBO spread and imbalance
* simulated market-order costs for fixed USD notionals
* completion/invalid-book rates

The output is not a signal backtest or final alpha/phase-robustness test; it is
a microstructure/execution-cost atlas over the materialized feature scope.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "tardis" / "features" / "rust_l2_engine_streaming_sweep"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "tardis" / "research" / "prelim_l2_198d"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "l2_preliminary_microstructure_current.md"

NOTIONALS = (1_000, 10_000, 100_000)
REFS = ("mid", "micro")
SIDES = ("buy", "sell")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    p.add_argument("--exchange", default="binance-futures")
    p.add_argument("--grid", default="1m")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--markdown-out", type=Path, default=DEFAULT_DOC)
    p.add_argument("--limit-files", type=int, help="debug only: read first N parquet files")
    p.add_argument("--start-date", help="inclusive YYYY-MM-DD date filter over hive date partition")
    p.add_argument("--end-date", help="inclusive YYYY-MM-DD date filter over hive date partition")
    p.add_argument("--stable-quote-only", action=argparse.BooleanOptionalAction, default=True, help="keep symbols ending in USDT/USDC")
    p.add_argument("--exclude-symbol", action="append", default=["FUTURES"], help="symbol to exclude; can repeat")
    return p.parse_args()


def collect_lazy(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def parquet_inputs(args: argparse.Namespace) -> str | list[str]:
    pattern = (
        args.feature_root
        / f"exchange={args.exchange}"
        / "symbol=*"
        / "date=*"
        / f"grid={args.grid}"
        / "slippage.parquet"
    )
    if args.limit_files is None:
        return str(pattern)
    paths = sorted(str(p) for p in pattern.parent.parent.parent.parent.glob(f"symbol=*/date=*/grid={args.grid}/slippage.parquet"))
    return paths[: args.limit_files]


def safe_cost(side: str, notional: int, ref: str) -> pl.Expr:
    cost = f"{side}_{notional}_{ref}_cost_bps"
    complete = f"{side}_{notional}_{ref}_complete"
    return pl.when(pl.col(complete) & pl.col(cost).is_finite()).then(pl.col(cost)).otherwise(None)


def base_scan(args: argparse.Namespace) -> pl.LazyFrame:
    inputs = parquet_inputs(args)
    lf = pl.scan_parquet(inputs, hive_partitioning=True)
    if args.start_date:
        lf = lf.filter(pl.col("date") >= pl.lit(args.start_date).str.strptime(pl.Date))
    if args.end_date:
        lf = lf.filter(pl.col("date") <= pl.lit(args.end_date).str.strptime(pl.Date))
    if args.stable_quote_only:
        lf = lf.filter(pl.col("symbol").str.ends_with("USDT") | pl.col("symbol").str.ends_with("USDC"))
    if args.exclude_symbol:
        lf = lf.filter(~pl.col("symbol").is_in(args.exclude_symbol))
    minute = ((pl.col("sample_local_timestamp_us") // 60_000_000) % 1_440).cast(pl.Int16)
    utc0_abs = pl.min_horizontal(minute, 1_440 - minute).cast(pl.Int16)
    return lf.with_columns(
        [
            minute.alias("minute_of_day"),
            (minute // 60).cast(pl.Int8).alias("hour_utc"),
            utc0_abs.alias("utc0_abs_min"),
            (utc0_abs <= 120).alias("is_utc0_2h"),
            pl.when(utc0_abs <= 15)
            .then(pl.lit("utc0_00_15m"))
            .when(utc0_abs <= 60)
            .then(pl.lit("utc0_15_60m"))
            .when(utc0_abs <= 120)
            .then(pl.lit("utc0_60_120m"))
            .otherwise(pl.lit("rest"))
            .alias("utc0_bucket"),
            ((~pl.col("invalid_book_flag")) & pl.col("spread_bps").is_finite()).alias("valid_bbo"),
            ((safe_cost("buy", 10_000, "mid") + safe_cost("sell", 10_000, "mid")) / 2).alias("avg_10000_mid_cost_bps"),
            ((safe_cost("buy", 100_000, "mid") + safe_cost("sell", 100_000, "mid")) / 2).alias("avg_100000_mid_cost_bps"),
            ((safe_cost("buy", 10_000, "micro") + safe_cost("sell", 10_000, "micro")) / 2).alias("avg_10000_micro_cost_bps"),
            ((safe_cost("buy", 100_000, "micro") + safe_cost("sell", 100_000, "micro")) / 2).alias("avg_100000_micro_cost_bps"),
        ]
    )


def metric_aggs(prefix: str = "") -> list[pl.Expr]:
    return [
        pl.len().alias(f"{prefix}rows"),
        pl.col("valid_bbo").mean().alias(f"{prefix}valid_bbo_rate"),
        pl.col("invalid_book_flag").mean().alias(f"{prefix}invalid_book_rate"),
        pl.col("spread_bps").filter(pl.col("valid_bbo")).mean().alias(f"{prefix}spread_mean_bps"),
        pl.col("spread_bps").filter(pl.col("valid_bbo")).median().alias(f"{prefix}spread_median_bps"),
        pl.col("spread_bps").filter(pl.col("valid_bbo")).quantile(0.90).alias(f"{prefix}spread_p90_bps"),
        pl.col("spread_bps").filter(pl.col("valid_bbo")).quantile(0.99).alias(f"{prefix}spread_p99_bps"),
        pl.col("avg_10000_mid_cost_bps").mean().alias(f"{prefix}avg_10000_mid_mean_bps"),
        pl.col("avg_10000_mid_cost_bps").quantile(0.90).alias(f"{prefix}avg_10000_mid_p90_bps"),
        pl.col("avg_100000_mid_cost_bps").mean().alias(f"{prefix}avg_100000_mid_mean_bps"),
        pl.col("avg_100000_mid_cost_bps").quantile(0.90).alias(f"{prefix}avg_100000_mid_p90_bps"),
        pl.col("avg_10000_micro_cost_bps").mean().alias(f"{prefix}avg_10000_micro_mean_bps"),
        pl.col("avg_100000_micro_cost_bps").mean().alias(f"{prefix}avg_100000_micro_mean_bps"),
        pl.col("buy_100000_mid_complete").mean().alias(f"{prefix}buy_100000_mid_complete_rate"),
        pl.col("sell_100000_mid_complete").mean().alias(f"{prefix}sell_100000_mid_complete_rate"),
    ]


def write_df(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    df.write_csv(path.with_suffix(".csv"))


def make_profiles(lf: pl.LazyFrame, out_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    coverage = collect_lazy(
        lf.select(
            [
                pl.len().alias("rows"),
                pl.col("symbol").n_unique().alias("symbols"),
                pl.col("date").n_unique().alias("dates"),
                pl.col("date").min().alias("first_date"),
                pl.col("date").max().alias("last_date"),
                pl.col("valid_bbo").mean().alias("valid_bbo_rate"),
                pl.col("invalid_book_flag").mean().alias("invalid_book_rate"),
                pl.col("spread_bps").filter(pl.col("valid_bbo")).mean().alias("spread_mean_bps"),
                pl.col("spread_bps").filter(pl.col("valid_bbo")).median().alias("spread_median_bps"),
                pl.col("spread_bps").filter(pl.col("valid_bbo")).quantile(0.90).alias("spread_p90_bps"),
                pl.col("avg_100000_mid_cost_bps").mean().alias("avg_100000_mid_mean_bps"),
                pl.col("avg_100000_mid_cost_bps").quantile(0.90).alias("avg_100000_mid_p90_bps"),
            ]
        )
    )
    write_df(coverage, out_dir / "coverage_summary.parquet")
    outputs["coverage"] = out_dir / "coverage_summary.parquet"

    minute = collect_lazy(
        lf.group_by("minute_of_day")
        .agg(metric_aggs())
        .with_columns(
            [
                (pl.col("minute_of_day") // 60).cast(pl.Int8).alias("hour_utc"),
                (pl.col("minute_of_day") % 60).cast(pl.Int8).alias("minute_utc"),
            ]
        )
        .sort("minute_of_day")
    )
    write_df(minute, out_dir / "minute_of_day_profile.parquet")
    outputs["minute"] = out_dir / "minute_of_day_profile.parquet"

    hour = collect_lazy(lf.group_by("hour_utc").agg(metric_aggs()).sort("hour_utc"))
    write_df(hour, out_dir / "hour_utc_profile.parquet")
    outputs["hour"] = out_dir / "hour_utc_profile.parquet"

    window = collect_lazy(lf.group_by("utc0_bucket").agg(metric_aggs()).sort("utc0_bucket"))
    write_df(window, out_dir / "utc0_window_profile.parquet")
    outputs["window"] = out_dir / "utc0_window_profile.parquet"

    symbol = collect_lazy(
        lf.group_by("symbol")
        .agg([pl.col("date").n_unique().alias("dates_seen"), *metric_aggs()])
        .sort(["spread_p90_bps", "avg_100000_mid_p90_bps"], descending=[True, True])
    )
    write_df(symbol, out_dir / "symbol_summary.parquet")
    outputs["symbol"] = out_dir / "symbol_summary.parquet"

    date = collect_lazy(lf.group_by("date").agg([pl.col("symbol").n_unique().alias("symbols"), *metric_aggs()]).sort("date"))
    write_df(date, out_dir / "date_summary.parquet")
    outputs["date"] = out_dir / "date_summary.parquet"

    by_symbol_window = collect_lazy(
        lf.group_by(["symbol", "is_utc0_2h"])
        .agg(
            [
                pl.col("date").n_unique().alias("dates_seen"),
                pl.len().alias("rows"),
                pl.col("spread_bps").filter(pl.col("valid_bbo")).mean().alias("spread_mean_bps"),
                pl.col("avg_100000_mid_cost_bps").mean().alias("avg_100000_mid_mean_bps"),
                pl.col("buy_100000_mid_complete").mean().alias("buy_100000_mid_complete_rate"),
                pl.col("sell_100000_mid_complete").mean().alias("sell_100000_mid_complete_rate"),
            ]
        )
        .sort(["symbol", "is_utc0_2h"])
    )
    write_df(by_symbol_window, out_dir / "symbol_utc0_vs_rest_profile.parquet")
    outputs["symbol_window"] = out_dir / "symbol_utc0_vs_rest_profile.parquet"

    by_date_window = collect_lazy(
        lf.group_by(["date", "is_utc0_2h"])
        .agg(
            [
                pl.col("symbol").n_unique().alias("symbols"),
                pl.len().alias("rows"),
                pl.col("spread_bps").filter(pl.col("valid_bbo")).mean().alias("spread_mean_bps"),
                pl.col("avg_100000_mid_cost_bps").mean().alias("avg_100000_mid_mean_bps"),
            ]
        )
        .sort(["date", "is_utc0_2h"])
    )
    write_df(by_date_window, out_dir / "date_utc0_vs_rest_profile.parquet")
    outputs["date_window"] = out_dir / "date_utc0_vs_rest_profile.parquet"

    return outputs


def pivot_delta(df: pl.DataFrame, index: str, metric: str) -> pl.DataFrame:
    wide = df.select([index, "is_utc0_2h", metric, "rows"]).pivot(
        values=[metric, "rows"], index=index, on="is_utc0_2h", aggregate_function="first"
    )
    false_col = f"{metric}_false"
    true_col = f"{metric}_true"
    if false_col not in wide.columns or true_col not in wide.columns:
        return wide
    return wide.with_columns(
        [
            (pl.col(true_col) - pl.col(false_col)).alias(f"{metric}_utc0_minus_rest"),
            (pl.col(true_col) / pl.col(false_col) - 1).alias(f"{metric}_utc0_rel_delta"),
        ]
    ).sort(f"{metric}_utc0_minus_rest", descending=True)


def format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.6g}"
    return str(value)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def top_rows(df: pl.DataFrame, cols: Iterable[str], n: int = 12) -> list[str]:
    if df.is_empty():
        return ["_no rows_"]
    selected = [c for c in cols if c in df.columns]
    rows = df.select(selected).head(n).to_dicts()
    if not rows:
        return ["_no rows_"]
    lines = [
        "| " + " | ".join(selected) + " |",
        "| " + " | ".join("---" for _ in selected) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(c)) for c in selected) + " |")
    return lines


def write_markdown(args: argparse.Namespace, outputs: dict[str, Path]) -> None:
    coverage = pl.read_parquet(outputs["coverage"])
    symbol = pl.read_parquet(outputs["symbol"])
    minute = pl.read_parquet(outputs["minute"])
    window = pl.read_parquet(outputs["window"])
    date = pl.read_parquet(outputs["date"])
    symbol_window = pl.read_parquet(outputs["symbol_window"])
    date_window = pl.read_parquet(outputs["date_window"])

    symbol_spread_delta = pivot_delta(symbol_window, "symbol", "spread_mean_bps")
    symbol_cost_delta = pivot_delta(symbol_window, "symbol", "avg_100000_mid_mean_bps")
    date_spread_delta = pivot_delta(date_window, "date", "spread_mean_bps")

    cov = coverage.row(0, named=True)
    top_minutes = minute.sort("avg_100000_mid_p90_bps", descending=True).head(20)
    best_minutes = minute.sort("avg_100000_mid_mean_bps").head(20)
    worst_symbols = symbol.sort("avg_100000_mid_p90_bps", descending=True).head(20)
    worst_spread_symbols = symbol.sort("spread_p90_bps", descending=True).head(20)
    worst_dates = date.sort("avg_100000_mid_p90_bps", descending=True).head(20)

    lines = [
        "# Preliminary L2 microstructure research from current feature lake",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Scope",
        "",
        f"- Exchange: `{args.exchange}`",
        f"- Grid: `{args.grid}`",
        f"- Symbol filter: stable quote only = `{args.stable_quote_only}`, excluded = `{', '.join(args.exclude_symbol or [])}`",
        f"- Rows: `{cov['rows']:,}`",
        f"- Symbols: `{cov['symbols']:,}`",
        f"- Dates: `{cov['dates']:,}` from `{cov['first_date']}` to `{cov['last_date']}`",
        "- Data state: materialized feature lake passed via `--feature-root`; this report covers exactly the rows and dates listed above.",
        "",
        "## Global execution-cost snapshot",
        "",
        f"- Valid BBO rate: `{cov['valid_bbo_rate']:.6f}`",
        f"- Invalid-book rate: `{cov['invalid_book_rate']:.6f}`",
        f"- Spread mean / median / p90: `{cov['spread_mean_bps']:.4f}` / `{cov['spread_median_bps']:.4f}` / `{cov['spread_p90_bps']:.4f}` bps",
        f"- Avg buy/sell 100k-mid cost mean / p90: `{cov['avg_100000_mid_mean_bps']:.4f}` / `{cov['avg_100000_mid_p90_bps']:.4f}` bps",
        "",
        "## UTC0 window profile",
        "",
        *top_rows(
            window,
            [
                "utc0_bucket",
                "rows",
                "spread_mean_bps",
                "spread_p90_bps",
                "avg_100000_mid_mean_bps",
                "avg_100000_mid_p90_bps",
                "buy_100000_mid_complete_rate",
            ],
            n=10,
        ),
        "",
        "## Worst minutes by 100k-mid p90 cost",
        "",
        *top_rows(
            top_minutes,
            [
                "minute_of_day",
                "hour_utc",
                "minute_utc",
                "rows",
                "spread_p90_bps",
                "avg_100000_mid_mean_bps",
                "avg_100000_mid_p90_bps",
                "buy_100000_mid_complete_rate",
            ],
            n=20,
        ),
        "",
        "## Cheapest minutes by 100k-mid mean cost",
        "",
        *top_rows(
            best_minutes,
            [
                "minute_of_day",
                "hour_utc",
                "minute_utc",
                "rows",
                "spread_mean_bps",
                "avg_100000_mid_mean_bps",
                "buy_100000_mid_complete_rate",
            ],
            n=20,
        ),
        "",
        "## Worst symbols by 100k-mid p90 cost",
        "",
        *top_rows(
            worst_symbols,
            [
                "symbol",
                "dates_seen",
                "rows",
                "spread_p90_bps",
                "avg_100000_mid_mean_bps",
                "avg_100000_mid_p90_bps",
                "buy_100000_mid_complete_rate",
            ],
            n=20,
        ),
        "",
        "## Worst symbols by spread p90",
        "",
        *top_rows(
            worst_spread_symbols,
            [
                "symbol",
                "dates_seen",
                "rows",
                "spread_mean_bps",
                "spread_p90_bps",
                "avg_100000_mid_mean_bps",
                "buy_100000_mid_complete_rate",
            ],
            n=20,
        ),
        "",
        "## Symbols with largest UTC0(±2h) spread penalty vs rest",
        "",
        *top_rows(
            symbol_spread_delta,
            [
                "symbol",
                "spread_mean_bps_false",
                "spread_mean_bps_true",
                "spread_mean_bps_utc0_minus_rest",
                "spread_mean_bps_utc0_rel_delta",
                "rows_false",
                "rows_true",
            ],
            n=20,
        ),
        "",
        "## Symbols with largest UTC0(±2h) 100k-mid cost penalty vs rest",
        "",
        *top_rows(
            symbol_cost_delta,
            [
                "symbol",
                "avg_100000_mid_mean_bps_false",
                "avg_100000_mid_mean_bps_true",
                "avg_100000_mid_mean_bps_utc0_minus_rest",
                "avg_100000_mid_mean_bps_utc0_rel_delta",
                "rows_false",
                "rows_true",
            ],
            n=20,
        ),
        "",
        "## Worst dates by 100k-mid p90 cost",
        "",
        *top_rows(
            worst_dates,
            [
                "date",
                "symbols",
                "rows",
                "spread_p90_bps",
                "avg_100000_mid_mean_bps",
                "avg_100000_mid_p90_bps",
            ],
            n=20,
        ),
        "",
        "## Dates with largest UTC0(±2h) spread penalty vs rest",
        "",
        *top_rows(
            date_spread_delta,
            [
                "date",
                "spread_mean_bps_false",
                "spread_mean_bps_true",
                "spread_mean_bps_utc0_minus_rest",
                "spread_mean_bps_utc0_rel_delta",
                "rows_false",
                "rows_true",
            ],
            n=20,
        ),
        "",
        "## Interpretation guardrails",
        "",
        "- This is useful for execution-cost atlas, UTC0-window diagnostics, and thin-alt ranking.",
        "- It should not be over-extrapolated beyond the date range shown above; extend the date range before making regime-wide claims.",
        "- It does not test signal-anchor phase robustness; it tests whether fill-local execution conditions vary by clock phase.",
        "- Re-run this report after the full 2024..2026-05 lake is complete, then stratify by historical symbol metadata/regime.",
        "",
        "## Output artifacts",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- `{name}`: `{display_path(path)}`")
    lines.append("")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lf = base_scan(args)
    outputs = make_profiles(lf, args.out_dir)
    manifest = {
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "featureRoot": str(args.feature_root),
        "startDate": args.start_date,
        "endDate": args.end_date,
        "exchange": args.exchange,
        "grid": args.grid,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "markdown": str(args.markdown_out),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args, outputs)
    print(json.dumps({"status": "ok", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
