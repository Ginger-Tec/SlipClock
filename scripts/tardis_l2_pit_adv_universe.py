#!/usr/bin/env python3
"""Point-in-time ADV universe tests for L2 execution costs.

This experiment turns the ex-post L2 liquidity frontier into a tradable
point-in-time universe construction test.  For each evaluation date t, symbols
are ranked only by quote-volume information observed before t.  The selected
universe is then evaluated on t using realized L2 market-order slippage.

The intended paper interpretation is:

* ex-post L2 cost ranks = oracle liquidity frontier / lower bound;
* trailing ADV ranks = live-feasible universe rule;
* the gap between them quantifies how much implementability is lost when the
  universe must be chosen from pre-trade information.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / "data" / "tardis" / "features" / "l2_1m_usdm_2024_2026_v1"
VISION_ROOT = PROJECT_ROOT / "data" / "binance_vision" / "parquet" / "market=futures_um"
ORACLE_ROOT = PROJECT_ROOT / "data" / "tardis" / "research" / "portfolio_execution_2024_01_02"
OUT_ROOT = PROJECT_ROOT / "data" / "tardis" / "research" / "portfolio_execution_pit_adv_2024_01_02"
DOC_PATH = PROJECT_ROOT / "docs" / "l2_pit_adv_universe_2024_01_02.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    p.add_argument("--vision-root", type=Path, default=VISION_ROOT)
    p.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument("--markdown-out", type=Path, default=DOC_PATH)
    p.add_argument("--exchange", default="binance-futures")
    p.add_argument("--grid", default="1m")
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--end-date", default="2024-02-29")
    p.add_argument("--lookbacks", default="7,14,30")
    p.add_argument("--volume-stats", default="mean,median")
    p.add_argument("--policies", default="daily,monthly")
    p.add_argument("--universe-sizes", default="25,50,100,150,200")
    p.add_argument("--notionals", default="100000")
    p.add_argument("--min-lookback-frac", type=float, default=0.80)
    p.add_argument("--min-selected-coverage", type=float, default=0.95)
    return p.parse_args()


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_tokens(raw: str) -> list[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def collect_lazy(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def write_df(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    df.write_csv(path.with_suffix(".csv"))


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
        if not math.isfinite(x):
            return ""
        if abs(x) >= 1_000:
            return f"{x:,.0f}"
        return f"{x:.{digits}f}"
    except Exception:
        return str(value)


def markdown_table(df: pl.DataFrame, cols: Iterable[str], n: int = 30) -> list[str]:
    selected = [c for c in cols if c in df.columns]
    rows = df.select(selected).head(n).to_dicts()
    if not rows:
        return ["_no rows_"]
    lines = ["| " + " | ".join(selected) + " |", "| " + " | ".join("---" for _ in selected) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in selected) + " |")
    return lines




def sort_universe(df: pl.DataFrame, extra: list[str] | None = None) -> pl.DataFrame:
    """Sort top25/top50/... numerically instead of lexicographically."""
    if df.is_empty() or "universe" not in df.columns:
        return df
    order_col = "_universe_order"
    sort_cols = [order_col, *(extra or [])]
    return (
        df.with_columns(
            pl.col("universe")
            .str.extract(r"^top(\d+)$", 1)
            .cast(pl.Int64, strict=False)
            .fill_null(1_000_000)
            .alias(order_col)
        )
        .sort(sort_cols)
        .drop(order_col)
    )

def stable_quote_expr() -> pl.Expr:
    return pl.col("symbol").str.ends_with("USDT") | pl.col("symbol").str.ends_with("USDC")


def safe_cost(side: str, notional: int, ref: str = "mid") -> pl.Expr:
    cost = f"{side}_{notional}_{ref}_cost_bps"
    complete = f"{side}_{notional}_{ref}_complete"
    return pl.when(pl.col(complete) & pl.col(cost).is_finite()).then(pl.col(cost)).otherwise(None)


def row_cost_expr(notional: int) -> pl.Expr:
    return ((safe_cost("buy", notional) + safe_cost("sell", notional)) / 2).alias(f"avg_{notional}_mid_cost_bps")


def load_l2_daily(args: argparse.Namespace, notionals: list[int]) -> pl.DataFrame:
    pattern = (
        args.feature_root
        / f"exchange={args.exchange}"
        / "symbol=*"
        / "date=*"
        / f"grid={args.grid}"
        / "slippage.parquet"
    )
    lf = (
        pl.scan_parquet(str(pattern), hive_partitioning=True)
        .filter(stable_quote_expr())
        .filter((pl.col("date") >= pl.lit(args.start_date).str.strptime(pl.Date)) & (pl.col("date") <= pl.lit(args.end_date).str.strptime(pl.Date)))
        .with_columns([row_cost_expr(n) for n in notionals])
        .group_by(["symbol", "date"])
        .agg(
            [
                pl.len().alias("l2_rows"),
                pl.col("spread_bps").mean().alias("spread_mean_bps"),
                pl.col("spread_bps").quantile(0.90).alias("spread_p90_bps"),
                pl.col("buy_100000_mid_complete").mean().alias("buy_100000_mid_complete_rate"),
                pl.col("sell_100000_mid_complete").mean().alias("sell_100000_mid_complete_rate"),
                *[pl.col(f"avg_{n}_mid_cost_bps").mean().alias(f"avg_{n}_mid_mean_bps") for n in notionals],
                *[pl.col(f"avg_{n}_mid_cost_bps").quantile(0.90).alias(f"avg_{n}_mid_p90_bps") for n in notionals],
            ]
        )
        .sort(["date", "symbol"])
    )
    return collect_lazy(lf)


def load_daily_quote_volume(args: argparse.Namespace, symbols: list[str], warmup_start: date, end: date) -> pl.DataFrame:
    pattern = args.vision_root / "data_type=klines" / "interval=1m" / "symbol=*" / "year=*" / "month=*" / "*.parquet"
    lf = (
        pl.scan_parquet(str(pattern), hive_partitioning=True)
        .filter(pl.col("source_symbol").is_in(symbols))
        .with_columns(pl.col("open_datetime").dt.date().alias("date"))
        .filter((pl.col("date") >= pl.lit(warmup_start).cast(pl.Date)) & (pl.col("date") <= pl.lit(end).cast(pl.Date)))
        .group_by(["source_symbol", "date"])
        .agg(
            [
                pl.col("quote_volume").sum().alias("quote_volume_usd"),
                pl.col("trade_count").sum().alias("trade_count"),
                pl.col("close").last().alias("close_price"),
                pl.len().alias("kline_minutes"),
            ]
        )
        .rename({"source_symbol": "symbol"})
        .sort(["date", "symbol"])
    )
    return collect_lazy(lf)


@dataclass(frozen=True)
class RankSpec:
    lookback_days: int
    volume_stat: str
    policy: str


def rank_for_date(vol: pl.DataFrame, eval_date: date, lookback_days: int, volume_stat: str, min_obs: int) -> pl.DataFrame:
    lo = eval_date - timedelta(days=lookback_days)
    hist = vol.filter((pl.col("date") >= lo) & (pl.col("date") < eval_date))
    if hist.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "rank_value": pl.Float64, "lookback_obs": pl.Int64})
    agg_expr = pl.col("quote_volume_usd").mean() if volume_stat == "mean" else pl.col("quote_volume_usd").median()
    return (
        hist.group_by("symbol")
        .agg(
            [
                agg_expr.alias("rank_value"),
                pl.len().alias("lookback_obs"),
                pl.col("quote_volume_usd").mean().alias("lookback_mean_quote_volume"),
                pl.col("quote_volume_usd").median().alias("lookback_median_quote_volume"),
                pl.col("trade_count").mean().alias("lookback_mean_trade_count"),
            ]
        )
        .filter((pl.col("lookback_obs") >= min_obs) & pl.col("rank_value").is_finite())
        .sort(["rank_value", "symbol"], descending=[True, False])
        .with_row_index("adv_rank", offset=1)
    )


def selection_dates(eval_dates: list[date], policy: str) -> dict[date, date]:
    if policy == "daily":
        return {d: d for d in eval_dates}
    if policy == "monthly":
        first_by_month: dict[tuple[int, int], date] = {}
        for d in eval_dates:
            first_by_month.setdefault((d.year, d.month), d)
        return {d: first_by_month[(d.year, d.month)] for d in eval_dates}
    if policy == "weekly":
        first_by_week: dict[tuple[int, int], date] = {}
        for d in eval_dates:
            iso = d.isocalendar()
            first_by_week.setdefault((iso.year, iso.week), d)
        return {d: first_by_week[(d.isocalendar().year, d.isocalendar().week)] for d in eval_dates}
    raise ValueError(f"unsupported policy: {policy}")


def build_pit_members(
    vol: pl.DataFrame,
    eval_dates: list[date],
    specs: list[RankSpec],
    universe_sizes: list[int],
    min_lookback_frac: float,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for spec in specs:
        min_obs = max(1, math.ceil(spec.lookback_days * min_lookback_frac))
        sel_map = selection_dates(eval_dates, spec.policy)
        rank_cache: dict[date, pl.DataFrame] = {}
        for d in sorted(set(sel_map.values())):
            rank_cache[d] = rank_for_date(vol, d, spec.lookback_days, spec.volume_stat, min_obs)
        for eval_date in eval_dates:
            sel_date = sel_map[eval_date]
            ranked = rank_cache[sel_date]
            if ranked.is_empty():
                continue
            max_n = min(max(universe_sizes), ranked.height)
            base = ranked.head(max_n)
            for n in universe_sizes:
                take_n = min(n, ranked.height)
                if take_n == 0:
                    continue
                uni = base.head(take_n).with_columns(
                    [
                        pl.lit(eval_date).alias("date"),
                        pl.lit(sel_date).alias("selection_date"),
                        pl.lit(spec.policy).alias("policy"),
                        pl.lit(spec.lookback_days).alias("lookback_days"),
                        pl.lit(spec.volume_stat).alias("volume_stat"),
                        pl.lit(f"top{n}").alias("universe"),
                        pl.lit(take_n).alias("selected_symbols"),
                        pl.lit(min_obs).alias("min_lookback_obs"),
                    ]
                )
                frames.append(uni)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").select(
        [
            "date",
            "selection_date",
            "policy",
            "lookback_days",
            "volume_stat",
            "universe",
            "selected_symbols",
            "symbol",
            "adv_rank",
            "rank_value",
            "lookback_obs",
            "min_lookback_obs",
            "lookback_mean_quote_volume",
            "lookback_median_quote_volume",
            "lookback_mean_trade_count",
        ]
    )


def build_daily_costs(members: pl.DataFrame, l2_daily: pl.DataFrame, notionals: list[int]) -> pl.DataFrame:
    joined = members.join(l2_daily, on=["date", "symbol"], how="left")
    group_keys = ["date", "policy", "lookback_days", "volume_stat", "universe", "selected_symbols"]
    aggs: list[pl.Expr] = [
        pl.col("symbol").n_unique().alias("member_rows"),
        pl.col("avg_100000_mid_mean_bps").is_not_null().sum().alias("evaluated_symbols"),
        (pl.col("avg_100000_mid_mean_bps").is_not_null().sum() / pl.col("symbol").n_unique()).alias("evaluation_coverage"),
        pl.col("rank_value").mean().alias("mean_rank_quote_volume"),
        pl.col("rank_value").min().alias("min_rank_quote_volume"),
        pl.col("lookback_obs").min().alias("min_lookback_obs_seen"),
        pl.col("spread_mean_bps").mean().alias("spread_mean_bps"),
        pl.col("spread_p90_bps").mean().alias("spread_p90_mean_bps"),
        pl.col("buy_100000_mid_complete_rate").min().alias("buy_100000_complete_min"),
    ]
    for n in notionals:
        aggs.extend(
            [
                pl.col(f"avg_{n}_mid_mean_bps").mean().alias(f"equal_weight_mean_cost_{n}_bps"),
                pl.col(f"avg_{n}_mid_mean_bps").quantile(0.90).alias(f"symbol_mean_cost_p90_{n}_bps"),
                pl.col(f"avg_{n}_mid_p90_bps").mean().alias(f"mean_of_symbol_p90_cost_{n}_bps"),
                (pl.col(f"avg_{n}_mid_mean_bps").mean() * pl.col("symbol").n_unique() * n / 10_000).alias(f"expected_cost_{n}_usd"),
            ]
        )
    return joined.group_by(group_keys).agg(aggs).sort(group_keys)


def summarize_daily_costs(daily: pl.DataFrame, notionals: list[int], min_selected_coverage: float) -> pl.DataFrame:
    keys = ["policy", "lookback_days", "volume_stat", "universe"]
    aggs: list[pl.Expr] = [
        pl.col("date").n_unique().alias("dates"),
        pl.col("selected_symbols").mean().alias("avg_selected_symbols"),
        pl.col("evaluated_symbols").mean().alias("avg_evaluated_symbols"),
        pl.col("evaluation_coverage").mean().alias("avg_evaluation_coverage"),
        (pl.col("evaluation_coverage") >= min_selected_coverage).mean().alias("date_pass_rate"),
        pl.col("mean_rank_quote_volume").mean().alias("mean_rank_quote_volume"),
        pl.col("min_rank_quote_volume").mean().alias("avg_min_rank_quote_volume"),
        pl.col("spread_mean_bps").mean().alias("spread_mean_bps"),
        pl.col("spread_p90_mean_bps").mean().alias("spread_p90_mean_bps"),
        pl.col("buy_100000_complete_min").min().alias("buy_100000_complete_min"),
    ]
    for n in notionals:
        aggs.extend(
            [
                pl.col(f"equal_weight_mean_cost_{n}_bps").mean().alias(f"mean_cost_{n}_bps"),
                pl.col(f"equal_weight_mean_cost_{n}_bps").quantile(0.50).alias(f"median_daily_cost_{n}_bps"),
                pl.col(f"equal_weight_mean_cost_{n}_bps").quantile(0.90).alias(f"p90_daily_cost_{n}_bps"),
                pl.col(f"symbol_mean_cost_p90_{n}_bps").mean().alias(f"symbol_mean_cost_p90_{n}_bps"),
                pl.col(f"mean_of_symbol_p90_cost_{n}_bps").mean().alias(f"mean_of_symbol_p90_cost_{n}_bps"),
                pl.col(f"expected_cost_{n}_usd").mean().alias(f"expected_cost_{n}_usd"),
            ]
        )
    return sort_universe(daily.group_by(keys).agg(aggs), extra=["policy", "lookback_days", "volume_stat"])


def build_oracle_comparison(args: argparse.Namespace, pit_summary: pl.DataFrame, notionals: list[int]) -> pl.DataFrame:
    path = args.oracle_root / "universe_capacity_curve.csv"
    if not path.exists():
        return pl.DataFrame()
    oracle = pl.read_csv(path).filter(pl.col("notional_per_symbol_usd").is_in(notionals))
    rows = []
    for n in notionals:
        pit_col = f"mean_cost_{n}_bps"
        if pit_col not in pit_summary.columns:
            continue
        pit = pit_summary.filter((pl.col("lookback_days") == 30) & (pl.col("volume_stat") == "mean") & (pl.col("policy") == "daily"))
        oracle_n = oracle.filter(pl.col("notional_per_symbol_usd") == n)
        for uni in pit["universe"].unique().to_list():
            o = oracle_n.filter(pl.col("universe") == uni)
            p = pit.filter(pl.col("universe") == uni)
            if o.is_empty() or p.is_empty():
                continue
            orow = o.row(0, named=True)
            prow = p.row(0, named=True)
            rows.append(
                {
                    "universe": uni,
                    "notional_per_symbol_usd": n,
                    "oracle_l2_cost_bps": orow["equal_weight_mean_cost_bps"],
                    "pit_adv30_mean_cost_bps": prow[pit_col],
                    "pit_minus_oracle_bps": prow[pit_col] - orow["equal_weight_mean_cost_bps"],
                    "pit_over_oracle": prow[pit_col] / orow["equal_weight_mean_cost_bps"] if orow["equal_weight_mean_cost_bps"] else None,
                }
            )
    if not rows:
        return pl.DataFrame()
    return sort_universe(pl.DataFrame(rows))


def write_markdown(
    args: argparse.Namespace,
    l2_daily: pl.DataFrame,
    vol_daily: pl.DataFrame,
    members: pl.DataFrame,
    daily: pl.DataFrame,
    summary: pl.DataFrame,
    comparison: pl.DataFrame,
    outputs: dict[str, Path],
) -> None:
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    main = summary.filter((pl.col("policy") == "daily") & (pl.col("lookback_days") == 30) & (pl.col("volume_stat") == "mean"))
    main = sort_universe(main) if not main.is_empty() else main
    lines = [
        "# Point-in-time ADV universe experiment",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Purpose",
        "",
        "The previous top-N table is an ex-post L2 liquidity frontier: it ranks symbols after observing realized 100k order-book costs.  This experiment builds live-feasible universes from pre-trade information only.  On each evaluation date `t`, symbols are ranked by trailing quote volume observed through `t-1`, and the selected universe is evaluated on date `t` with realized L2 slippage.",
        "",
        "This is also consistent with standard investability practice outside crypto: MSCI-style liquidity screens use prior traded-value ratios/frequency, and S&P index examples use prior average daily trading-volume screens before inclusion rather than future execution costs.",
        "",
        "## Scope",
        "",
        f"- Evaluation dates: `{start}`..`{end}`.",
        f"- L2 daily panel: `{l2_daily.height:,}` symbol-date rows, `{l2_daily['symbol'].n_unique()}` symbols, `{l2_daily['date'].n_unique()}` dates.",
        f"- Binance Vision quote-volume panel: `{vol_daily.height:,}` symbol-date rows, `{vol_daily['symbol'].n_unique()}` symbols.",
        f"- Ranking windows: `{args.lookbacks}` days, using only dates before the evaluation date.",
        f"- Universe policies: `{args.policies}`. Daily recomputes every UTC date; monthly holds the first available universe for each month.",
        "- Primary ranking rule for the paper: trailing 30-day mean quote volume, daily recomputation.",
        "",
        "## Main PIT result: daily trailing 30-day mean quote volume",
        "",
        *markdown_table(
            main,
            [
                "universe",
                "dates",
                "avg_selected_symbols",
                "avg_evaluated_symbols",
                "avg_evaluation_coverage",
                "mean_cost_100000_bps",
                "p90_daily_cost_100000_bps",
                "expected_cost_100000_usd",
                "avg_min_rank_quote_volume",
                "buy_100000_complete_min",
            ],
            n=20,
        ),
        "",
        "## Oracle L2 frontier versus tradable PIT ADV rule",
        "",
        *markdown_table(
            comparison,
            [
                "universe",
                "oracle_l2_cost_bps",
                "pit_adv30_mean_cost_bps",
                "pit_minus_oracle_bps",
                "pit_over_oracle",
            ],
            n=20,
        ),
        "",
        "## Robustness grid",
        "",
        *markdown_table(
            sort_universe(summary.filter(pl.col("universe").is_in(["top25", "top100", "top200"])), extra=["policy", "lookback_days", "volume_stat"]),
            [
                "universe",
                "policy",
                "lookback_days",
                "volume_stat",
                "dates",
                "avg_evaluation_coverage",
                "mean_cost_100000_bps",
                "p90_daily_cost_100000_bps",
                "expected_cost_100000_usd",
            ],
            n=80,
        ),
        "",
        "## Interpretation",
        "",
        "- The ex-post L2 ranking remains useful, but it should be labeled as an oracle frontier rather than a tradable rule.",
        "- The point-in-time ADV rule answers the live question: if we only knew trailing quote volume yesterday, what universe would we trade today and what L2 cost would we realize?",
        "- The gap between PIT ADV and oracle L2 costs is the price of using implementable pre-trade information instead of hindsight L2 costs.",
        "- Median quote-volume ranks are retained as a robustness check because index providers often use median traded value to reduce one-day volume spikes.",
        "- External methodology anchors: MSCI Liquidity Data Methodology (`https://www.msci.com/index/methodology/latest/LiquidityData`) and S&P Dow Jones Indices methodology examples (`https://www.spglobal.com/spdji/en/research-insights/index-literacy/methodology-matters/`).",
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
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    lookbacks = parse_ints(args.lookbacks)
    volume_stats = parse_tokens(args.volume_stats)
    policies = parse_tokens(args.policies)
    universe_sizes = parse_ints(args.universe_sizes)
    notionals = parse_ints(args.notionals)
    if not lookbacks:
        raise SystemExit("no lookbacks")
    if any(s not in {"mean", "median"} for s in volume_stats):
        raise SystemExit("--volume-stats supports only mean,median")
    if any(p not in {"daily", "weekly", "monthly"} for p in policies):
        raise SystemExit("--policies supports daily,weekly,monthly")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    l2_daily = load_l2_daily(args, notionals)
    if l2_daily.is_empty():
        raise SystemExit("empty L2 daily panel")
    eval_dates = [d for d in sorted(l2_daily["date"].unique().to_list()) if start <= d <= end]
    symbols = sorted(l2_daily["symbol"].unique().to_list())
    warmup_start = start - timedelta(days=max(lookbacks))
    vol_daily = load_daily_quote_volume(args, symbols, warmup_start, end)
    if vol_daily.is_empty():
        raise SystemExit("empty Binance Vision daily quote-volume panel")

    specs = [RankSpec(lb, stat, policy) for policy in policies for lb in lookbacks for stat in volume_stats]
    members = build_pit_members(vol_daily, eval_dates, specs, universe_sizes, args.min_lookback_frac)
    if members.is_empty():
        raise SystemExit("empty PIT member table")
    daily = build_daily_costs(members, l2_daily, notionals)
    summary = summarize_daily_costs(daily, notionals, args.min_selected_coverage)
    comparison = build_oracle_comparison(args, summary, notionals)

    outputs = {
        "l2_daily_costs": args.out_dir / "l2_daily_costs.parquet",
        "daily_quote_volume": args.out_dir / "daily_quote_volume.parquet",
        "pit_universe_members": args.out_dir / "pit_universe_members.parquet",
        "pit_daily_costs": args.out_dir / "pit_daily_costs.parquet",
        "pit_summary": args.out_dir / "pit_summary.parquet",
        "oracle_vs_pit_adv30": args.out_dir / "oracle_vs_pit_adv30.parquet",
    }
    write_df(l2_daily, outputs["l2_daily_costs"])
    write_df(vol_daily, outputs["daily_quote_volume"])
    write_df(members, outputs["pit_universe_members"])
    write_df(daily, outputs["pit_daily_costs"])
    write_df(summary, outputs["pit_summary"])
    if not comparison.is_empty():
        write_df(comparison, outputs["oracle_vs_pit_adv30"])
    else:
        comparison.write_parquet(outputs["oracle_vs_pit_adv30"])
        comparison.write_csv(outputs["oracle_vs_pit_adv30"].with_suffix(".csv"))

    manifest = {
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "startDate": args.start_date,
        "endDate": args.end_date,
        "lookbacks": lookbacks,
        "volumeStats": volume_stats,
        "policies": policies,
        "universeSizes": universe_sizes,
        "notionals": notionals,
        "minLookbackFrac": args.min_lookback_frac,
        "l2DailyRows": l2_daily.height,
        "l2Symbols": l2_daily["symbol"].n_unique(),
        "evalDates": len(eval_dates),
        "quoteVolumeRows": vol_daily.height,
        "memberRows": members.height,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "markdown": str(args.markdown_out),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args, l2_daily, vol_daily, members, daily, summary, comparison, outputs)
    print(json.dumps({"status": "ok", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
