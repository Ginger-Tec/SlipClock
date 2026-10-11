#!/usr/bin/env python3
"""Post-hoc fixed-clock backtests on the already frozen PIT order vectors.

This diagnostic deliberately chooses the highest- and lowest-cost UTC minute
with full-sample information.  It is an oracle sensitivity test, not an OOS
strategy result, and writes only under ``results/posthoc_clock_oracle``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from experiment_core import ExperimentConfig, STRATEGIES


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SUITE = (
    REPO
    / "data/tardis/research/released_l2_analysis"
    / "baseline"
)
RAW_CLOCK = SUITE / "analysis_mart/cost_panel_1m_long.parquet"
RAW_DAILY_MINUTE = SUITE / "pit_exact_clock/pit_adv30_exact_daily_minute.parquet"
RAW_KLINE_ROOT = (
    REPO
    / "data/binance_vision/parquet/market=futures_um/data_type=klines/interval=1m"
)
BASELINE_ORDERS = HERE / "results/submitted_orders.parquet"
BASELINE_METRICS = HERE / "results/strategy_metrics.csv"
DATA_DIR = HERE / "data/posthoc_clock_oracle"
OUTPUT_DIR = HERE / "results/posthoc_clock_oracle"
CLOCK_SELECTION_COMPACT = DATA_DIR / "clock_selection.csv"

BASELINE_MINUTE = 12 * 60
CLOCK_ORDER = ("baseline_1200", "oracle_low", "oracle_high")
TREATMENTS = ("no_cost", "slipclock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def minute_label(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def write_csv(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path, float_precision=12)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_posthoc_clocks(
    surface: pl.DataFrame, cfg: ExperimentConfig
) -> pl.DataFrame:
    """Choose exact high/low minutes from the full evaluation sample."""

    scoped = surface.filter(
        (pl.col("universe") == "top100")
        & pl.col("date").is_between(cfg.eval_start, cfg.eval_end, closed="both")
    )
    expected_dates = scoped.get_column("date").n_unique()
    if expected_dates == 0:
        raise RuntimeError("the evaluation window has no observed L2 dates")
    minute_stats = (
        scoped.group_by("minute_of_day")
        .agg(
            pl.col("cost_100000_bps").mean().alias("mean_probe_cost_bps"),
            pl.col("cost_100000_bps").median().alias("median_probe_cost_bps"),
            pl.col("completion_rate").mean().alias("mean_completion_rate"),
            pl.col("date").n_unique().alias("observed_dates"),
        )
        .filter(pl.col("observed_dates") == expected_dates)
        .sort("minute_of_day")
    )
    if minute_stats.is_empty():
        raise RuntimeError("no complete minute profile is available")

    low = minute_stats.sort(["mean_probe_cost_bps", "minute_of_day"]).row(
        0, named=True
    )
    high = minute_stats.sort(
        ["mean_probe_cost_bps", "minute_of_day"], descending=[True, False]
    ).row(0, named=True)
    baseline = minute_stats.filter(pl.col("minute_of_day") == BASELINE_MINUTE)
    if baseline.height != 1:
        raise RuntimeError("12:00 UTC is absent from the complete minute profile")

    rows: list[dict[str, object]] = []
    for clock, row in (
        ("baseline_1200", baseline.row(0, named=True)),
        ("oracle_low", low),
        ("oracle_high", high),
    ):
        minute = int(row["minute_of_day"])
        rows.append(
            {
                "clock": clock,
                "minute_of_day": minute,
                "time_utc": minute_label(minute),
                "mean_probe_cost_bps": float(row["mean_probe_cost_bps"]),
                "median_probe_cost_bps": float(row["median_probe_cost_bps"]),
                "mean_completion_rate": float(row["mean_completion_rate"]),
                "observed_dates": int(row["observed_dates"]),
                "selection_universe": "PIT ADV30-mean top100",
                "selection_notional_usd": 100_000,
                "selection_is_posthoc": clock != "baseline_1200",
            }
        )
    clocks = pl.DataFrame(rows)
    if clocks.get_column("minute_of_day").n_unique() != len(CLOCK_ORDER):
        raise RuntimeError("oracle and baseline clocks must be distinct")
    return clocks


def extract_clock_states(
    orders_path: Path,
    clocks: pl.DataFrame,
    cfg: ExperimentConfig,
    output_path: Path,
) -> None:
    minutes = ",".join(str(value) for value in clocks["minute_of_day"].to_list())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute(
        f"""
        COPY (
            SELECT c.timestamp_utc, c.date, c.minute_of_day, c.symbol,
                   c.mid_price, c.spread_bps,
                   c.buy_100000_mid_cost_bps, c.sell_100000_mid_cost_bps,
                   c.buy_100000_mid_complete, c.sell_100000_mid_complete,
                   c.valid_bbo
            FROM read_parquet('{RAW_CLOCK.as_posix()}') c
            INNER JOIN (
                SELECT DISTINCT symbol
                FROM read_parquet('{orders_path.as_posix()}')
            ) o USING (symbol)
            WHERE c.date BETWEEN DATE '{cfg.eval_start.isoformat()}'
                             AND DATE '{(cfg.eval_end + timedelta(days=1)).isoformat()}'
              AND c.minute_of_day IN ({minutes})
            ORDER BY c.date, c.minute_of_day, c.symbol
        ) TO '{output_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    con.close()


def extract_clock_prices(
    orders: pl.DataFrame,
    clocks: pl.DataFrame,
    cfg: ExperimentConfig,
    output_path: Path,
) -> None:
    """Extract the close immediately preceding each L2 clock boundary."""

    symbols = sorted(orders.get_column("symbol").unique().to_list())
    clock_to_source = {
        int(row["minute_of_day"]): (int(row["minute_of_day"]) - 1) % 1440
        for row in clocks.iter_rows(named=True)
    }
    source_to_clock = {source: clock for clock, source in clock_to_source.items()}
    source_minutes = np.asarray(sorted(source_to_clock), dtype=np.int64)
    min_day = np.datetime64(cfg.eval_start.isoformat())
    max_day = np.datetime64((cfg.eval_end + timedelta(days=1)).isoformat())
    day_ms = 86_400_000
    minute_ms = 60_000
    records: list[pl.DataFrame] = []

    for symbol in symbols:
        symbol_root = RAW_KLINE_ROOT / f"symbol={symbol}"
        for path in sorted(symbol_root.glob("year=*/month=*/part-*.parquet")):
            table = pq.ParquetFile(path).read(columns=["open_time", "close"])
            open_time = table.column("open_time").combine_chunks().to_numpy(
                zero_copy_only=False
            )
            close = (
                table.column("close")
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
                .astype(np.float64)
            )
            source_minute = (open_time % day_ms) // minute_ms
            mask = np.isin(source_minute, source_minutes) & np.isfinite(close) & (close > 0)
            if not mask.any():
                continue
            days = (open_time[mask] // day_ms).astype(np.int32)
            day_values = days.astype("datetime64[D]")
            range_mask = (day_values >= min_day) & (day_values <= max_day)
            if not range_mask.any():
                continue
            selected_source = source_minute[mask][range_mask]
            records.append(
                pl.DataFrame(
                    {
                        "date": day_values[range_mask],
                        "symbol": [symbol] * int(range_mask.sum()),
                        "minute_of_day": [
                            source_to_clock[int(value)] for value in selected_source
                        ],
                        "source_minute_of_day": selected_source.astype(np.int16),
                        "kline_mark": close[mask][range_mask],
                    }
                )
            )

    if not records:
        raise RuntimeError("no clock-aligned Binance kline prices were extracted")
    prices = (
        pl.concat(records, how="vertical_relaxed")
        .unique(["date", "symbol", "minute_of_day"], keep="last")
        .sort(["date", "minute_of_day", "symbol"])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.write_parquet(output_path, compression="zstd")


def price_frozen_orders(
    orders: pl.DataFrame,
    states: pl.DataFrame,
    prices: pl.DataFrame,
    clocks: pl.DataFrame,
    cfg: ExperimentConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    clock_map = clocks.select("clock", "minute_of_day")
    entries = orders.drop("timestamp_utc").join(clock_map, how="cross")
    entry_state = states.join(clock_map, on="minute_of_day", how="inner")
    next_state = entry_state.select(
        pl.col("date").dt.offset_by("-1d").alias("date"),
        "symbol",
        "clock",
        pl.col("mid_price").alias("next_l2_mid_price"),
    )
    next_fallback = prices.select(
        pl.col("date").dt.offset_by("-1d").alias("date"),
        "symbol",
        "minute_of_day",
        pl.col("kline_mark").alias("next_kline_mark"),
    )

    priced = (
        entries.join(
            entry_state,
            on=["date", "symbol", "clock", "minute_of_day"],
            how="left",
            validate="m:1",
        )
        .join(
            next_state,
            on=["date", "symbol", "clock"],
            how="left",
            validate="m:1",
        )
        .join(
            next_fallback,
            on=["date", "symbol", "minute_of_day"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.when(pl.col("side") == 1)
            .then(pl.col("buy_100000_mid_cost_bps"))
            .otherwise(pl.col("sell_100000_mid_cost_bps"))
            .alias("slipclock_cost_bps"),
            pl.when(pl.col("side") == 1)
            .then(pl.col("buy_100000_mid_complete").fill_null(False))
            .otherwise(pl.col("sell_100000_mid_complete").fill_null(False))
            .alias("relevant_side_complete"),
            pl.coalesce("next_l2_mid_price", "next_kline_mark").alias("end_mark"),
            pl.when(pl.col("next_l2_mid_price").is_finite().fill_null(False))
            .then(pl.lit("l2_mid"))
            .when(pl.col("next_kline_mark").is_finite().fill_null(False))
            .then(pl.lit("kline_previous_minute_close"))
            .otherwise(pl.lit("missing"))
            .alias("markout_source"),
        )
        .with_columns(
            (
                pl.col("valid_bbo").fill_null(False)
                & pl.col("mid_price").is_finite().fill_null(False)
                & pl.col("spread_bps").is_finite().fill_null(False)
                & pl.col("slipclock_cost_bps").is_finite().fill_null(False)
                & pl.col("relevant_side_complete")
            ).alias("entry_valid"),
            (pl.col("end_mark") / pl.col("mid_price") - 1.0).alias(
                "raw_evaluation_return"
            ),
        )
        .with_columns(
            (
                pl.col("raw_evaluation_return").is_finite().fill_null(False)
                & (
                    pl.col("raw_evaluation_return").abs()
                    > cfg.max_abs_daily_contract_return
                )
            ).alias("return_outlier")
        )
        .with_columns(
            pl.when(pl.col("return_outlier"))
            .then(pl.lit(0.0))
            .otherwise(pl.col("raw_evaluation_return"))
            .alias("evaluation_return")
        )
        .with_columns(
            (
                pl.col("entry_valid")
                & pl.col("evaluation_return").is_finite().fill_null(False)
            ).alias("order_valid")
        )
    )

    entry_audit = priced.group_by("clock", "date", "strategy").agg(
        pl.len().alias("orders"),
        pl.col("entry_valid").sum().alias("entry_valid_orders"),
        pl.col("order_valid").sum().alias("markout_valid_orders"),
        (pl.col("markout_source") == "kline_previous_minute_close")
        .sum()
        .alias("kline_fallback_orders"),
        pl.col("return_outlier").sum().alias("return_outlier_orders"),
    )
    if entry_audit.filter(
        pl.col("entry_valid_orders") != cfg.orders_per_strategy_day
    ).height:
        raise RuntimeError("a frozen order is not executable at every compared clock")

    common = priced.group_by("order_id").agg(
        pl.col("order_valid").all().alias("common_order_valid")
    )
    retained = (
        priced.join(common, on="order_id", how="left")
        .filter(pl.col("common_order_valid"))
        .with_columns(
            pl.len().over("date", "strategy", "clock").alias("evaluated_order_count")
        )
        .with_columns(
            (
                cfg.submitted_gross_usd / pl.col("evaluated_order_count")
            ).alias("evaluation_notional_usd")
        )
        .sort(["clock", "date", "strategy", "symbol"])
    )
    support = retained.group_by("clock", "date", "strategy").agg(
        pl.len().alias("evaluated_orders")
    )
    if support.select(pl.col("date").n_unique()).item() != orders.select(
        pl.col("date").n_unique()
    ).item():
        raise RuntimeError("ordinary rebalance dates are not common across clocks")
    return retained, entry_audit.sort(["clock", "date", "strategy"])


def ordinary_daily(
    retained: pl.DataFrame, cfg: ExperimentConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    wide = retained.with_columns(
        (
            pl.col("side").cast(pl.Float64)
            * pl.col("evaluation_notional_usd")
            * pl.col("evaluation_return")
        ).alias("gross_pnl_usd"),
        (
            pl.col("evaluation_notional_usd")
            * pl.col("slipclock_cost_bps")
            / 10_000.0
        ).alias("slipclock_cost_usd"),
    )
    frames = []
    for treatment in TREATMENTS:
        cost = (
            pl.lit(0.0)
            if treatment == "no_cost"
            else pl.col("slipclock_cost_usd")
        )
        frames.append(
            wide.select(
                "order_id",
                "date",
                "clock",
                "minute_of_day",
                "strategy",
                "symbol",
                "side",
                "evaluation_notional_usd",
                "gross_pnl_usd",
                pl.lit(treatment).alias("treatment"),
                cost.alias("cost_usd"),
            ).with_columns(
                (pl.col("gross_pnl_usd") - pl.col("cost_usd")).alias(
                    "net_pnl_usd"
                )
            )
        )
    order_long = pl.concat(frames, how="vertical")
    daily = (
        order_long.group_by("clock", "minute_of_day", "date", "strategy", "treatment")
        .agg(
            pl.len().alias("order_count"),
            pl.col("evaluation_notional_usd").sum().alias("evaluation_gross_usd"),
            pl.col("gross_pnl_usd").sum().alias("gross_pnl_usd"),
            pl.col("cost_usd").sum().alias("cost_usd"),
            pl.col("net_pnl_usd").sum().alias("net_pnl_usd"),
        )
        .with_columns(
            pl.lit(cfg.submitted_gross_usd).alias("submitted_gross_usd"),
            pl.lit(False).alias("is_carry_day"),
        )
        .with_columns(
            (pl.col("net_pnl_usd") / pl.col("evaluation_gross_usd")).alias(
                "net_return"
            )
        )
    )
    return order_long.sort(["clock", "date", "strategy", "treatment", "symbol"]), daily


def missing_l2_dates(
    states: pl.DataFrame, cfg: ExperimentConfig
) -> list[date]:
    observed = set(states.get_column("date").unique().to_list())
    return [
        cfg.eval_start + timedelta(days=offset)
        for offset in range((cfg.eval_end - cfg.eval_start).days + 1)
        if cfg.eval_start + timedelta(days=offset) not in observed
    ]


def carry_rows(
    retained: pl.DataFrame,
    prices: pl.DataFrame,
    carry_dates: Iterable[date],
    cfg: ExperimentConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    carry_dates = sorted(set(carry_dates))
    if not carry_dates:
        return pl.DataFrame(), pl.DataFrame()
    price_lookup = {
        (row["date"], row["symbol"], int(row["minute_of_day"])): float(
            row["kline_mark"]
        )
        for row in prices.iter_rows(named=True)
        if row["kline_mark"] is not None
        and np.isfinite(row["kline_mark"])
        and row["kline_mark"] > 0
    }
    blocks = {
        (block["clock"][0], block["date"][0], block["strategy"][0]): block
        for block in retained.partition_by(
            ["clock", "date", "strategy"], maintain_order=True
        )
    }
    position_dates = {
        (clock, strategy): sorted(
            {
                current_date
                for block_clock, current_date, block_strategy in blocks
                if block_clock == clock and block_strategy == strategy
            }
        )
        for clock in CLOCK_ORDER
        for strategy in STRATEGIES
    }
    rows: list[dict[str, object]] = []
    for current_date in carry_dates:
        for clock in CLOCK_ORDER:
            minute = int(
                retained.filter(pl.col("clock") == clock)["minute_of_day"][0]
            )
            for strategy in STRATEGIES:
                prior = [
                    value
                    for value in position_dates[clock, strategy]
                    if value < current_date
                ]
                if not prior:
                    continue
                position_date = prior[-1]
                positions = blocks[clock, position_date, strategy]
                for row in positions.iter_rows(named=True):
                    start = price_lookup.get((current_date, row["symbol"], minute))
                    end = price_lookup.get(
                        (current_date + timedelta(days=1), row["symbol"], minute)
                    )
                    available = bool(
                        start is not None
                        and end is not None
                        and np.isfinite(start)
                        and np.isfinite(end)
                        and start > 0
                        and end > 0
                    )
                    raw_return = float(end / start - 1.0) if available else np.nan
                    return_outlier = bool(
                        available
                        and abs(raw_return) > cfg.max_abs_daily_contract_return
                    )
                    rows.append(
                        {
                            "date": current_date,
                            "position_entry_date": position_date,
                            "clock": clock,
                            "minute_of_day": minute,
                            "strategy": strategy,
                            "symbol": row["symbol"],
                            "side": int(row["side"]),
                            "base_notional_usd": float(row["evaluation_notional_usd"]),
                            "entry_mid_price": float(row["mid_price"]),
                            "entry_quantity": float(row["evaluation_notional_usd"])
                            / float(row["mid_price"]),
                            "price_start": start,
                            "price_end": end,
                            "price_available": available,
                            "raw_evaluation_return": raw_return,
                            "return_outlier": return_outlier,
                        }
                    )
    audit = pl.DataFrame(rows).sort(["date", "strategy", "symbol", "clock"])
    common = audit.group_by("date", "strategy", "symbol").agg(
        pl.col("price_available").all().alias("common_price_available")
    )
    audit = audit.join(common, on=["date", "strategy", "symbol"], how="left")
    valid = (
        audit.filter(pl.col("common_price_available"))
        .with_columns(
            pl.col("base_notional_usd")
            .sum()
            .over("date", "strategy", "clock")
            .alias("valid_base_notional_usd")
        )
        .with_columns(
            (
                cfg.submitted_gross_usd / pl.col("valid_base_notional_usd")
            ).alias("notional_scale")
        )
        .with_columns(
            (
                pl.col("base_notional_usd") * pl.col("notional_scale")
            ).alias("evaluation_notional_usd"),
            pl.when(pl.col("return_outlier"))
            .then(pl.lit(0.0))
            .otherwise(pl.col("raw_evaluation_return"))
            .alias("evaluation_return"),
        )
        .with_columns(
            pl.when(pl.col("return_outlier"))
            .then(pl.lit(0.0))
            .otherwise(
                pl.col("side").cast(pl.Float64)
                * pl.col("entry_quantity")
                * pl.col("notional_scale")
                * (pl.col("price_end") - pl.col("price_start"))
            )
            .alias("gross_pnl_usd")
        )
    )
    daily_base = valid.group_by(
        "clock", "minute_of_day", "date", "strategy"
    ).agg(
        pl.len().alias("order_count"),
        pl.col("evaluation_notional_usd").sum().alias("evaluation_gross_usd"),
        pl.col("gross_pnl_usd").sum().alias("gross_pnl_usd"),
    )
    daily = pl.concat(
        [
            daily_base.with_columns(
                pl.lit(treatment).alias("treatment"),
                pl.lit(0.0).alias("cost_usd"),
                pl.col("gross_pnl_usd").alias("net_pnl_usd"),
                pl.lit(0.0).alias("submitted_gross_usd"),
                pl.lit(True).alias("is_carry_day"),
            ).with_columns(
                (pl.col("net_pnl_usd") / pl.col("evaluation_gross_usd")).alias(
                    "net_return"
                )
            )
            for treatment in TREATMENTS
        ],
        how="vertical",
    )
    return daily, audit


def finalize_daily(ordinary: pl.DataFrame, carried: pl.DataFrame) -> pl.DataFrame:
    frames = [ordinary] if carried.is_empty() else [ordinary, carried]
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .sort(["clock", "strategy", "treatment", "date"])
        .with_columns(
            pl.col("net_pnl_usd")
            .cum_sum()
            .over("clock", "strategy", "treatment")
            .alias("cumulative_net_pnl_usd")
        )
    )


def summarize(daily: pl.DataFrame, cfg: ExperimentConfig) -> pl.DataFrame:
    raw = daily.group_by("clock", "minute_of_day", "strategy", "treatment").agg(
        pl.len().alias("n_days"),
        pl.col("is_carry_day").sum().alias("n_carry_days"),
        pl.col("submitted_gross_usd").sum().alias("cumulative_submitted_gross_usd"),
        pl.col("evaluation_gross_usd").sum().alias("cumulative_evaluation_gross_usd"),
        pl.col("gross_pnl_usd").sum().alias("cumulative_gross_pnl_usd"),
        pl.col("cost_usd").sum().alias("cumulative_cost_usd"),
        pl.col("net_pnl_usd").sum().alias("cumulative_net_pnl_usd"),
        pl.col("net_return").mean().alias("mean_daily_net_return"),
        pl.col("net_return").std(ddof=1).alias("sd_daily_net_return"),
    ).with_columns(
        (
            pl.col("mean_daily_net_return")
            / pl.col("sd_daily_net_return")
            * np.sqrt(cfg.annualization_days)
        ).alias("net_sharpe"),
        (
            pl.when(pl.col("cumulative_submitted_gross_usd") > 0)
            .then(
                10_000.0
                * pl.col("cumulative_cost_usd")
                / pl.col("cumulative_submitted_gross_usd")
            )
            .otherwise(0.0)
        ).alias("mean_entry_cost_bps"),
    )
    rows: list[dict[str, object]] = []
    for clock in CLOCK_ORDER:
        for treatment in TREATMENTS:
            block = raw.filter(
                (pl.col("clock") == clock) & (pl.col("treatment") == treatment)
            )
            sharpe_order = block.sort(
                ["net_sharpe", "strategy"], descending=[True, False]
            )["strategy"].to_list()
            pnl_order = block.sort(
                ["cumulative_net_pnl_usd", "strategy"], descending=[True, False]
            )["strategy"].to_list()
            for row in block.iter_rows(named=True):
                row["sharpe_rank"] = sharpe_order.index(row["strategy"]) + 1
                row["pnl_rank"] = pnl_order.index(row["strategy"]) + 1
                rows.append(row)
    return pl.DataFrame(rows).sort(["clock", "treatment", "sharpe_rank"])


def paired_clock_effects(
    daily: pl.DataFrame,
    metrics: pl.DataFrame,
    *,
    bootstrap_reps: int = 10_000,
    seed: int = 20_260_803,
) -> pl.DataFrame:
    """Summarize paired high-versus-low clock effects by strategy.

    Cost uncertainty uses ordinary rebalance dates because carry dates have no
    new submitted order.  P&L decompositions use the full evaluation calendar.
    Calendar months are resampled as blocks to preserve within-month dependence.
    """

    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive")
    slip = daily.filter(pl.col("treatment") == "slipclock")
    low = slip.filter(pl.col("clock") == "oracle_low").select(
        "date",
        "strategy",
        "is_carry_day",
        "submitted_gross_usd",
        pl.col("gross_pnl_usd").alias("gross_low_usd"),
        pl.col("cost_usd").alias("cost_low_usd"),
        pl.col("net_pnl_usd").alias("net_low_usd"),
    )
    high = slip.filter(pl.col("clock") == "oracle_high").select(
        "date",
        "strategy",
        pl.col("is_carry_day").alias("high_is_carry_day"),
        pl.col("submitted_gross_usd").alias("high_submitted_gross_usd"),
        pl.col("gross_pnl_usd").alias("gross_high_usd"),
        pl.col("cost_usd").alias("cost_high_usd"),
        pl.col("net_pnl_usd").alias("net_high_usd"),
    )
    paired = (
        low.join(high, on=["date", "strategy"], how="inner", validate="1:1")
        .with_columns(
            (pl.col("gross_low_usd") - pl.col("gross_high_usd")).alias(
                "gross_low_minus_high_usd"
            ),
            (pl.col("cost_high_usd") - pl.col("cost_low_usd")).alias(
                "cost_saving_low_usd"
            ),
            (pl.col("net_low_usd") - pl.col("net_high_usd")).alias(
                "net_low_minus_high_usd"
            ),
            pl.col("date").dt.strftime("%Y-%m").alias("month"),
        )
        .with_columns(
            pl.when(~pl.col("is_carry_day"))
            .then(
                10_000.0
                * (pl.col("cost_high_usd") - pl.col("cost_low_usd"))
                / pl.col("submitted_gross_usd")
            )
            .otherwise(None)
            .alias("high_minus_low_cost_bps")
        )
    )
    if paired.height != low.height or paired.select(
        (pl.col("is_carry_day") == pl.col("high_is_carry_day")).all()
        & (
            (pl.col("submitted_gross_usd") - pl.col("high_submitted_gross_usd"))
            .abs()
            .max()
            <= 1e-9
        )
    ).item() is not True:
        raise RuntimeError("high and low clocks do not share an identical calendar")

    slip_metrics = metrics.filter(pl.col("treatment") == "slipclock")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for strategy in STRATEGIES:
        strategy_daily = paired.filter(pl.col("strategy") == strategy)
        monthly = (
            strategy_daily.group_by("month")
            .agg(
                pl.col("high_minus_low_cost_bps")
                .drop_nulls()
                .sum()
                .alias("cost_delta_sum_bps"),
                pl.col("high_minus_low_cost_bps")
                .drop_nulls()
                .len()
                .alias("cost_n_days"),
                pl.col("gross_low_minus_high_usd")
                .sum()
                .alias("gross_delta_usd"),
                pl.col("cost_saving_low_usd").sum().alias("cost_saving_usd"),
                pl.col("net_low_minus_high_usd").sum().alias("net_delta_usd"),
            )
            .sort("month")
        )
        if monthly.is_empty():
            raise RuntimeError(f"no paired calendar for {strategy}")
        cost_sums = monthly["cost_delta_sum_bps"].to_numpy()
        cost_counts = monthly["cost_n_days"].to_numpy()
        net_sums = monthly["net_delta_usd"].to_numpy()
        draws = rng.integers(
            0, monthly.height, size=(bootstrap_reps, monthly.height)
        )
        cost_draws = cost_sums[draws].sum(axis=1) / cost_counts[draws].sum(axis=1)
        net_draws_m = net_sums[draws].sum(axis=1) / 1_000_000.0

        def metric(clock: str, column: str) -> float:
            block = slip_metrics.filter(
                (pl.col("strategy") == strategy) & (pl.col("clock") == clock)
            )
            if block.height != 1:
                raise RuntimeError(f"missing {clock} metric for {strategy}")
            return float(block[column][0])

        gross_delta_m = float(strategy_daily["gross_low_minus_high_usd"].sum()) / 1_000_000.0
        cost_saving_m = float(strategy_daily["cost_saving_low_usd"].sum()) / 1_000_000.0
        net_delta_m = float(strategy_daily["net_low_minus_high_usd"].sum()) / 1_000_000.0
        rows.append(
            {
                "strategy": strategy,
                "low_cost_bps": metric("oracle_low", "mean_entry_cost_bps"),
                "baseline_cost_bps": metric(
                    "baseline_1200", "mean_entry_cost_bps"
                ),
                "high_cost_bps": metric("oracle_high", "mean_entry_cost_bps"),
                "high_minus_low_cost_bps": float(
                    strategy_daily["high_minus_low_cost_bps"].drop_nulls().mean()
                ),
                "cost_ci_low_bps": float(np.quantile(cost_draws, 0.025)),
                "cost_ci_high_bps": float(np.quantile(cost_draws, 0.975)),
                "gross_low_minus_high_m": gross_delta_m,
                "cost_saving_low_m": cost_saving_m,
                "net_low_minus_high_m": net_delta_m,
                "net_ci_low_m": float(np.quantile(net_draws_m, 0.025)),
                "net_ci_high_m": float(np.quantile(net_draws_m, 0.975)),
                "positive_net_probability": float(np.mean(net_draws_m > 0.0)),
                "ordinary_rebalance_dates": int(
                    strategy_daily.filter(~pl.col("is_carry_day")).height
                ),
                "evaluated_calendar_dates": int(strategy_daily.height),
                "calendar_month_blocks": int(monthly.height),
            }
        )
    return pl.DataFrame(rows)


def make_figure(daily: pl.DataFrame, output: Path) -> None:
    slipclock = daily.filter(pl.col("treatment") == "slipclock")
    colors = {
        "baseline_1200": "#4c78a8",
        "oracle_low": "#2ca02c",
        "oracle_high": "#d62728",
    }
    labels = {
        "baseline_1200": "12:00 baseline",
        "oracle_low": "post-hoc low",
        "oracle_high": "post-hoc high",
    }
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for axis, strategy in zip(axes.flat, STRATEGIES):
        for clock in CLOCK_ORDER:
            block = slipclock.filter(
                (pl.col("strategy") == strategy) & (pl.col("clock") == clock)
            ).sort("date")
            axis.plot(
                block["date"].to_list(),
                block["cumulative_net_pnl_usd"].to_numpy() / 1_000_000.0,
                label=labels[clock],
                color=colors[clock],
                linewidth=1.2,
            )
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_title(strategy)
        axis.set_ylabel("Cumulative net P&L ($m)")
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower right", bbox_to_anchor=(0.95, 0.08))
    fig.suptitle("Post-hoc fixed-clock diagnostic (not OOS)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_cost_exposure_figure(effects: pl.DataFrame, output_stem: Path) -> None:
    """Render a publication-size paired clock-cost comparison."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    labels = {
        "ts_momentum": "TS momentum",
        "cs_momentum": "CS momentum",
        "mean_reversion": "Mean reversion",
        "vol_scaled": "Vol.-scaled momentum",
        "mlp": "MLP",
    }
    ordered = effects.with_columns(
        pl.col("strategy")
        .replace_strict({strategy: rank for rank, strategy in enumerate(STRATEGIES)})
        .alias("strategy_order")
    ).sort("strategy_order", descending=True)
    y = np.arange(ordered.height)
    low = ordered["low_cost_bps"].to_numpy()
    baseline = ordered["baseline_cost_bps"].to_numpy()
    high = ordered["high_cost_bps"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.9, 3.8))
    for index in range(ordered.height):
        ax.plot(
            [low[index], high[index]],
            [y[index], y[index]],
            color="#BBBBBB",
            linewidth=1.4,
            zorder=1,
        )
    ax.scatter(low, y, color="#009988", marker="o", s=42, label="Low 11:45", zorder=3)
    ax.scatter(
        baseline,
        y,
        color="#0077BB",
        marker="D",
        s=38,
        label="Baseline 12:00",
        zorder=3,
    )
    ax.scatter(high, y, color="#EE7733", marker="s", s=42, label="High 08:00", zorder=3)
    for index, row in enumerate(ordered.iter_rows(named=True)):
        ax.text(
            float(row["high_cost_bps"]) + 1.1,
            y[index],
            f"+{row['high_minus_low_cost_bps']:.2f} bp",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([labels[value] for value in ordered["strategy"].to_list()])
    ax.set_xlabel("Mean displayed-L2 entry cost (bp)")
    ax.set_ylabel("Strategy")
    ax.set_xlim(0.0, max(70.0, float(high.max()) + 12.0))
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        metadata={"Creator": "SlipClock", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def build_qa(
    orders: pl.DataFrame,
    retained: pl.DataFrame,
    daily: pl.DataFrame,
    metrics: pl.DataFrame,
    effects: pl.DataFrame,
    clocks: pl.DataFrame,
    carry_dates: list[date],
    cfg: ExperimentConfig,
) -> dict[str, object]:
    expected_repriced = orders.height * len(CLOCK_ORDER)
    days_by_clock = daily.group_by("clock").agg(
        pl.col("date").n_unique().alias("n_days")
    )
    all_days = days_by_clock.select(
        (pl.col("n_days") == days_by_clock["n_days"][0]).all()
    ).item()
    low_cost = float(
        clocks.filter(pl.col("clock") == "oracle_low")["mean_probe_cost_bps"][0]
    )
    high_cost = float(
        clocks.filter(pl.col("clock") == "oracle_high")["mean_probe_cost_bps"][0]
    )
    slip = metrics.filter(pl.col("treatment") == "slipclock")
    baseline_reference = pl.read_csv(BASELINE_METRICS).filter(
        pl.col("treatment").is_in(TREATMENTS)
    )
    baseline_repriced = metrics.filter(pl.col("clock") == "baseline_1200")
    baseline_comparison = baseline_repriced.join(
        baseline_reference.select(
            "strategy",
            "treatment",
            pl.col("net_sharpe").alias("reference_net_sharpe"),
            pl.col("cumulative_net_pnl_usd").alias(
                "reference_cumulative_net_pnl_usd"
            ),
        ),
        on=["strategy", "treatment"],
        how="inner",
        validate="1:1",
    )
    baseline_matches = bool(
        baseline_comparison.height == len(STRATEGIES) * len(TREATMENTS)
        and baseline_comparison.select(
            (
                (pl.col("net_sharpe") - pl.col("reference_net_sharpe")).abs()
                <= 1e-10
            ).all()
            & (
                (
                    pl.col("cumulative_net_pnl_usd")
                    - pl.col("reference_cumulative_net_pnl_usd")
                ).abs()
                <= 1e-6
            ).all()
        ).item()
    )
    cost_direction = []
    for strategy in STRATEGIES:
        low = float(
            slip.filter(
                (pl.col("clock") == "oracle_low")
                & (pl.col("strategy") == strategy)
            )["mean_entry_cost_bps"][0]
        )
        high = float(
            slip.filter(
                (pl.col("clock") == "oracle_high")
                & (pl.col("strategy") == strategy)
            )["mean_entry_cost_bps"][0]
        )
        cost_direction.append(high > low)
    checks = {
        "oracle_selection_explicitly_posthoc": bool(
            clocks.filter(pl.col("clock") != "baseline_1200")[
                "selection_is_posthoc"
            ].all()
        ),
        "oracle_high_probe_cost_exceeds_low": high_cost > low_cost,
        "same_frozen_order_ids_at_all_clocks": retained.group_by("order_id").agg(
            pl.col("clock").n_unique().alias("n")
        ).select((pl.col("n") == len(CLOCK_ORDER)).all()).item(),
        "all_rebalance_orders_repriced": retained.height == expected_repriced,
        "baseline_1200_reproduces_canonical_metrics": baseline_matches,
        "same_calendar_at_all_clocks": bool(all_days),
        "calendar_includes_only_declared_carry_days": int(
            daily.filter(pl.col("is_carry_day"))["date"].n_unique()
        )
        == len(carry_dates),
        "each_clock_has_expected_calendar_days": days_by_clock.select(
            (
                pl.col("n_days")
                == orders["date"].n_unique() + len(carry_dates)
            ).all()
        ).item(),
        "high_clock_strategy_cost_exceeds_low_for_all_strategies": all(
            cost_direction
        ),
        "low_baseline_high_cost_order_holds_for_all_strategies": effects.select(
            (
                (pl.col("low_cost_bps") < pl.col("baseline_cost_bps"))
                & (pl.col("baseline_cost_bps") < pl.col("high_cost_bps"))
            ).all()
        ).item(),
        "month_block_cost_intervals_are_positive_for_all_strategies": effects.select(
            (pl.col("cost_ci_low_bps") > 0).all()
        ).item(),
        "finite_daily_returns": daily.select(pl.col("net_return").is_finite().all()).item(),
        "fixed_evaluation_gross": daily.select(
            (
                (pl.col("evaluation_gross_usd") - cfg.submitted_gross_usd).abs()
                <= 1e-6
            ).all()
        ).item(),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": {
            "evaluation_start": cfg.eval_start.isoformat(),
            "evaluation_end": cfg.eval_end.isoformat(),
            "submitted_orders": orders.height,
            "ordinary_rebalance_dates": orders["date"].n_unique(),
            "carry_dates": len(carry_dates),
            "evaluated_calendar_dates": daily["date"].n_unique(),
            "clocks": len(CLOCK_ORDER),
        },
        "checks": checks,
        "interpretation_guard": (
            "The high/low minutes use the same evaluation sample being scored. "
            "They are oracle sensitivity diagnostics and are not OOS evidence."
        ),
    }


def write_result_report(
    clocks: pl.DataFrame,
    metrics: pl.DataFrame,
    effects: pl.DataFrame,
    qa: dict[str, object],
    output_files: list[Path],
) -> None:
    slip = metrics.filter(pl.col("treatment") == "slipclock")
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-08-03",
        "- Verification Status: ANALYZED",
        "- Version Label: posthoc_clock_oracle_v1",
        "",
        "## Experiment Result",
        "",
        "- **ID**: posthoc_clock_oracle_v1",
        "- **Type**: analysis",
        f"- **Status**: {'completed' if qa['status'] == 'PASS' else 'failed'}",
        "- **Command**: `uv run python rewrite2/paper/expander/frozen_order_strategy_cost_v1_20260801/posthoc_clock_oracle.py`",
        "- **Working Directory**: `paper/liq_execution`",
        "- **Interpretation**: Same-sample fixed-clock sensitivity used as paired cost-exposure evidence; it is not a deployable timing-rule result.",
        "",
        "### Selected fixed clocks",
        "",
        "| Clock | UTC | Top-100 probe mean cost (bp) | Post-hoc |",
        "|---|---:|---:|---:|",
    ]
    for row in clocks.iter_rows(named=True):
        lines.append(
            f"| {row['clock']} | {row['time_utc']} | "
            f"{row['mean_probe_cost_bps']:.3f} | "
            f"{'yes' if row['selection_is_posthoc'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "### SlipClock-adjusted strategy results",
            "",
            "| Clock | Strategy | Mean entry cost (bp) | No-cost P&L ($m) | Net Sharpe | Net P&L ($m) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for clock in CLOCK_ORDER:
        for row in slip.filter(pl.col("clock") == clock).sort("strategy").iter_rows(
            named=True
        ):
            lines.append(
                f"| {clock} | {row['strategy']} | {row['mean_entry_cost_bps']:.3f} | "
                f"{row['cumulative_gross_pnl_usd'] / 1_000_000.0:.3f} | "
                f"{row['net_sharpe']:.3f} | "
                f"{row['cumulative_net_pnl_usd'] / 1_000_000.0:.3f} |"
            )
    lines.extend(
        [
            "",
            "### Paired strategy clock effects",
            "",
            "| Strategy | Low (bp) | Baseline (bp) | High (bp) | High-low (bp) | Month-block 95% CI | Gross low-high ($m) | Cost saving ($m) | Net low-high ($m) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effects.iter_rows(named=True):
        lines.append(
            f"| {row['strategy']} | {row['low_cost_bps']:.3f} | "
            f"{row['baseline_cost_bps']:.3f} | {row['high_cost_bps']:.3f} | "
            f"{row['high_minus_low_cost_bps']:.3f} | "
            f"[{row['cost_ci_low_bps']:.3f}, {row['cost_ci_high_bps']:.3f}] | "
            f"{row['gross_low_minus_high_m']:.3f} | {row['cost_saving_low_m']:.3f} | "
            f"{row['net_low_minus_high_m']:.3f} |"
        )
    lines.extend(["", "### Output Files", "", "| File | Size |", "|---|---:|"])
    for path in sorted(output_files):
        lines.append(f"| `{path.relative_to(HERE)}` | {path.stat().st_size} |")
    lines.extend(
        [
            "",
            "### Anomalies Detected",
            "",
            "- The clock choice is intentionally contaminated by full-sample future information.",
            "- The oracle selects an exact minute; it is distinct from the manuscript's cross-fitted 60-minute window comparison.",
            "- No order replacement was needed: the same frozen PIT order IDs were executable at all three clocks.",
            "- The manuscript may cite the paired cost-exposure result, but the clock comparison does not establish an implementable timing rule.",
            "",
        ]
    )
    (OUTPUT_DIR / "EXPERIMENT_RESULT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_manifest(output_files: list[Path], input_files: list[Path]) -> None:
    payload = {
        "experiment": "posthoc_clock_oracle_v1",
        "command": (
            "uv run python rewrite2/paper/expander/"
            "frozen_order_strategy_cost_v1_20260801/posthoc_clock_oracle.py"
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            str(path.relative_to(REPO)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_files
        },
        "outputs": {
            str(path.relative_to(OUTPUT_DIR)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output_files)
        },
        "raw_inputs_not_hashed_due_size": [
            str(RAW_CLOCK.relative_to(REPO)),
            str(RAW_KLINE_ROOT.relative_to(REPO)),
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orders = pl.read_parquet(BASELINE_ORDERS)
    state_path = DATA_DIR / "clock_states.parquet"
    price_path = DATA_DIR / "clock_kline_marks.parquet"
    if args.skip_extract:
        clocks = pl.read_csv(CLOCK_SELECTION_COMPACT)
        input_files = [CLOCK_SELECTION_COMPACT, state_path, price_path, BASELINE_ORDERS]
    else:
        surface = pl.read_parquet(RAW_DAILY_MINUTE)
        clocks = select_posthoc_clocks(surface, cfg)
        clocks.write_csv(CLOCK_SELECTION_COMPACT)
        input_files = [RAW_DAILY_MINUTE, BASELINE_ORDERS]
    if not args.skip_extract or not state_path.exists():
        extract_clock_states(BASELINE_ORDERS, clocks, cfg, state_path)
    if not args.skip_extract or not price_path.exists():
        extract_clock_prices(orders, clocks, cfg, price_path)

    states = pl.read_parquet(state_path)
    prices = pl.read_parquet(price_path)
    retained, entry_audit = price_frozen_orders(
        orders, states, prices, clocks, cfg
    )
    order_long, ordinary = ordinary_daily(retained, cfg)
    carry_dates = missing_l2_dates(states, cfg)
    carried, carry_audit = carry_rows(retained, prices, carry_dates, cfg)
    daily = finalize_daily(ordinary, carried)
    metrics = summarize(daily, cfg)
    effects = paired_clock_effects(daily, metrics)
    qa = build_qa(
        orders, retained, daily, metrics, effects, clocks, carry_dates, cfg
    )

    write_csv(clocks, OUTPUT_DIR / "clock_selection.csv")
    retained.write_parquet(OUTPUT_DIR / "order_pricing_inputs.parquet", compression="zstd")
    order_long.write_parquet(OUTPUT_DIR / "order_treatments.parquet", compression="zstd")
    write_csv(entry_audit, OUTPUT_DIR / "entry_and_markout_audit.csv")
    write_csv(carry_audit, OUTPUT_DIR / "carry_audit.csv")
    write_csv(daily, OUTPUT_DIR / "daily_results.csv")
    write_csv(metrics, OUTPUT_DIR / "strategy_metrics.csv")
    write_csv(effects, OUTPUT_DIR / "paired_clock_effects.csv")
    (OUTPUT_DIR / "qa.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.no_figure:
        make_figure(daily, OUTPUT_DIR / "cumulative_slipclock_pnl.png")
        make_cost_exposure_figure(
            effects, OUTPUT_DIR / "strategy_clock_cost_exposure"
        )

    output_files = [
        OUTPUT_DIR / "clock_selection.csv",
        OUTPUT_DIR / "order_pricing_inputs.parquet",
        OUTPUT_DIR / "order_treatments.parquet",
        OUTPUT_DIR / "entry_and_markout_audit.csv",
        OUTPUT_DIR / "carry_audit.csv",
        OUTPUT_DIR / "daily_results.csv",
        OUTPUT_DIR / "strategy_metrics.csv",
        OUTPUT_DIR / "paired_clock_effects.csv",
        OUTPUT_DIR / "qa.json",
    ]
    if not args.no_figure:
        output_files.append(OUTPUT_DIR / "cumulative_slipclock_pnl.png")
        output_files.extend(
            [
                OUTPUT_DIR / "strategy_clock_cost_exposure.pdf",
                OUTPUT_DIR / "strategy_clock_cost_exposure.png",
            ]
        )
    write_result_report(clocks, metrics, effects, qa, output_files)
    output_files.append(OUTPUT_DIR / "EXPERIMENT_RESULT.md")
    write_manifest(output_files, input_files)
    print(json.dumps(qa, indent=2, sort_keys=True))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
