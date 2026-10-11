#!/usr/bin/env python3
"""Reproducible PIT clock-anchor and funding-calendar regime diagnostics.

This is an associative diagnostic: each contrast compares a portfolio's event
minutes with its own same-date, same-universe pre-event minutes.  It does not
identify a causal funding or session effect.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT / "data/tardis/research/released_l2_analysis/baseline"
DEFAULT_DAILY_MINUTE = RESEARCH_ROOT / "pit_exact_clock/pit_adv30_exact_daily_minute.parquet"
DEFAULT_MEMBERS = RESEARCH_ROOT / "analysis_mart/pit_universe_daily_members.parquet"
DEFAULT_DAILY_COSTS = RESEARCH_ROOT / "pit_adv_universe/pit_daily_costs.parquet"
DEFAULT_FUNDING_GLOB = PROJECT_ROOT / "data/binance_vision/parquet/market=futures_um/data_type=fundingRate/symbol=*/year=*/month=*/part-*.parquet"
DEFAULT_OUT_DIR = RESEARCH_ROOT / "pit_clock_anchors"
UNIVERSES = ("top25", "top100", "top200")
VALID_INTERVALS = {1, 2, 4, 8}
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20_260_717


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-minute", type=Path, default=DEFAULT_DAILY_MINUTE)
    parser.add_argument("--members", type=Path, default=DEFAULT_MEMBERS)
    parser.add_argument("--daily-costs", type=Path, default=DEFAULT_DAILY_COSTS)
    parser.add_argument("--funding-glob", type=Path, default=DEFAULT_FUNDING_GLOB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--policy", default="daily")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--volume-stat", default="mean")
    parser.add_argument("--notional", type=int, default=100_000)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--force", action="store_true", help="replace existing output files")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_glob(pattern: Path) -> list[Path]:
    files = sorted(pattern.parent.glob(pattern.name))
    # Path.glob cannot expand multiple parent wildcards, so use pathlib from root.
    if not files:
        import glob

        files = [Path(value) for value in sorted(glob.glob(str(pattern)))]
    if not files:
        raise FileNotFoundError(f"funding parquet glob matched no files: {pattern}")
    return files


def require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}; found {frame.columns}")


def assert_unique(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    duplicates = frame.group_by(keys).len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(f"{label} has {duplicates.height} duplicate keys on {keys}")


def normalize_symbol(column: str) -> pl.Expr:
    """Canonical Binance USD-M symbol: trim and uppercase, with no aliases."""
    return pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_uppercase()


def finite_or_null(frame: pl.DataFrame, label: str) -> None:
    invalid: list[str] = []
    for name, dtype in frame.schema.items():
        if dtype in (pl.Float32, pl.Float64):
            count = frame.filter(pl.col(name).is_not_null() & ~pl.col(name).is_finite()).height
            if count:
                invalid.append(f"{name} ({count})")
    if invalid:
        raise ValueError(f"{label} contains nonfinite values: {', '.join(invalid)}")


def date_expr(value: str) -> pl.Expr:
    return pl.lit(value).str.strptime(pl.Date, strict=True)


def mean_or_null(expression: pl.Expr) -> pl.Expr:
    return expression.mean()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of no values")
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def month_block_ci(values: list[tuple[date, float]], reps: int, seed: int, key: str) -> tuple[float, float]:
    if reps < 2:
        raise ValueError("bootstrap reps must be at least 2")
    blocks: dict[str, list[float]] = {}
    for event_date, value in values:
        if not math.isfinite(value):
            raise ValueError(f"nonfinite cost delta submitted to bootstrap for {key}")
        blocks.setdefault(event_date.strftime("%Y-%m"), []).append(value)
    months = sorted(blocks)
    if len(months) < 2:
        raise ValueError(f"month-block bootstrap requires at least two calendar months for {key}")
    samples = [blocks[month] for month in months]
    stream_seed = seed + zlib.crc32(key.encode("utf-8"))
    rng = random.Random(stream_seed)
    estimates: list[float] = []
    for _ in range(reps):
        chosen = [samples[rng.randrange(len(samples))] for _ in months]
        total = sum(sum(block) for block in chosen)
        count = sum(len(block) for block in chosen)
        estimates.append(total / count)
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def write_outputs(frame: pl.DataFrame, stem: str, out_dir: Path, force: bool, csv: bool = True) -> dict[str, int]:
    parquet = out_dir / f"{stem}.parquet"
    csv_path = out_dir / f"{stem}.csv"
    targets = [parquet, csv_path] if csv else [parquet]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to replace existing output(s): {existing}; pass --force")
    frame.write_parquet(parquet, compression="zstd")
    if csv:
        frame.write_csv(csv_path)
    return {"parquet": frame.height, "csv": frame.height if csv else 0}


def xny_events(start: date, end: date) -> pl.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    records: list[dict[str, Any]] = []
    for session in sessions:
        session_date = session.date()
        for anchor, timestamp in (("us_cash_open", calendar.session_open(session)), ("us_cash_close", calendar.session_close(session))):
            stamp = timestamp.tz_convert("UTC")
            records.append(
                {
                    "date": session_date,
                    "anchor": anchor,
                    "event_minute": stamp.hour * 60 + stamp.minute,
                    "event_utc_labels": stamp.strftime("%H:%M UTC"),
                    "event_datetime_utc": stamp.to_pydatetime(),
                }
            )
    if not records:
        raise ValueError("XNYS schedule returned no sessions in the requested sample")
    return pl.DataFrame(records)


def load_panel(args: argparse.Namespace) -> pl.DataFrame:
    frame = pl.read_parquet(require_file(args.daily_minute, "daily-minute input"))
    require_columns(frame, {"date", "universe", "minute_of_day", "cost_100000_bps", "spread_bps", "completion_rate"}, "daily-minute input")
    frame = frame.filter(
        (pl.col("date") >= date_expr(args.start_date))
        & (pl.col("date") <= date_expr(args.end_date))
        & pl.col("universe").is_in(UNIVERSES)
    ).select(
        "date", "universe", pl.col("minute_of_day").cast(pl.Int16),
        pl.col("cost_100000_bps").cast(pl.Float64), pl.col("spread_bps").cast(pl.Float64),
        pl.col("completion_rate").cast(pl.Float64),
    )
    if frame.is_empty():
        raise ValueError("daily-minute input has no requested sample rows")
    if frame.filter((pl.col("minute_of_day") < 0) | (pl.col("minute_of_day") >= 1440)).height:
        raise ValueError("daily-minute input contains an invalid minute_of_day")
    assert_unique(frame, ["date", "universe", "minute_of_day"], "daily-minute input")
    available = set(frame.get_column("universe").unique().to_list())
    missing = set(UNIVERSES) - available
    if missing:
        raise ValueError(f"daily-minute input missing expected universes: {sorted(missing)}")
    finite_or_null(frame, "daily-minute input")
    return frame


def load_members(args: argparse.Namespace) -> pl.DataFrame:
    frame = pl.read_parquet(require_file(args.members, "PIT members input"))
    require_columns(frame, {"date", "symbol", "policy", "lookback_days", "volume_stat", "universe"}, "PIT members input")
    frame = frame.filter(
        (pl.col("date") >= date_expr(args.start_date)) & (pl.col("date") <= date_expr(args.end_date))
        & (pl.col("policy") == args.policy) & (pl.col("lookback_days") == args.lookback_days)
        & (pl.col("volume_stat") == args.volume_stat) & pl.col("universe").is_in(UNIVERSES)
    ).select("date", normalize_symbol("symbol").alias("symbol"), "universe")
    if frame.is_empty():
        raise ValueError("PIT members input has no requested policy/lookback/volume sample rows")
    assert_unique(frame, ["date", "universe", "symbol"], "PIT members input")
    counts = frame.group_by(["date", "universe"]).len().rename({"len": "selected_symbols"})
    missing = set(UNIVERSES) - set(counts.get_column("universe").unique().to_list())
    if missing:
        raise ValueError(f"PIT members input missing expected universes: {sorted(missing)}")
    return frame.join(counts, on=["date", "universe"], how="left")


def load_funding(args: argparse.Namespace) -> tuple[pl.DataFrame, list[Path]]:
    files = require_glob(args.funding_glob)
    frame = pl.read_parquet([str(path) for path in files])
    require_columns(frame, {"calc_datetime", "funding_interval_hours", "last_funding_rate", "source_symbol"}, "funding input")
    frame = frame.with_columns(
        [
            pl.col("calc_datetime").cast(pl.Datetime(time_zone="UTC"), strict=False).alias("calc_datetime"),
            pl.col("funding_interval_hours").cast(pl.Int8, strict=False),
            pl.col("last_funding_rate").cast(pl.Float64, strict=False),
            normalize_symbol("source_symbol").alias("symbol"),
        ]
    ).filter(pl.col("calc_datetime").is_not_null()).with_columns(
        [
            pl.col("calc_datetime").dt.date().alias("date"),
            (pl.col("calc_datetime").dt.hour().cast(pl.Int16) * 60 + pl.col("calc_datetime").dt.minute().cast(pl.Int16)).alias("event_minute"),
        ]
    ).filter((pl.col("date") >= date_expr(args.start_date)) & (pl.col("date") <= date_expr(args.end_date)))
    if frame.is_empty():
        raise ValueError("funding input has no rows in the requested sample")
    intervals = set(frame.get_column("funding_interval_hours").drop_nulls().unique().to_list())
    if frame.get_column("funding_interval_hours").null_count() or not intervals <= VALID_INTERVALS:
        raise ValueError(f"funding intervals must be contained in {sorted(VALID_INTERVALS)}, found {sorted(intervals)}")
    if frame.filter(pl.col("symbol") == "").height:
        raise ValueError("funding input contains an empty normalized source symbol")
    assert_unique(frame, ["symbol", "calc_datetime"], "funding input")
    assert_unique(frame, ["symbol", "date", "event_minute"], "funding input at minute resolution")
    finite_or_null(frame, "funding input")
    return frame.select("date", "symbol", "calc_datetime", "event_minute", "funding_interval_hours", "last_funding_rate"), files


def funding_membership(funding: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    joined = funding.join(members, on=["date", "symbol"], how="inner")
    if joined.is_empty():
        raise ValueError("no actual funding rows overlap primary PIT membership")
    return joined


def density_by_event(events: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    """Attach membership-normalized actual funding densities to event keys."""
    keys = ["date", "universe", "event_minute"]
    funded = events.group_by(keys).agg(
        pl.col("symbol").n_unique().alias("funding_symbols"),
        pl.col("funding_interval_hours").eq(1).sum().alias("funding_symbols_1h"),
        pl.col("funding_interval_hours").eq(2).sum().alias("funding_symbols_2h"),
        pl.col("funding_interval_hours").eq(4).sum().alias("funding_symbols_4h"),
        pl.col("funding_interval_hours").eq(8).sum().alias("funding_symbols_8h"),
    )
    denominators = members.select("date", "universe", "selected_symbols").unique()
    assert_unique(denominators, ["date", "universe"], "PIT membership denominators")
    return funded.join(denominators, on=["date", "universe"], how="inner").with_columns(
        [
            (pl.col("funding_symbols") / pl.col("selected_symbols")).alias("mean_actual_funding_share"),
            *( (pl.col(f"funding_symbols_{hours}h") / pl.col("selected_symbols")).alias(f"funding_share_{hours}h") for hours in (1, 2, 4, 8) ),
        ]
    )


def attach_outcomes(events: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Calculate all event/pre windows from the fixed portfolio minute panel."""
    event_keys = ["date", "universe", "anchor", "event_minute"]
    assert_unique(events, event_keys, "anchor event table")
    points = events.select(event_keys).with_row_index("event_id").join(
        pl.DataFrame({"offset": list(range(-10, 3))}), how="cross"
    ).with_columns((pl.col("event_minute") + pl.col("offset")).cast(pl.Int16).alias("minute_of_day"))
    # Events requiring a prior UTC-day baseline are not silently wrapped.
    points = points.filter((pl.col("minute_of_day") >= 0) & (pl.col("minute_of_day") < 1440))
    joined = points.join(panel, on=["date", "universe", "minute_of_day"], how="left")
    window = joined.group_by("event_id").agg(
        [
            mean_or_null(pl.when(pl.col("offset") >= 0).then(pl.col("cost_100000_bps"))).alias("event_cost_bps"),
            mean_or_null(pl.when(pl.col("offset") <= -2).then(pl.col("cost_100000_bps"))).alias("baseline_cost_bps"),
            mean_or_null(pl.when(pl.col("offset") >= 0).then(pl.col("spread_bps"))).alias("event_spread_bps"),
            mean_or_null(pl.when(pl.col("offset") <= -2).then(pl.col("spread_bps"))).alias("baseline_spread_bps"),
            mean_or_null(pl.when(pl.col("offset") >= 0).then(pl.col("completion_rate"))).alias("event_completion_rate"),
            mean_or_null(pl.when(pl.col("offset") <= -2).then(pl.col("completion_rate"))).alias("baseline_completion_rate"),
        ]
    )
    return events.with_row_index("event_id").join(window, on="event_id", how="left").with_columns(
        [
            (pl.col("event_cost_bps") - pl.col("baseline_cost_bps")).alias("cost_delta_bps"),
            (pl.col("event_spread_bps") - pl.col("baseline_spread_bps")).alias("spread_delta_bps"),
            ((pl.col("event_cost_bps") - pl.col("event_spread_bps") / 2) - (pl.col("baseline_cost_bps") - pl.col("baseline_spread_bps") / 2)).alias("beyond_half_spread_delta_bps"),
            ((pl.col("event_completion_rate") - pl.col("baseline_completion_rate")) * 100).alias("completion_delta_pp"),
        ]
    ).drop("event_id")


def aggregate_event_observations(detail: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Reduce simultaneous clock observations to one date/universe/anchor record."""
    outcome_columns = [
        "event_cost_bps", "baseline_cost_bps", "event_spread_bps", "baseline_spread_bps",
        "event_completion_rate", "baseline_completion_rate", "cost_delta_bps",
        "spread_delta_bps", "beyond_half_spread_delta_bps", "completion_delta_pp",
        "mean_actual_funding_share", "funding_share_1h", "funding_share_2h",
        "funding_share_4h", "funding_share_8h", "mean_last_funding_rate",
    ]
    sum_columns = [
        "event_count", "funding_symbols", "funding_symbols_1h", "funding_symbols_2h",
        "funding_symbols_4h", "funding_symbols_8h",
    ]
    aggregations: list[pl.Expr] = []
    aggregations.extend(pl.col(column).mean().alias(column) for column in outcome_columns if column in detail.columns)
    aggregations.extend(pl.col(column).sum().alias(column) for column in sum_columns if column in detail.columns)
    if "selected_symbols" in detail.columns:
        aggregations.append(pl.col("selected_symbols").first().alias("selected_symbols"))
    if "event_utc_labels" in detail.columns:
        aggregations.append(pl.col("event_utc_labels").sort().str.join("; ").alias("event_utc_labels"))
    if "event_datetime_utc" in detail.columns:
        aggregations.append(pl.col("event_datetime_utc").min().alias("event_datetime_utc"))
    if "interval_hours_present" in detail.columns:
        aggregations.append(pl.col("interval_hours_present").sort().str.join("; ").alias("interval_hours_present"))
    return detail.group_by(keys).agg(aggregations).sort(keys)


def summarize_events(detail: pl.DataFrame, group_columns: list[str], reps: int, seed: int) -> pl.DataFrame:
    metrics = ["cost_delta_bps", "spread_delta_bps", "beyond_half_spread_delta_bps", "completion_delta_pp"]
    rows: list[dict[str, Any]] = []
    for keys, group in detail.group_by(group_columns, maintain_order=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row["dates"] = group.get_column("date").n_unique()
        row["events"] = int(group.get_column("event_count").sum())
        for field in ("mean_actual_funding_share", "funding_share_1h", "funding_share_2h", "funding_share_4h", "funding_share_8h"):
            row[field] = group.get_column(field).mean()
        if "event_utc_labels" in group.columns:
            row["event_utc_labels"] = "; ".join(sorted(group.get_column("event_utc_labels").drop_nulls().unique().to_list()))
        for metric in metrics:
            row[f"mean_{metric}"] = group.get_column(metric).mean()
        bootstrap_values = [
            (event_date, float(value))
            for event_date, value in group.select("date", "cost_delta_bps").iter_rows()
            if value is not None
        ]
        bootstrap_months = len({event_date.strftime("%Y-%m") for event_date, _ in bootstrap_values})
        row["bootstrap_months"] = bootstrap_months
        if bootstrap_months >= 2:
            ci_low, ci_high = month_block_ci(bootstrap_values, reps, seed, "|".join(map(str, keys)))
            row["cost_delta_ci95_low_bps"] = ci_low
            row["cost_delta_ci95_high_bps"] = ci_high
        else:
            row["cost_delta_ci95_low_bps"] = None
            row["cost_delta_ci95_high_bps"] = None
        rows.append(row)
    summary = pl.DataFrame(rows).sort(group_columns)
    finite_or_null(summary, "event summary")
    return summary


def build_anchor_events(funding_members: pl.DataFrame, members: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    density = density_by_event(funding_members, members)
    common = ["date", "universe", "event_minute"]
    four = density.filter(pl.col("event_minute").is_in([240, 720, 1200])).with_columns(
        pl.lit("four_hour_nonoverlap").alias("anchor"),
        pl.col("event_minute").map_elements(lambda minute: f"{minute // 60:02d}:00 UTC", return_dtype=pl.Utf8).alias("event_utc_labels"),
    )
    eight = density.filter(pl.col("event_minute").is_in([480, 960])).with_columns(
        pl.lit("eight_hour_overlap").alias("anchor"),
        pl.col("event_minute").map_elements(lambda minute: f"{minute // 60:02d}:00 UTC", return_dtype=pl.Utf8).alias("event_utc_labels"),
    )
    schedule = xny_events(date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)).drop("event_datetime_utc")
    denominators = members.select("date", "universe", "selected_symbols").unique()
    cash = schedule.join(denominators, on="date", how="inner").join(density, on=common, how="left", suffix="_funding")
    cash = cash.with_columns(
        [pl.col(column).fill_null(0) for column in ["funding_symbols", "funding_symbols_1h", "funding_symbols_2h", "funding_symbols_4h", "funding_symbols_8h"]]
    ).with_columns(
        [
            (pl.col("funding_symbols") / pl.col("selected_symbols")).alias("mean_actual_funding_share"),
            *((pl.col(f"funding_symbols_{hours}h") / pl.col("selected_symbols")).alias(f"funding_share_{hours}h") for hours in (1, 2, 4, 8)),
        ]
    )
    all_events = pl.concat([four, eight, cash], how="diagonal_relaxed").with_columns(pl.lit(1).alias("event_count"))
    assert_unique(all_events, ["date", "universe", "anchor", "event_minute"], "anchor event table")
    if all_events.is_empty():
        raise ValueError("no anchor events overlap the requested PIT sample")
    return all_events


def build_funding_events(funding_members: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    keys = ["date", "universe", "event_minute"]
    density = density_by_event(funding_members, members)
    flags = funding_members.group_by(keys).agg(
        [
            pl.col("funding_interval_hours").eq(8).any().alias("has_8h_component"),
            pl.col("funding_interval_hours").n_unique().alias("interval_count"),
            pl.col("funding_interval_hours").sort().unique().cast(pl.Utf8).str.join("+").alias("interval_hours_present"),
            pl.col("calc_datetime").min().alias("event_datetime_utc"),
            pl.col("last_funding_rate").mean().alias("mean_last_funding_rate"),
        ]
    )
    return density.join(flags, on=keys, how="inner").with_columns(
        [
            pl.when(pl.col("has_8h_component")).then(pl.lit("with_8h_component")).otherwise(pl.lit("without_8h_component")).alias("eight_hour_component"),
            pl.when(pl.col("interval_count") > 1).then(pl.lit("multi_interval")).otherwise(pl.lit("single_interval")).alias("interval_mix"),
            pl.col("event_datetime_utc").dt.strftime("%H:%M UTC").alias("event_utc_labels"),
            pl.lit("actual_funding").alias("anchor"),
            pl.lit(1).alias("event_count"),
        ]
    )


def build_regime(daily_costs: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    required = {"date", "policy", "lookback_days", "volume_stat", "universe", "equal_weight_mean_cost_100000_bps", "spread_mean_bps"}
    require_columns(daily_costs, required, "daily costs input")
    p90 = "mean_of_symbol_p90_cost_100000_bps"
    if p90 not in daily_costs.columns:
        p90 = "symbol_mean_cost_p90_100000_bps"
    if p90 not in daily_costs.columns:
        raise ValueError("daily costs input has no supported daily p90 cost column")
    frame = daily_costs.filter(
        (pl.col("date") >= date_expr(args.start_date)) & (pl.col("date") <= date_expr(args.end_date))
        & (pl.col("policy") == args.policy) & (pl.col("lookback_days") == args.lookback_days)
        & (pl.col("volume_stat") == args.volume_stat) & pl.col("universe").is_in(UNIVERSES)
    ).select(
        "date", "universe", pl.col("equal_weight_mean_cost_100000_bps").cast(pl.Float64).alias("daily_mean_cost_bps"),
        pl.col("spread_mean_bps").cast(pl.Float64).alias("daily_mean_spread_bps"), pl.col(p90).cast(pl.Float64).alias("daily_p90_cost_bps"),
    ).sort(["universe", "date"])
    if frame.is_empty():
        raise ValueError("daily costs input has no requested PIT rows")
    assert_unique(frame, ["date", "universe"], "daily costs input")
    missing = set(UNIVERSES) - set(frame.get_column("universe").unique().to_list())
    if missing:
        raise ValueError(f"daily costs input missing expected universes: {sorted(missing)}")
    frame = frame.with_columns(
        [
            pl.col("daily_mean_cost_bps").rolling_mean(window_size=30, min_samples=30).over("universe").alias("daily_mean_cost_30_calendar_observation_ma_bps"),
            pl.col("daily_mean_spread_bps").rolling_mean(window_size=30, min_samples=30).over("universe").alias("daily_mean_spread_30_calendar_observation_ma_bps"),
            pl.col("daily_p90_cost_bps").rolling_mean(window_size=30, min_samples=30).over("universe").alias("daily_p90_cost_30_calendar_observation_ma_bps"),
        ]
    )
    finite_or_null(frame, "daily regime")
    return frame


def regime_summary(regime: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for universe_key, group in regime.group_by("universe", maintain_order=True):
        universe = universe_key[0] if isinstance(universe_key, tuple) else universe_key
        metric_values = {
            "daily_cost": [float(value) for value in group.get_column("daily_mean_cost_bps").drop_nulls().to_list()],
            "daily_spread": [float(value) for value in group.get_column("daily_mean_spread_bps").drop_nulls().to_list()],
            "daily_p90_cost": [float(value) for value in group.get_column("daily_p90_cost_bps").drop_nulls().to_list()],
        }
        if not metric_values["daily_cost"]:
            raise ValueError(f"daily regime has no finite daily costs for {universe}")
        row: dict[str, Any] = {"universe": universe, "dates": group.get_column("date").n_unique()}
        for prefix, values in metric_values.items():
            row[f"full_mean_{prefix}_bps"] = sum(values) / len(values) if values else None
            row[f"{prefix}_p10_bps"] = percentile(values, 0.10) if values else None
            row[f"{prefix}_p50_bps"] = percentile(values, 0.50) if values else None
            row[f"{prefix}_p90_bps"] = percentile(values, 0.90) if values else None
        for prefix, column in (
            ("cost", "daily_mean_cost_30_calendar_observation_ma_bps"),
            ("spread", "daily_mean_spread_30_calendar_observation_ma_bps"),
            ("p90_cost", "daily_p90_cost_30_calendar_observation_ma_bps"),
        ):
            ma = group.filter(pl.col(column).is_not_null()).select("date", column)
            if ma.is_empty():
                row[f"{prefix}_30_calendar_observation_ma_min_bps"] = None
                row[f"{prefix}_30_calendar_observation_ma_min_date"] = None
                row[f"{prefix}_30_calendar_observation_ma_max_bps"] = None
                row[f"{prefix}_30_calendar_observation_ma_max_date"] = None
                continue
            min_row = ma.sort(column).row(0, named=True)
            max_row = ma.sort(column, descending=True).row(0, named=True)
            row[f"{prefix}_30_calendar_observation_ma_min_bps"] = min_row[column]
            row[f"{prefix}_30_calendar_observation_ma_min_date"] = min_row["date"]
            row[f"{prefix}_30_calendar_observation_ma_max_bps"] = max_row[column]
            row[f"{prefix}_30_calendar_observation_ma_max_date"] = max_row["date"]
        rows.append(row)
    output = pl.DataFrame(rows).sort("universe")
    finite_or_null(output, "daily regime summary")
    return output


def main() -> None:
    args = parse_args()
    if args.notional != 100_000:
        raise ValueError("this diagnostic is defined only for the primary $100k PIT estimand")
    if date.fromisoformat(args.end_date) < date.fromisoformat(args.start_date):
        raise ValueError("end-date must not precede start-date")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(args)
    members = load_members(args)
    funding, funding_files = load_funding(args)
    funding_members = funding_membership(funding, members)

    anchor_events = build_anchor_events(funding_members, members, args)
    anchor_detail = aggregate_event_observations(
        attach_outcomes(anchor_events, panel), ["date", "universe", "anchor"]
    ).sort(["anchor", "universe", "date"])
    if anchor_detail.filter(pl.col("cost_delta_bps").is_not_null()).height == 0:
        raise ValueError("no anchor events have sufficient PIT event/pre-event cost overlap")
    anchor_summary = summarize_events(anchor_detail, ["anchor", "universe"], args.bootstrap_reps, args.bootstrap_seed)

    funding_events = build_funding_events(funding_members, members)
    funding_detail = aggregate_event_observations(
        attach_outcomes(funding_events, panel),
        ["date", "universe", "anchor", "eight_hour_component", "interval_mix"],
    ).sort(["universe", "date", "eight_hour_component", "interval_mix"])
    if funding_detail.filter(pl.col("cost_delta_bps").is_not_null()).height == 0:
        raise ValueError("no actual funding events have sufficient PIT event/pre-event cost overlap")
    funding_summary = summarize_events(funding_detail, ["universe", "eight_hour_component", "interval_mix"], args.bootstrap_reps, args.bootstrap_seed)

    daily_costs = pl.read_parquet(require_file(args.daily_costs, "daily costs input"))
    regime = build_regime(daily_costs, args)
    regime_summary_frame = regime_summary(regime)

    counts = {
        "clock_anchor_event_detail": write_outputs(anchor_detail, "clock_anchor_event_detail", args.out_dir, args.force),
        "clock_anchor_summary": write_outputs(anchor_summary, "clock_anchor_summary", args.out_dir, args.force),
        "funding_event_detail": write_outputs(funding_detail, "funding_event_detail", args.out_dir, args.force, csv=False),
        "funding_event_summary": write_outputs(funding_summary, "funding_event_summary", args.out_dir, args.force),
        "pit_daily_regime": write_outputs(regime, "pit_daily_regime", args.out_dir, args.force),
        "pit_daily_regime_summary": write_outputs(regime_summary_frame, "pit_daily_regime_summary", args.out_dir, args.force),
    }
    coverage = funding_members.select(
        pl.col("date").min().alias("first_date"), pl.col("date").max().alias("last_date"),
        pl.len().alias("membership_funding_rows"), pl.col("symbol").n_unique().alias("symbols"),
        pl.col("funding_interval_hours").value_counts(sort=True).alias("interval_counts"),
    ).row(0, named=True)
    manifest = {
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"dailyMinute": str(args.daily_minute), "members": str(args.members), "dailyCosts": str(args.daily_costs), "fundingGlob": str(args.funding_glob), "fundingFiles": [str(path) for path in funding_files]},
        "sample": {"startDate": args.start_date, "endDate": args.end_date, "policy": args.policy, "lookbackDays": args.lookback_days, "volumeStat": args.volume_stat, "notionalUsd": args.notional, "universes": list(UNIVERSES)},
        "definitions": {"eventWindowMinutes": [0, 2], "preEventBaselineMinutes": [-10, -2], "eventAggregation": "one date/universe/anchor record before inference", "funding": "actual Binance Vision calc_datetime rows joined to same-date normalized primary PIT membership", "xnys": "exchange_calendars XNYS official sessions, including DST, holidays, and early closes", "interpretation": "associative diagnostic; no causal claim"},
        "dependencies": {"polars": pl.__version__, "pandas": pd.__version__, "exchange_calendars": xcals.__version__},
        "bootstrap": {"seed": args.bootstrap_seed, "reps": args.bootstrap_reps, "method": "calendar-month block bootstrap of date/universe event cost deltas"},
        "actualFundingCoverage": coverage,
        "outputRowCounts": counts,
    }
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"refusing to replace existing output: {manifest_path}; pass --force")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"outDir": str(args.out_dir), "outputRowCounts": counts}, default=str))


if __name__ == "__main__":
    main()
