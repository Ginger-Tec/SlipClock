#!/usr/bin/env python3
"""Build exact PIT ADV30 date-symbol-minute clock surfaces for the L2 paper.

The experiment fixes the point-in-time daily universe from prior quote-volume
information, then joins the selected symbols to the realized one-minute L2
book-walk rows on the exact (date, symbol, minute_of_day) key.  The resulting
minute surface is therefore not a membership-frequency projection: each plotted
minute is built from actual selected-symbol book states observed at that date
and UTC minute.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESEARCH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tardis"
    / "research"
    / "released_l2_analysis"
    / "baseline"
)
DEFAULT_MART_ROOT = DEFAULT_RESEARCH_ROOT / "analysis_mart"
DEFAULT_OUT_DIR = DEFAULT_RESEARCH_ROOT / "pit_exact_clock"
DEFAULT_DOC_OUT = (
    PROJECT_ROOT
    / "docs"
    / "released_l2_analysis"
    / "baseline"
    / "l2_pit_exact_clock.md"
)
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2026-04-30"
UNIVERSE_THRESHOLDS = [("top25", 25), ("top50", 50), ("top100", 100), ("top150", 150), ("top200", 200)]
MAIN_UNIVERSES = ["top25", "top100", "top200"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    p.add_argument("--mart-root", type=Path, default=DEFAULT_MART_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--markdown-out", type=Path, default=DEFAULT_DOC_OUT)
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--policy", default="daily")
    p.add_argument("--lookback-days", type=int, default=30)
    p.add_argument("--volume-stat", default="mean")
    p.add_argument("--threads", type=int, default=max(1, min(32, (os.cpu_count() or 8) // 2)))
    p.add_argument("--bootstrap-reps", type=int, default=10_000)
    p.add_argument("--bootstrap-seed", type=int, default=20_260_712)
    p.add_argument("--memory-limit", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def sql_str(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {rel(path)}")
    return path


def run_copy(con: duckdb.DuckDBPyConnection, query: str, path: Path, *, force: bool) -> None:
    ensure_parent(path)
    if path.exists() and not force:
        print(json.dumps({"event": "skip_existing", "path": rel(path)}, ensure_ascii=False), flush=True)
        return
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(json.dumps({"event": "write_start", "path": rel(path)}, ensure_ascii=False), flush=True)
    con.execute(
        f"""
        COPY (
        {query}
        ) TO {sql_str(tmp)} (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 1000000
        )
        """
    )
    tmp.replace(path)
    print(json.dumps({"event": "write_done", "path": rel(path)}, ensure_ascii=False), flush=True)


def parquet_meta(path: Path) -> dict[str, Any]:
    meta = pq.ParquetFile(path).metadata
    schema = pq.read_schema(path)
    return {
        "path": rel(path),
        "rows": int(meta.num_rows),
        "columns": int(len(schema.names)),
        "sizeBytes": int(path.stat().st_size),
        "schema": {name: str(schema.field(name).type) for name in schema.names},
    }


def write_csv_from_parquet(con: duckdb.DuckDBPyConnection, parquet_path: Path, csv_path: Path) -> None:
    ensure_parent(csv_path)
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet({sql_str(parquet_path)})
        ) TO {sql_str(csv_path)} (HEADER, DELIMITER ',')
        """
    )


def circular_window_values(minutes: list[int], values: list[float], width: int) -> list[float]:
    """Average observed minute values inside each UTC-clock window.

    The L2 replay starts at 00:01 because the first usable book snapshot arrives
    after midnight. Selecting windows by minute labels rather than adjacent row
    positions keeps wraparound windows consistent with the date-level scorer
    when the 00:00 state is unavailable.
    """
    if len(minutes) != len(values):
        raise ValueError("minute and value arrays must have equal length")
    if not values:
        return []
    pairs = [
        (int(minute), float(value))
        for minute, value in zip(minutes, values, strict=True)
        if value is not None and math.isfinite(value)
    ]
    means: list[float] = []
    for start in minutes:
        window = [value for minute, value in pairs if in_circular_window(minute, int(start), width)]
        if not window:
            raise ValueError(f"clock window starting at {start} has no observed minutes")
        means.append(sum(window) / len(window))
    return means


def minute_label(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def in_circular_window(minute: int, start: int, width: int) -> bool:
    return (minute - start) % 1440 < width


def standard_error(values: list[float]) -> float:
    values = [v for v in values if v is not None and math.isfinite(v)]
    n = len(values)
    if n <= 1:
        return float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def month_block_bootstrap(
    values_by_month: dict[str, list[float]], *, reps: int, seed: int
) -> dict[str, float]:
    if reps < 2:
        raise ValueError(f"bootstrap reps must be at least 2, got {reps}")
    months = sorted(values_by_month)
    if len(months) < 2:
        raise ValueError(f"month-block bootstrap requires at least 2 months, got {len(months)}")

    blocks = [
        [float(value) for value in values_by_month[month] if value is not None and math.isfinite(value)]
        for month in months
    ]
    if any(not block for block in blocks):
        raise ValueError("month-block bootstrap received an empty month block")

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(reps):
        total = 0.0
        count = 0
        for _ in months:
            block = blocks[rng.randrange(len(blocks))]
            total += sum(block)
            count += len(block)
        estimates.append(total / count)

    bootstrap_mean = sum(estimates) / len(estimates)
    variance = sum((value - bootstrap_mean) ** 2 for value in estimates) / (len(estimates) - 1)
    return {
        "se": math.sqrt(variance),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def build_contrast_tables(out_dir: Path, paths: dict[str, Path]) -> None:
    surface = pl.read_parquet(paths["minute_surface"])
    daily = pl.read_parquet(paths["daily_minute"])
    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for universe in MAIN_UNIVERSES:
        sub = surface.filter(pl.col("universe") == universe).sort("minute_of_day")
        values = sub["mean_cost_100000_bps"].to_list()
        minute_numbers = sub["minute_of_day"].to_list()
        if len(values) < 1000:
            raise RuntimeError(f"Too few minute rows for {universe}: {len(values)}")
        for width in (30, 60, 120):
            window_means = circular_window_values(minute_numbers, values, width)
            best_idx = min(range(len(window_means)), key=window_means.__getitem__)
            worst_idx = max(range(len(window_means)), key=window_means.__getitem__)
            best_start = int(minute_numbers[best_idx])
            worst_start = int(minute_numbers[worst_idx])
            dsub = daily.filter(pl.col("universe") == universe)
            by_date = (
                dsub.with_columns(
                    [
                        pl.col("minute_of_day").map_elements(
                            lambda m, s=best_start, w=width: in_circular_window(int(m), s, w), return_dtype=pl.Boolean
                        ).alias("is_best_window"),
                        pl.col("minute_of_day").map_elements(
                            lambda m, s=worst_start, w=width: in_circular_window(int(m), s, w), return_dtype=pl.Boolean
                        ).alias("is_worst_window"),
                    ]
                )
                .group_by("date")
                .agg(
                    [
                        pl.col("cost_100000_bps").filter(pl.col("is_best_window")).mean().alias("best_cost_bps"),
                        pl.col("cost_100000_bps").filter(pl.col("is_worst_window")).mean().alias("worst_cost_bps"),
                        pl.col("completion_rate").filter(pl.col("is_best_window")).mean().alias("best_completion_rate"),
                        pl.col("completion_rate").filter(pl.col("is_worst_window")).mean().alias("worst_completion_rate"),
                    ]
                )
                .with_columns((pl.col("worst_cost_bps") - pl.col("best_cost_bps")).alias("worst_minus_best_bps"))
                .filter(pl.col("worst_minus_best_bps").is_not_null())
                .sort("date")
            )
            diffs = by_date["worst_minus_best_bps"].to_list()
            mean_diff = sum(diffs) / len(diffs)
            se_diff = standard_error(diffs)
            rows.append(
                {
                    "universe": universe,
                    "window_minutes": width,
                    "best_start_utc": minute_label(best_start),
                    "best_mean_cost_bps": float(window_means[best_idx]),
                    "worst_start_utc": minute_label(worst_start),
                    "worst_mean_cost_bps": float(window_means[worst_idx]),
                    "surface_range_bps": float(window_means[worst_idx] - window_means[best_idx]),
                    "paired_dates": len(diffs),
                    "paired_worst_minus_best_bps": float(mean_diff),
                    "paired_se_bps": float(se_diff),
                    "paired_t_stat": float(mean_diff / se_diff) if se_diff and math.isfinite(se_diff) and se_diff > 0 else None,
                    "best_completion_rate": float(by_date["best_completion_rate"].mean()),
                    "worst_completion_rate": float(by_date["worst_completion_rate"].mean()),
                }
            )
            for row in by_date.iter_rows(named=True):
                daily_rows.append(
                    {
                        "universe": universe,
                        "window_minutes": width,
                        "date": row["date"],
                        "best_start_utc": minute_label(best_start),
                        "worst_start_utc": minute_label(worst_start),
                        "best_cost_bps": row["best_cost_bps"],
                        "worst_cost_bps": row["worst_cost_bps"],
                        "worst_minus_best_bps": row["worst_minus_best_bps"],
                        "best_completion_rate": row["best_completion_rate"],
                        "worst_completion_rate": row["worst_completion_rate"],
                    }
                )
    pl.DataFrame(rows).write_parquet(paths["window_contrasts"], compression="zstd")
    pl.DataFrame(rows).write_csv(out_dir / "pit_adv30_exact_window_contrasts.csv")
    pl.DataFrame(daily_rows).write_parquet(paths["window_contrast_daily"], compression="zstd")


def build_leave_one_month_window_validation(
    out_dir: Path, paths: dict[str, Path], *, bootstrap_reps: int, bootstrap_seed: int
) -> None:
    daily = (
        pl.read_parquet(paths["daily_minute"])
        .filter((pl.col("universe").is_in(MAIN_UNIVERSES)) & pl.col("cost_100000_bps").is_not_null())
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
    )
    months = daily.select("month").unique().sort("month")["month"].to_list()
    heldout_rows: list[dict[str, Any]] = []

    for universe in MAIN_UNIVERSES:
        dsub = daily.filter(pl.col("universe") == universe)
        for month in months:
            train_surface = (
                dsub.filter(pl.col("month") != month)
                .group_by("minute_of_day")
                .agg(pl.col("cost_100000_bps").mean().alias("mean_cost_100000_bps"))
                .sort("minute_of_day")
            )
            values = train_surface["mean_cost_100000_bps"].to_list()
            minute_numbers = train_surface["minute_of_day"].to_list()
            if len(values) < 1000:
                raise RuntimeError(f"Too few training minute rows for {universe} excluding {month}: {len(values)}")

            width = 60
            window_means = circular_window_values(minute_numbers, values, width)
            best_idx = min(range(len(window_means)), key=window_means.__getitem__)
            worst_idx = max(range(len(window_means)), key=window_means.__getitem__)
            best_start = int(minute_numbers[best_idx])
            worst_start = int(minute_numbers[worst_idx])

            heldout = (
                dsub.filter(pl.col("month") == month)
                .with_columns(
                    [
                        pl.col("minute_of_day").map_elements(
                            lambda m, s=best_start, w=width: in_circular_window(int(m), s, w), return_dtype=pl.Boolean
                        ).alias("is_best_window"),
                        pl.col("minute_of_day").map_elements(
                            lambda m, s=worst_start, w=width: in_circular_window(int(m), s, w), return_dtype=pl.Boolean
                        ).alias("is_worst_window"),
                    ]
                )
                .group_by("date")
                .agg(
                    [
                        pl.first("month").alias("heldout_month"),
                        pl.col("cost_100000_bps").filter(pl.col("is_best_window")).mean().alias("best_cost_bps"),
                        pl.col("cost_100000_bps").filter(pl.col("is_worst_window")).mean().alias("worst_cost_bps"),
                        pl.col("completion_rate").filter(pl.col("is_best_window")).mean().alias("best_completion_rate"),
                        pl.col("completion_rate").filter(pl.col("is_worst_window")).mean().alias("worst_completion_rate"),
                    ]
                )
                .with_columns((pl.col("worst_cost_bps") - pl.col("best_cost_bps")).alias("worst_minus_best_bps"))
                .filter(pl.col("worst_minus_best_bps").is_not_null())
                .sort("date")
            )
            for row in heldout.iter_rows(named=True):
                heldout_rows.append(
                    {
                        "universe": universe,
                        "window_minutes": width,
                        "heldout_month": month,
                        "date": row["date"],
                        "best_start_utc": minute_label(best_start),
                        "worst_start_utc": minute_label(worst_start),
                        "best_cost_bps": row["best_cost_bps"],
                        "worst_cost_bps": row["worst_cost_bps"],
                        "worst_minus_best_bps": row["worst_minus_best_bps"],
                        "best_completion_rate": row["best_completion_rate"],
                        "worst_completion_rate": row["worst_completion_rate"],
                    }
                )

    detail = pl.DataFrame(heldout_rows)
    summary_rows: list[dict[str, Any]] = []
    for universe in MAIN_UNIVERSES:
        sub = detail.filter(pl.col("universe") == universe)
        diffs = sub["worst_minus_best_bps"].to_list()
        mean_diff = sum(diffs) / len(diffs)
        iid_se_diff = standard_error(diffs)
        values_by_month = {
            row["heldout_month"]: row["diffs"]
            for row in (
                sub.group_by("heldout_month")
                .agg(pl.col("worst_minus_best_bps").alias("diffs"))
                .sort("heldout_month")
                .iter_rows(named=True)
            )
        }
        bootstrap = month_block_bootstrap(values_by_month, reps=bootstrap_reps, seed=bootstrap_seed)
        positive_share = sum(1 for value in diffs if value > 0) / len(diffs)
        summary_rows.append(
            {
                "universe": universe,
                "window_minutes": 60,
                "heldout_months": sub["heldout_month"].n_unique(),
                "heldout_dates": len(diffs),
                "heldout_worst_minus_best_bps": float(mean_diff),
                "inference_method": "calendar_month_block_bootstrap_percentile",
                "bootstrap_reps": bootstrap_reps,
                "bootstrap_seed": bootstrap_seed,
                "month_block_bootstrap_se_bps": bootstrap["se"],
                "month_block_bootstrap_ci95_low_bps": bootstrap["ci95_low"],
                "month_block_bootstrap_ci95_high_bps": bootstrap["ci95_high"],
                "se_bps": bootstrap["se"],
                "ci95_low_bps": bootstrap["ci95_low"],
                "ci95_high_bps": bootstrap["ci95_high"],
                "iid_se_bps": float(iid_se_diff),
                "iid_ci95_low_bps": float(mean_diff - 1.96 * iid_se_diff),
                "iid_ci95_high_bps": float(mean_diff + 1.96 * iid_se_diff),
                "iid_t_stat": (
                    float(mean_diff / iid_se_diff)
                    if iid_se_diff and math.isfinite(iid_se_diff) and iid_se_diff > 0
                    else None
                ),
                "positive_date_share": float(positive_share),
                "avg_best_cost_bps": float(sub["best_cost_bps"].mean()),
                "avg_worst_cost_bps": float(sub["worst_cost_bps"].mean()),
                "avg_best_completion_rate": float(sub["best_completion_rate"].mean()),
                "avg_worst_completion_rate": float(sub["worst_completion_rate"].mean()),
            }
        )

    detail.write_parquet(paths["lomo_window_detail"], compression="zstd")
    detail.write_csv(out_dir / "pit_adv30_exact_lomo_window_detail.csv")
    summary = pl.DataFrame(summary_rows).sort(pl.col("universe").str.replace("top", "").cast(pl.Int32))
    summary.write_parquet(paths["lomo_window_summary"], compression="zstd")
    summary.write_csv(paths["lomo_window_summary_csv"])


def markdown_table(df: pl.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.iter_rows(named=True):
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(args: argparse.Namespace, paths: dict[str, Path], manifest: dict[str, Any]) -> None:
    ensure_parent(args.markdown_out)
    surface = pl.read_parquet(paths["minute_surface"])
    hour = pl.read_parquet(paths["hour_surface"])
    contrasts = pl.read_parquet(paths["window_contrasts"])
    lomo = pl.read_parquet(paths["lomo_window_summary"]) if paths["lomo_window_summary"].exists() else pl.DataFrame()
    top_table = (
        surface.filter(pl.col("universe").is_in(MAIN_UNIVERSES))
        .group_by("universe")
        .agg(
            [
                pl.col("mean_cost_100000_bps").mean().alias("mean_minute_cost_bps"),
                pl.col("mean_cost_100000_bps").min().alias("best_minute_bps"),
                pl.col("mean_cost_100000_bps").max().alias("worst_minute_bps"),
                pl.col("avg_completion_rate").mean().alias("avg_completion_rate"),
                pl.col("dates").min().alias("dates"),
            ]
        )
        .sort(pl.col("universe").str.replace("top", "").cast(pl.Int32))
    )
    hour_table = (
        hour.filter(pl.col("universe").is_in(MAIN_UNIVERSES))
        .sort(["universe", "mean_cost_100000_bps"], descending=[False, True])
        .group_by("universe", maintain_order=True)
        .head(3)
        .select(["universe", "hour_utc", "mean_cost_100000_bps", "avg_completion_rate"])
    )
    lomo_table = (
        lomo.select(
            [
                "universe",
                "heldout_months",
                "heldout_dates",
                "heldout_worst_minus_best_bps",
                "month_block_bootstrap_se_bps",
                "month_block_bootstrap_ci95_low_bps",
                "month_block_bootstrap_ci95_high_bps",
                "positive_date_share",
                "avg_best_completion_rate",
                "avg_worst_completion_rate",
            ]
        )
        if not lomo.is_empty()
        else lomo
    )
    completion_audit = (
        "\n".join(
            (
                f"- {row['universe']}: low/high completion = "
                f"{100 * row['avg_best_completion_rate']:.4f}% / "
                f"{100 * row['avg_worst_completion_rate']:.4f}%; "
                f"high-minus-low = "
                f"{100 * (row['avg_worst_completion_rate'] - row['avg_best_completion_rate']):.4f} "
                "percentage points."
            )
            for row in lomo_table.iter_rows(named=True)
        )
        if not lomo_table.is_empty()
        else "Not yet generated."
    )
    doc = f"""# Exact PIT ADV30 clock surface

Generated: {manifest['generatedUtc']}

## Definition

This experiment fixes the point-in-time ADV30 universe for each UTC date and joins those selected symbols to the realized one-minute L2 book-walk rows on the exact `(date, symbol, minute_of_day)` key.  The headline surface is therefore an exact selected-member minute surface for the current L2-only sample, not a symbol-profile projection.

- Sample: {args.start_date} through {args.end_date}.
- Minute coverage: 1,439 observed UTC positions from 00:01 through 23:59. The 00:00 state is unavailable because the replay cannot form a usable book before the first after-midnight L2 snapshot.
- Universe rule: policy={args.policy}, lookback_days={args.lookback_days}, volume_stat={args.volume_stat}.
- Cost: buy/sell averaged displayed-book $100k mid-reference sweep cost, missing when either side cannot complete.
- Date weighting: daily symbol means are formed first for each universe-minute, then minute surfaces average those daily observations.
- LOMO uncertainty: percentile 95% confidence intervals from {args.bootstrap_reps:,} deterministic calendar-month block-bootstrap replications (seed={args.bootstrap_seed}); dates remain grouped within month.

## Main universe summary

{markdown_table(top_table)}

## Window contrasts

{markdown_table(contrasts.filter(pl.col('universe').is_in(MAIN_UNIVERSES)).sort(['universe', 'window_minutes']))}

## Leave-one-month-out validation

{markdown_table(lomo_table) if not lomo_table.is_empty() else 'Not yet generated.\n'}
### Completion accounting

{completion_audit}

These aggregate rates make a large differential-completion explanation unlikely, but they do not establish identical completed-order composition across windows.

## Highest-cost UTC hours among headline universes

{markdown_table(hour_table)}

## Outputs

- `{rel(paths['daily_minute'])}`
- `{rel(paths['minute_surface'])}` and `{rel(paths['minute_surface_csv'])}`
- `{rel(paths['hour_surface'])}` and `{rel(paths['hour_surface_csv'])}`
- `{rel(paths['window_contrasts'])}` and `{rel(paths['window_contrasts_csv'])}`
- `{rel(paths['lomo_window_summary'])}` and `{rel(paths['lomo_window_summary_csv'])}`
- `{rel(paths['manifest'])}`
"""
    args.markdown_out.write_text(doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cost_panel = require_path(args.mart_root / "cost_panel_1m_long.parquet", "cost_panel_1m_long")
    members = require_path(args.mart_root / "pit_universe_daily_members.parquet", "pit_universe_daily_members")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    if args.memory_limit:
        con.execute(f"PRAGMA memory_limit={sql_str(args.memory_limit)}")
    tmp_dir = args.out_dir / "_duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory={sql_str(tmp_dir)}")

    paths = {
        "daily_wide": args.out_dir / "pit_adv30_exact_daily_minute_wide.parquet",
        "daily_minute": args.out_dir / "pit_adv30_exact_daily_minute.parquet",
        "minute_surface": args.out_dir / "pit_adv30_exact_minute_surface.parquet",
        "minute_surface_csv": args.out_dir / "pit_adv30_exact_minute_surface.csv",
        "hour_surface": args.out_dir / "pit_adv30_exact_hour_surface.parquet",
        "hour_surface_csv": args.out_dir / "pit_adv30_exact_hour_surface.csv",
        "window_contrasts": args.out_dir / "pit_adv30_exact_window_contrasts.parquet",
        "window_contrasts_csv": args.out_dir / "pit_adv30_exact_window_contrasts.csv",
        "window_contrast_daily": args.out_dir / "pit_adv30_exact_window_contrast_daily.parquet",
        "lomo_window_summary": args.out_dir / "pit_adv30_exact_lomo_window_validation.parquet",
        "lomo_window_summary_csv": args.out_dir / "pit_adv30_exact_lomo_window_validation.csv",
        "lomo_window_detail": args.out_dir / "pit_adv30_exact_lomo_window_detail.parquet",
        "manifest": args.out_dir / "manifest.json",
    }

    count_exprs: list[str] = []
    long_selects: list[str] = []
    for universe, threshold in UNIVERSE_THRESHOLDS:
        suffix = universe
        count_exprs.extend(
            [
                f"COUNT(*) FILTER (WHERE adv_rank <= {threshold}) AS observed_members_{suffix}",
                f"COUNT(cost_100000_bps) FILTER (WHERE adv_rank <= {threshold}) AS complete_members_{suffix}",
                f"AVG(cost_100000_bps) FILTER (WHERE adv_rank <= {threshold}) AS cost_100000_bps_{suffix}",
                f"AVG(spread_bps) FILTER (WHERE adv_rank <= {threshold} AND valid_bbo) AS spread_bps_{suffix}",
            ]
        )
        long_selects.append(
            f"""
            SELECT
                date,
                {sql_str(universe)} AS universe,
                {threshold} AS selected_symbols,
                minute_of_day,
                hour_utc,
                observed_members_{suffix} AS observed_members,
                complete_members_{suffix} AS complete_members,
                CAST(complete_members_{suffix} AS DOUBLE) / {threshold} AS completion_rate,
                cost_100000_bps_{suffix} AS cost_100000_bps,
                spread_bps_{suffix} AS spread_bps
            FROM read_parquet({sql_str(paths['daily_wide'])})
            """
        )

    daily_wide_query = f"""
        WITH members AS (
            SELECT date, symbol, adv_rank
            FROM read_parquet({sql_str(members)})
            WHERE policy = {sql_str(args.policy)}
              AND lookback_days = {args.lookback_days}
              AND volume_stat = {sql_str(args.volume_stat)}
              AND universe = 'top200'
              AND date >= DATE {sql_str(args.start_date)}
              AND date <= DATE {sql_str(args.end_date)}
        ),
        joined AS (
            SELECT
                c.date,
                c.minute_of_day,
                c.hour_utc,
                m.adv_rank,
                c.avg_100000_mid_cost_bps AS cost_100000_bps,
                c.spread_bps,
                c.valid_bbo
            FROM read_parquet({sql_str(cost_panel)}) AS c
            JOIN members AS m
              ON c.date = m.date AND c.symbol = m.symbol
            WHERE c.date >= DATE {sql_str(args.start_date)}
              AND c.date <= DATE {sql_str(args.end_date)}
        )
        SELECT
            date,
            minute_of_day,
            any_value(hour_utc) AS hour_utc,
            {', '.join(count_exprs)}
        FROM joined
        GROUP BY date, minute_of_day
        ORDER BY date, minute_of_day
    """
    run_copy(con, daily_wide_query, paths["daily_wide"], force=args.force)

    daily_long_query = "\nUNION ALL\n".join(long_selects) + "\nORDER BY date, universe, minute_of_day"
    run_copy(con, daily_long_query, paths["daily_minute"], force=args.force)

    surface_query = f"""
        SELECT
            universe,
            any_value(selected_symbols) AS selected_symbols,
            minute_of_day,
            any_value(hour_utc) AS hour_utc,
            COUNT(*) AS dates,
            AVG(observed_members) AS avg_observed_members,
            AVG(complete_members) AS avg_complete_members,
            AVG(completion_rate) AS avg_completion_rate,
            AVG(cost_100000_bps) AS mean_cost_100000_bps,
            median(cost_100000_bps) AS median_daily_minute_cost_100000_bps,
            quantile_cont(cost_100000_bps, 0.10) AS p10_daily_minute_cost_100000_bps,
            quantile_cont(cost_100000_bps, 0.90) AS p90_daily_minute_cost_100000_bps,
            stddev_samp(cost_100000_bps) AS sd_daily_minute_cost_100000_bps,
            AVG(spread_bps) AS mean_spread_bps
        FROM read_parquet({sql_str(paths['daily_minute'])})
        GROUP BY universe, minute_of_day
        ORDER BY CAST(replace(universe, 'top', '') AS INTEGER), minute_of_day
    """
    run_copy(con, surface_query, paths["minute_surface"], force=args.force)
    write_csv_from_parquet(con, paths["minute_surface"], paths["minute_surface_csv"])

    hour_query = f"""
        SELECT
            universe,
            any_value(selected_symbols) AS selected_symbols,
            hour_utc,
            COUNT(DISTINCT date) AS dates,
            COUNT(*) AS date_minutes,
            AVG(observed_members) AS avg_observed_members,
            AVG(complete_members) AS avg_complete_members,
            AVG(completion_rate) AS avg_completion_rate,
            AVG(cost_100000_bps) AS mean_cost_100000_bps,
            median(cost_100000_bps) AS median_daily_minute_cost_100000_bps,
            quantile_cont(cost_100000_bps, 0.10) AS p10_daily_minute_cost_100000_bps,
            quantile_cont(cost_100000_bps, 0.90) AS p90_daily_minute_cost_100000_bps,
            AVG(spread_bps) AS mean_spread_bps
        FROM read_parquet({sql_str(paths['daily_minute'])})
        GROUP BY universe, hour_utc
        ORDER BY CAST(replace(universe, 'top', '') AS INTEGER), hour_utc
    """
    run_copy(con, hour_query, paths["hour_surface"], force=args.force)
    write_csv_from_parquet(con, paths["hour_surface"], paths["hour_surface_csv"])

    build_contrast_tables(args.out_dir, paths)
    build_leave_one_month_window_validation(
        args.out_dir,
        paths,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )

    manifest = {
        "generatedUtc": datetime.now(UTC).isoformat(),
        "script": rel(Path(__file__)),
        "researchRoot": rel(args.research_root),
        "martRoot": rel(args.mart_root),
        "outDir": rel(args.out_dir),
        "sampleWindow": {"startDate": args.start_date, "endDate": args.end_date},
        "universeRule": {
            "policy": args.policy,
            "lookbackDays": args.lookback_days,
            "volumeStat": args.volume_stat,
            "source": rel(members),
            "rankUniverseJoined": "top200, with adv_rank thresholds used for top25/top50/top100/top150/top200",
        },
        "costInput": rel(cost_panel),
        "definition": "Exact PIT date-symbol-minute join: selected daily PIT members are joined to realized one-minute L2 cost rows by date and symbol, then grouped by date and minute before averaging across dates.",
        "minuteCoverage": {
            "observedPositions": 1439,
            "firstMinuteUtc": "00:01",
            "lastMinuteUtc": "23:59",
            "omittedMinuteUtc": "00:00",
            "omissionReason": "The replay cannot form a usable book before the first after-midnight L2 snapshot.",
        },
        "costMetric": "buy/sell averaged $100k displayed-book mid-reference cost in bps; null if either side is incomplete",
        "lomoInference": {
            "method": "calendar-month block bootstrap with percentile 95% confidence interval",
            "reps": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "resamplingUnit": "held-out calendar month; all dates within a sampled month remain grouped",
        },
        "outputs": {key: parquet_meta(path) for key, path in paths.items() if path.suffix == ".parquet" and path.exists()},
        "csvOutputs": {key: rel(path) for key, path in paths.items() if path.suffix == ".csv" and path.exists()},
        "environment": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "polars": pl.__version__,
            "threads": args.threads,
            "memoryLimit": args.memory_limit,
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args, paths, manifest)
    print(json.dumps({"event": "manifest_done", "path": rel(paths["manifest"]), "doc": rel(args.markdown_out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
