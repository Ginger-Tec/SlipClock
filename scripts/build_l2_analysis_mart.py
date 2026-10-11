#!/usr/bin/env python3
"""Build compact analysis-mart parquet files for the current L2-only paper run."""
from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import duckdb
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "tardis" / "features" / "l2_1m_usdm_2024_2026_v1"
DEFAULT_RESEARCH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tardis"
    / "research"
    / "released_l2_analysis"
    / "baseline"
)
DEFAULT_OUT_DIR = DEFAULT_RESEARCH_ROOT / "analysis_mart"
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2026-04-30"
DEFAULT_EXCHANGE = "binance-futures"
DEFAULT_GRID = "1m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    p.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--exchange", default=DEFAULT_EXCHANGE)
    p.add_argument("--grid", default=DEFAULT_GRID)
    p.add_argument("--threads", type=int, default=max(1, min(32, (os.cpu_count() or 8) // 2)))
    p.add_argument("--memory-limit", default=None, help="DuckDB memory limit, for example 64GB")
    p.add_argument("--force", action="store_true", help="replace existing mart files")
    return p.parse_args()


SQL_DATE = "DATE '{}'"


def sql_str(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_copy(con: duckdb.DuckDBPyConnection, query: str, path: Path, *, force: bool) -> None:
    ensure_parent(path)
    if path.exists() and not force:
        print(json.dumps({"event": "skip_existing", "path": str(path)}, ensure_ascii=False), flush=True)
        return
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(json.dumps({"event": "write_start", "path": str(path)}, ensure_ascii=False), flush=True)
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
    print(json.dumps({"event": "write_done", "path": str(path)}, ensure_ascii=False), flush=True)


def parquet_meta(path: Path) -> dict[str, object]:
    meta = pq.ParquetFile(path).metadata
    schema = pq.read_schema(path)
    return {
        "path": str(path),
        "rows": int(meta.num_rows),
        "columns": int(len(schema.names)),
        "sizeBytes": int(path.stat().st_size),
        "schema": {name: str(schema.field(name).type) for name in schema.names},
    }


def existing_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def build_source_view(args: argparse.Namespace) -> str:
    pattern = (
        args.feature_root
        / f"exchange={args.exchange}"
        / "symbol=*"
        / "date=*"
        / f"grid={args.grid}"
        / "slippage.parquet"
    )
    return f"""
        SELECT *
        FROM read_parquet({sql_str(pattern)}, hive_partitioning = true)
        WHERE date >= DATE {sql_str(args.start_date)}
          AND date <= DATE {sql_str(args.end_date)}
          AND (ends_with(symbol, 'USDT') OR ends_with(symbol, 'USDC'))
          AND symbol <> 'FUTURES'
    """


def safe_cost_expr(column: str, complete_column: str) -> str:
    return f"CASE WHEN {complete_column} AND isfinite({column}) THEN CAST({column} AS REAL) ELSE NULL END"


def common_time_columns() -> str:
    minute_expr = "CAST(((sample_local_timestamp_us // 60000000) % 1440) AS SMALLINT)"
    return f"""
        make_timestamp(sample_local_timestamp_us) AS timestamp_utc,
        sample_local_timestamp_us,
        date,
        CAST({minute_expr} // 60 AS TINYINT) AS hour_utc,
        {minute_expr} AS minute_of_day,
        symbol
    """


def build_wide_query(panel_path: Path, calendar_path: Path, value_expr: str, value_alias: str) -> str:
    return f"""
        WITH long_values AS (
            SELECT timestamp_utc, symbol, {value_expr} AS {value_alias}
            FROM read_parquet({sql_str(panel_path)})
        ),
        wide_values AS (
            PIVOT long_values
            ON symbol
            USING first({value_alias})
            GROUP BY timestamp_utc
        )
        SELECT cal.timestamp_utc, wide_values.* EXCLUDE (timestamp_utc)
        FROM read_parquet({sql_str(calendar_path)}) AS cal
        LEFT JOIN wide_values USING (timestamp_utc)
        ORDER BY cal.timestamp_utc
    """


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    if args.memory_limit:
        con.execute(f"PRAGMA memory_limit={sql_str(args.memory_limit)}")
    tmp_dir = args.out_dir / "_duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory={sql_str(tmp_dir)}")

    source = build_source_view(args)
    out = args.out_dir
    prelim_date_summary = existing_file(args.research_root / "prelim_l2" / "date_summary.parquet", "date_summary")
    symbol_rank = existing_file(args.research_root / "portfolio_execution" / "symbol_liquidity_rank.parquet", "symbol_liquidity_rank")
    pit_members = existing_file(args.research_root / "pit_adv_universe" / "pit_universe_members.parquet", "pit_universe_members")
    pit_daily_costs = existing_file(args.research_root / "pit_adv_universe" / "pit_daily_costs.parquet", "pit_daily_costs")

    paths = {
        "symbol_master": out / "symbol_master.parquet",
        "minute_calendar": out / "minute_calendar.parquet",
        "cost_panel_1m_long": out / "cost_panel_1m_long.parquet",
        "feature_panel_1m_long": out / "feature_panel_1m_long.parquet",
        "cost_100k_mid_bps_wide": out / "cost_100k_mid_bps_wide.parquet",
        "cost_100k_complete_wide": out / "cost_100k_complete_wide.parquet",
        "spread_bps_wide": out / "spread_bps_wide.parquet",
        "mid_price_wide": out / "mid_price_wide.parquet",
        "bbo_imbalance_wide": out / "bbo_imbalance_wide.parquet",
        "top_bbo_amount_wide": out / "top_bbo_amount_wide.parquet",
        "pit_universe_daily_members": out / "pit_universe_daily_members.parquet",
        "pit_universe_cost_surface": out / "pit_universe_cost_surface.parquet",
        "pit_membership_rank_wide": out / "pit_membership_rank_wide.parquet",
    }

    run_copy(
        con,
        f"""
        WITH base AS (
            SELECT
                symbol,
                min(date) AS first_date,
                max(date) AS last_date,
                min(make_timestamp(sample_local_timestamp_us)) AS first_timestamp_utc,
                max(make_timestamp(sample_local_timestamp_us)) AS last_timestamp_utc,
                count(*) AS rows,
                count(DISTINCT date) AS dates_seen,
                avg(CASE WHEN NOT invalid_book_flag AND isfinite(spread_bps) THEN spread_bps ELSE NULL END) AS spread_mean_bps,
                avg(({safe_cost_expr('buy_100000_mid_cost_bps', 'buy_100000_mid_complete')} + {safe_cost_expr('sell_100000_mid_cost_bps', 'sell_100000_mid_complete')}) / 2.0) AS avg_100000_mid_cost_mean_bps
            FROM ({source})
            GROUP BY symbol
        ),
        ranks AS (
            SELECT * FROM read_parquet({sql_str(symbol_rank)})
        )
        SELECT
            base.symbol,
            ranks.liquidity_rank,
            base.first_date,
            base.last_date,
            base.first_timestamp_utc,
            base.last_timestamp_utc,
            CAST(base.rows AS UBIGINT) AS rows,
            CAST(base.dates_seen AS UINTEGER) AS dates_seen,
            CAST(base.last_date = DATE {sql_str(args.end_date)} AS BOOLEAN) AS seen_on_end_date,
            CAST(base.last_date < DATE {sql_str(args.end_date)} AS BOOLEAN) AS inactive_by_end_date,
            CAST(base.spread_mean_bps AS REAL) AS spread_mean_bps,
            CAST(base.avg_100000_mid_cost_mean_bps AS REAL) AS avg_100000_mid_cost_mean_bps,
            ranks.buy_100000_mid_complete_rate,
            ranks.avg_100000_mid_p90_bps
        FROM base
        LEFT JOIN ranks USING (symbol)
        ORDER BY base.symbol
        """,
        paths["symbol_master"],
        force=args.force,
    )

    run_copy(
        con,
        f"""
        WITH observed_dates AS (
            SELECT DISTINCT date
            FROM read_parquet({sql_str(prelim_date_summary)})
            WHERE date >= DATE {sql_str(args.start_date)} AND date <= DATE {sql_str(args.end_date)}
        ),
        minute_numbers AS (
            SELECT CAST(range AS SMALLINT) AS minute_of_day
            FROM range(0, 1440)
        ),
        cal AS (
            SELECT
                CAST(observed_dates.date AS TIMESTAMP) + minute_numbers.minute_of_day * INTERVAL 1 MINUTE AS timestamp_utc,
                epoch_us(CAST(observed_dates.date AS TIMESTAMP) + minute_numbers.minute_of_day * INTERVAL 1 MINUTE) AS sample_local_timestamp_us,
                observed_dates.date,
                CAST(minute_numbers.minute_of_day // 60 AS TINYINT) AS hour_utc,
                CAST(minute_numbers.minute_of_day % 60 AS TINYINT) AS minute_utc,
                minute_numbers.minute_of_day,
                CAST(strftime(observed_dates.date, '%w') AS TINYINT) AS day_of_week_utc,
                CAST(strftime(observed_dates.date, '%w') IN ('0', '6') AS BOOLEAN) AS is_weekend_utc,
                CAST(least(minute_numbers.minute_of_day, 1440 - minute_numbers.minute_of_day) AS SMALLINT) AS utc0_abs_min,
                CAST(least(minute_numbers.minute_of_day, 1440 - minute_numbers.minute_of_day) <= 120 AS BOOLEAN) AS is_utc0_2h,
                CASE
                    WHEN least(minute_numbers.minute_of_day, 1440 - minute_numbers.minute_of_day) <= 15 THEN 'utc0_00_15m'
                    WHEN least(minute_numbers.minute_of_day, 1440 - minute_numbers.minute_of_day) <= 60 THEN 'utc0_15_60m'
                    WHEN least(minute_numbers.minute_of_day, 1440 - minute_numbers.minute_of_day) <= 120 THEN 'utc0_60_120m'
                    ELSE 'rest'
                END AS utc0_bucket
            FROM observed_dates
            CROSS JOIN minute_numbers
        )
        SELECT * FROM cal ORDER BY timestamp_utc
        """,
        paths["minute_calendar"],
        force=args.force,
    )

    buy_safe = safe_cost_expr("buy_100000_mid_cost_bps", "buy_100000_mid_complete")
    sell_safe = safe_cost_expr("sell_100000_mid_cost_bps", "sell_100000_mid_complete")
    run_copy(
        con,
        f"""
        WITH base AS (
            SELECT
                {common_time_columns()},
                {buy_safe} AS buy_100000_mid_cost_bps,
                {sell_safe} AS sell_100000_mid_cost_bps,
                buy_100000_mid_complete,
                sell_100000_mid_complete,
                CAST(buy_100000_mid_levels_used AS UINTEGER) AS buy_100000_mid_levels_used,
                CAST(sell_100000_mid_levels_used AS UINTEGER) AS sell_100000_mid_levels_used,
                CAST(spread_bps AS REAL) AS spread_bps,
                CAST(mid_price AS DOUBLE) AS mid_price,
                CAST((NOT invalid_book_flag) AND isfinite(spread_bps) AS BOOLEAN) AS valid_bbo
            FROM ({source})
        )
        SELECT
            timestamp_utc,
            sample_local_timestamp_us,
            date,
            hour_utc,
            minute_of_day,
            symbol,
            buy_100000_mid_cost_bps,
            sell_100000_mid_cost_bps,
            CAST((buy_100000_mid_cost_bps + sell_100000_mid_cost_bps) / 2.0 AS REAL) AS avg_100000_mid_cost_bps,
            buy_100000_mid_complete,
            sell_100000_mid_complete,
            CAST(buy_100000_mid_complete AND sell_100000_mid_complete AS BOOLEAN) AS complete_100000_mid,
            buy_100000_mid_levels_used,
            sell_100000_mid_levels_used,
            spread_bps,
            mid_price,
            valid_bbo
        FROM base
        """,
        paths["cost_panel_1m_long"],
        force=args.force,
    )

    run_copy(
        con,
        f"""
        SELECT
            {common_time_columns()},
            CAST(bid_price AS DOUBLE) AS bid_price,
            CAST(bid_amount AS REAL) AS bid_amount,
            CAST(ask_price AS DOUBLE) AS ask_price,
            CAST(ask_amount AS REAL) AS ask_amount,
            CAST(mid_price AS DOUBLE) AS mid_price,
            CAST(micro_price AS DOUBLE) AS micro_price,
            CAST(spread_bps AS REAL) AS spread_bps,
            CAST(bbo_imbalance AS REAL) AS bbo_imbalance,
            CAST(book_bid_levels AS UINTEGER) AS book_bid_levels,
            CAST(book_ask_levels AS UINTEGER) AS book_ask_levels,
            invalid_book_flag,
            CAST((NOT invalid_book_flag) AND isfinite(spread_bps) AS BOOLEAN) AS valid_bbo,
            CAST(bid_amount + ask_amount AS REAL) AS top_bbo_amount,
            CAST(bid_amount * bid_price + ask_amount * ask_price AS DOUBLE) AS top_bbo_notional_usd,
            CAST(source_file_count AS UINTEGER) AS source_file_count
        FROM ({source})
        """,
        paths["feature_panel_1m_long"],
        force=args.force,
    )

    wide_specs: Iterable[tuple[str, Path, str, str]] = [
        ("cost_100k_mid_bps_wide", paths["cost_panel_1m_long"], "avg_100000_mid_cost_bps", "metric_value"),
        ("cost_100k_complete_wide", paths["cost_panel_1m_long"], "complete_100000_mid", "metric_value"),
        ("spread_bps_wide", paths["feature_panel_1m_long"], "spread_bps", "metric_value"),
        ("mid_price_wide", paths["feature_panel_1m_long"], "mid_price", "metric_value"),
        ("bbo_imbalance_wide", paths["feature_panel_1m_long"], "bbo_imbalance", "metric_value"),
        ("top_bbo_amount_wide", paths["feature_panel_1m_long"], "top_bbo_amount", "metric_value"),
    ]
    for name, panel, value_expr, value_alias in wide_specs:
        run_copy(
            con,
            build_wide_query(panel, paths["minute_calendar"], value_expr, value_alias),
            paths[name],
            force=args.force,
        )

    run_copy(
        con,
        f"""
        SELECT
            date,
            selection_date,
            policy,
            lookback_days,
            volume_stat,
            universe,
            selected_symbols,
            symbol,
            adv_rank,
            rank_value,
            lookback_obs,
            min_lookback_obs,
            lookback_mean_quote_volume,
            lookback_median_quote_volume,
            lookback_mean_trade_count,
            CAST(adv_rank <= 25 AS BOOLEAN) AS in_top25,
            CAST(adv_rank <= 50 AS BOOLEAN) AS in_top50,
            CAST(adv_rank <= 100 AS BOOLEAN) AS in_top100,
            CAST(adv_rank <= 150 AS BOOLEAN) AS in_top150,
            CAST(adv_rank <= 200 AS BOOLEAN) AS in_top200
        FROM read_parquet({sql_str(pit_members)})
        WHERE date >= DATE {sql_str(args.start_date)} AND date <= DATE {sql_str(args.end_date)}
        ORDER BY date, policy, lookback_days, volume_stat, universe, adv_rank, symbol
        """,
        paths["pit_universe_daily_members"],
        force=args.force,
    )

    run_copy(
        con,
        f"""
        SELECT
            date,
            policy,
            lookback_days,
            volume_stat,
            universe,
            selected_symbols,
            member_rows,
            evaluated_symbols,
            evaluation_coverage,
            mean_rank_quote_volume,
            min_rank_quote_volume,
            min_lookback_obs_seen,
            spread_mean_bps,
            spread_p90_mean_bps,
            buy_100000_complete_min,
            equal_weight_mean_cost_100000_bps,
            symbol_mean_cost_p90_100000_bps,
            mean_of_symbol_p90_cost_100000_bps,
            expected_cost_100000_usd
        FROM read_parquet({sql_str(pit_daily_costs)})
        WHERE date >= DATE {sql_str(args.start_date)} AND date <= DATE {sql_str(args.end_date)}
        ORDER BY date, policy, lookback_days, volume_stat, universe
        """,
        paths["pit_universe_cost_surface"],
        force=args.force,
    )

    run_copy(
        con,
        f"""
        WITH rank_values AS (
            SELECT date, symbol, adv_rank
            FROM read_parquet({sql_str(paths['pit_universe_daily_members'])})
            WHERE policy = 'daily'
              AND lookback_days = 30
              AND volume_stat = 'mean'
              AND universe = 'top200'
        ),
        wide_values AS (
            PIVOT rank_values
            ON symbol
            USING min(adv_rank)
            GROUP BY date
        )
        SELECT * FROM wide_values ORDER BY date
        """,
        paths["pit_membership_rank_wide"],
        force=args.force,
    )

    outputs_meta = {name: parquet_meta(path) for name, path in paths.items() if path.exists()}
    manifest = {
        "generatedUtc": datetime.now(UTC).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "featureRoot": str(args.feature_root),
        "researchRoot": str(args.research_root),
        "outDir": str(args.out_dir),
        "exchange": args.exchange,
        "grid": args.grid,
        "startDate": args.start_date,
        "endDate": args.end_date,
        "filters": {
            "stableQuoteOnly": "symbol이 USDT 또는 USDC로 끝나는 행만 사용",
            "excludedSymbols": ["FUTURES"],
            "dateCalendar": "prelim_l2/date_summary.parquet의 관측 날짜에 UTC 1440분을 교차 결합",
        },
        "definitions": {
            "avg_100000_mid_cost_bps": "safe buy_100000_mid_cost_bps와 safe sell_100000_mid_cost_bps의 평균; safe는 complete flag가 true이고 값이 finite인 경우만 의미",
            "complete_100000_mid": "buy_100000_mid_complete AND sell_100000_mid_complete",
            "wideMissingValueRule": "symbol-minute 관측치가 없으면 null로 남기며 0으로 채우지 않음",
            "top_bbo_amount": "최우선 bid_amount + 최우선 ask_amount",
            "pit_membership_rank_wide": "daily, 30일 lookback, mean ADV 기준 top200 PIT rank; null이면 비구성종목 또는 관측 없음",
        },
        "environment": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "threads": args.threads,
            "memoryLimit": args.memory_limit,
        },
        "outputs": outputs_meta,
    }
    manifest_path = out / "analysis_mart_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "manifest_done", "path": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
