#!/usr/bin/env python3
"""Run the predefined frozen-order strategy cost experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from experiment_core import (
    MLP_FEATURES,
    STRATEGIES,
    TREATMENTS,
    ExperimentConfig,
    add_walk_forward_mlp_scores,
    apply_cost_treatments,
    augment_with_carried_position_days,
    assert_order_invariants,
    assert_treatment_support,
    attach_execution_state,
    build_lagged_feature_panel,
    equation17_residuals,
    generate_frozen_orders,
    pairwise_reversal_matrix,
    retain_common_valid_dates,
    summarize_metrics,
    verify_equation17_pairwise,
)


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parents[1]
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))

from paper_figure_style import (  # noqa: E402
    INK,
    KOREAN_FONT_FAMILY,
    PDF_METADATA,
    STRATEGY_COLORS,
    TEXT_WIDTH_IN,
    apply_paper_style,
)


apply_paper_style()

REPO = HERE.parents[3]
MART = REPO / "data/tardis/research/released_l2_analysis/baseline/analysis_mart"
RAW_CLOCK = MART / "cost_panel_1m_long.parquet"
RAW_MEMBERS = MART / "pit_universe_daily_members.parquet"
RAW_SIGNAL_ROOT = REPO / "data/binance_vision/parquet/market=futures_um/data_type=klines/interval=1m"
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
TABLES_DIR = HERE / "tables"
LOGS_DIR = HERE / "logs"
CLOCK_COMPACT = DATA_DIR / "clock_panel_1200_all.parquet"
MEMBERS_COMPACT = DATA_DIR / "pit_adv30_mean_top100.parquet"
SIGNAL_COMPACT = DATA_DIR / "signal_prices_1159.parquet"
SIGNAL_INVENTORY = DATA_DIR / "signal_source_inventory.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true", help="reuse compact inputs")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_compact_inputs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=8")
    log("extracting the 12:00 UTC clock panel from the full analysis mart")
    con.execute(
        f"""
        COPY (
            SELECT timestamp_utc, date, symbol, mid_price, spread_bps,
                   buy_100000_mid_cost_bps, sell_100000_mid_cost_bps,
                   buy_100000_mid_complete, sell_100000_mid_complete, valid_bbo
            FROM read_parquet('{RAW_CLOCK.as_posix()}')
            WHERE minute_of_day = 720
            ORDER BY date, symbol
        ) TO '{CLOCK_COMPACT.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    log("extracting daily PIT ADV30-mean top-100 membership")
    con.execute(
        f"""
        COPY (
            SELECT date, selection_date, symbol, adv_rank, rank_value,
                   lookback_obs, min_lookback_obs
            FROM read_parquet('{RAW_MEMBERS.as_posix()}')
            WHERE policy = 'daily'
              AND lookback_days = 30
              AND volume_stat = 'mean'
              AND universe = 'top100'
            ORDER BY date, adv_rank, symbol
        ) TO '{MEMBERS_COMPACT.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    con.close()
    extract_signal_prices()


def extract_signal_prices() -> None:
    """Extract one pre-order 11:59 close per symbol-calendar day.

    Reading two Parquet columns directly avoids loading the full 30 GB kline
    data set.  The inventory records every monthly source file consumed.
    """

    members = pl.read_parquet(MEMBERS_COMPACT, columns=["symbol"])
    symbols = sorted(members.get_column("symbol").unique().to_list())
    min_day = np.datetime64("2023-10-01")
    max_day = np.datetime64("2026-04-30")
    target_millisecond = (11 * 60 + 59) * 60 * 1000
    day_ms = 86_400_000
    day_chunks: list[np.ndarray] = []
    symbol_chunks: list[np.ndarray] = []
    price_chunks: list[np.ndarray] = []
    inventory: list[dict[str, object]] = []

    log(f"extracting 11:59 UTC signal closes for {len(symbols)} PIT symbols")
    for symbol_idx, symbol in enumerate(symbols, start=1):
        symbol_root = RAW_SIGNAL_ROOT / f"symbol={symbol}"
        for path in sorted(symbol_root.glob("year=*/month=*/part-*.parquet")):
            try:
                year = int(path.parent.parent.name.split("=", 1)[1])
                month = int(path.parent.name.split("=", 1)[1])
            except (IndexError, ValueError):
                continue
            month_day = np.datetime64(f"{year:04d}-{month:02d}-01")
            if month_day > max_day or month_day + np.timedelta64(31, "D") < min_day:
                continue
            table = pq.ParquetFile(path).read(columns=["open_time", "close"])
            open_time = table.column("open_time").combine_chunks().to_numpy(zero_copy_only=False)
            close = table.column("close").combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
            mask = (open_time % day_ms == target_millisecond) & np.isfinite(close) & (close > 0)
            if not mask.any():
                continue
            days = (open_time[mask] // day_ms).astype(np.int32)
            day_values = days.astype("datetime64[D]")
            range_mask = (day_values >= min_day) & (day_values <= max_day)
            if not range_mask.any():
                continue
            days = days[range_mask]
            prices = close[mask][range_mask]
            day_chunks.append(days)
            symbol_chunks.append(np.repeat(symbol, len(days)))
            price_chunks.append(prices)
            inventory.append(
                {
                    "path": str(path.relative_to(REPO)),
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "selected_rows": int(len(days)),
                }
            )
        if symbol_idx % 100 == 0:
            log(f"signal extraction progress: {symbol_idx}/{len(symbols)} symbols")

    if not day_chunks:
        raise RuntimeError("no Binance Vision signal prices were extracted")
    signal = (
        pl.DataFrame(
            {
                "date_day": np.concatenate(day_chunks),
                "symbol": np.concatenate(symbol_chunks),
                "signal_price": np.concatenate(price_chunks),
            }
        )
        .with_columns(pl.col("date_day").cast(pl.Date).alias("date"))
        .drop("date_day")
        .unique(["date", "symbol"], keep="last")
        .sort(["date", "symbol"])
    )
    signal.write_parquet(SIGNAL_COMPACT, compression="zstd")
    write_csv(pl.DataFrame(inventory).sort("path"), SIGNAL_INVENTORY)
    log(f"extracted {signal.height} symbol-days of pre-order signal prices")


def write_csv(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path, float_precision=12, datetime_format="%Y-%m-%dT%H:%M:%S%.6f", date_format="%Y-%m-%d")


def make_cumulative_figure(daily: pl.DataFrame, output: Path, *, language: str) -> None:
    colors = STRATEGY_COLORS
    if language == "en":
        titles = {
            "no_cost": "No cost",
            "fixed_5bp": "5 bp",
            "fixed_10bp": "10 bp",
            "fixed_20bp": "20 bp",
            "half_spread": "Half-spread",
            "slipclock": "SlipClock",
        }
        labels = {
            "ts_momentum": "TS momentum",
            "cs_momentum": "CS momentum",
            "mean_reversion": "Mean reversion",
            "vol_scaled": "Vol.-normalized",
            "mlp": "MLP",
        }
        y_label = "Cumulative entry-cost-adjusted markout (USD m)"
        x_label = "Order date (UTC)"
        font_family = "Linux Biolinum O"
    elif language == "ko":
        titles = {
            "no_cost": "무비용",
            "fixed_5bp": "5 bp",
            "fixed_10bp": "10 bp",
            "fixed_20bp": "20 bp",
            "half_spread": "반스프레드",
            "slipclock": "SlipClock",
        }
        labels = {
            "ts_momentum": "시계열 모멘텀",
            "cs_momentum": "횡단면 모멘텀",
            "mean_reversion": "평균회귀",
            "vol_scaled": "변동성 정규화",
            "mlp": "MLP",
        }
        y_label = "누적 진입비용 조정 1일 중간가 평가손익 (백만 USD)"
        x_label = "주문일 (UTC)"
        font_family = ["Linux Biolinum O", KOREAN_FONT_FAMILY]
    else:
        raise ValueError(f"unsupported figure language: {language}")

    with plt.rc_context({"font.family": font_family, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH_IN, 4.10), sharex=True)
        first_date = daily.get_column("date").min()
        last_date = daily.get_column("date").max()
        calendar_dates = [
            first_date + timedelta(days=offset)
            for offset in range((last_date - first_date).days + 1)
        ]
        for axis, treatment in zip(axes.flat, TREATMENTS, strict=True):
            for strategy in STRATEGIES:
                block = daily.filter(
                    (pl.col("treatment") == treatment) & (pl.col("strategy") == strategy)
                ).sort("date")
                cumulative_by_date = dict(
                    zip(
                        block.get_column("date").to_list(),
                        block.get_column("cumulative_net_pnl_usd").to_list(),
                        strict=True,
                    )
                )
                # A date without a newly evaluated order vector contributes no
                # new cumulative P&L. Carry the last cumulative balance forward
                # so the figure remains a continuous wealth path without
                # inventing an interpolated return.
                carried_value = np.nan
                plot_values_list: list[float] = []
                for value in calendar_dates:
                    if value in cumulative_by_date:
                        carried_value = float(cumulative_by_date[value])
                    plot_values_list.append(carried_value)
                plot_values = np.asarray(plot_values_list, dtype=np.float64)
                axis.plot(
                    calendar_dates,
                    plot_values / 1_000_000.0,
                    label=labels[strategy],
                    color=colors[strategy],
                    linewidth=1.0,
                )
            panel = chr(ord("a") + list(TREATMENTS).index(treatment))
            axis.axhline(0.0, color=INK, linewidth=0.55)
            axis.set_title(f"({panel}) {titles[treatment]}", loc="left", pad=3)
            axis.grid(axis="y")
            locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.supylabel(y_label, fontsize=8.2, x=0.018)
        fig.supxlabel(x_label, fontsize=8.2, y=0.080)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=5,
            frameon=False,
        )
        fig.subplots_adjust(left=0.095, right=0.985, top=0.965, bottom=0.205, hspace=0.30, wspace=0.22)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, metadata=PDF_METADATA)
        fig.savefig(output.with_suffix(".png"), dpi=300)
        plt.close(fig)


def make_metric_table(metrics: pl.DataFrame, output: Path) -> None:
    names = {
        "ts_momentum": "TS momentum",
        "cs_momentum": "CS momentum",
        "mean_reversion": "Mean reversion",
        "vol_scaled": "Vol.-normalized",
        "mlp": "MLP",
    }
    treatment_names = {
        "no_cost": "No cost",
        "fixed_5bp": "5 bp",
        "fixed_10bp": "10 bp",
        "fixed_20bp": "20 bp",
        "half_spread": "Half-spread",
        "slipclock": "SlipClock",
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Cost rule & Strategy & Terminal markout (\$m) & Net Sharpe & Rank \\",
        r"\midrule",
    ]
    for t_idx, treatment in enumerate(TREATMENTS):
        block = metrics.filter(pl.col("treatment") == treatment).sort("sharpe_rank")
        for idx, row in enumerate(block.iter_rows(named=True)):
            treatment_label = treatment_names[treatment] if idx == 0 else ""
            lines.append(
                f"{treatment_label} & {names[row['strategy']]} & "
                f"{row['cumulative_net_pnl_usd'] / 1_000_000.0:.3f} & "
                f"{row['net_sharpe']:.3f} & {row['sharpe_rank']} \\\\"
            )
        if t_idx != len(TREATMENTS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reversal_count(reversals: pl.DataFrame, metric: str, treatment: str) -> int:
    cell = reversals.filter(
        (pl.col("metric") == metric)
        & (pl.col("treatment_a") == "no_cost")
        & (pl.col("treatment_b") == treatment)
    )
    if cell.height != 1:
        raise AssertionError(f"missing reversal cell for {metric}: no_cost -> {treatment}")
    return int(cell.get_column("rank_reversals")[0])


def write_manuscript_assets(
    metrics: pl.DataFrame,
    reversals: pl.DataFrame,
    residual_summary: pl.DataFrame,
    order_invariants: dict[str, int | float],
    sample_audit: dict[str, int | float],
    cfg: ExperimentConfig,
) -> dict[str, object]:
    """Serialize the authoritative manuscript numbers and bilingual table rows."""

    strategy_names_en = {
        "ts_momentum": "Time-series momentum",
        "cs_momentum": "Cross-sectional momentum",
        "mean_reversion": "Mean reversion",
        "vol_scaled": "Volatility-normalized momentum",
        "mlp": "MLP",
    }
    strategy_names_ko = {
        "ts_momentum": "시계열 모멘텀",
        "cs_momentum": "횡단면 모멘텀",
        "mean_reversion": "평균회귀",
        "vol_scaled": "변동성 정규화 모멘텀",
        "mlp": "MLP",
    }
    strategies: dict[str, object] = {}
    for strategy in STRATEGIES:
        strategy_metrics: dict[str, object] = {}
        for treatment in TREATMENTS:
            cell = metrics.filter(
                (pl.col("strategy") == strategy) & (pl.col("treatment") == treatment)
            )
            if cell.height != 1:
                raise AssertionError(f"missing metric cell for {strategy}/{treatment}")
            row = cell.row(0, named=True)
            strategy_metrics[treatment] = {
                "cumulative_net_pnl_usd": float(row["cumulative_net_pnl_usd"]),
                "terminal_entry_cost_adjusted_markout_usd": float(
                    row["cumulative_net_pnl_usd"]
                ),
                "net_sharpe": float(row["net_sharpe"]),
                "sharpe_rank": int(row["sharpe_rank"]),
                "pnl_rank": int(row["pnl_rank"]),
            }
        slip = metrics.filter(
            (pl.col("strategy") == strategy) & (pl.col("treatment") == "slipclock")
        ).row(0, named=True)
        residuals_by_rate: dict[str, float] = {}
        for rate in (5, 10, 20):
            residual = residual_summary.filter(
                (pl.col("strategy") == strategy) & (pl.col("lambda_bps") == float(rate))
            )
            if residual.height != 1:
                raise AssertionError(f"missing {rate} bp residual for {strategy}")
            residuals_by_rate[f"fixed_{rate}bp"] = float(residual["mean_residual_bps"][0])
        strategies[strategy] = {
            "label_en": strategy_names_en[strategy],
            "label_ko": strategy_names_ko[strategy],
            "slipclock_mean_cost_bps": float(
                slip["cumulative_cost_usd"]
                / float(slip["cumulative_submitted_gross_usd"])
                * 10_000.0
            ),
            "mean_residual_bps_by_constant_rate": residuals_by_rate,
            "mean_residual_10bp_bps": residuals_by_rate["fixed_10bp"],
            "treatments": strategy_metrics,
        }

    comparisons: dict[str, object] = {}
    for treatment in TREATMENTS[1:]:
        comparisons[f"no_cost_vs_{treatment}"] = {
            "strategy_pairs": 10,
            "net_sharpe_reversals": _reversal_count(reversals, "net_sharpe", treatment),
            "cumulative_net_pnl_reversals": _reversal_count(
                reversals, "cumulative_net_pnl_usd", treatment
            ),
            "terminal_markout_reversals": _reversal_count(
                reversals, "cumulative_net_pnl_usd", treatment
            ),
        }

    claims: dict[str, object] = {
        "experiment": "frozen_order_strategy_cost_v1_20260801",
        "sample": {
            "retained_dates": int(order_invariants["retained_dates"]),
            "frozen_orders": int(order_invariants["orders"]),
            "orders_per_strategy_day": int(order_invariants["orders_per_strategy_day"]),
            "submitted_gross_usd": float(order_invariants["submitted_gross_usd"]),
            **sample_audit,
        },
        "comparisons": comparisons,
        "strategies": strategies,
    }
    (RESULTS_DIR / "manuscript_claims.json").write_text(
        json.dumps(claims, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def table_lines(
        labels: dict[str, str],
        markout_label: str,
        sharpe_label: str,
        sharpe_panel_label: str,
        markout_panel_label: str,
    ) -> list[str]:
        lines = [f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{sharpe_panel_label}}}}} \\\\"]
        for strategy in STRATEGIES:
            item = strategies[strategy]
            treatments = item["treatments"]
            cells = [
                f"{treatments[treatment]['net_sharpe']:.3f} "
                f"({treatments[treatment]['sharpe_rank']})"
                for treatment in TREATMENTS
            ]
            lines.append(
                f"{labels[strategy]} & "
                + " & ".join(cells)
                + f" & {item['slipclock_mean_cost_bps']:.3f} \\\\"
            )
        lines.extend(
            [
                r"\addlinespace[2pt]",
                f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{markout_panel_label}}}}} \\\\",
            ]
        )
        for strategy in STRATEGIES:
            item = strategies[strategy]
            treatments = item["treatments"]
            cells = [
                f"{treatments[treatment]['terminal_entry_cost_adjusted_markout_usd'] / 1_000_000.0:.3f} "
                f"({treatments[treatment]['pnl_rank']})"
                for treatment in TREATMENTS
            ]
            lines.append(f"{labels[strategy]} & " + " & ".join(cells) + r" & -- \\")
        lines.append(r"\midrule")
        sharpe_counts = ["--"] + [
            str(comparisons[f"no_cost_vs_{treatment}"]["net_sharpe_reversals"])
            for treatment in TREATMENTS[1:]
        ]
        profit_counts = ["--"] + [
            str(comparisons[f"no_cost_vs_{treatment}"]["cumulative_net_pnl_reversals"])
            for treatment in TREATMENTS[1:]
        ]
        lines.append(f"{sharpe_label} & " + " & ".join(sharpe_counts) + r" & -- \\")
        lines.append(f"{markout_label} & " + " & ".join(profit_counts) + r" & -- \\")
        return lines

    en_lines = table_lines(
        strategy_names_en,
        "Terminal markout",
        "Net Sharpe",
        "Panel A: Net Sharpe (rank)",
        "Panel B: Terminal entry-cost-adjusted markout, USD million (rank)",
    )
    ko_lines = table_lines(
        strategy_names_ko,
        "최종 평가손익",
        "순 Sharpe",
        "패널 A: 순 Sharpe (순위)",
        "패널 B: 최종 진입비용 조정 평가손익, 백만 USD (순위)",
    )
    (TABLES_DIR / "strategy_cost_audit_en.tex").write_text(
        "\n".join(en_lines) + "\n", encoding="utf-8"
    )
    (TABLES_DIR / "strategy_cost_audit_ko.tex").write_text(
        "\n".join(ko_lines) + "\n", encoding="utf-8"
    )

    def residual_lines(labels: dict[str, str]) -> list[str]:
        lines = []
        for strategy in STRATEGIES:
            residuals = strategies[strategy]["mean_residual_bps_by_constant_rate"]
            lines.append(
                f"{labels[strategy]} & {residuals['fixed_5bp']:.3f} & "
                f"{residuals['fixed_10bp']:.3f} & {residuals['fixed_20bp']:.3f} \\\\"
            )
        return lines

    (TABLES_DIR / "strategy_cost_residuals_en.tex").write_text(
        "\n".join(residual_lines(strategy_names_en)) + "\n", encoding="utf-8"
    )
    (TABLES_DIR / "strategy_cost_residuals_ko.tex").write_text(
        "\n".join(residual_lines(strategy_names_ko)) + "\n", encoding="utf-8"
    )
    return claims


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig()
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "run.log").write_text("", encoding="utf-8")

    if (
        not args.skip_extract
        or not CLOCK_COMPACT.exists()
        or not MEMBERS_COMPACT.exists()
        or not SIGNAL_COMPACT.exists()
        or not SIGNAL_INVENTORY.exists()
    ):
        extract_compact_inputs()
    else:
        log("reusing compact inputs")

    log("loading compact inputs and constructing strictly lagged features")
    clock = pl.read_parquet(CLOCK_COMPACT)
    members = pl.read_parquet(MEMBERS_COMPACT)
    signal_prices = pl.read_parquet(SIGNAL_COMPACT)
    panel = build_lagged_feature_panel(clock, signal_prices, members)
    panel.write_parquet(DATA_DIR / "strategy_feature_panel.parquet", compression="zstd")

    log("fitting deterministic walk-forward monthly MLP models")
    panel, mlp_log = add_walk_forward_mlp_scores(panel, cfg)
    write_csv(mlp_log, RESULTS_DIR / "mlp_training_log.csv")
    mlp_audit_columns = ["date", "symbol", "adv_rank", *MLP_FEATURES, "mlp_score"]
    write_csv(
        panel.filter(pl.col("date").is_between(cfg.eval_start, cfg.eval_end, closed="both"))
        .select(mlp_audit_columns)
        .sort(["date", "adv_rank", "symbol"]),
        RESULTS_DIR / "mlp_candidate_scores.csv",
    )

    log("generating fixed-time entry-qualified strategy order vectors")
    submitted, generation_log = generate_frozen_orders(panel, cfg)
    vector_columns = [
        "order_id",
        "date",
        "timestamp_utc",
        "strategy",
        "symbol",
        "side",
        "notional_usd",
        "signal_score",
        "adv_rank",
    ]
    submitted.select(vector_columns).write_parquet(
        RESULTS_DIR / "submitted_orders.parquet", compression="zstd"
    )
    write_csv(submitted.select(vector_columns), RESULTS_DIR / "submitted_orders.csv")

    replacement_count = int(
        generation_log.get_column("replaced_signal_only_orders").sum()
    )
    replacement_strategy_dates = int(
        generation_log.filter(pl.col("replaced_signal_only_orders") > 0).height
    )
    log(
        f"replaced {replacement_count} unavailable signal-ranked candidates "
        f"across {replacement_strategy_dates} strategy-dates"
    )
    log("joining frozen orders to L2 marks with kline fallback and no hindsight replacement")
    frozen_all = attach_execution_state(submitted, panel, cfg)
    frozen, date_status = retain_common_valid_dates(frozen_all, cfg)
    frozen.select(vector_columns).write_parquet(RESULTS_DIR / "frozen_orders.parquet", compression="zstd")
    write_csv(frozen.select(vector_columns), RESULTS_DIR / "frozen_orders.csv")
    frozen.write_parquet(RESULTS_DIR / "order_pricing_inputs.parquet", compression="zstd")
    write_csv(generation_log, RESULTS_DIR / "order_generation_log.csv")
    write_csv(date_status, RESULTS_DIR / "date_retention_status.csv")

    validity = frozen_all.with_columns(
        [
            (~pl.col("matched_entry_state_valid")).alias("missing_or_invalid_entry_state"),
            (~pl.col("target_return").is_finite().fill_null(False)).alias("missing_next_day_l2_mid"),
            (pl.col("markout_source") == "kline_1159").alias("used_kline_markout"),
            (~pl.col("evaluation_return").is_finite().fill_null(False)).alias(
                "missing_all_markout_sources"
            ),
            (~pl.col("spread_bps").is_finite().fill_null(False)).alias("missing_spread"),
            (~pl.col("slipclock_cost_bps").is_finite().fill_null(False)).alias("missing_slipclock_cost"),
            (~pl.col("relevant_side_complete")).alias("incomplete_relevant_side"),
        ]
    )
    validity_audit = validity.group_by("date", "strategy").agg(
        pl.len().alias("generated_orders"),
        pl.col("order_valid").sum().alias("valid_orders"),
        pl.col("missing_or_invalid_entry_state").sum().alias("missing_or_invalid_entry_state"),
        pl.col("missing_next_day_l2_mid").sum().alias("missing_next_day_l2_mid"),
        pl.col("used_kline_markout").sum().alias("used_kline_markout"),
        pl.col("missing_all_markout_sources").sum().alias(
            "missing_all_markout_sources"
        ),
        pl.col("missing_spread").sum().alias("missing_spread"),
        pl.col("missing_slipclock_cost").sum().alias("missing_slipclock_cost"),
        pl.col("incomplete_relevant_side").sum().alias("incomplete_relevant_side"),
    ).sort(["date", "strategy"])
    write_csv(validity_audit, RESULTS_DIR / "order_validity_audit.csv")
    write_csv(
        validity.filter(~pl.col("order_valid")).select(
            "order_id",
            "date",
            "strategy",
            "symbol",
            "side",
            "missing_or_invalid_entry_state",
            "missing_next_day_l2_mid",
            "used_kline_markout",
            "missing_all_markout_sources",
            "missing_spread",
            "missing_slipclock_cost",
            "incomplete_relevant_side",
        ),
        RESULTS_DIR / "invalid_selected_orders.csv",
    )

    order_invariants = assert_order_invariants(frozen, cfg)
    log(f"retained {order_invariants['retained_dates']} common dates and {order_invariants['orders']} frozen orders")

    expected_eval_days = (cfg.eval_end - cfg.eval_start).days + 1
    observed_eval_dates = sorted(
        value
        for value in clock.get_column("date").unique().to_list()
        if cfg.eval_start <= value <= cfg.eval_end
    )
    observed_eval_set = set(observed_eval_dates)
    missing_l2_dates = [
        cfg.eval_start + timedelta(days=offset)
        for offset in range(expected_eval_days)
        if cfg.eval_start + timedelta(days=offset) not in observed_eval_set
    ]

    log("applying six cost treatments to the serialized vector")
    order_wide, order_long, daily = apply_cost_treatments(frozen, cfg)
    daily, carry_audit = augment_with_carried_position_days(
        daily,
        frozen,
        signal_prices,
        cfg,
        carry_dates=missing_l2_dates,
    )
    log(
        f"evaluated {daily.filter(pl.col('is_carry_day')).get_column('date').n_unique()} "
        "missing-L2 dates from carried positions and Binance 11:59 closes"
    )
    order_wide.write_parquet(RESULTS_DIR / "order_costs_wide.parquet", compression="zstd")
    write_csv(order_wide, RESULTS_DIR / "order_costs_wide.csv")
    order_long.write_parquet(RESULTS_DIR / "order_treatments_long.parquet", compression="zstd")
    write_csv(order_long, RESULTS_DIR / "order_treatments_long.csv")
    write_csv(carry_audit, RESULTS_DIR / "carried_position_returns.csv")
    return_outlier_audit = pl.concat(
        [
            order_wide.filter(pl.col("return_outlier")).select(
                "date",
                "strategy",
                "symbol",
                "side",
                "markout_source",
                "raw_evaluation_return",
                "evaluation_return",
            ).with_columns(pl.lit(False).alias("is_carry_day")),
            carry_audit.filter(pl.col("return_outlier")).select(
                "date",
                "strategy",
                "symbol",
                pl.col("side").cast(pl.Int8),
                pl.lit("kline_1159_close_to_close").alias("markout_source"),
                "raw_evaluation_return",
                "evaluation_return",
            ).with_columns(pl.lit(True).alias("is_carry_day")),
        ],
        how="vertical",
    ).sort(["date", "strategy", "symbol"])
    write_csv(return_outlier_audit, RESULTS_DIR / "return_outlier_audit.csv")
    write_csv(daily, RESULTS_DIR / "daily_net_results.csv")
    treatment_support = assert_treatment_support(order_long, frozen)

    metrics = summarize_metrics(daily, cfg)
    write_csv(metrics, RESULTS_DIR / "strategy_metrics.csv")
    reversals = pl.concat(
        [
            pairwise_reversal_matrix(metrics, "net_sharpe"),
            pairwise_reversal_matrix(metrics, "cumulative_net_pnl_usd"),
        ],
        how="vertical",
    )
    write_csv(reversals, RESULTS_DIR / "pairwise_rank_reversals.csv")

    residuals, residual_summary = equation17_residuals(daily)
    write_csv(residuals, RESULTS_DIR / "equation17_residuals_daily.csv")
    write_csv(residual_summary, RESULTS_DIR / "equation17_residual_summary.csv")
    identity = verify_equation17_pairwise(residuals)

    invalid_orders = frozen_all.filter(~pl.col("order_valid"))
    reason_counts = {
        "missing_or_invalid_entry_state": int(
            validity.get_column("missing_or_invalid_entry_state").sum()
        ),
        "missing_next_day_l2_mid": int(validity.get_column("missing_next_day_l2_mid").sum()),
        "used_kline_markout": int(validity.get_column("used_kline_markout").sum()),
        "missing_all_markout_sources": int(
            validity.get_column("missing_all_markout_sources").sum()
        ),
        "missing_spread": int(validity.get_column("missing_spread").sum()),
        "missing_slipclock_cost": int(validity.get_column("missing_slipclock_cost").sum()),
        "incomplete_relevant_side": int(validity.get_column("incomplete_relevant_side").sum()),
    }
    submitted_orders = int(submitted.height)
    vector_generation_dates = int(frozen_all.get_column("date").n_unique())
    carried_evaluation_dates = int(
        daily.filter(pl.col("is_carry_day")).get_column("date").n_unique()
    )
    evaluated_calendar_dates = int(daily.get_column("date").n_unique())
    carry_missing_price_pairs = int(
        carry_audit.select((~pl.col("price_available")).sum()).item()
    )
    ordinary_return_outlier_orders = int(
        order_wide.get_column("return_outlier").sum()
    )
    carried_return_outlier_orders = int(
        carry_audit.get_column("return_outlier").sum()
    )
    sample_audit: dict[str, int | float] = {
        "calendar_dates": expected_eval_days,
        "evaluated_calendar_dates": evaluated_calendar_dates,
        "carried_evaluation_dates": carried_evaluation_dates,
        "vector_generation_dates": vector_generation_dates,
        "missing_l2_dates": len(missing_l2_dates),
        "carry_missing_price_pairs": carry_missing_price_pairs,
        "ordinary_return_outlier_orders_zeroed": ordinary_return_outlier_orders,
        "carried_return_outlier_orders_zeroed": carried_return_outlier_orders,
        "unique_return_outlier_symbol_days_zeroed": int(
            return_outlier_audit.select(
                pl.struct("date", "symbol").n_unique()
            ).item()
        ),
        "entry_unavailable_signal_rank_replacements": replacement_count,
        "strategy_dates_with_entry_replacements": replacement_strategy_dates,
        "submitted_orders_before_complete_case_filter": submitted_orders,
        "invalid_selected_orders_before_complete_case_filter": int(invalid_orders.height),
        "additional_complete_case_excluded_dates": int(
            date_status.filter(~pl.col("retained")).height
        ),
        "missing_or_invalid_entry_state_orders": reason_counts[
            "missing_or_invalid_entry_state"
        ],
        "incomplete_relevant_side_orders": reason_counts["incomplete_relevant_side"],
        "missing_next_day_l2_mid_orders": reason_counts["missing_next_day_l2_mid"],
        "kline_fallback_markout_orders": reason_counts["used_kline_markout"],
        "missing_all_markout_source_orders": reason_counts[
            "missing_all_markout_sources"
        ],
        "entry_state_valid_orders": submitted_orders
        - reason_counts["missing_or_invalid_entry_state"],
        "entry_state_coverage": (
            submitted_orders - reason_counts["missing_or_invalid_entry_state"]
        )
        / submitted_orders,
        "relevant_side_complete_orders": submitted_orders
        - reason_counts["incomplete_relevant_side"],
        "relevant_side_completion": (
            submitted_orders - reason_counts["incomplete_relevant_side"]
        )
        / submitted_orders,
        "next_day_l2_markout_available_orders": submitted_orders
        - reason_counts["missing_next_day_l2_mid"],
        "next_day_l2_markout_availability": (
            submitted_orders - reason_counts["missing_next_day_l2_mid"]
        )
        / submitted_orders,
        "evaluation_markout_available_orders": submitted_orders
        - reason_counts["missing_all_markout_sources"],
        "evaluation_markout_availability": (
            submitted_orders - reason_counts["missing_all_markout_sources"]
        )
        / submitted_orders,
        "common_date_retention": int(order_invariants["retained_dates"])
        / vector_generation_dates,
    }
    write_manuscript_assets(
        metrics,
        reversals,
        residual_summary,
        order_invariants,
        sample_audit,
        cfg,
    )

    if not args.no_figures:
        figure_en = FIGURES_DIR / "cumulative_entry_cost_adjusted_markout_en.pdf"
        make_cumulative_figure(daily, figure_en, language="en")
        shutil.copyfile(figure_en, FIGURES_DIR / "cumulative_net_pnl_by_cost.pdf")
        shutil.copyfile(
            figure_en.with_suffix(".png"), FIGURES_DIR / "cumulative_net_pnl_by_cost.png"
        )
        make_cumulative_figure(
            daily,
            FIGURES_DIR / "cumulative_entry_cost_adjusted_markout_ko.pdf",
            language="ko",
        )
    make_metric_table(metrics, TABLES_DIR / "strategy_cost_metrics.tex")

    all_outputs_finite = bool(
        metrics.select(
            pl.all_horizontal(
                pl.col("cumulative_net_pnl_usd").is_finite(),
                pl.col("net_sharpe").is_finite(),
            ).all()
        ).item()
    )
    qa_checks = {
        "all_outputs_finite": all_outputs_finite,
        "all_submitted_orders_entry_executable": all(
            reason_counts[key] == 0
            for key in (
                "missing_or_invalid_entry_state",
                "missing_spread",
                "missing_slipclock_cost",
                "incomplete_relevant_side",
            )
        ),
        "all_retained_orders_have_evaluation_markout": bool(
            frozen.select(pl.col("evaluation_return").is_finite().all()).item()
        ),
        "treatment_support_valid": bool(
            treatment_support["one_row_per_treatment_per_order"]
            and treatment_support["each_treatment_matches_frozen_ids"]
        ),
        "equation17_identity_within_tolerance": bool(identity["max_abs_error_usd"] <= 1e-8),
        "all_calendar_days_evaluated": evaluated_calendar_dates == expected_eval_days,
        "all_carried_contracts_have_binance_marks": carry_missing_price_pairs == 0,
        "all_flagged_returns_zeroed": bool(
            return_outlier_audit.select(
                (pl.col("evaluation_return") == 0.0).all()
            ).item()
        ),
    }
    qa = {
        "status": "PASS" if all(qa_checks.values()) else "FAIL",
        "config": {
            "eval_start": cfg.eval_start.isoformat(),
            "eval_end": cfg.eval_end.isoformat(),
            "order_notional_usd": cfg.order_notional_usd,
            "orders_per_strategy_day": cfg.orders_per_strategy_day,
            "submitted_gross_usd": cfg.submitted_gross_usd,
            "annualization_days": cfg.annualization_days,
            "mlp_features": list(MLP_FEATURES),
            "mlp_hidden_units": cfg.mlp_hidden_units,
            "mlp_epochs": cfg.mlp_epochs,
            "mlp_learning_rate": cfg.mlp_learning_rate,
            "mlp_l2": cfg.mlp_l2,
            "max_abs_daily_contract_return": cfg.max_abs_daily_contract_return,
            "seed": cfg.seed,
        },
        "input_rows": {
            "clock_1200": clock.height,
            "signal_prices_1159": signal_prices.height,
            "pit_members": members.height,
            "feature_panel": panel.height,
        },
        "order_invariants": order_invariants,
        "submitted_orders_before_common_date_filter": submitted_orders,
        "entry_unavailable_signal_rank_replacements": replacement_count,
        "strategy_dates_with_entry_replacements": replacement_strategy_dates,
        "generated_dates": vector_generation_dates,
        "excluded_common_dates": int(date_status.filter(~pl.col("retained")).height),
        "invalid_selected_orders_before_common_date_filter": invalid_orders.height,
        "invalid_order_reason_counts": reason_counts,
        "submitted_denominator_rates": {
            "entry_state_coverage": sample_audit["entry_state_coverage"],
            "relevant_side_completion": sample_audit["relevant_side_completion"],
            "next_day_l2_markout_availability": sample_audit[
                "next_day_l2_markout_availability"
            ],
            "evaluation_markout_availability": sample_audit[
                "evaluation_markout_availability"
            ],
        },
        "calendar": {
            "expected_eval_days": expected_eval_days,
            "evaluated_calendar_days": evaluated_calendar_dates,
            "carried_evaluation_days": carried_evaluation_dates,
            "observed_l2_eval_days": len(observed_eval_dates),
            "missing_l2_eval_days": len(missing_l2_dates),
            "missing_l2_dates": [value.isoformat() for value in missing_l2_dates],
        },
        "return_outlier_policy": {
            "max_abs_daily_contract_return": cfg.max_abs_daily_contract_return,
            "ordinary_orders_zeroed": ordinary_return_outlier_orders,
            "carried_orders_zeroed": carried_return_outlier_orders,
            "unique_symbol_days_zeroed": sample_audit[
                "unique_return_outlier_symbol_days_zeroed"
            ],
        },
        "checks": qa_checks,
        "treatment_support": treatment_support,
        "equation17_pairwise_identity": identity,
        "all_outputs_finite": all_outputs_finite,
    }
    (RESULTS_DIR / "qa.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log("hashing inputs and outputs for the reproducibility manifest")
    input_paths = [CLOCK_COMPACT, MEMBERS_COMPACT, SIGNAL_COMPACT, SIGNAL_INVENTORY]
    if not args.skip_extract:
        input_paths = [RAW_CLOCK, RAW_MEMBERS, *input_paths]
    output_paths = sorted(
        path
        for directory in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR)
        for path in directory.glob("*")
        if path.is_file() and path.name != "reproducibility_verification.json"
    )
    signal_inventory = pl.read_csv(SIGNAL_INVENTORY)
    implementation_paths = [HERE / "experiment_core.py", HERE / "run_experiment.py"]
    manifest = {
        "experiment": "frozen_order_strategy_cost_v1_20260801",
        "version_date": "2026-08-01",
        "command": ".venv/bin/python rewrite2/paper/expander/frozen_order_strategy_cost_v1_20260801/run_experiment.py",
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "polars": pl.__version__,
            "duckdb": duckdb.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "inputs": {
            str(path.relative_to(REPO)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        },
        "raw_signal_inventory": {
            "entries": signal_inventory.height,
            "declared_bytes": int(signal_inventory.get_column("bytes").sum()),
            "inventory_sha256": sha256_file(SIGNAL_INVENTORY),
        },
        "implementation": {
            str(path.relative_to(HERE)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in implementation_paths
        },
        "signal_source_root": str(RAW_SIGNAL_ROOT.relative_to(REPO)),
        "outputs": {
            str(path.relative_to(HERE)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log("experiment completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
