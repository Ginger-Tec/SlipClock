#!/usr/bin/env python3
"""Build reproducible assets for the SlipClock empirical portfolio-cost paper.

The builder is intentionally self-contained and uses only the current L2-only
research suite.  It writes paper-facing CSV/TeX/Markdown tables, PNG/PDF
figures, and a compact manifest describing sources and interpretation notes.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, Rectangle
import polars as pl
import pyarrow.parquet as pq

from paper_figure_style import (
    COLUMN_WIDTH_IN,
    PDF_METADATA,
    UNIVERSE_COLORS,
    UNIVERSE_LABELS,
    apply_paper_style,
)


apply_paper_style()


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
RUN_ROOT = REPO_ROOT / "data" / "derived"
TABLE_DIR = PACKAGE_ROOT / "tables"
FIGURE_DIR = PACKAGE_ROOT / "figures"
BUILD_DIR = PACKAGE_ROOT / "build"
MANIFEST_PATH = BUILD_DIR / "paper_assets_manifest.json"

PIT_POLICY = "daily"
PIT_LOOKBACK_DAYS = 30
PIT_VOLUME_STAT = "mean"
MAIN_UNIVERSES = ["top25", "top100", "top200"]
SAMPLE_WINDOW = {
    "start_date": "2024-01-01",
    "end_date": "2026-04-30",
    "symbols": 718,
    "dates": 833,
    "observed_utc_minute_positions": 1_439,
    "omitted_utc_minute": "00:00",
}


TABLE_SOURCES = {
    "pit_adv30_topn_cost": RUN_ROOT / "pit_adv_universe" / "pit_summary.csv",
    "pit_exact_minute_completion": RUN_ROOT / "pit_exact_clock" / "pit_adv30_exact_minute_surface.parquet",
    "oracle_vs_pit_adv30": RUN_ROOT / "pit_adv_universe" / "oracle_vs_pit_adv30.csv",
    "broad_tail_capacity_curve": RUN_ROOT / "portfolio_execution" / "universe_capacity_curve.csv",
    "factor_significance_100k_mid_cost": RUN_ROOT / "factor_significance" / "factor_test_summary.csv",
    "pit_exact_lomo_window_validation": RUN_ROOT / "pit_exact_clock" / "pit_adv30_exact_lomo_window_validation.csv",
    "pit_exact_clock_window_horizons": RUN_ROOT / "pit_exact_clock" / "pit_adv30_exact_window_contrasts.parquet",
    "pit_exact_hour_surface": RUN_ROOT / "pit_exact_clock" / "pit_adv30_exact_hour_surface.parquet",
    "clock_anchor_summary": RUN_ROOT / "pit_clock_anchors" / "clock_anchor_summary.csv",
}
FIGURE_SOURCES = {
    "pit_exact_minute_100k": [
        RUN_ROOT / "pit_exact_clock" / "pit_adv30_exact_minute_surface.parquet",
        RUN_ROOT / "pit_exact_clock" / "pit_adv30_exact_window_contrasts.parquet",
    ],
    "pit_notional_scaling": [RUN_ROOT / "explorer" / "pit_daily_costs_all_notionals.parquet"],
    "pit_exact_lomo_monthly_contrasts": [RUN_ROOT / "pit_exact_clock" / "pit_adv30_exact_lomo_window_detail.csv"],
    "pit_daily_regime_and_lomo": [
        RUN_ROOT / "pit_clock_anchors" / "pit_daily_regime.parquet",
        RUN_ROOT / "pit_exact_clock" / "pit_adv30_exact_lomo_window_detail.csv",
    ],
}


def rel(path: Path) -> str:
    """Return a repo-relative path when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required source artifact is missing: {rel(path)}")


def ensure_dirs() -> None:
    for path in (TABLE_DIR, FIGURE_DIR, BUILD_DIR):
        path.mkdir(parents=True, exist_ok=True)


def universe_sort_expr() -> pl.Expr:
    return pl.col("universe").str.replace("top", "").cast(pl.Int32, strict=False).fill_null(10_000)


def read_csv(path: Path) -> pl.DataFrame:
    require_path(path)
    return pl.read_csv(path)


_latex_replacements = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        if abs(value) >= 100:
            return f"{value:,.2f}"
        return f"{value:,.{digits}f}"
    return str(value)


def latex_escape(value: str) -> str:
    return "".join(_latex_replacements.get(ch, ch) for ch in value)


def column_label(column: str) -> str:
    labels = {
        "universe": "Universe",
        "dates": "Dates",
        "avg_selected_symbols": "Selected symbols",
        "avg_evaluated_symbols": "Evaluated symbols",
        "avg_evaluation_coverage": "Eval. coverage",
        "exact_100k_mean_completion_rate": "$100k exact completion",
        "date_pass_rate": "Date pass rate",
        "mean_rank_quote_volume": "ADV quote volume",
        "avg_min_rank_quote_volume": "Min rank ADV",
        "spread_mean_bps": "Spread mean (bps)",
        "spread_p90_mean_bps": "Spread p90 (bps)",
        "mean_cost_100000_bps": "$100k mean cost (bps)",
        "median_daily_cost_100000_bps": "$100k median daily cost (bps)",
        "p90_daily_cost_100000_bps": "$100k p90 daily cost (bps)",
        "mean_of_symbol_p90_cost_100000_bps": "Mean symbol p90 cost (bps)",
        "expected_cost_100000_usd": "$100k reference dollars (USD)",
        "notional_per_symbol_usd": "Notional / symbol (USD)",
        "oracle_l2_cost_bps": "Oracle lower bound (bps)",
        "pit_adv30_mean_cost_bps": "PIT ADV30 cost (bps)",
        "pit_minus_oracle_bps": "PIT - oracle (bps)",
        "pit_over_oracle": "PIT / oracle",
        "symbols": "Symbols",
        "gross_rebalance_usd": "Gross rebalance (USD)",
        "equal_weight_mean_cost_bps": "Equal-weight mean cost (bps)",
        "symbol_mean_cost_p90_bps": "Symbol mean cost p90 (bps)",
        "mean_of_symbol_p90_cost_bps": "Mean of symbol p90 cost (bps)",
        "expected_cost_usd": "Expected cost (USD)",
        "complete_rate_min": "Min complete rate",
        "worst_symbol_by_mean_cost": "Worst symbol",
        "factor": "Factor",
        "raw_factor": "Raw factor",
        "response": "Response",
        "contrast": "Contrast",
        "n_dates": "Dates",
        "effect_bps": "Effect (bps)",
        "se_bps": "SE (bps)",
        "t_stat": "t-stat",
        "p_value": "p-value",
        "q_value_bh": "BH q-value",
        "ci95_low": "95% CI low",
        "ci95_high": "95% CI high",
        "hour_utc": "UTC hour",
        "low_mean_bps": "Low mean (bps)",
        "high_mean_bps": "High mean (bps)",
        "rows_low": "Low rows",
        "rows_high": "High rows",
        "window_minutes": "Window (min)",
        "best_start_utc": "Low-cost start",
        "best_mean_cost_bps": "Low-window cost (bps)",
        "worst_start_utc": "High-cost start",
        "worst_mean_cost_bps": "High-window cost (bps)",
        "surface_range_bps": "Surface range (bps)",
        "paired_dates": "Paired dates",
        "paired_worst_minus_best_bps": "Paired high-low (bps)",
        "paired_t_stat": "Paired t-stat",
        "best_completion_rate": "Low-window complete",
        "worst_completion_rate": "High-window complete",
        "heldout_months": "Held-out months",
        "heldout_dates": "Held-out dates",
        "heldout_month": "Held-out month",
        "heldout_60m_high_minus_low_cost_bps": "Held-out 60-minute high-minus-low cost (bps)",
        "heldout_worst_minus_best_bps": "Held-out high-low (bps)",
        "se_bps": "Month-block SE (bps)",
        "ci95_low_bps": "Month-block 95% CI low",
        "ci95_high_bps": "Month-block 95% CI high",
        "positive_date_share": "Positive date share",
        "avg_best_completion_rate": "Low-window complete",
        "avg_worst_completion_rate": "High-window complete",
        "pit_gate_count": "PIT gates",
        "mean_cost_min_bps": "Mean min (bps)",
        "mean_cost_max_bps": "Mean max (bps)",
        "p90_cost_min_bps": "Daily p90 min (bps)",
        "p90_cost_max_bps": "Daily p90 max (bps)",
        "coverage_min_pct": "Coverage min (%)",
        "coverage_max_pct": "Coverage max (%)",
        "anchor": "Clock anchor",
        "funding_share_pct": "Actual funding share (%)",
        "cost_delta_bps": "Cost delta (bps)",
        "spread_delta_bps": "Spread delta (bps)",
        "beyond_half_spread_delta_bps": "Beyond-half-spread delta (bps)",
        "cost_delta_ci95_bps": "Cost delta block 95% CI",
    }
    return labels.get(column, column.replace("_", " ").title())


def dataframe_to_markdown(df: pl.DataFrame) -> str:
    columns = df.columns
    header = "| " + " | ".join(column_label(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in df.iter_rows(named=True):
        body.append("| " + " | ".join(format_value(row.get(c)).replace("|", "\\|") for c in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def dataframe_to_latex(df: pl.DataFrame, caption: str, label: str) -> str:
    columns = df.columns
    spec = "l" + "r" * max(0, len(columns) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(latex_escape(column_label(c)) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in df.iter_rows(named=True):
        lines.append(" & ".join(latex_escape(format_value(row.get(c))) for c in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_table_assets(df: pl.DataFrame, stem: str, caption: str, note: str, manifest: dict[str, Any]) -> dict[str, str]:
    csv_path = TABLE_DIR / f"{stem}.csv"
    md_path = TABLE_DIR / f"{stem}.md"
    tex_path = TABLE_DIR / f"{stem}.tex"
    df.write_csv(csv_path)
    md_path.write_text(dataframe_to_markdown(df) + f"\nNote: {note}\n", encoding="utf-8")
    tex_path.write_text(dataframe_to_latex(df, caption, f"tab:{stem}"), encoding="utf-8")
    outputs = {"csv": rel(csv_path), "md": rel(md_path), "tex": rel(tex_path)}
    manifest["assets"][stem] = {"kind": "table", "outputs": outputs, "note": note, "rows": df.height, "columns": df.columns}
    return outputs


def build_pit_adv30_cost_table(manifest: dict[str, Any]) -> None:
    path = TABLE_SOURCES["pit_adv30_topn_cost"]
    completion_path = TABLE_SOURCES["pit_exact_minute_completion"]
    require_path(completion_path)
    completion = (
        pl.read_parquet(completion_path)
        .group_by("universe")
        .agg(pl.col("avg_completion_rate").mean().alias("exact_100k_mean_completion_rate"))
    )
    df = (
        read_csv(path)
        .filter(
            (pl.col("policy") == PIT_POLICY)
            & (pl.col("lookback_days") == PIT_LOOKBACK_DAYS)
            & (pl.col("volume_stat") == PIT_VOLUME_STAT)
        )
        .select(
            [
                "universe",
                "dates",
                "avg_selected_symbols",
                "avg_evaluated_symbols",
                "avg_evaluation_coverage",
                "date_pass_rate",
                "mean_rank_quote_volume",
                "avg_min_rank_quote_volume",
                "spread_mean_bps",
                "mean_cost_100000_bps",
                "median_daily_cost_100000_bps",
                "p90_daily_cost_100000_bps",
                "mean_of_symbol_p90_cost_100000_bps",
                "expected_cost_100000_usd",
            ]
        )
        .join(completion, on="universe", how="left")
        .sort(universe_sort_expr())
    )
    write_table_assets(
        df,
        "pit_adv30_daily_mean_topn_costs",
        "PIT ADV30 daily mean quote-volume top-N displayed-book audit scorecard.",
        "Daily PIT evaluation coverage is joined to fixed-denominator exact date-symbol-minute $100k completion averaged over the 1,439 observed UTC minute positions from 00:01 through 23:59; costs remain completed-only and must be read jointly with both failure metrics.",
        manifest,
    )
    manifest["sources"]["pit_adv30_topn_cost"] = rel(path)
    manifest["sources"]["pit_adv30_topn_completion"] = rel(completion_path)
def build_pit_gate_robustness_table(manifest: dict[str, Any]) -> None:
    path = TABLE_SOURCES["pit_adv30_topn_cost"]
    gate_keys = ["policy", "lookback_days", "volume_stat"]
    base = read_csv(path).filter(pl.col("universe").is_in(MAIN_UNIVERSES))
    gate_count = base.select(gate_keys).unique().height
    if gate_count != 12:
        raise RuntimeError(f"Expected 12 PIT liquidity gates, found {gate_count}")

    top25 = (
        base.filter(pl.col("universe") == "top25")
        .select([*gate_keys, pl.col("mean_cost_100000_bps").alias("top25_mean_bps")])
    )
    top200 = (
        base.filter(pl.col("universe") == "top200")
        .select([*gate_keys, pl.col("mean_cost_100000_bps").alias("top200_mean_bps")])
    )
    matched = top25.join(top200, on=gate_keys, how="inner")
    if matched.height != gate_count or not matched.select(
        (pl.col("top200_mean_bps") > pl.col("top25_mean_bps")).all()
    ).item():
        raise RuntimeError("Top-200 mean cost does not exceed top-25 for every matched PIT gate")

    df = (
        base.group_by("universe")
        .agg(
            [
                pl.len().alias("pit_gate_count"),
                pl.col("mean_cost_100000_bps").min().alias("mean_cost_min_bps"),
                pl.col("mean_cost_100000_bps").max().alias("mean_cost_max_bps"),
                pl.col("p90_daily_cost_100000_bps").min().alias("p90_cost_min_bps"),
                pl.col("p90_daily_cost_100000_bps").max().alias("p90_cost_max_bps"),
                (100 * pl.col("avg_evaluation_coverage").min()).alias("coverage_min_pct"),
                (100 * pl.col("avg_evaluation_coverage").max()).alias("coverage_max_pct"),
            ]
        )
        .sort(universe_sort_expr())
    )
    write_table_assets(
        df,
        "pit_gate_robustness_ranges",
        "Range of $100k date-level outcomes across 12 PIT liquidity gates.",
        "The grid crosses daily/monthly holding, 7/14/30-day lookbacks, and mean/median trailing quote volume. All 12 matched gates satisfy top-200 mean cost greater than top-25 mean cost; the output reports full-grid ranges rather than selected specifications.",
        manifest,
    )
    manifest["sources"]["pit_gate_robustness_ranges"] = rel(path)




def build_oracle_benchmark_table(manifest: dict[str, Any]) -> None:
    path = TABLE_SOURCES["oracle_vs_pit_adv30"]
    df = read_csv(path).sort(universe_sort_expr()).select(
        ["universe", "notional_per_symbol_usd", "oracle_l2_cost_bps", "pit_adv30_mean_cost_bps", "pit_minus_oracle_bps", "pit_over_oracle"]
    )
    write_table_assets(
        df,
        "oracle_vs_pit_adv30_benchmark",
        "Oracle lower-bound benchmark versus PIT ADV30 costs.",
        "Oracle is an ex-post lower-bound benchmark only; PIT ADV30 is the paper-facing investable universe construction.",
        manifest,
    )
    manifest["sources"]["oracle_vs_pit_adv30"] = rel(path)


def build_broad_tail_capacity_table(manifest: dict[str, Any]) -> None:
    path = TABLE_SOURCES["broad_tail_capacity_curve"]
    df = (
        read_csv(path)
        .select(
            [
                "universe",
                "symbols",
                "notional_per_symbol_usd",
                "gross_rebalance_usd",
                "equal_weight_mean_cost_bps",
                "symbol_mean_cost_p90_bps",
                "mean_of_symbol_p90_cost_bps",
                "expected_cost_usd",
                "spread_p90_mean_bps",
                "complete_rate_min",
                "worst_symbol_by_mean_cost",
            ]
        )
        .sort([universe_sort_expr(), "notional_per_symbol_usd"])
    )
    write_table_assets(
        df,
        "broad_tail_capacity_curve",
        "All-symbol broad-tail capacity stress curve.",
        "Uses portfolio_execution/universe_capacity_curve.csv as broad-tail stress evidence, not as the main PIT universe framing.",
        manifest,
    )
    manifest["sources"]["broad_tail_capacity_curve"] = rel(path)


def build_factor_significance_table(manifest: dict[str, Any]) -> None:
    path = TABLE_SOURCES["factor_significance_100k_mid_cost"]
    base = read_csv(path).filter(pl.col("response") == "cost_100000_mid_bps")
    significant = base.filter(pl.col("significant_5pct_bh") == True)
    if significant.is_empty():
        significant = base
        note = "No BH-significant rows found for cost_100000_mid_bps; table shows the lowest q-value diagnostics."
    else:
        note = "BH-significant rows for cost_100000_mid_bps, sorted by absolute t-statistic; table is a compact paper-facing subset."
    df = (
        significant.with_columns(pl.col("t_stat").abs().alias("abs_t_stat"))
        .sort(["abs_t_stat", "q_value_bh"], descending=[True, False])
        .head(12)
        .select(
            [
                "factor",
                "raw_factor",
                "response",
                "contrast",
                "n_dates",
                "effect_bps",
                "se_bps",
                "t_stat",
                "p_value",
                "q_value_bh",
                "ci95_low",
                "ci95_high",
                "hour_utc",
                "low_mean_bps",
                "high_mean_bps",
                "rows_low",
                "rows_high",
            ]
        )
    )
    write_table_assets(
        df,
        "factor_significance_100k_mid_cost",
        "Significant factor contrasts for $100k mid-cost response.",
        note,
        manifest,
    )
    manifest["sources"]["factor_significance_100k_mid_cost"] = rel(path)


def build_pit_exact_minute_surface() -> pl.DataFrame:
    surface_path, contrasts_path = FIGURE_SOURCES["pit_exact_minute_100k"]
    require_path(surface_path)
    require_path(contrasts_path)
    return (
        pl.read_parquet(surface_path)
        .filter(pl.col("universe").is_in(MAIN_UNIVERSES))
        .with_columns((pl.col("minute_of_day") % 60).cast(pl.Int8).alias("minute_utc"))
        .sort([universe_sort_expr(), "minute_of_day"])
    )


def build_exact_window_contrast_table(manifest: dict[str, Any]) -> None:
    _surface_path, contrasts_path = FIGURE_SOURCES["pit_exact_minute_100k"]
    require_path(contrasts_path)
    df = (
        pl.read_parquet(contrasts_path)
        .filter((pl.col("universe").is_in(MAIN_UNIVERSES)) & (pl.col("window_minutes") == 60))
        .select(
            [
                "universe",
                "window_minutes",
                "best_start_utc",
                "best_mean_cost_bps",
                "worst_start_utc",
                "worst_mean_cost_bps",
                "surface_range_bps",
                "paired_dates",
                "paired_worst_minus_best_bps",
                "paired_t_stat",
                "best_completion_rate",
                "worst_completion_rate",
            ]
        )
        .sort(universe_sort_expr())
    )
    write_table_assets(
        df,
        "pit_exact_clock_60m_contrasts",
        "Exact PIT ADV30 60-minute clock-window contrasts.",
        "Each universe is selected point-in-time by prior ADV30 quote volume; selected symbols are joined to realized L2 rows on exact date, symbol, and minute-of-day keys. Windows are selected from the exact minute surface and paired by UTC date.",
        manifest,
    )
    manifest["sources"]["pit_exact_clock_60m_contrasts"] = rel(contrasts_path)
def build_exact_clock_window_horizons_table(manifest: dict[str, Any]) -> None:
    path = TABLE_SOURCES["pit_exact_clock_window_horizons"]
    require_path(path)
    df = (
        pl.read_parquet(path)
        .filter(
            (pl.col("universe").is_in(MAIN_UNIVERSES))
            & (pl.col("window_minutes").is_in([30, 60, 120]))
        )
        .select(
            [
                "universe",
                "window_minutes",
                "best_start_utc",
                "worst_start_utc",
                "paired_worst_minus_best_bps",
                "paired_t_stat",
                "paired_dates",
            ]
        )
        .sort([universe_sort_expr(), "window_minutes"])
    )
    write_table_assets(
        df,
        "pit_exact_clock_window_horizons",
        "Exact PIT $100k clock-window contrasts across 30-, 60-, and 120-minute horizons.",
        "Each universe is selected point-in-time by prior ADV30 quote volume and evaluated on the exact date-symbol-minute $100k displayed-book panel. Starts identify in-sample low- and high-cost UTC windows; the paired contrast is high-cost minus low-cost across UTC dates.",
        manifest,
    )
    manifest["sources"]["pit_exact_clock_window_horizons"] = rel(path)

def build_exact_hour_cost_spread_table(manifest: dict[str, Any]) -> None:
    path = TABLE_SOURCES["pit_exact_hour_surface"]
    require_path(path)
    hourly = pl.read_parquet(path).filter(
        pl.col("universe").is_in(MAIN_UNIVERSES) & pl.col("hour_utc").is_in([11, 21])
    )
    rows: list[dict[str, Any]] = []
    for universe in MAIN_UNIVERSES:
        low = hourly.filter(
            (pl.col("universe") == universe) & (pl.col("hour_utc") == 11)
        ).row(0, named=True)
        high = hourly.filter(
            (pl.col("universe") == universe) & (pl.col("hour_utc") == 21)
        ).row(0, named=True)
        low_cost = float(low["mean_cost_100000_bps"])
        high_cost = float(high["mean_cost_100000_bps"])
        low_spread = float(low["mean_spread_bps"])
        high_spread = float(high["mean_spread_bps"])
        rows.append(
            {
                "universe": universe,
                "cost_11_utc_bps": low_cost,
                "cost_21_utc_bps": high_cost,
                "cost_difference_bps": high_cost - low_cost,
                "spread_difference_bps": high_spread - low_spread,
                "beyond_half_spread_difference_bps": (
                    high_cost - high_spread / 2
                ) - (low_cost - low_spread / 2),
            }
        )
    df = pl.DataFrame(rows)
    write_table_assets(
        df,
        "pit_exact_hour_cost_spread_decomposition",
        "Exact PIT $100k hour-level cost and spread decomposition.",
        "The comparison uses the same PIT portfolios at 11 and 21 UTC. Because a mid-referenced top-of-book fill pays one-half of the quoted bid-ask spread, cost minus one-half spread is the displayed-depth component of the snapshot-implied book walk. This is a descriptive accounting decomposition, not a causal mechanism estimate.",
        manifest,
    )
    manifest["sources"]["pit_exact_hour_cost_spread_decomposition"] = rel(path)




def build_lomo_window_validation_table(manifest: dict[str, Any]) -> None:
    path = TABLE_SOURCES["pit_exact_lomo_window_validation"]
    require_path(path)
    df = (
        read_csv(path)
        .filter(pl.col("universe").is_in(MAIN_UNIVERSES))
        .select(
            [
                "universe",
                "heldout_months",
                "heldout_dates",
                "heldout_worst_minus_best_bps",
                "se_bps",
                "ci95_low_bps",
                "ci95_high_bps",
                "positive_date_share",
                "avg_best_completion_rate",
                "avg_worst_completion_rate",
            ]
        )
        .sort(universe_sort_expr())
    )
    write_table_assets(
        df,
        "pit_exact_lomo_window_validation",
        "Leave-one-month-out validation with calendar-month block-bootstrap inference.",
        "For each held-out month, low- and high-cost 60-minute windows are selected from all other months, then evaluated on the held-out dates. The contrast is high-cost minus low-cost. Percentile 95% confidence intervals use 10,000 deterministic calendar-month block-bootstrap replications, preserving all dates within each sampled month.",
        manifest,
    )
    manifest["sources"]["pit_exact_lomo_window_validation"] = rel(path)


def build_clock_anchor_table(manifest: dict[str, Any]) -> None:
    path = TABLE_SOURCES["clock_anchor_summary"]
    require_path(path)
    labels = {
        "four_hour_nonoverlap": "4h boundary, no 8h (04/12/20)",
        "eight_hour_overlap": "4h+8h overlap (08/16)",
        "us_cash_open": "U.S. cash open",
        "us_cash_close": "U.S. cash close",
    }
    order = {key: index for index, key in enumerate(labels)}
    source = read_csv(path).filter(pl.col("universe") == "top200")
    rows: list[dict[str, Any]] = []
    for row in source.iter_rows(named=True):
        anchor = row["anchor"]
        if anchor not in labels:
            continue
        rows.append(
            {
                "anchor_order": order[anchor],
                "anchor": labels[anchor],
                "funding_share_pct": 100 * row["mean_actual_funding_share"],
                "cost_delta_bps": row["mean_cost_delta_bps"],
                "spread_delta_bps": row["mean_spread_delta_bps"],
                "beyond_half_spread_delta_bps": row["mean_beyond_half_spread_delta_bps"],
                "cost_delta_ci95_bps": (
                    f"[{row['cost_delta_ci95_low_bps']:.3f}, "
                    f"{row['cost_delta_ci95_high_bps']:.3f}]"
                ),
            }
        )
    df = pl.DataFrame(rows).sort("anchor_order").drop("anchor_order")
    if df.height != 4:
        raise ValueError(f"Expected four top-200 clock-anchor rows, found {df.height}")
    write_table_assets(
        df,
        "clock_anchor_top200",
        "Top-200 exact-clock event contrasts at $100k per contract.",
        "Each row compares the event window [0,+2] minutes with the same date and PIT universe over [-10,-2] minutes. Actual funding share uses contract-date Binance funding timestamps with observed 1-, 2-, 4-, and 8-hour intervals. U.S. cash sessions use the official XNYS calendar, including DST, holidays, and early closes. Cost confidence intervals use 10,000 calendar-month block-bootstrap replications. These are associative event-time diagnostics, not causal effects.",
        manifest,
    )
    manifest["sources"]["clock_anchor_top200"] = rel(path)

def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(fig)
    return {"png": rel(png_path), "pdf": rel(pdf_path)}
def build_slipclock_workflow_figure(manifest: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.5))
    ax.set_axis_off()
    stages = [
        ("Submitted dated\norder vector", "contract and side\nmidpoint-marked USD notional\nUTC sampling instant", "#dbe9f6"),
        ("Prior-information\nPIT ADV30 gate", "prior-date membership\n30-day trailing mean\nno ex-post replacement", "#e8f0db"),
        ("Exact-minute L2\nladder walk", "book at timestamp\nQ = a / midpoint\nlevel fills and VWAP", "#f9e4cb"),
        ("Portfolio\noutputs", "atomic bps and USD\nfull-gross reference budget\ncoverage and completion", "#eadff1"),
    ]
    x_positions = [0.02, 0.27, 0.52, 0.77]
    width, height, y = 0.21, 0.64, 0.18
    for x, (title, detail, color) in zip(x_positions, stages):
        ax.add_patch(Rectangle((x, y), width, height, transform=ax.transAxes, facecolor=color, edgecolor="#3f3f3f", linewidth=1.0))
        ax.text(x + width / 2, y + height * 0.72, title, transform=ax.transAxes, ha="center", va="center", fontsize=10, fontweight="bold", wrap=True)
        ax.text(x + width / 2, y + height * 0.34, detail, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, linespacing=1.35)
    for left, right in zip(x_positions, x_positions[1:]):
        ax.add_patch(FancyArrowPatch((left + width, y + height / 2), (right - 0.012, y + height / 2), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#3f3f3f"))
    outputs = save_figure(fig, "slipclock_workflow")
    manifest["assets"]["slipclock_workflow"] = {
        "kind": "figure",
        "outputs": outputs,
        "note": "Measurement workflow for submitted dated order vectors: point-in-time ADV30 eligibility precedes the exact-minute L2 displayed-book ladder walk, which reports immediate displayed-book book-walk cost, full-gross reference budget, coverage, and completion. It does not represent optimal execution or implementation shortfall.",
    }


def build_pit_exact_lomo_monthly_contrasts_figure(manifest: dict[str, Any]) -> None:
    path = FIGURE_SOURCES["pit_exact_lomo_monthly_contrasts"][0]
    require_path(path)
    monthly = (
        read_csv(path)
        .filter(
            (pl.col("universe").is_in(MAIN_UNIVERSES))
            & (pl.col("window_minutes") == 60)
        )
        .group_by(["heldout_month", "universe"])
        .agg(pl.col("worst_minus_best_bps").mean().alias("heldout_60m_high_minus_low_cost_bps"))
        .sort(["heldout_month", "universe"])
    )
    monthly_csv = TABLE_DIR / "pit_exact_lomo_monthly_contrasts.csv"
    monthly.write_csv(monthly_csv)

    months = monthly.get_column("heldout_month").unique(maintain_order=True).to_list()
    month_index = {month: index for index, month in enumerate(months)}
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for universe in MAIN_UNIVERSES:
        sub = monthly.filter(pl.col("universe") == universe).sort("heldout_month")
        if sub.is_empty():
            continue
        ax.plot(
            [month_index[month] for month in sub["heldout_month"].to_list()],
            sub["heldout_60m_high_minus_low_cost_bps"].to_list(),
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=universe,
            color=UNIVERSE_COLORS[universe],
        )
    ax.axhline(0, color="#444444", linewidth=1.0)
    ax.set_title("Monthly held-out 60-minute high-minus-low cost")
    ax.set_xlabel("Held-out UTC month")
    ax.set_ylabel("Held-out 60-minute high-minus-low cost (bps)")
    tick_positions = list(range(0, len(months), 3))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([months[index] for index in tick_positions], rotation=35, ha="right")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Universe", frameon=False, ncol=3)
    outputs = save_figure(fig, "pit_exact_lomo_monthly_contrasts")
    manifest["assets"]["pit_exact_lomo_monthly_contrasts"] = {
        "kind": "figure",
        "outputs": outputs,
        "data_csv": rel(monthly_csv),
        "source_artifacts": [rel(path)],
        "note": "For each held-out UTC calendar month, low- and high-cost 60-minute windows are selected using all other months, then the exact PIT $100k date-symbol-minute displayed-book costs are paired by held-out date. The plotted monthly mean is high-cost minus low-cost; it is a descriptive held-out contrast, not a causal or prospective-savings estimate.",
    }
    manifest["sources"]["pit_exact_lomo_monthly_contrasts"] = [rel(path)]


def build_pit_daily_regime_and_lomo_figure(manifest: dict[str, Any]) -> None:
    regime_path, lomo_path = FIGURE_SOURCES["pit_daily_regime_and_lomo"]
    require_path(regime_path)
    require_path(lomo_path)
    regime = (
        pl.read_parquet(regime_path)
        .filter(pl.col("universe").is_in(MAIN_UNIVERSES))
        .select(
            "date",
            "universe",
            "daily_mean_cost_bps",
            "daily_mean_cost_30_calendar_observation_ma_bps",
        )
        .sort(["date", "universe"])
    )
    if regime.get_column("date").n_unique() != SAMPLE_WINDOW["dates"]:
        raise ValueError("Daily regime series does not cover the full disclosed sample")
    regime_csv = TABLE_DIR / "pit_daily_regime_series.csv"
    regime.write_csv(regime_csv)

    monthly = (
        read_csv(lomo_path)
        .filter(
            (pl.col("universe").is_in(MAIN_UNIVERSES))
            & (pl.col("window_minutes") == 60)
        )
        .group_by(["heldout_month", "universe"])
        .agg(pl.col("worst_minus_best_bps").mean().alias("heldout_60m_high_minus_low_cost_bps"))
        .sort(["heldout_month", "universe"])
    )
    monthly_csv = TABLE_DIR / "pit_exact_lomo_monthly_contrasts.csv"
    monthly.write_csv(monthly_csv)

    colors = {"top25": "#1f77b4", "top100": "#ff7f0e", "top200": "#2ca02c"}
    fig, (daily_ax, clock_ax) = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for universe in MAIN_UNIVERSES:
        daily_sub = regime.filter(pl.col("universe") == universe).sort("date")
        daily_ax.plot(
            daily_sub["date"].to_list(),
            daily_sub["daily_mean_cost_bps"].to_list(),
            color=colors[universe],
            linewidth=0.45,
            alpha=0.16,
        )
        daily_ax.plot(
            daily_sub["date"].to_list(),
            daily_sub["daily_mean_cost_30_calendar_observation_ma_bps"].to_list(),
            color=colors[universe],
            linewidth=1.8,
            label=universe,
        )

        monthly_sub = monthly.filter(pl.col("universe") == universe).sort("heldout_month")
        clock_ax.plot(
            [datetime.strptime(month, "%Y-%m") for month in monthly_sub["heldout_month"].to_list()],
            monthly_sub["heldout_60m_high_minus_low_cost_bps"].to_list(),
            marker="o",
            markersize=3.0,
            linewidth=1.6,
            label=universe,
            color=colors[universe],
        )

    daily_ax.set_title("(a) Daily mean cost regime")
    daily_ax.set_ylabel("Daily mean displayed-book cost (bps)")
    daily_ax.legend(title="Universe", frameon=False, ncol=3, loc="upper left")
    clock_ax.set_title("(b) Held-out clock contrast")
    clock_ax.set_ylabel("60-minute high-minus-low cost (bps)")
    clock_ax.axhline(0, color="#444444", linewidth=0.9)
    for axis in (daily_ax, clock_ax):
        axis.set_xlabel("UTC date")
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", rotation=35)
    fig.tight_layout(w_pad=2.0)
    outputs = save_figure(fig, "pit_daily_regime_and_lomo")
    manifest["assets"]["pit_daily_regime_and_lomo"] = {
        "kind": "figure",
        "outputs": outputs,
        "data_csv": [rel(regime_csv), rel(monthly_csv)],
        "source_artifacts": [rel(regime_path), rel(lomo_path)],
        "note": "Panel (a) plots each PIT portfolio's raw daily mean $100k displayed-book cost as a faint line and its trailing 30-calendar-day mean as a solid line. Panel (b) plots the leave-one-month-out 60-minute high-minus-low cost contrast. Both panels are historical calendar-regime descriptions, not trend forecasts or causal event estimates.",
    }
    manifest["sources"]["pit_daily_regime_and_lomo"] = [rel(regime_path), rel(lomo_path)]

def build_pit_exact_minute_figure(manifest: dict[str, Any]) -> None:
    df = build_pit_exact_minute_surface()
    surface_csv = TABLE_DIR / "pit_exact_minute_100k_surface.csv"
    df.write_csv(surface_csv)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.32))
    line_styles = {"top25": "-", "top100": "--", "top200": "-."}
    for universe in MAIN_UNIVERSES:
        sub = df.filter(pl.col("universe") == universe).sort("minute_of_day")
        if sub.is_empty():
            continue
        x = [m / 60 for m in sub["minute_of_day"].to_list()]
        y = sub["mean_cost_100000_bps"].to_list()
        ax.plot(
            x,
            y,
            label=UNIVERSE_LABELS[universe],
            color=UNIVERSE_COLORS[universe],
            linestyle=line_styles[universe],
            linewidth=1.15,
        )
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("Displayed L2 cost (bp)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=3, borderaxespad=0.15)
    fig.subplots_adjust(left=0.205, right=0.985, top=0.965, bottom=0.215)
    outputs = save_figure(fig, "pit_exact_minute_100k_top25_top100_top200")
    manifest["assets"]["pit_exact_minute_100k"] = {
        "kind": "figure",
        "outputs": outputs,
        "data_csv": rel(surface_csv),
        "source_artifacts": [rel(p) for p in FIGURE_SOURCES["pit_exact_minute_100k"]],
        "note": "Exact PIT date-symbol-minute surface over 1,439 observed UTC minute positions from 00:01 through 23:59: PIT ADV30 daily members are joined to realized one-minute L2 cost rows by date and symbol, then averaged by date and minute before the cross-date minute profile is formed. The unavailable 00:00 state is excluded because the first usable L2 snapshot arrives after midnight.",
    }
    manifest["sources"]["pit_exact_minute_100k"] = [rel(p) for p in FIGURE_SOURCES["pit_exact_minute_100k"]]


def build_notional_scaling_figure(manifest: dict[str, Any]) -> None:
    path = FIGURE_SOURCES["pit_notional_scaling"][0]
    require_path(path)
    df_raw = (
        pl.read_parquet(path)
        .filter(
            (pl.col("policy") == PIT_POLICY)
            & (pl.col("lookback_days") == PIT_LOOKBACK_DAYS)
            & (pl.col("volume_stat") == PIT_VOLUME_STAT)
            & (pl.col("universe").is_in(MAIN_UNIVERSES))
        )
        .group_by("universe")
        .agg(
            [
                pl.col("cost_mean_1000").mean().alias("cost_1000_bps"),
                pl.col("cost_mean_10000").mean().alias("cost_10000_bps"),
                pl.col("cost_mean_100000").mean().alias("cost_100000_bps"),
                pl.col("symbol_p90_1000").mean().alias("symbol_p90_1000_bps"),
                pl.col("symbol_p90_10000").mean().alias("symbol_p90_10000_bps"),
                pl.col("symbol_p90_100000").mean().alias("symbol_p90_100000_bps"),
                pl.col("date").n_unique().alias("dates"),
                pl.col("selected_symbols").mean().alias("avg_selected_symbols"),
            ]
        )
    )
    df = df_raw.sort(universe_sort_expr())
    scaling_csv = TABLE_DIR / "pit_notional_scaling_summary.csv"
    df.write_csv(scaling_csv)

    notionals = [1_000, 10_000, 100_000]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = {"top25": "#1f77b4", "top100": "#ff7f0e", "top200": "#2ca02c"}
    for row in df.iter_rows(named=True):
        universe = row["universe"]
        y = [row[f"cost_{n}_bps"] for n in notionals]
        ax.plot(notionals, y, marker="o", linewidth=1.9, label=universe, color=colors.get(universe))
    ax.set_xscale("log")
    ax.set_xticks(notionals)
    ax.set_xticklabels(["$1k", "$10k", "$100k"])
    ax.set_title("PIT ADV30 notional scaling of displayed-book costs")
    ax.set_xlabel("Notional per symbol")
    ax.set_ylabel("Mean mid-cost (bps)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Universe", frameon=False)
    outputs = save_figure(fig, "pit_notional_scaling_top25_top100_top200")
    manifest["assets"]["pit_notional_scaling"] = {
        "kind": "figure",
        "outputs": outputs,
        "data_csv": rel(scaling_csv),
        "source_artifacts": [rel(path)],
        "note": "Summarizes the common 718-symbol PIT source panel in explorer/pit_daily_costs_all_notionals.parquet for $1k, $10k, and $100k.",
    }
    manifest["sources"]["pit_notional_scaling"] = [rel(path)]


def parquet_schema(path: Path) -> list[str]:
    require_path(path)
    return [field.name for field in pq.ParquetFile(path).schema_arrow]


def initial_manifest() -> dict[str, Any]:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": rel(PACKAGE_ROOT),
        "research_root": rel(RUN_ROOT),
        "sample_window": dict(SAMPLE_WINDOW),
        "main_framing": "Empirical study of intraday displayed-book costs for broad point-in-time cross-sectional crypto-futures portfolios; SlipClock is the measurement framework.",
        "scope_boundary": "Integrated message-factor extensions are outside this L2-only release.",
        "sources": {},
        "assets": {},
        "interpretation_notes": [
            "PIT ADV30 tables and figures characterize broad cross-sectional portfolio costs across UTC clock, breadth, per-name notional, and eligibility timing; they do not evaluate predictive-model quality.",
            "Oracle-vs-PIT evidence is a hindsight lower-bound diagnostic only and should not be presented as an attainable strategy.",
            "All-symbol/broad-tail capacity evidence is a stress diagnostic, not the main universe definition.",
            "Displayed-book immediate book-walk cost is measured from the current L2-only one-minute suite over 2024-01-01 through 2026-04-30.",
            "LOMO clock-window uncertainty uses 10,000 deterministic calendar-month block-bootstrap replications with dates grouped within month.",
            "Funding and XNYS clock-anchor contrasts are same-date event-time associations; they do not identify causal funding or U.S. session effects.",
            "The measurement protocol holds universe eligibility, notional probes, clock policy, cost reference, completion treatment, aggregation, and resampling invariant within a run; this frozen status is not historical preregistration.",
            "The disclosed 2024-2026 panel is a historical characterization, not a prospective deployment test.",
        ],
        "scenario_analysis": {},
    }
    pit_manifest = RUN_ROOT / "pit_adv_universe" / "manifest.json"
    if pit_manifest.exists():
        data = json.loads(pit_manifest.read_text(encoding="utf-8"))
        manifest["sample_window"].update(
            {
                "start_date": data.get("startDate", SAMPLE_WINDOW["start_date"]),
                "end_date": data.get("endDate", SAMPLE_WINDOW["end_date"]),
                "symbols": data.get("l2Symbols", SAMPLE_WINDOW["symbols"]),
                "dates": data.get("evalDates", SAMPLE_WINDOW["dates"]),
            }
        )
        manifest["sources"]["pit_adv_universe_manifest"] = rel(pit_manifest)
    for key, paths in FIGURE_SOURCES.items():
        for path in paths:
            if path.suffix == ".parquet" and path.exists():
                manifest.setdefault("source_schemas", {})[rel(path)] = parquet_schema(path)
    return manifest


def main() -> None:
    require_path(RUN_ROOT)
    ensure_dirs()
    manifest = initial_manifest()
    build_pit_adv30_cost_table(manifest)
    build_pit_gate_robustness_table(manifest)
    build_oracle_benchmark_table(manifest)
    build_broad_tail_capacity_table(manifest)
    build_factor_significance_table(manifest)
    build_exact_window_contrast_table(manifest)
    build_exact_clock_window_horizons_table(manifest)
    build_exact_hour_cost_spread_table(manifest)
    build_lomo_window_validation_table(manifest)
    build_clock_anchor_table(manifest)
    build_pit_exact_lomo_monthly_contrasts_figure(manifest)
    build_pit_daily_regime_and_lomo_figure(manifest)
    build_pit_exact_minute_figure(manifest)
    build_notional_scaling_figure(manifest)
    build_slipclock_workflow_figure(manifest)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote SlipClock paper assets manifest: {rel(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
