#!/usr/bin/env python3
"""Build compact single-column figures for the eight-page SlipClock manuscripts."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from paper_figure_style import (
    BLUE,
    COLUMN_WIDTH_IN,
    GREEN,
    INK,
    LIGHT_GREY,
    MUTED,
    PDF_METADATA,
    UNIVERSE_COLORS,
    UNIVERSE_LABELS,
    VERMILLION,
    apply_paper_style,
)


apply_paper_style()


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
CLOCK_EFFECTS = (
    ROOT
    / "expander"
    / "frozen_order_strategy_cost_v1_20260801"
    / "results"
    / "posthoc_clock_oracle"
    / "paired_clock_effects.csv"
)
DAILY_REGIME = ROOT / "tables" / "pit_daily_regime_series.csv"
LOMO_CONTRASTS = ROOT / "tables" / "pit_exact_lomo_monthly_contrasts.csv"


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        metadata=PDF_METADATA,
    )
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=300,
    )
    plt.close(fig)


def build_clock_exposure() -> None:
    with CLOCK_EFFECTS.open(newline="", encoding="utf-8") as handle:
        rows = {row["strategy"]: row for row in csv.DictReader(handle)}

    order = ["ts_momentum", "cs_momentum", "mean_reversion", "vol_scaled", "mlp"]
    labels = {
        "ts_momentum": "TS mom.",
        "cs_momentum": "CS mom.",
        "mean_reversion": "Mean rev.",
        "vol_scaled": "Vol. mom.",
        "mlp": "MLP",
    }
    low = [float(rows[key]["low_cost_bps"]) for key in order]
    base = [float(rows[key]["baseline_cost_bps"]) for key in order]
    high = [float(rows[key]["high_cost_bps"]) for key in order]
    delta = [float(rows[key]["high_minus_low_cost_bps"]) for key in order]
    y = list(reversed(range(len(order))))

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.28))
    for index in range(len(order)):
        ax.plot(
            [low[index], high[index]],
            [y[index], y[index]],
            color=LIGHT_GREY,
            linewidth=0.85,
            zorder=1,
        )
    ax.scatter(low, y, color=GREEN, marker="o", s=19, label="11:45", zorder=3)
    ax.scatter(base, y, color=BLUE, marker="D", s=18, label="12:00", zorder=3)
    ax.scatter(high, y, color=VERMILLION, marker="s", s=19, label="08:00", zorder=3)
    for index, value in enumerate(delta):
        ax.text(
            high[index] + 0.8,
            y[index],
            f"+{value:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=MUTED,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([labels[key] for key in order])
    ax.set_xlabel("Displayed L2 entry cost (bp)")
    ax.set_xlim(0.0, 68.0)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.1,
    )
    fig.subplots_adjust(left=0.245, right=0.965, top=0.825, bottom=0.205)
    save(fig, "strategy_clock_cost_exposure_compact")


def build_daily_regime_and_lomo() -> None:
    colors = UNIVERSE_COLORS
    labels = UNIVERSE_LABELS
    markers = {"top25": "o", "top100": "s", "top200": "^"}
    universes = ["top25", "top100", "top200"]

    daily = {universe: {"date": [], "raw": [], "ma30": []} for universe in universes}
    with DAILY_REGIME.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            universe = row["universe"]
            if universe not in daily:
                continue
            daily[universe]["date"].append(datetime.strptime(row["date"], "%Y-%m-%d"))
            daily[universe]["raw"].append(float(row["daily_mean_cost_bps"]))
            daily[universe]["ma30"].append(
                float(row["daily_mean_cost_30_calendar_observation_ma_bps"])
                if row["daily_mean_cost_30_calendar_observation_ma_bps"]
                else float("nan")
            )

    monthly = {universe: {"date": [], "contrast": []} for universe in universes}
    with LOMO_CONTRASTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            universe = row["universe"]
            if universe not in monthly:
                continue
            monthly[universe]["date"].append(datetime.strptime(row["heldout_month"], "%Y-%m"))
            monthly[universe]["contrast"].append(
                float(row["heldout_60m_high_minus_low_cost_bps"])
            )

    fig, (daily_ax, clock_ax) = plt.subplots(
        2,
        1,
        figsize=(COLUMN_WIDTH_IN, 4.42),
        sharex=True,
        gridspec_kw={"height_ratios": [1.28, 1.0]},
    )
    for universe in universes:
        daily_ax.plot(
            daily[universe]["date"],
            daily[universe]["raw"],
            color=colors[universe],
            linewidth=0.35,
            alpha=0.14,
        )
        daily_ax.plot(
            daily[universe]["date"],
            daily[universe]["ma30"],
            color=colors[universe],
            linewidth=1.15,
            label=labels[universe],
        )
        clock_ax.plot(
            monthly[universe]["date"],
            monthly[universe]["contrast"],
            color=colors[universe],
            marker=markers[universe],
            markersize=2.8,
            markeredgewidth=0,
            linewidth=1.0,
        )

    daily_ax.set_title("(a) Daily cost level", loc="left", pad=3)
    clock_ax.set_title(
        "(b) Cross-fitted high–low clock contrast",
        loc="left",
        pad=3,
    )
    daily_ax.set_ylabel("Displayed L2 cost (bp)")
    clock_ax.set_ylabel("High–low cost (bp)")
    clock_ax.set_xlabel("UTC date")
    clock_ax.axhline(0, color=INK, linewidth=0.65)
    for axis in (daily_ax, clock_ax):
        axis.grid(axis="y")
    clock_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    clock_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    clock_ax.tick_params(axis="x", rotation=28)
    daily_ax.legend(
        frameon=False,
        loc="upper left",
        ncol=3,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.7,
        borderaxespad=0.15,
    )
    fig.subplots_adjust(left=0.205, right=0.985, top=0.965, bottom=0.125, hspace=0.25)
    save(fig, "pit_daily_regime_and_lomo_compact")


def main() -> None:
    build_clock_exposure()
    build_daily_regime_and_lomo()


if __name__ == "__main__":
    main()
