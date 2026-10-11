#!/usr/bin/env python3
"""Summarize entry-date turnover of the frozen strategy target vectors."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Mapping


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
STRATEGY_ORDER = (
    "ts_momentum",
    "cs_momentum",
    "mean_reversion",
    "vol_scaled",
    "mlp",
)
STRATEGY_LABELS = {
    "ts_momentum": "Time-series momentum",
    "cs_momentum": "Cross-sectional momentum",
    "mean_reversion": "Mean reversion",
    "vol_scaled": "Volatility-scaled momentum",
    "mlp": "MLP",
}


def one_way_target_turnover(
    previous: Mapping[str, float],
    current: Mapping[str, float],
    target_gross_usd: float,
) -> float:
    """Return one-way turnover between two signed target-notional vectors."""
    if target_gross_usd <= 0.0:
        raise ValueError("target gross must be positive")
    traded_notional = sum(
        abs(float(current.get(symbol, 0.0)) - float(previous.get(symbol, 0.0)))
        for symbol in set(previous) | set(current)
    )
    return traded_notional / (2.0 * target_gross_usd)


def read_target_vectors(
    path: Path,
) -> dict[str, list[tuple[date, dict[str, float]]]]:
    grouped: dict[tuple[str, date], dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            strategy = row["strategy"]
            if strategy not in STRATEGY_ORDER:
                raise ValueError(f"unexpected strategy: {strategy}")
            current_date = date.fromisoformat(row["date"])
            symbol = row["symbol"]
            side = int(row["side"])
            notional = float(row["notional_usd"])
            if side not in (-1, 1) or notional <= 0.0:
                raise ValueError("side and notional must define a signed target")
            key = (strategy, current_date)
            if symbol in grouped[key]:
                raise ValueError(f"duplicate target: {strategy} {current_date} {symbol}")
            grouped[key][symbol] = side * notional

    vectors: dict[str, list[tuple[date, dict[str, float]]]] = {}
    expected_dates: list[date] | None = None
    for strategy in STRATEGY_ORDER:
        rows = sorted(
            (current_date, positions)
            for (row_strategy, current_date), positions in grouped.items()
            if row_strategy == strategy
        )
        dates = [current_date for current_date, _ in rows]
        if not rows:
            raise ValueError(f"no target vectors for {strategy}")
        if expected_dates is None:
            expected_dates = dates
        elif dates != expected_dates:
            raise ValueError("entry-date calendars differ across strategies")
        for current_date, positions in rows:
            target_gross = sum(abs(value) for value in positions.values())
            if len(positions) != 20 or not math.isclose(
                target_gross, 2_000_000.0, rel_tol=0.0, abs_tol=1e-8
            ):
                raise ValueError(
                    f"invalid frozen target block: {strategy} {current_date}"
                )
        vectors[strategy] = rows
    return vectors


def summarize(
    vectors: Mapping[str, list[tuple[date, Mapping[str, float]]]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        strategy_vectors = vectors[strategy]
        turnovers = [
            one_way_target_turnover(previous, current, 2_000_000.0)
            for (_, previous), (_, current) in zip(
                strategy_vectors, strategy_vectors[1:], strict=False
            )
        ]
        if not turnovers:
            raise ValueError(f"at least two target vectors are required for {strategy}")
        mean_turnover = statistics.fmean(turnovers)
        summaries.append(
            {
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "transition_days": len(turnovers),
                "mean_one_way_target_turnover": mean_turnover,
                "mean_one_way_target_turnover_pct": 100.0 * mean_turnover,
            }
        )
    return summaries


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.12f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-orders",
        type=Path,
        default=RESULTS_DIR / "frozen_orders.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "strategy_turnover_summary.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vectors = read_target_vectors(args.frozen_orders)
    rows = summarize(vectors)
    write_summary(args.output, rows)
    print(f"wrote {len(rows)} strategy turnover rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
