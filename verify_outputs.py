#!/usr/bin/env python3
"""Compare regenerated paper-facing outputs with frozen anonymous targets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "rewrite2" / "paper"
EXPERIMENT = PAPER / "expander" / "frozen_order_strategy_cost_v1_20260801"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_equal(expected: Path, actual: Path) -> None:
    if not expected.is_file() or not actual.is_file():
        raise FileNotFoundError(f"missing comparison input: {expected} or {actual}")
    if expected.suffix == ".json":
        expected_value = json.loads(expected.read_text(encoding="utf-8"))
        actual_value = json.loads(actual.read_text(encoding="utf-8"))
        if expected_value != actual_value:
            raise AssertionError(f"JSON mismatch: {actual}")
    elif expected.suffix == ".csv":
        with expected.open(newline="", encoding="utf-8") as handle:
            expected_rows = list(csv.reader(handle))
        with actual.open(newline="", encoding="utf-8") as handle:
            actual_rows = list(csv.reader(handle))
        if len(expected_rows) != len(actual_rows):
            raise AssertionError(f"CSV row-count mismatch: {actual}")
        for row_index, (expected_row, actual_row) in enumerate(
            zip(expected_rows, actual_rows, strict=True)
        ):
            if len(expected_row) != len(actual_row):
                raise AssertionError(f"CSV column-count mismatch: {actual}:{row_index}")
            for column_index, (expected_cell, actual_cell) in enumerate(
                zip(expected_row, actual_row, strict=True)
            ):
                try:
                    matches = math.isclose(
                        float(expected_cell),
                        float(actual_cell),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                except ValueError:
                    matches = expected_cell == actual_cell
                if not matches:
                    raise AssertionError(
                        f"CSV mismatch: {actual}:{row_index + 1}:{column_index + 1}"
                    )
    elif expected.read_bytes() != actual.read_bytes():
        raise AssertionError(
            f"byte mismatch: {actual}; expected={sha256(expected)} actual={sha256(actual)}"
        )


def main() -> None:
    comparisons = (
        (
            ROOT / "expected" / "pit_adv30_daily_mean_topn_costs.csv",
            PAPER / "tables" / "pit_adv30_daily_mean_topn_costs.csv",
        ),
        (
            ROOT / "expected" / "pit_exact_clock_60m_contrasts.csv",
            PAPER / "tables" / "pit_exact_clock_60m_contrasts.csv",
        ),
        (
            ROOT / "expected" / "strategy_claims.json",
            EXPERIMENT / "results" / "manuscript_claims.json",
        ),
        (
            ROOT / "expected" / "strategy_turnover_summary.csv",
            EXPERIMENT / "results" / "strategy_turnover_summary.csv",
        ),
        (
            ROOT / "expected" / "paired_clock_effects.csv",
            EXPERIMENT / "results" / "posthoc_clock_oracle" / "paired_clock_effects.csv",
        ),
    )
    for expected, actual in comparisons:
        require_equal(expected, actual)

    required_figures = (
        "pit_exact_minute_100k_top25_top100_top200.pdf",
        "pit_daily_regime_and_lomo_compact.pdf",
        "strategy_clock_cost_exposure_compact.pdf",
        "frozen_strategy_cost_cumulative_en.pdf",
    )
    for name in required_figures:
        path = PAPER / "figures" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing or empty figure: {path}")
    print(json.dumps({"status": "PASS", "comparisons": 5, "figures": 4}))


if __name__ == "__main__":
    main()
