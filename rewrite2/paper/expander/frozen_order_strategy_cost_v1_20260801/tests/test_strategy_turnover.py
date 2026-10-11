from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy_turnover import one_way_target_turnover


def test_unchanged_target_has_zero_turnover() -> None:
    target = {"A": 100.0, "B": -100.0}
    assert one_way_target_turnover(target, target, 200.0) == 0.0


def test_full_replacement_has_one_way_turnover_of_one() -> None:
    previous = {"A": 100.0, "B": -100.0}
    current = {"C": 100.0, "D": -100.0}
    assert math.isclose(one_way_target_turnover(previous, current, 200.0), 1.0)


def test_sign_flip_counts_close_and_reopen() -> None:
    previous = {"A": 100.0}
    current = {"A": -100.0}
    assert math.isclose(one_way_target_turnover(previous, current, 100.0), 1.0)
