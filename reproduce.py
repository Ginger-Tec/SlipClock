#!/usr/bin/env python3
"""Rebuild and verify all paper-facing assets from compact released inputs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "rewrite2" / "paper"
EXPERIMENT = PAPER / "expander" / "frozen_order_strategy_cost_v1_20260801"


def run(*parts: str) -> None:
    command = [sys.executable, *parts]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def copy_asset(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main() -> None:
    run(str(EXPERIMENT / "run_experiment.py"), "--skip-extract")
    run(str(EXPERIMENT / "strategy_turnover.py"))
    run(str(EXPERIMENT / "posthoc_clock_oracle.py"), "--skip-extract")
    run(str(PAPER / "build_assets.py"))

    for suffix in ("pdf", "png"):
        copy_asset(
            EXPERIMENT / "figures" / f"cumulative_entry_cost_adjusted_markout_en.{suffix}",
            PAPER / "figures" / f"frozen_strategy_cost_cumulative_en.{suffix}",
        )
    run(str(PAPER / "build_compact_assets.py"))
    run(str(ROOT / "verify_outputs.py"))
    print("Reproduction completed and verified.")


if __name__ == "__main__":
    main()
