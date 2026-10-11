from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_core import ExperimentConfig  # noqa: E402
from posthoc_clock_oracle import (  # noqa: E402
    ordinary_daily,
    paired_clock_effects,
    price_frozen_orders,
    select_posthoc_clocks,
    summarize,
)


class PosthocClockOracleTests(unittest.TestCase):
    @staticmethod
    def _clocks() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "clock": ["baseline_1200", "oracle_low", "oracle_high"],
                "minute_of_day": [720, 705, 480],
                "time_utc": ["12:00", "11:45", "08:00"],
            }
        )

    def test_full_sample_clock_selection_is_explicit_and_deterministic(self) -> None:
        cfg = ExperimentConfig(
            eval_start=date(2025, 1, 1),
            eval_end=date(2025, 1, 2),
        )
        rows = []
        costs = {480: 30.0, 705: 20.0, 720: 25.0}
        for current_date in (date(2025, 1, 1), date(2025, 1, 2)):
            for minute, cost in costs.items():
                rows.append(
                    {
                        "date": current_date,
                        "universe": "top100",
                        "minute_of_day": minute,
                        "cost_100000_bps": cost,
                        "completion_rate": 1.0,
                    }
                )
        clocks = select_posthoc_clocks(pl.DataFrame(rows), cfg)
        lookup = {
            row["clock"]: row for row in clocks.iter_rows(named=True)
        }
        self.assertEqual(lookup["oracle_low"]["minute_of_day"], 705)
        self.assertEqual(lookup["oracle_high"]["minute_of_day"], 480)
        self.assertFalse(lookup["baseline_1200"]["selection_is_posthoc"])
        self.assertTrue(lookup["oracle_low"]["selection_is_posthoc"])
        self.assertTrue(lookup["oracle_high"]["selection_is_posthoc"])

    def test_pricing_reuses_one_order_at_every_clock(self) -> None:
        cfg = ExperimentConfig(
            eval_start=date(2025, 1, 1),
            eval_end=date(2025, 1, 1),
            orders_per_strategy_day=1,
        )
        orders = pl.DataFrame(
            {
                "order_id": ["ts_momentum|2025-01-01|BTCUSDT"],
                "date": [date(2025, 1, 1)],
                "timestamp_utc": [
                    datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
                ],
                "strategy": ["ts_momentum"],
                "symbol": ["BTCUSDT"],
                "side": [1],
                "notional_usd": [100_000.0],
                "signal_score": [0.1],
                "adv_rank": [1],
            }
        )
        state_rows = []
        for minute, start, end, cost in (
            (720, 100.0, 101.0, 4.0),
            (705, 100.0, 102.0, 2.0),
            (480, 100.0, 99.0, 8.0),
        ):
            for current_date, mid in (
                (date(2025, 1, 1), start),
                (date(2025, 1, 2), end),
            ):
                state_rows.append(
                    {
                        "timestamp_utc": None,
                        "date": current_date,
                        "minute_of_day": minute,
                        "symbol": "BTCUSDT",
                        "mid_price": mid,
                        "spread_bps": 1.0,
                        "buy_100000_mid_cost_bps": cost,
                        "sell_100000_mid_cost_bps": cost + 1.0,
                        "buy_100000_mid_complete": True,
                        "sell_100000_mid_complete": True,
                        "valid_bbo": True,
                    }
                )
        prices = pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.String,
                "minute_of_day": pl.Int64,
                "source_minute_of_day": pl.Int64,
                "kline_mark": pl.Float64,
            }
        )
        retained, audit = price_frozen_orders(
            orders,
            pl.DataFrame(state_rows),
            prices,
            self._clocks(),
            cfg,
        )
        self.assertEqual(retained.height, 3)
        self.assertEqual(retained["order_id"].n_unique(), 1)
        self.assertEqual(retained["clock"].n_unique(), 3)
        self.assertTrue(audit.select((pl.col("entry_valid_orders") == 1).all()).item())
        low = retained.filter(pl.col("clock") == "oracle_low")
        high = retained.filter(pl.col("clock") == "oracle_high")
        self.assertAlmostEqual(low["evaluation_return"][0], 0.02)
        self.assertAlmostEqual(high["evaluation_return"][0], -0.01)
        self.assertAlmostEqual(low["slipclock_cost_bps"][0], 2.0)
        self.assertAlmostEqual(high["slipclock_cost_bps"][0], 8.0)

    def test_daily_cost_and_metric_summary_keep_clock_dimension(self) -> None:
        cfg = ExperimentConfig(
            eval_start=date(2025, 1, 1),
            eval_end=date(2025, 1, 2),
            orders_per_strategy_day=1,
        )
        rows = []
        for clock, minute, cost in (
            ("baseline_1200", 720, 4.0),
            ("oracle_low", 705, 2.0),
            ("oracle_high", 480, 8.0),
        ):
            for current_date, daily_return in (
                (date(2025, 1, 1), 0.01),
                (date(2025, 1, 2), -0.005),
            ):
                rows.append(
                    {
                        "order_id": f"{current_date}",
                        "date": current_date,
                        "clock": clock,
                        "minute_of_day": minute,
                        "strategy": "ts_momentum",
                        "symbol": "BTCUSDT",
                        "side": 1,
                        "evaluation_notional_usd": 100_000.0,
                        "evaluation_return": daily_return,
                        "slipclock_cost_bps": cost,
                    }
                )
        _, daily = ordinary_daily(pl.DataFrame(rows), cfg)
        daily = daily.with_columns(
            pl.col("net_pnl_usd")
            .cum_sum()
            .over("clock", "strategy", "treatment")
            .alias("cumulative_net_pnl_usd")
        )
        metrics = summarize(daily, cfg)
        self.assertEqual(metrics["clock"].n_unique(), 3)
        self.assertEqual(metrics["treatment"].n_unique(), 2)
        slip = metrics.filter(pl.col("treatment") == "slipclock")
        low = slip.filter(pl.col("clock") == "oracle_low")
        high = slip.filter(pl.col("clock") == "oracle_high")
        self.assertAlmostEqual(low["mean_entry_cost_bps"][0], 2.0)
        self.assertAlmostEqual(high["mean_entry_cost_bps"][0], 8.0)
        self.assertGreater(
            low["cumulative_net_pnl_usd"][0],
            high["cumulative_net_pnl_usd"][0],
        )

    def test_paired_effects_preserve_cost_and_pnl_decomposition(self) -> None:
        cfg = ExperimentConfig(
            eval_start=date(2025, 1, 1),
            eval_end=date(2025, 2, 1),
            orders_per_strategy_day=1,
        )
        rows = []
        for strategy in (
            "ts_momentum",
            "cs_momentum",
            "mean_reversion",
            "vol_scaled",
            "mlp",
        ):
            for current_date in (date(2025, 1, 1), date(2025, 2, 1)):
                for clock, minute, cost, gross in (
                    ("baseline_1200", 720, 4.0, 100.0),
                    ("oracle_low", 705, 2.0, 120.0),
                    ("oracle_high", 480, 8.0, 80.0),
                ):
                    for treatment in ("no_cost", "slipclock"):
                        cost_usd = (
                            0.0
                            if treatment == "no_cost"
                            else cfg.submitted_gross_usd * cost / 10_000.0
                        )
                        rows.append(
                            {
                                "clock": clock,
                                "minute_of_day": minute,
                                "date": current_date,
                                "strategy": strategy,
                                "treatment": treatment,
                                "order_count": 1,
                                "evaluation_gross_usd": cfg.submitted_gross_usd,
                                "gross_pnl_usd": gross,
                                "cost_usd": cost_usd,
                                "net_pnl_usd": gross - cost_usd,
                                "submitted_gross_usd": cfg.submitted_gross_usd,
                                "is_carry_day": False,
                                "net_return": (gross - cost_usd)
                                / cfg.submitted_gross_usd,
                                "cumulative_net_pnl_usd": 0.0,
                            }
                        )
        daily = pl.DataFrame(rows)
        metrics = summarize(daily, cfg)
        effects = paired_clock_effects(
            daily, metrics, bootstrap_reps=200, seed=7
        )
        self.assertEqual(effects.height, 5)
        self.assertTrue(
            effects.select(
                (pl.col("high_minus_low_cost_bps") == 6.0).all()
            ).item()
        )
        self.assertTrue(
            effects.select((pl.col("cost_ci_low_bps") == 6.0).all()).item()
        )
        self.assertTrue(
            effects.select(
                (
                    pl.col("net_low_minus_high_m")
                    - (2.0 * (40.0 + 60.0) / 1_000_000.0)
                )
                .abs()
                .lt(1e-12)
                .all()
            ).item()
        )


if __name__ == "__main__":
    unittest.main()
