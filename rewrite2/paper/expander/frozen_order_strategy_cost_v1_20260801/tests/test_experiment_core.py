from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_core import (  # noqa: E402
    MLP_FEATURES,
    ExperimentConfig,
    SimpleMLPRegressor,
    add_walk_forward_mlp_scores,
    apply_cost_treatments,
    attach_execution_state,
    augment_with_carried_position_days,
    build_lagged_feature_panel,
    equation17_residuals,
    generate_frozen_orders,
    pairwise_reversal_matrix,
    retain_common_valid_dates,
    verify_equation17_pairwise,
)


class ExperimentCoreTests(unittest.TestCase):
    @staticmethod
    def _order_selection_panel() -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for idx in range(30):
            row: dict[str, object] = {
                "date": date(2025, 1, 1),
                "symbol": f"S{idx:02d}",
                "adv_rank": idx + 1,
                "ret_1_lag": (idx - 15) / 100.0,
                "ret_5_lag": (idx - 15) / 80.0,
                "ret_20_lag": (idx - 15) / 50.0,
                "ret_60_lag": (idx - 15) / 40.0,
                "vol_5_lag": 0.02 + idx / 10_000.0,
                "vol_20_lag": 0.03 + idx / 10_000.0,
                "vol_60_lag": 0.04 + idx / 10_000.0,
                "adv_rank_norm": 1.0 - 2.0 * idx / 99.0,
                "mlp_score": (idx - 15) / 60.0,
                "mid_price": 100.0 + idx,
                "target_return": 0.001,
                "fallback_markout_return": 0.001,
                "target_end_date": date(2025, 1, 2),
                "spread_bps": 2.0,
                "buy_100000_mid_cost_bps": 3.0,
                "sell_100000_mid_cost_bps": 3.5,
                "buy_100000_mid_complete": True,
                "sell_100000_mid_complete": True,
                "valid_bbo": True,
            }
            rows.append(row)
        return pl.DataFrame(rows)

    def test_simple_mlp_is_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(250, 8))
        y = 0.3 * x[:, 0] - 0.2 * x[:, 3] + rng.normal(scale=0.01, size=250)
        first = SimpleMLPRegressor(epochs=20, seed=11).fit(x, y).predict(x[:12])
        second = SimpleMLPRegressor(epochs=20, seed=11).fit(x, y).predict(x[:12])
        np.testing.assert_array_equal(first, second)

    def test_cost_arithmetic_uses_one_way_ticket(self) -> None:
        cfg = ExperimentConfig()
        frozen = pl.DataFrame(
            {
                "order_id": ["x"],
                "date": [date(2025, 1, 1)],
                "strategy": ["ts_momentum"],
                "symbol": ["BTCUSDT"],
                "side": [1],
                "notional_usd": [100_000.0],
                "target_return": [0.01],
                "spread_bps": [2.0],
                "slipclock_cost_bps": [7.0],
            }
        )
        wide, order_long, daily = apply_cost_treatments(frozen, cfg)
        self.assertAlmostEqual(wide["cost_fixed_5bp_usd"][0], 50.0)
        self.assertAlmostEqual(wide["cost_half_spread_usd"][0], 10.0)
        self.assertAlmostEqual(wide["cost_slipclock_usd"][0], 70.0)
        self.assertEqual(order_long.height, 6)
        self.assertEqual(order_long["treatment"].n_unique(), 6)
        slipclock = daily.filter(pl.col("treatment") == "slipclock")
        self.assertAlmostEqual(slipclock["net_pnl_usd"][0], 930.0)

    def test_calendar_gap_is_not_collapsed(self) -> None:
        signal = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
                "symbol": ["X", "X", "X", "X"],
                "signal_price": [100.0, 101.0, 103.0, 104.0],
            }
        )
        clock = pl.DataFrame(
            {
                "timestamp_utc": [None, None, None],
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4)],
                "symbol": ["X", "X", "X"],
                "mid_price": [100.0, 102.0, 108.0],
                "spread_bps": [1.0, 1.0, 1.0],
                "buy_100000_mid_cost_bps": [2.0, 2.0, 2.0],
                "sell_100000_mid_cost_bps": [2.0, 2.0, 2.0],
                "buy_100000_mid_complete": [True, True, True],
                "sell_100000_mid_complete": [True, True, True],
                "valid_bbo": [True, True, True],
            }
        )
        members = pl.DataFrame(
            {
                "date": [date(2024, 1, 2), date(2024, 1, 4)],
                "selection_date": [date(2024, 1, 2), date(2024, 1, 4)],
                "symbol": ["X", "X"],
                "adv_rank": [1, 1],
                "rank_value": [1.0, 1.0],
                "lookback_obs": [30, 30],
                "min_lookback_obs": [24, 24],
            }
        )
        panel = build_lagged_feature_panel(clock, signal, members)
        jan2 = panel.filter(pl.col("date") == date(2024, 1, 2))
        jan4 = panel.filter(pl.col("date") == date(2024, 1, 4))
        # Jan 2 cannot have a next-calendar-day L2 mark because Jan 3 is absent.
        self.assertTrue(np.isnan(jan2["target_return"][0]))
        # The next-day 11:59 close still supplies a held-position markout.
        self.assertAlmostEqual(
            jan2["fallback_markout_return"][0],
            103.0 / 102.0 - 1.0,
        )
        # Signal data remain complete, so Jan 4 uses Jan 3 / Jan 2 exactly.
        self.assertAlmostEqual(jan4["ret_1_lag"][0], 103.0 / 101.0 - 1.0)

    def test_mlp_refit_uses_calendar_month_boundary(self) -> None:
        symbols = [f"S{idx:03d}" for idx in range(120)]
        rows: list[dict[str, object]] = []
        for panel_date, target_end in (
            (date(2026, 2, 27), date(2026, 2, 28)),
            (date(2026, 3, 2), date(2026, 3, 3)),
        ):
            for idx, symbol in enumerate(symbols):
                row: dict[str, object] = {
                    "date": panel_date,
                    "symbol": symbol,
                    "target_return": 0.001 * ((idx % 7) - 3),
                    "target_end_date": target_end,
                    "adv_rank": idx + 1,
                }
                row.update({name: 0.01 * (feature_idx + 1) + idx / 10_000 for feature_idx, name in enumerate(MLP_FEATURES)})
                rows.append(row)
        cfg = ExperimentConfig(
            eval_start=date(2026, 3, 2),
            eval_end=date(2026, 3, 2),
            mlp_epochs=1,
        )
        _, fit_log = add_walk_forward_mlp_scores(pl.DataFrame(rows), cfg)
        self.assertEqual(fit_log.height, 1)
        self.assertEqual(fit_log["fit_date"][0], date(2026, 3, 1))
        self.assertEqual(fit_log["train_end"][0], date(2026, 2, 27))
        self.assertEqual(fit_log["status"][0], "fitted")

    def test_equation17_pairwise_identity(self) -> None:
        rows = []
        for strategy_idx, strategy in enumerate(("ts_momentum", "cs_momentum", "mean_reversion", "vol_scaled", "mlp")):
            gross = 1000.0 + strategy_idx * 10.0
            for treatment, cost in (
                ("fixed_5bp", 100.0),
                ("fixed_10bp", 200.0),
                ("fixed_20bp", 400.0),
                ("slipclock", 130.0 + strategy_idx),
            ):
                rows.append(
                    {
                        "date": date(2025, 1, 1),
                        "strategy": strategy,
                        "treatment": treatment,
                        "submitted_gross_usd": 200_000.0,
                        "gross_pnl_usd": gross,
                        "cost_usd": cost,
                        "net_pnl_usd": gross - cost,
                    }
                )
        residuals, _ = equation17_residuals(pl.DataFrame(rows))
        result = verify_equation17_pairwise(residuals)
        self.assertEqual(result["checks"], 30)
        self.assertLessEqual(result["max_abs_error_usd"], 1e-8)

    def test_pairwise_reversal_counter(self) -> None:
        strategies = ("ts_momentum", "cs_momentum", "mean_reversion", "vol_scaled", "mlp")
        treatments = ("no_cost", "fixed_5bp", "fixed_10bp", "fixed_20bp", "half_spread", "slipclock")
        rows = []
        for treatment in treatments:
            ordered = list(strategies)
            if treatment == "slipclock":
                ordered[0], ordered[1] = ordered[1], ordered[0]
            values = {strategy: 5.0 - idx for idx, strategy in enumerate(ordered)}
            rows.extend(
                {"strategy": strategy, "treatment": treatment, "net_sharpe": values[strategy]}
                for strategy in strategies
            )
        matrix = pairwise_reversal_matrix(pl.DataFrame(rows), "net_sharpe")
        cell = matrix.filter(
            (pl.col("treatment_a") == "no_cost") & (pl.col("treatment_b") == "slipclock")
        )
        self.assertEqual(cell["rank_reversals"][0], 1)

    def test_entry_cost_values_do_not_change_executable_order_ids(self) -> None:
        cfg = ExperimentConfig(eval_start=date(2025, 1, 1), eval_end=date(2025, 1, 1))
        panel = self._order_selection_panel()
        baseline, _ = generate_frozen_orders(panel, cfg)
        mutated_panel = panel.with_columns(
            [
                pl.when(pl.col("symbol") == "S00")
                .then(999.0)
                .otherwise(pl.col("buy_100000_mid_cost_bps"))
                .alias("buy_100000_mid_cost_bps"),
                pl.when(pl.col("symbol") == "S00")
                .then(999.0)
                .otherwise(pl.col("sell_100000_mid_cost_bps"))
                .alias("sell_100000_mid_cost_bps"),
            ]
        )
        mutated, _ = generate_frozen_orders(mutated_panel, cfg)
        self.assertEqual(
            baseline.select("order_id", "side").rows(),
            mutated.select("order_id", "side").rows(),
        )

    def test_unavailable_candidate_is_replaced_before_order_freeze(self) -> None:
        cfg = ExperimentConfig(eval_start=date(2025, 1, 1), eval_end=date(2025, 1, 1))
        panel = self._order_selection_panel().with_columns(
            [
                pl.when(pl.col("symbol") == "S00")
                .then(None)
                .otherwise(pl.col("mid_price"))
                .alias("mid_price"),
                pl.when(pl.col("symbol") == "S00")
                .then(False)
                .otherwise(pl.col("valid_bbo"))
                .alias("valid_bbo"),
            ]
        )
        submitted, generation_log = generate_frozen_orders(panel, cfg)
        ts_symbols = submitted.filter(pl.col("strategy") == "ts_momentum").get_column("symbol").to_list()
        cs_symbols = submitted.filter(pl.col("strategy") == "cs_momentum").get_column("symbol").to_list()
        self.assertNotIn("S00", ts_symbols)
        self.assertIn("S20", ts_symbols)
        self.assertNotIn("S00", cs_symbols)
        self.assertIn("S10", cs_symbols)
        self.assertGreater(generation_log.get_column("replaced_signal_only_orders").sum(), 0)

        priced = attach_execution_state(submitted, panel)
        self.assertTrue(priced.get_column("order_valid").all())

        retained, date_status = retain_common_valid_dates(priced, cfg)
        self.assertEqual(retained.height, len(("ts_momentum", "cs_momentum", "mean_reversion", "vol_scaled", "mlp")) * 20)
        self.assertEqual(date_status.height, 1)
        self.assertTrue(date_status.get_column("retained")[0])

    def test_next_day_l2_gap_uses_kline_markout_without_replacement(self) -> None:
        cfg = ExperimentConfig(eval_start=date(2025, 1, 1), eval_end=date(2025, 1, 1))
        panel = self._order_selection_panel().with_columns(
            pl.when(pl.col("symbol") == "S00")
            .then(float("nan"))
            .otherwise(pl.col("target_return"))
            .alias("target_return")
        )
        submitted, _ = generate_frozen_orders(panel, cfg)
        ts_symbols = submitted.filter(pl.col("strategy") == "ts_momentum").get_column("symbol").to_list()
        cs_symbols = submitted.filter(pl.col("strategy") == "cs_momentum").get_column("symbol").to_list()
        self.assertIn("S00", ts_symbols)
        self.assertIn("S00", cs_symbols)

        priced = attach_execution_state(submitted, panel)
        retained, date_status = retain_common_valid_dates(priced, cfg)
        fallback_rows = priced.filter(pl.col("symbol") == "S00")
        self.assertTrue(fallback_rows.get_column("order_valid").all())
        self.assertTrue(
            fallback_rows.select((pl.col("markout_source") == "kline_1159").all()).item()
        )
        self.assertFalse(retained.is_empty())
        self.assertTrue(date_status.get_column("retained")[0])

    def test_missing_all_markouts_excludes_order_and_renormalizes(self) -> None:
        cfg = ExperimentConfig(eval_start=date(2025, 1, 1), eval_end=date(2025, 1, 1))
        panel = self._order_selection_panel().with_columns(
            [
                pl.when(pl.col("symbol") == "S00")
                .then(float("nan"))
                .otherwise(pl.col("target_return"))
                .alias("target_return"),
                pl.when(pl.col("symbol") == "S00")
                .then(float("nan"))
                .otherwise(pl.col("fallback_markout_return"))
                .alias("fallback_markout_return"),
            ]
        )
        submitted, _ = generate_frozen_orders(panel, cfg)
        submitted_ids = set(submitted.get_column("order_id").to_list())
        self.assertTrue(any(order_id.endswith("|S00") for order_id in submitted_ids))

        priced = attach_execution_state(submitted, panel)
        retained, date_status = retain_common_valid_dates(priced, cfg)
        self.assertTrue(date_status.get_column("retained")[0])
        self.assertFalse(any(order_id.endswith("|S00") for order_id in retained["order_id"]))

        _, _, daily = apply_cost_treatments(retained, cfg)
        self.assertTrue(
            daily.select(
                (pl.col("submitted_gross_usd") - cfg.submitted_gross_usd)
                .abs()
                .le(1e-8)
                .all()
            ).item()
        )

    def test_extreme_contract_return_is_audited_and_zeroed(self) -> None:
        cfg = ExperimentConfig(
            eval_start=date(2025, 1, 1),
            eval_end=date(2025, 1, 1),
        )
        panel = self._order_selection_panel().with_columns(
            pl.when(pl.col("symbol") == "S00")
            .then(1.5)
            .otherwise(pl.col("target_return"))
            .alias("target_return")
        )
        submitted, _ = generate_frozen_orders(panel, cfg)
        priced = attach_execution_state(submitted, panel, cfg)
        flagged = priced.filter(pl.col("symbol") == "S00")
        self.assertGreater(flagged.height, 0)
        self.assertTrue(flagged.get_column("return_outlier").all())
        self.assertTrue(
            flagged.select((pl.col("raw_evaluation_return") == 1.5).all()).item()
        )
        self.assertTrue(
            flagged.select((pl.col("evaluation_return") == 0.0).all()).item()
        )
        self.assertTrue(flagged.get_column("order_valid").all())

    def test_missing_entry_date_carries_positions_with_binance_marks(self) -> None:
        cfg = ExperimentConfig(
            eval_start=date(2025, 1, 1),
            eval_end=date(2025, 1, 3),
            orders_per_strategy_day=2,
        )
        frozen = pl.DataFrame(
            [
                {
                    "order_id": f"{current_date}|{symbol}",
                    "date": current_date,
                    "strategy": "ts_momentum",
                    "symbol": symbol,
                    "side": 1,
                    "notional_usd": 100_000.0,
                    "evaluation_notional_usd": 100_000.0,
                    "mid_price": 50.0 if symbol == "X" else 100.0,
                    "evaluation_return": 0.0,
                    "spread_bps": 2.0,
                    "slipclock_cost_bps": 3.0,
                }
                for current_date in (date(2025, 1, 1), date(2025, 1, 3))
                for symbol in ("X", "Y")
            ]
        )
        _, _, daily = apply_cost_treatments(frozen, cfg)
        signal_prices = pl.DataFrame(
            {
                "date": [
                    date(2025, 1, 2),
                    date(2025, 1, 2),
                    date(2025, 1, 3),
                    date(2025, 1, 3),
                ],
                "symbol": ["X", "Y", "X", "Y"],
                "signal_price": [100.0, 100.0, 110.0, 250.0],
            }
        )
        augmented, audit = augment_with_carried_position_days(
            daily,
            frozen,
            signal_prices,
            cfg,
        )
        carry = augmented.filter(pl.col("is_carry_day"))
        self.assertEqual(carry.get_column("date").n_unique(), 1)
        self.assertEqual(carry.height, 6)
        self.assertTrue(
            carry.select((pl.col("submitted_gross_usd") == 0.0).all()).item()
        )
        self.assertTrue(
            carry.select((pl.col("evaluation_gross_usd") == 200_000.0).all()).item()
        )
        self.assertTrue(
            carry.select((pl.col("gross_pnl_usd") - 20_000.0).abs().le(1e-8).all()).item()
        )
        y_row = audit.filter(pl.col("symbol") == "Y")
        self.assertTrue(y_row.get_column("return_outlier")[0])
        self.assertEqual(y_row.get_column("evaluation_return")[0], 0.0)


if __name__ == "__main__":
    unittest.main()
