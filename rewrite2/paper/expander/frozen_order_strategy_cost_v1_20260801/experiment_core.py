"""Core logic for the frozen-order strategy cost comparison.

The module is deliberately dependency-light.  It uses only packages already in
the repository environment and keeps signal generation separate from cost
application so the order-vector invariance can be tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


STRATEGIES = (
    "ts_momentum",
    "cs_momentum",
    "mean_reversion",
    "vol_scaled",
    "mlp",
)

TREATMENTS = (
    "no_cost",
    "fixed_5bp",
    "fixed_10bp",
    "fixed_20bp",
    "half_spread",
    "slipclock",
)

MLP_FEATURES = (
    "ret_1_lag",
    "ret_5_lag",
    "ret_20_lag",
    "ret_60_lag",
    "vol_5_lag",
    "vol_20_lag",
    "vol_60_lag",
    "adv_rank_norm",
)


@dataclass(frozen=True)
class ExperimentConfig:
    eval_start: date = date(2025, 1, 1)
    eval_end: date = date(2026, 4, 29)
    order_notional_usd: float = 100_000.0
    orders_per_strategy_day: int = 20
    annualization_days: int = 365
    mlp_train_days: int = 365
    mlp_hidden_units: int = 8
    mlp_epochs: int = 150
    mlp_learning_rate: float = 0.01
    mlp_l2: float = 1e-4
    max_abs_daily_contract_return: float = 1.0
    seed: int = 20_260_801

    @property
    def submitted_gross_usd(self) -> float:
        return self.order_notional_usd * self.orders_per_strategy_day


class SimpleMLPRegressor:
    """Small deterministic one-hidden-layer MLP implemented with NumPy.

    Training statistics are stored on the fitted instance so prediction uses
    only the corresponding historical training window.
    """

    def __init__(
        self,
        *,
        hidden_units: int = 8,
        epochs: int = 150,
        learning_rate: float = 0.01,
        l2: float = 1e-4,
        seed: int = 20_260_801,
    ) -> None:
        self.hidden_units = int(hidden_units)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: float | None = None
        self.y_scale_: float | None = None
        self.params_: dict[str, np.ndarray] | None = None
        self.training_loss_: float | None = None

    @staticmethod
    def _safe_scale(values: np.ndarray, axis: int | None = None) -> np.ndarray:
        scale = np.std(values, axis=axis, ddof=0)
        return np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SimpleMLPRegressor":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if x.ndim != 2 or y.shape[0] != x.shape[0]:
            raise ValueError("x and y must have aligned two-dimensional rows")
        if x.shape[0] < 100:
            raise ValueError("at least 100 training rows are required")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training arrays must be finite")

        self.x_mean_ = np.mean(x, axis=0)
        self.x_scale_ = self._safe_scale(x, axis=0)
        xz = np.clip((x - self.x_mean_) / self.x_scale_, -8.0, 8.0)

        q01, q99 = np.quantile(y[:, 0], [0.01, 0.99])
        yw = np.clip(y[:, 0], q01, q99)
        self.y_mean_ = float(np.mean(yw))
        self.y_scale_ = float(self._safe_scale(yw))
        yz = ((yw - self.y_mean_) / self.y_scale_).reshape(-1, 1)

        rng = np.random.default_rng(self.seed)
        n_features = x.shape[1]
        w1 = rng.normal(0.0, 1.0 / np.sqrt(n_features), size=(n_features, self.hidden_units))
        b1 = np.zeros((1, self.hidden_units), dtype=np.float64)
        w2 = rng.normal(0.0, 1.0 / np.sqrt(self.hidden_units), size=(self.hidden_units, 1))
        b2 = np.zeros((1, 1), dtype=np.float64)

        params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        first = {name: np.zeros_like(value) for name, value in params.items()}
        second = {name: np.zeros_like(value) for name, value in params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        n = float(xz.shape[0])

        for step in range(1, self.epochs + 1):
            hidden = np.tanh(xz @ params["w1"] + params["b1"])
            pred = hidden @ params["w2"] + params["b2"]
            err = pred - yz

            d_pred = (2.0 / n) * err
            grads = {
                "w2": hidden.T @ d_pred + 2.0 * self.l2 * params["w2"],
                "b2": np.sum(d_pred, axis=0, keepdims=True),
            }
            d_hidden = (d_pred @ params["w2"].T) * (1.0 - hidden * hidden)
            grads["w1"] = xz.T @ d_hidden + 2.0 * self.l2 * params["w1"]
            grads["b1"] = np.sum(d_hidden, axis=0, keepdims=True)

            for name in params:
                first[name] = beta1 * first[name] + (1.0 - beta1) * grads[name]
                second[name] = beta2 * second[name] + (1.0 - beta2) * (grads[name] ** 2)
                first_hat = first[name] / (1.0 - beta1**step)
                second_hat = second[name] / (1.0 - beta2**step)
                params[name] -= self.learning_rate * first_hat / (np.sqrt(second_hat) + eps)

        hidden = np.tanh(xz @ params["w1"] + params["b1"])
        pred = hidden @ params["w2"] + params["b2"]
        self.training_loss_ = float(np.mean((pred - yz) ** 2))
        self.params_ = params
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.params_ is None or self.x_mean_ is None or self.x_scale_ is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=np.float64)
        xz = np.clip((x - self.x_mean_) / self.x_scale_, -8.0, 8.0)
        hidden = np.tanh(xz @ self.params_["w1"] + self.params_["b1"])
        yz = hidden @ self.params_["w2"] + self.params_["b2"]
        return (yz[:, 0] * float(self.y_scale_)) + float(self.y_mean_)


def build_lagged_feature_panel(
    clock: pl.DataFrame, signal_prices: pl.DataFrame, members: pl.DataFrame
) -> pl.DataFrame:
    """Join PIT states and add features on an explicit calendar-day grid.

    Signal features use 11:59 UTC kline closes.  Paper P&L uses current and
    next-calendar-day 12:00 L2 midpoints.  Keeping the two matrices separate
    prevents an L2 archive gap from being collapsed by row-position lags.
    """

    min_date = min(signal_prices.get_column("date").min(), clock.get_column("date").min())
    max_date = max(signal_prices.get_column("date").max(), clock.get_column("date").max())
    dates = [min_date + timedelta(days=offset) for offset in range((max_date - min_date).days + 1)]
    symbols = sorted(
        set(signal_prices.get_column("symbol").unique().to_list())
        | set(clock.get_column("symbol").unique().to_list())
    )
    date_to_idx = {value: idx for idx, value in enumerate(dates)}
    symbol_to_idx = {value: idx for idx, value in enumerate(symbols)}

    prices = np.full((len(symbols), len(dates)), np.nan, dtype=np.float64)
    for row in signal_prices.select("date", "symbol", "signal_price").iter_rows(named=True):
        value = row["signal_price"]
        if value is not None and np.isfinite(value) and value > 0:
            prices[symbol_to_idx[row["symbol"]], date_to_idx[row["date"]]] = float(value)

    l2_mid = np.full((len(symbols), len(dates)), np.nan, dtype=np.float64)
    for row in clock.select("date", "symbol", "mid_price").iter_rows(named=True):
        value = row["mid_price"]
        if value is not None and np.isfinite(value) and value > 0:
            l2_mid[symbol_to_idx[row["symbol"]], date_to_idx[row["date"]]] = float(value)

    log_returns = np.full_like(prices, np.nan)
    valid_pair = np.isfinite(prices[:, 1:]) & np.isfinite(prices[:, :-1])
    log_returns[:, 1:][valid_pair] = np.log(prices[:, 1:][valid_pair] / prices[:, :-1][valid_pair])

    n_symbols, n_dates = prices.shape
    ret_arrays: dict[int, np.ndarray] = {}
    for horizon in (1, 5, 20, 60):
        values = np.full((n_symbols, n_dates), np.nan, dtype=np.float64)
        for idx in range(horizon + 1, n_dates):
            end = prices[:, idx - 1]
            start = prices[:, idx - horizon - 1]
            valid = np.isfinite(end) & np.isfinite(start) & (start > 0)
            values[valid, idx] = end[valid] / start[valid] - 1.0
        ret_arrays[horizon] = values

    vol_arrays: dict[int, np.ndarray] = {}
    for horizon in (5, 20, 60):
        values = np.full((n_symbols, n_dates), np.nan, dtype=np.float64)
        for idx in range(horizon + 1, n_dates):
            window = log_returns[:, idx - horizon : idx]
            valid = np.isfinite(window).all(axis=1)
            if valid.any():
                values[valid, idx] = np.std(window[valid], axis=1, ddof=1)
        vol_arrays[horizon] = values

    target = np.full((n_symbols, n_dates), np.nan, dtype=np.float64)
    fallback_markout = np.full((n_symbols, n_dates), np.nan, dtype=np.float64)
    for idx in range(0, n_dates - 1):
        start = l2_mid[:, idx]
        end = l2_mid[:, idx + 1]
        valid = np.isfinite(start) & np.isfinite(end) & (start > 0)
        target[valid, idx] = end[valid] / start[valid] - 1.0

        fallback_end = prices[:, idx + 1]
        fallback_valid = np.isfinite(start) & np.isfinite(fallback_end) & (start > 0)
        fallback_markout[fallback_valid, idx] = (
            fallback_end[fallback_valid] / start[fallback_valid] - 1.0
        )

    current = members.join(clock, on=["date", "symbol"], how="left").sort(["date", "adv_rank", "symbol"])
    row_dates = current.get_column("date").to_list()
    row_symbols = current.get_column("symbol").to_list()
    date_idx = np.fromiter((date_to_idx[value] for value in row_dates), dtype=np.int32)
    symbol_idx = np.fromiter((symbol_to_idx[value] for value in row_symbols), dtype=np.int32)

    columns: list[pl.Series] = []
    for horizon in (1, 5, 20, 60):
        columns.append(pl.Series(f"ret_{horizon}_lag", ret_arrays[horizon][symbol_idx, date_idx]))
    for horizon in (5, 20, 60):
        columns.append(pl.Series(f"vol_{horizon}_lag", vol_arrays[horizon][symbol_idx, date_idx]))
    columns.extend(
        [
            pl.Series("target_return", target[symbol_idx, date_idx]),
            pl.Series(
                "fallback_markout_return",
                fallback_markout[symbol_idx, date_idx],
            ),
            pl.Series("target_end_date", [value + timedelta(days=1) for value in row_dates], dtype=pl.Date),
            (1.0 - 2.0 * (pl.col("adv_rank").cast(pl.Float64) - 1.0) / 99.0).alias("adv_rank_norm"),
        ]
    )
    return current.with_columns(columns)


def add_walk_forward_mlp_scores(panel: pl.DataFrame, cfg: ExperimentConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit the pre-registered MLP monthly and attach out-of-sample scores."""

    work = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias("mlp_score"))
    eval_rows = work.filter(pl.col("date").is_between(cfg.eval_start, cfg.eval_end, closed="both"))
    months = sorted({(value.year, value.month) for value in eval_rows.get_column("date").to_list()})
    logs: list[dict[str, object]] = []

    feature_finite = pl.all_horizontal([pl.col(name).is_finite() for name in MLP_FEATURES])
    for year, month in months:
        month_rows = eval_rows.filter((pl.col("date").dt.year() == year) & (pl.col("date").dt.month() == month))
        if month_rows.is_empty():
            continue
        # The refit boundary is a calendar convention, not the first available
        # L2 row. This matters when the archive misses the first day of a month.
        fit_date = date(year, month, 1)
        train_start = fit_date - timedelta(days=cfg.mlp_train_days)
        label_cutoff = fit_date - timedelta(days=1)
        train = work.filter(
            pl.col("date").is_between(train_start, fit_date - timedelta(days=2), closed="both")
            & (pl.col("target_end_date") <= label_cutoff)
            & feature_finite
            & pl.col("target_return").is_finite()
        )
        predict = month_rows.filter(feature_finite)
        if train.height < 100 or predict.is_empty():
            logs.append(
                {
                    "fit_date": fit_date,
                    "train_start": train_start,
                    "train_end": fit_date - timedelta(days=2),
                    "train_rows": train.height,
                    "prediction_rows": predict.height,
                    "status": "insufficient_data",
                    "training_loss": None,
                }
            )
            continue

        model = SimpleMLPRegressor(
            hidden_units=cfg.mlp_hidden_units,
            epochs=cfg.mlp_epochs,
            learning_rate=cfg.mlp_learning_rate,
            l2=cfg.mlp_l2,
            seed=cfg.seed,
        )
        x_train = train.select(MLP_FEATURES).to_numpy()
        y_train = train.get_column("target_return").to_numpy()
        model.fit(x_train, y_train)
        scores = model.predict(predict.select(MLP_FEATURES).to_numpy())

        score_frame = predict.select("date", "symbol").with_columns(pl.Series("score_update", scores))
        work = (
            work.join(score_frame, on=["date", "symbol"], how="left")
            .with_columns(pl.coalesce("score_update", "mlp_score").alias("mlp_score"))
            .drop("score_update")
        )
        logs.append(
            {
                "fit_date": fit_date,
                "train_start": train_start,
                "train_end": fit_date - timedelta(days=2),
                "train_rows": train.height,
                "prediction_rows": predict.height,
                "status": "fitted",
                "training_loss": model.training_loss_,
            }
        )

    return work.sort(["date", "adv_rank", "symbol"]), pl.DataFrame(logs)


def _entry_executable_expr(side: int) -> pl.Expr:
    """Return the contemporaneous $100k execution-feasibility gate for one side."""

    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    cost_column = "buy_100000_mid_cost_bps" if side == 1 else "sell_100000_mid_cost_bps"
    complete_column = "buy_100000_mid_complete" if side == 1 else "sell_100000_mid_complete"
    return (
        pl.col("valid_bbo").fill_null(False)
        & pl.col("mid_price").is_finite().fill_null(False)
        & pl.col("spread_bps").is_finite().fill_null(False)
        & pl.col(cost_column).is_finite().fill_null(False)
        & pl.col(complete_column).fill_null(False)
    )


def _rank_long_short(
    day: pl.DataFrame,
    score_col: str,
    *,
    require_entry_executable: bool = True,
) -> pl.DataFrame:
    finite = day.filter(pl.col(score_col).is_finite())
    if finite.height < 20:
        return pl.DataFrame()
    long_candidates = (
        finite.filter(pl.col("buy_entry_executable"))
        if require_entry_executable
        else finite
    )
    short_candidates = (
        finite.filter(pl.col("sell_entry_executable"))
        if require_entry_executable
        else finite
    )
    longs = (
        long_candidates.sort([score_col, "symbol"], descending=[True, False])
        .head(10)
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("side"))
    )
    shorts = (
        short_candidates.sort([score_col, "symbol"], descending=[False, False])
        .head(10)
        .with_columns(pl.lit(-1, dtype=pl.Int8).alias("side"))
    )
    if longs.height != 10 or shorts.height != 10:
        return pl.DataFrame()
    selected = pl.concat([longs, shorts], how="vertical")
    if selected.get_column("symbol").n_unique() != 20:
        return pl.DataFrame()
    return selected


def generate_frozen_orders(panel: pl.DataFrame, cfg: ExperimentConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate five executable order vectors, then freeze them before costing.

    Signals and ranks use only PIT membership and lagged features. At the fixed
    submission time, candidates without a valid BBO or enough displayed depth
    for the relevant $100k side are skipped in favor of the next signal-ranked
    contract. The numeric cost never enters the ranking, and next-day markout
    availability is deliberately excluded from the gate. Once selected, the
    order vector is shared unchanged by all cost treatments.
    """

    selection_columns = (
        "date",
        "symbol",
        "adv_rank",
        "ret_1_lag",
        "ret_20_lag",
        "vol_20_lag",
        "mlp_score",
        "mid_price",
        "spread_bps",
        "buy_100000_mid_cost_bps",
        "sell_100000_mid_cost_bps",
        "buy_100000_mid_complete",
        "sell_100000_mid_complete",
        "valid_bbo",
    )
    eval_panel = panel.select(selection_columns).with_columns(
        _entry_executable_expr(1).alias("buy_entry_executable"),
        _entry_executable_expr(-1).alias("sell_entry_executable"),
    ).filter(
        pl.col("date").is_between(cfg.eval_start, cfg.eval_end, closed="both")
    )
    dates = sorted(eval_panel.get_column("date").unique().to_list())
    orders: list[pl.DataFrame] = []
    generation_log: list[dict[str, object]] = []

    for current_date in dates:
        day = eval_panel.filter(pl.col("date") == current_date)

        definitions: list[tuple[str, pl.DataFrame, pl.DataFrame, str]] = []
        ts_candidates = day.filter(pl.col("ret_20_lag").is_finite()).with_columns(
            pl.when(pl.col("ret_20_lag") >= 0.0)
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(-1, dtype=pl.Int8))
            .alias("side")
        )
        ts_signal_only = ts_candidates.sort(["adv_rank", "symbol"]).head(20)
        ts = (
            ts_candidates.with_columns(
                pl.when(pl.col("side") == 1)
                .then(pl.col("buy_entry_executable"))
                .otherwise(pl.col("sell_entry_executable"))
                .alias("entry_executable")
            )
            .filter(pl.col("entry_executable"))
            .sort(["adv_rank", "symbol"])
            .head(20)
        )
        definitions.append(("ts_momentum", ts, ts_signal_only, "ret_20_lag"))

        definitions.append(
            (
                "cs_momentum",
                _rank_long_short(day, "ret_20_lag"),
                _rank_long_short(day, "ret_20_lag", require_entry_executable=False),
                "ret_20_lag",
            )
        )

        mean_rev_day = day.with_columns((-pl.col("ret_1_lag")).alias("mean_reversion_score"))
        definitions.append(
            (
                "mean_reversion",
                _rank_long_short(mean_rev_day, "mean_reversion_score"),
                _rank_long_short(
                    mean_rev_day,
                    "mean_reversion_score",
                    require_entry_executable=False,
                ),
                "mean_reversion_score",
            )
        )

        vol_day = day.with_columns(
            pl.when(pl.col("vol_20_lag").is_finite() & (pl.col("vol_20_lag") > 1e-12))
            .then(pl.col("ret_20_lag") / pl.col("vol_20_lag"))
            .otherwise(None)
            .alias("vol_scaled_score")
        )
        definitions.append(
            (
                "vol_scaled",
                _rank_long_short(vol_day, "vol_scaled_score"),
                _rank_long_short(
                    vol_day,
                    "vol_scaled_score",
                    require_entry_executable=False,
                ),
                "vol_scaled_score",
            )
        )
        definitions.append(
            (
                "mlp",
                _rank_long_short(day, "mlp_score"),
                _rank_long_short(day, "mlp_score", require_entry_executable=False),
                "mlp_score",
            )
        )

        for strategy, selected, signal_only, score_col in definitions:
            signal_only_symbols = (
                set(signal_only.get_column("symbol").to_list())
                if not signal_only.is_empty()
                else set()
            )
            selected_symbols = (
                set(selected.get_column("symbol").to_list()) if not selected.is_empty() else set()
            )
            replacement_count = len(signal_only_symbols - selected_symbols)
            if selected.height != cfg.orders_per_strategy_day:
                generation_log.append(
                    {
                        "date": current_date,
                        "strategy": strategy,
                        "status": "generation_failed",
                        "order_count": selected.height,
                        "signal_only_order_count": signal_only.height,
                        "replaced_signal_only_orders": replacement_count,
                    }
                )
                continue
            frame = selected.with_columns(
                [
                    pl.lit(strategy).alias("strategy"),
                    pl.col(score_col).cast(pl.Float64).alias("signal_score"),
                    pl.lit(cfg.order_notional_usd).alias("notional_usd"),
                    pl.format("{}|{}|{}", pl.lit(strategy), pl.col("date"), pl.col("symbol")).alias("order_id"),
                    pl.col("date")
                    .cast(pl.Datetime("us"))
                    .dt.offset_by("12h")
                    .dt.replace_time_zone("UTC")
                    .alias("timestamp_utc"),
                ]
            ).select(
                "order_id",
                "date",
                "timestamp_utc",
                "strategy",
                "symbol",
                "side",
                "notional_usd",
                "signal_score",
                "adv_rank",
            )
            orders.append(frame)
            generation_log.append(
                {
                    "date": current_date,
                    "strategy": strategy,
                    "status": "generated",
                    "order_count": selected.height,
                    "signal_only_order_count": signal_only.height,
                    "replaced_signal_only_orders": replacement_count,
                }
            )

    if not orders:
        raise RuntimeError("no order vectors were generated")
    frozen = pl.concat(orders, how="vertical").sort(["date", "strategy", "symbol"])
    return frozen, pl.DataFrame(generation_log).sort(["date", "strategy"])


def attach_execution_state(
    frozen: pl.DataFrame,
    panel: pl.DataFrame,
    cfg: ExperimentConfig | None = None,
) -> pl.DataFrame:
    """Attach L2 entry/markout/cost support after order-vector construction.

    This is a left join by the already frozen ``date``/``symbol`` keys. Entry
    feasibility has already been applied during candidate selection. A missing
    next-day L2 midpoint is filled from the same contract's next-day
    11:59 kline mark. If neither mark exists, the order remains unevaluable and
    is removed only after the already-frozen vector has been serialized; it can
    never trigger a hindsight replacement.
    """

    cfg = cfg or ExperimentConfig()
    state_columns = (
        "date",
        "symbol",
        "mid_price",
        "target_return",
        "target_end_date",
        "spread_bps",
        "buy_100000_mid_cost_bps",
        "sell_100000_mid_cost_bps",
        "buy_100000_mid_complete",
        "sell_100000_mid_complete",
        "valid_bbo",
    )
    if "fallback_markout_return" in panel.columns:
        state = panel.select(*state_columns, "fallback_markout_return")
    else:
        state = panel.select(state_columns).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("fallback_markout_return")
        )
    if state.select(pl.struct("date", "symbol").n_unique()).item() != state.height:
        raise AssertionError("execution-state panel must be unique by date and symbol")

    priced = frozen.join(state, on=["date", "symbol"], how="left", validate="m:1")
    priced = priced.with_columns(
        [
            pl.when(pl.col("side") == 1)
            .then(pl.col("buy_100000_mid_complete").fill_null(False))
            .otherwise(pl.col("sell_100000_mid_complete").fill_null(False))
            .alias("relevant_side_complete"),
            pl.when(pl.col("side") == 1)
            .then(pl.col("buy_100000_mid_cost_bps"))
            .otherwise(pl.col("sell_100000_mid_cost_bps"))
            .alias("slipclock_cost_bps"),
            (
                pl.col("valid_bbo").fill_null(False)
                & pl.col("mid_price").is_finite().fill_null(False)
            ).alias("matched_entry_state_valid"),
            pl.when(pl.col("target_return").is_finite().fill_null(False))
            .then(pl.col("target_return"))
            .otherwise(pl.col("fallback_markout_return"))
            .alias("raw_evaluation_return"),
            pl.when(pl.col("target_return").is_finite().fill_null(False))
            .then(pl.lit("l2_mid"))
            .when(pl.col("fallback_markout_return").is_finite().fill_null(False))
            .then(pl.lit("kline_1159"))
            .otherwise(pl.lit("missing"))
            .alias("markout_source"),
        ]
    ).with_columns(
        [
            (
                pl.col("raw_evaluation_return").is_finite().fill_null(False)
                & (
                    pl.col("raw_evaluation_return").abs()
                    > cfg.max_abs_daily_contract_return
                )
            ).alias("return_outlier"),
            pl.when(
                pl.col("raw_evaluation_return").is_finite().fill_null(False)
                & (
                    pl.col("raw_evaluation_return").abs()
                    > cfg.max_abs_daily_contract_return
                )
            )
            .then(pl.lit(0.0))
            .otherwise(pl.col("raw_evaluation_return"))
            .alias("evaluation_return"),
        ]
    ).with_columns(
        (
            pl.col("matched_entry_state_valid")
            & pl.col("evaluation_return").is_finite().fill_null(False)
            & pl.col("spread_bps").is_finite().fill_null(False)
            & pl.col("slipclock_cost_bps").is_finite().fill_null(False)
            & pl.col("relevant_side_complete")
        ).alias("order_valid")
    )
    return priced.sort(["date", "strategy", "symbol"])


def retain_common_valid_dates(frozen: pl.DataFrame, cfg: ExperimentConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep common evaluable dates and renormalize after irrecoverable mark gaps.

    Entry-invalid orders still invalidate the strategy-date. A frozen order
    whose L2 and kline markouts are both unavailable is excluded without
    replacement. Remaining orders are equal-weighted back to the predeclared
    strategy gross so cross-strategy daily returns remain comparable.
    """

    by_strategy = frozen.group_by("date", "strategy").agg(
        pl.len().alias("order_count"),
        pl.col("order_valid").sum().alias("evaluable_order_count"),
        (
            pl.col("matched_entry_state_valid")
            & pl.col("spread_bps").is_finite().fill_null(False)
            & pl.col("slipclock_cost_bps").is_finite().fill_null(False)
            & pl.col("relevant_side_complete")
        ).all().alias("all_entry_cost_valid"),
        pl.col("symbol").n_unique().alias("distinct_symbols"),
        pl.col("notional_usd").sum().alias("submitted_gross_usd"),
    )
    date_status = by_strategy.group_by("date").agg(
        pl.len().alias("strategy_count"),
        (pl.col("order_count") == cfg.orders_per_strategy_day).all().alias("all_counts_valid"),
        (pl.col("evaluable_order_count") > 0).all().alias("all_cost_markout_valid"),
        pl.col("evaluable_order_count").min().alias("min_evaluable_orders"),
        pl.col("all_entry_cost_valid").all().alias("all_entry_cost_valid"),
        (pl.col("distinct_symbols") == cfg.orders_per_strategy_day).all().alias("all_distinct"),
        (pl.col("submitted_gross_usd") == cfg.submitted_gross_usd).all().alias("all_gross_valid"),
    ).with_columns(
        (
            (pl.col("strategy_count") == len(STRATEGIES))
            & pl.col("all_counts_valid")
            & pl.col("all_cost_markout_valid")
            & pl.col("all_entry_cost_valid")
            & pl.col("all_distinct")
            & pl.col("all_gross_valid")
        ).alias("retained")
    ).sort("date")
    valid_dates = date_status.filter(pl.col("retained")).get_column("date")
    retained = (
        frozen.filter(
            pl.col("date").is_in(valid_dates.implode()) & pl.col("order_valid")
        )
        .with_columns(
            pl.len().over("date", "strategy").alias("evaluated_order_count")
        )
        .with_columns(
            (
                pl.col("notional_usd")
                * cfg.orders_per_strategy_day
                / pl.col("evaluated_order_count")
            ).alias("evaluation_notional_usd")
        )
        .sort(["date", "strategy", "symbol"])
    )
    return retained, date_status


def apply_cost_treatments(
    frozen: pl.DataFrame, cfg: ExperimentConfig
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply all six treatments after the order vector has been frozen."""

    return_column = "evaluation_return" if "evaluation_return" in frozen.columns else "target_return"
    notional_column = (
        "evaluation_notional_usd"
        if "evaluation_notional_usd" in frozen.columns
        else "notional_usd"
    )
    orders = frozen.with_columns(
        [
            (pl.col("side").cast(pl.Float64) * pl.col(notional_column) * pl.col(return_column)).alias(
                "gross_pnl_usd"
            ),
            pl.col(notional_column).alias("evaluated_notional_usd"),
            pl.lit(0.0).alias("cost_no_cost_usd"),
            (pl.col(notional_column) * 5.0 / 10_000.0).alias("cost_fixed_5bp_usd"),
            (pl.col(notional_column) * 10.0 / 10_000.0).alias("cost_fixed_10bp_usd"),
            (pl.col(notional_column) * 20.0 / 10_000.0).alias("cost_fixed_20bp_usd"),
            (pl.col(notional_column) * pl.col("spread_bps") / 2.0 / 10_000.0).alias(
                "cost_half_spread_usd"
            ),
            (pl.col(notional_column) * pl.col("slipclock_cost_bps") / 10_000.0).alias(
                "cost_slipclock_usd"
            ),
        ]
    )

    cost_columns = {
        "no_cost": "cost_no_cost_usd",
        "fixed_5bp": "cost_fixed_5bp_usd",
        "fixed_10bp": "cost_fixed_10bp_usd",
        "fixed_20bp": "cost_fixed_20bp_usd",
        "half_spread": "cost_half_spread_usd",
        "slipclock": "cost_slipclock_usd",
    }
    long_frames: list[pl.DataFrame] = []
    for treatment, cost_col in cost_columns.items():
        long_frames.append(
            orders.select(
                "order_id",
                "date",
                "strategy",
                "symbol",
                "side",
                "notional_usd",
                "evaluated_notional_usd",
                "gross_pnl_usd",
                pl.lit(treatment).alias("treatment"),
                pl.col(cost_col).alias("cost_usd"),
                (pl.col("gross_pnl_usd") - pl.col(cost_col)).alias("net_pnl_usd"),
            )
        )
    order_long = pl.concat(long_frames, how="vertical").sort(
        ["date", "strategy", "treatment", "symbol"]
    )
    daily = (
        order_long.group_by("date", "strategy", "treatment")
        .agg(
            pl.len().alias("order_count"),
            pl.col("evaluated_notional_usd").sum().alias("submitted_gross_usd"),
            pl.col("gross_pnl_usd").sum().alias("gross_pnl_usd"),
            pl.col("cost_usd").sum().alias("cost_usd"),
            pl.col("net_pnl_usd").sum().alias("net_pnl_usd"),
        )
        .with_columns(
            [
                pl.col("submitted_gross_usd").alias("evaluation_gross_usd"),
                pl.lit(False).alias("is_carry_day"),
            ]
        )
        .with_columns(
            (pl.col("net_pnl_usd") / pl.col("evaluation_gross_usd")).alias(
                "net_return"
            )
        )
        .sort(["strategy", "treatment", "date"])
        .with_columns(pl.col("net_pnl_usd").cum_sum().over("strategy", "treatment").alias("cumulative_net_pnl_usd"))
    )
    return orders.sort(["date", "strategy", "symbol"]), order_long, daily


def augment_with_carried_position_days(
    daily: pl.DataFrame,
    frozen: pl.DataFrame,
    signal_prices: pl.DataFrame,
    cfg: ExperimentConfig,
    carry_dates: Iterable[date] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate missing entry dates by carrying the latest frozen vector.

    Carry days reuse the latest contracts, sides, and entry quantities, Binance
    11:59 close-to-close marks, and no new entry cost. Missing price pairs
    exclude the affected contract and renormalize the remaining marked tickets
    to the fixed evaluation gross. Extreme returns follow the same zero-return
    rule as ordinary one-day markouts.
    """

    if daily.is_empty() or frozen.is_empty():
        return daily, pl.DataFrame()

    base_dates = set(daily.get_column("date").unique().to_list())
    frozen_dates = sorted(frozen.get_column("date").unique().to_list())
    strategies = sorted(daily.get_column("strategy").unique().to_list())
    candidate_dates = (
        [
            cfg.eval_start + timedelta(days=offset)
            for offset in range((cfg.eval_end - cfg.eval_start).days + 1)
        ]
        if carry_dates is None
        else sorted(set(carry_dates))
    )
    missing_dates = [
        value
        for value in candidate_dates
        if cfg.eval_start <= value <= cfg.eval_end and value not in base_dates
    ]
    if not missing_dates:
        return daily, pl.DataFrame()

    required_price_dates = set(missing_dates) | {
        value + timedelta(days=1) for value in missing_dates
    }
    price_lookup = {
        (row["date"], row["symbol"]): float(row["signal_price"])
        for row in signal_prices.filter(
            pl.col("date").is_in(sorted(required_price_dates))
        )
        .select("date", "symbol", "signal_price")
        .iter_rows(named=True)
        if row["signal_price"] is not None
        and np.isfinite(row["signal_price"])
        and row["signal_price"] > 0
    }

    notional_column = (
        "evaluation_notional_usd"
        if "evaluation_notional_usd" in frozen.columns
        else "notional_usd"
    )
    position_blocks = {
        (block.get_column("date")[0], block.get_column("strategy")[0]): block
        for block in frozen.partition_by(["date", "strategy"], maintain_order=True)
    }
    position_dates = {
        strategy: [
            current_date
            for current_date in frozen_dates
            if (current_date, strategy) in position_blocks
        ]
        for strategy in strategies
    }
    audit_rows: list[dict[str, object]] = []
    carry_daily_rows: list[dict[str, object]] = []
    for current_date in missing_dates:
        for strategy in strategies:
            prior_dates = [
                value for value in position_dates[strategy] if value < current_date
            ]
            if not prior_dates:
                continue
            position_date = prior_dates[-1]
            positions = position_blocks[position_date, strategy]
            position_rows: list[dict[str, object]] = []
            valid_notional = 0.0
            for row in positions.iter_rows(named=True):
                start_price = price_lookup.get((current_date, row["symbol"]))
                end_price = price_lookup.get(
                    (current_date + timedelta(days=1), row["symbol"])
                )
                price_available = bool(
                    start_price is not None
                    and end_price is not None
                    and np.isfinite(start_price)
                    and np.isfinite(end_price)
                    and start_price > 0
                    and end_price > 0
                )
                raw_return = (
                    float(end_price / start_price - 1.0)
                    if price_available
                    else np.nan
                )
                return_outlier = bool(
                    price_available
                    and abs(raw_return) > cfg.max_abs_daily_contract_return
                )
                evaluation_return = 0.0 if return_outlier else raw_return
                base_notional = float(row[notional_column])
                entry_mid_price = float(row["mid_price"])
                entry_quantity = (
                    base_notional / entry_mid_price
                    if np.isfinite(entry_mid_price) and entry_mid_price > 0
                    else np.nan
                )
                quantity_available = bool(
                    price_available and np.isfinite(entry_quantity)
                )
                if quantity_available:
                    valid_notional += base_notional
                position_rows.append(
                    {
                        "date": current_date,
                        "position_entry_date": position_date,
                        "strategy": strategy,
                        "symbol": row["symbol"],
                        "side": int(row["side"]),
                        "base_notional_usd": base_notional,
                        "entry_mid_price": entry_mid_price,
                        "entry_quantity": entry_quantity,
                        "price_start_1159": start_price,
                        "price_end_1159": end_price,
                        "price_available": quantity_available,
                        "raw_evaluation_return": raw_return,
                        "return_outlier": return_outlier,
                        "evaluation_return": evaluation_return,
                    }
                )
            if valid_notional <= 0:
                continue

            gross_pnl = 0.0
            valid_orders = 0
            scale = cfg.submitted_gross_usd / valid_notional
            for row in position_rows:
                if row["price_available"]:
                    evaluated_notional = float(row["base_notional_usd"]) * scale
                    contribution = 0.0 if row["return_outlier"] else (
                        float(row["side"])
                        * float(row["entry_quantity"])
                        * scale
                        * (
                            float(row["price_end_1159"])
                            - float(row["price_start_1159"])
                        )
                    )
                    valid_orders += 1
                    gross_pnl += contribution
                else:
                    evaluated_notional = 0.0
                    contribution = 0.0
                audit_rows.append(
                    {
                        **row,
                        "evaluated_notional_usd": evaluated_notional,
                        "gross_pnl_usd": contribution,
                    }
                )

            for treatment in TREATMENTS:
                carry_daily_rows.append(
                    {
                        "date": current_date,
                        "strategy": strategy,
                        "treatment": treatment,
                        "order_count": valid_orders,
                        "submitted_gross_usd": 0.0,
                        "evaluation_gross_usd": cfg.submitted_gross_usd,
                        "gross_pnl_usd": gross_pnl,
                        "cost_usd": 0.0,
                        "net_pnl_usd": gross_pnl,
                        "net_return": gross_pnl / cfg.submitted_gross_usd,
                        "is_carry_day": True,
                    }
                )

    carry_audit = (
        pl.DataFrame(audit_rows).sort(["date", "strategy", "symbol"])
        if audit_rows
        else pl.DataFrame()
    )
    if not carry_daily_rows:
        return daily, carry_audit

    normal = daily.drop("cumulative_net_pnl_usd")
    carried = pl.DataFrame(carry_daily_rows)
    augmented = (
        pl.concat([normal, carried], how="diagonal_relaxed")
        .sort(["strategy", "treatment", "date"])
        .with_columns(
            pl.col("net_pnl_usd")
            .cum_sum()
            .over("strategy", "treatment")
            .alias("cumulative_net_pnl_usd")
        )
    )
    return augmented, carry_audit


def summarize_metrics(daily: pl.DataFrame, cfg: ExperimentConfig) -> pl.DataFrame:
    metrics = (
        daily.group_by("strategy", "treatment")
        .agg(
            pl.len().alias("n_days"),
            pl.col("is_carry_day").sum().alias("n_carry_days"),
            pl.col("submitted_gross_usd").sum().alias(
                "cumulative_submitted_gross_usd"
            ),
            pl.col("evaluation_gross_usd").sum().alias(
                "cumulative_evaluation_gross_usd"
            ),
            pl.col("gross_pnl_usd").sum().alias("cumulative_gross_pnl_usd"),
            pl.col("cost_usd").sum().alias("cumulative_cost_usd"),
            pl.col("net_pnl_usd").sum().alias("cumulative_net_pnl_usd"),
            pl.col("net_return").mean().alias("mean_daily_net_return"),
            pl.col("net_return").std(ddof=1).alias("sd_daily_net_return"),
        )
        .with_columns(
            (
                pl.col("mean_daily_net_return")
                / pl.col("sd_daily_net_return")
                * np.sqrt(cfg.annualization_days)
            ).alias("net_sharpe")
        )
    )

    treatment_order = {value: idx for idx, value in enumerate(TREATMENTS)}
    rows: list[dict[str, object]] = []
    for treatment in TREATMENTS:
        block = metrics.filter(pl.col("treatment") == treatment)
        sharpe_order = block.sort(["net_sharpe", "strategy"], descending=[True, False])
        pnl_order = block.sort(["cumulative_net_pnl_usd", "strategy"], descending=[True, False])
        sharpe_rank = {name: idx + 1 for idx, name in enumerate(sharpe_order.get_column("strategy").to_list())}
        pnl_rank = {name: idx + 1 for idx, name in enumerate(pnl_order.get_column("strategy").to_list())}
        for row in block.iter_rows(named=True):
            row["sharpe_rank"] = sharpe_rank[row["strategy"]]
            row["pnl_rank"] = pnl_rank[row["strategy"]]
            row["treatment_order"] = treatment_order[treatment]
            rows.append(row)
    return pl.DataFrame(rows).sort(["treatment_order", "sharpe_rank"]).drop("treatment_order")


def pairwise_reversal_matrix(metrics: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Count strict pairwise order reversals between every treatment pair."""

    values = {
        treatment: {
            row["strategy"]: float(row[metric])
            for row in metrics.filter(pl.col("treatment") == treatment).iter_rows(named=True)
        }
        for treatment in TREATMENTS
    }
    rows: list[dict[str, object]] = []
    pairs = list(combinations(STRATEGIES, 2))
    for left in TREATMENTS:
        for right in TREATMENTS:
            reversals = 0
            ties = 0
            for strategy_a, strategy_b in pairs:
                diff_left = values[left][strategy_a] - values[left][strategy_b]
                diff_right = values[right][strategy_a] - values[right][strategy_b]
                sign_left = int(np.sign(diff_left))
                sign_right = int(np.sign(diff_right))
                if sign_left == 0 or sign_right == 0:
                    ties += 1
                elif sign_left != sign_right:
                    reversals += 1
            rows.append(
                {
                    "metric": metric,
                    "treatment_a": left,
                    "treatment_b": right,
                    "strategy_pairs": len(pairs),
                    "rank_reversals": reversals,
                    "ties": ties,
                }
            )
    return pl.DataFrame(rows)


def equation17_residuals(daily: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    base = daily.filter(pl.col("submitted_gross_usd") > 0).select(
        "date", "strategy", "treatment", "submitted_gross_usd", "gross_pnl_usd", "cost_usd", "net_pnl_usd"
    )
    sc = base.filter(pl.col("treatment") == "slipclock").rename(
        {"cost_usd": "slipclock_cost_usd", "net_pnl_usd": "slipclock_net_pnl_usd"}
    ).drop("treatment")
    rows: list[pl.DataFrame] = []
    for bps, treatment in ((5.0, "fixed_5bp"), (10.0, "fixed_10bp"), (20.0, "fixed_20bp")):
        fixed = base.filter(pl.col("treatment") == treatment).select(
            "date", "strategy", pl.col("cost_usd").alias("constant_cost_usd"), pl.col("net_pnl_usd").alias("constant_net_pnl_usd")
        )
        rows.append(
            sc.join(fixed, on=["date", "strategy"], how="inner").with_columns(
                [
                    pl.lit(bps).alias("lambda_bps"),
                    (pl.col("slipclock_cost_usd") - pl.col("constant_cost_usd")).alias("residual_usd"),
                ]
            ).with_columns(
                (pl.col("residual_usd") / pl.col("submitted_gross_usd") * 10_000.0).alias("residual_bps")
            )
        )
    residuals = pl.concat(rows, how="vertical").sort(["lambda_bps", "date", "strategy"])
    summary = residuals.group_by("strategy", "lambda_bps").agg(
        pl.len().alias("n_days"),
        pl.col("residual_usd").mean().alias("mean_residual_usd"),
        pl.col("residual_usd").median().alias("median_residual_usd"),
        pl.col("residual_usd").abs().quantile(0.90, interpolation="linear").alias("p90_abs_residual_usd"),
        pl.col("residual_usd").sum().alias("cumulative_residual_usd"),
        pl.col("residual_bps").mean().alias("mean_residual_bps"),
    ).sort(["lambda_bps", "strategy"])
    return residuals, summary


def verify_equation17_pairwise(residuals: pl.DataFrame, tolerance: float = 1e-8) -> dict[str, float | int]:
    max_error = 0.0
    checks = 0
    for bps in (5.0, 10.0, 20.0):
        block = residuals.filter(pl.col("lambda_bps") == bps)
        for current_date in block.get_column("date").unique().to_list():
            day = {row["strategy"]: row for row in block.filter(pl.col("date") == current_date).iter_rows(named=True)}
            for strategy_a, strategy_b in combinations(STRATEGIES, 2):
                a, b = day[strategy_a], day[strategy_b]
                lhs = (
                    (a["constant_net_pnl_usd"] - b["constant_net_pnl_usd"])
                    - (a["slipclock_net_pnl_usd"] - b["slipclock_net_pnl_usd"])
                )
                rhs = a["residual_usd"] - b["residual_usd"]
                max_error = max(max_error, abs(float(lhs - rhs)))
                checks += 1
    if max_error > tolerance:
        raise AssertionError(f"Equation-17 pairwise identity error {max_error} exceeds {tolerance}")
    return {"checks": checks, "max_abs_error_usd": max_error, "tolerance_usd": tolerance}


def assert_order_invariants(frozen: pl.DataFrame, cfg: ExperimentConfig) -> dict[str, int | float]:
    grouped = frozen.group_by("date", "strategy").agg(
        pl.len().alias("n"),
        pl.col("symbol").n_unique().alias("n_symbols"),
        pl.col("evaluation_notional_usd").sum().alias("evaluation_gross"),
        pl.col("order_valid").all().alias("valid"),
    )
    if not grouped.select(
        ((pl.col("n") > 0) & (pl.col("n") <= cfg.orders_per_strategy_day)).all()
    ).item():
        raise AssertionError("evaluated strategy-date order count is outside 1..20")
    if not grouped.select((pl.col("n_symbols") == pl.col("n")).all()).item():
        raise AssertionError("a strategy-date contains duplicate symbols")
    if not grouped.select(
        (pl.col("evaluation_gross") - cfg.submitted_gross_usd).abs().le(1e-8).all()
    ).item():
        raise AssertionError("renormalized evaluation gross is not constant")
    if not grouped.select(pl.col("valid").all()).item():
        raise AssertionError("retained vector contains invalid orders")
    return {
        "strategy_dates": grouped.height,
        "retained_dates": grouped.get_column("date").n_unique(),
        "orders": frozen.height,
        "orders_per_strategy_day": cfg.orders_per_strategy_day,
        "min_evaluated_orders_per_strategy_day": grouped.get_column("n").min(),
        "max_evaluated_orders_per_strategy_day": grouped.get_column("n").max(),
        "submitted_gross_usd": cfg.submitted_gross_usd,
    }


def assert_treatment_support(order_long: pl.DataFrame, frozen: pl.DataFrame) -> dict[str, int | bool]:
    """Check actual emitted treatment rows against the serialized vector."""

    frozen_ids = set(frozen.get_column("order_id").to_list())
    by_order = order_long.group_by("order_id").agg(
        pl.len().alias("rows"),
        pl.col("treatment").n_unique().alias("treatments"),
    )
    one_each = bool(
        by_order.select(
            ((pl.col("rows") == len(TREATMENTS)) & (pl.col("treatments") == len(TREATMENTS))).all()
        ).item()
    )
    exact_sets = True
    for treatment in TREATMENTS:
        emitted_ids = set(
            order_long.filter(pl.col("treatment") == treatment).get_column("order_id").to_list()
        )
        if emitted_ids != frozen_ids:
            exact_sets = False
            break
    if by_order.height != len(frozen_ids) or not one_each or not exact_sets:
        raise AssertionError("actual treatment output does not contain one row per treatment per frozen order")
    return {
        "frozen_order_ids": len(frozen_ids),
        "emitted_order_ids": by_order.height,
        "treatments_per_order": len(TREATMENTS),
        "one_row_per_treatment_per_order": one_each,
        "each_treatment_matches_frozen_ids": exact_sets,
    }
