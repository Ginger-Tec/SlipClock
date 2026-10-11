# L2 displayed-book replay engine

Rust baseline engine for the Tardis `incremental_book_L2` execution-cost experiment.

## Current scope

Implemented:

- Streaming Arrow/Parquet reader with required-column projection for `timestamp`, `local_timestamp`, `is_snapshot`, `side`, `price`, and `amount`.
- Tardis-compatible `side` dictionary-to-UTF8 cast and `decimal128(24,10)` fixed-point price/amount handling.
- Atomic grouping by identical `local_timestamp` so a snapshot/update group is applied as one state transition.
- Fixed-point `BTreeMap` order book baseline behind the `OrderBook` trait.
- Local-clock grid sampler (`1m`, `1s`, etc.) with mid/micro references and buy/sell book-walk metrics for configurable USD notionals.
- Full and compact Parquet feature output plus QC JSON output. Partition/meta fields (`exchange`, `symbol`, `date`, `grid`, engine variant, sample clock) are kept in the Hive path and QC JSON, not duplicated inside the Parquet body, so PyArrow/DuckDB dataset readers do not see partition-column type conflicts.
- `run-range` one-symbol multi-day batching that writes one Hive feature partition per UTC date.

Streaming requires accepted rows to be sorted by `local_timestamp`; it fails closed if a partition regresses, because correct as-of sampling would otherwise require an explicit normalization/sort pass.

## Build and test

```bash
cd rust/liq-l2-engine
. "$HOME/.cargo/env"
cargo fmt
cargo test -q
```

## CLI smoke example

```bash
cargo run -q -- run \
  --input ../../data/tardis/parquet/exchange=binance-futures/data_type=incremental_book_L2/symbol=ARBUSDT/date=2024-07-01 \
  --exchange binance-futures \
  --symbol ARBUSDT \
  --date 2024-07-01 \
  --grid 1m \
  --output-layout compact \
  --out ../../data/tardis/features/rust_l2_engine_smoke/exchange=binance-futures/symbol=ARBUSDT/date=2024-07-01/grid=1m/slippage.parquet \
  --qc-out ../../data/tardis/features/rust_l2_engine_smoke/exchange=binance-futures/symbol=ARBUSDT/date=2024-07-01/grid=1m/qc.json
```

Observed smoke on ARBUSDT 2024-07-01: 5,199,183 input rows, 1,439 1-minute samples, 20 record batches, 7 row groups, 0 unknown-side rows, 0 invalid price rows and 0 invalid amount rows.

## Multi-day one-symbol batch

```bash
cargo run -q -- run-range \
  --input-root ../../data/tardis/parquet/exchange=binance-futures/data_type=incremental_book_L2/symbol=ARBUSDT \
  --exchange binance-futures \
  --symbol ARBUSDT \
  --start-date 2024-07-01 \
  --end-date 2024-07-31 \
  --grid 1m \
  --engine btree \
  --output-layout compact \
  --out-root ../../data/tardis/features/rust_l2_engine_stream
```

`run-range` keeps each day as `exchange=/symbol=/date=/grid=/slippage.parquet` plus a colocated QC JSON, so workers can resume/retry date partitions independently while avoiding repeated process setup for one-symbol ranges.

## Output layouts

- `--output-layout full`: legacy/debug layout with VWAP, fill quantities, target quantity, last price, and shortfall for every walk.
- `--output-layout compact`: broad-sweep layout with BBO/spread/mid/micro/book-level fields plus walk `cost_bps`, `complete`, and `levels_used` only.

## Output policy

Generated feature/QC files under `data/` are ignored by git. Commit source code, tests, and markdown only.

## Engine variants

- `--engine btree`: default production baseline. Evidence in `docs/l2_orderbook_candidate_competition_2024-07-01.md` keeps this as the broad default.
- `--engine vec`: sorted-vector candidate retained for thin-symbol experiments only; it is output-equivalent but too slow for BTCUSDT/deep books.
