# SlipClock

Reproduction code for **SlipClock: A Conditional Displayed-Depth Cost Benchmark for Evaluating Financial AI Strategies in Crypto Futures**, accepted at ICAIF 2026.

**Authors:** Seunghan Son, Doohwi Cha, Minjae Lee, Juyeong Lee, Minsuk Sung, Donghwa Seo, and Yoontae Hwang.

Repository: https://github.com/Ginger-Tec/SlipClock

```bash
git clone https://github.com/Ginger-Tec/SlipClock.git
cd SlipClock
uv sync --frozen
uv run python reproduce.py
```

This repository contains the code and compact derived inputs needed
to reproduce the paper-facing tables and figures for a minute-level displayed
L2 execution-cost benchmark. It includes the control-order analyses and the
five-system frozen-order experiment. The source repository and its history are
not included.

## Repository layout

```text
.
├── data/derived/                 # compact control-order aggregates
├── expected/                     # frozen paper-facing verification targets
├── rewrite2/paper/               # camera-ready manuscript and asset builders
│   ├── fonts/                    # pinned publication-figure fonts and licenses
│   └── expander/.../             # frozen-order strategy-cost experiment
├── rust/liq-l2-engine/           # L2 replay and fixed-notional book walk
├── scripts/                      # raw-to-analysis pipeline components
├── reproduce.py                  # compact end-to-end reproduction
├── verify_outputs.py             # paper-facing output comparison
├── pyproject.toml
├── uv.lock
└── REPRODUCIBILITY_MANIFEST.json
```

## Data availability

Raw provider-licensed order-book archives are not redistributed. The compact
files in this repository contain derived portfolio/date/minute aggregates,
fixed strategy inputs, and public-price extracts needed for the paper-facing
rerun. They contain neither raw message streams nor credentials.

The raw pipeline expects one-minute L2 feature partitions of the form:

```text
data/features/l2_1m/
  exchange=binance-futures/symbol=<SYMBOL>/date=YYYY-MM-DD/grid=1m/slippage.parquet
```

Each feature row contains BBO/depth fields and side-specific book-walk costs,
completion flags, and levels used at USD 1,000, 10,000, and 100,000. The Rust
engine consumes ordered `incremental_book_L2` rows with exchange timestamp,
local timestamp, snapshot flag, side, price, and amount.

## Environment

Python 3.12+, `uv`, and Rust stable are required. XeLaTeX plus `latexmk` is
optional and is needed only to compile the manuscript after assets are built.

```bash
uv sync --frozen
cargo test --manifest-path rust/liq-l2-engine/Cargo.toml
```

Python and Rust dependency resolutions are locked in `uv.lock` and
`rust/liq-l2-engine/Cargo.lock`.

## Compact end-to-end reproduction

No licensed raw archive is needed for this path.

```bash
uv run python reproduce.py
```

The command reruns the frozen-order experiment, computes entry-date target-vector
turnover, reprices the same orders at the three reported UTC clocks, rebuilds the
control-order tables and all manuscript figures, and compares paper-facing
outputs with the frozen targets in `expected/`.

Expected headline checks include:

- held-out 60-minute high-minus-low costs of 0.874, 2.309, and 3.477 bp for
  PIT top-25, top-100, and top-200 portfolios;
- USD 100,000 date-level mean costs of 7.215 and 33.555 bp for top-25 and
  top-200 portfolios;
- 46,600 frozen orders with strategy mean costs spanning 6.709--56.111 bp;
- mean one-way target-vector turnover spanning 11.591--79.247%, with
  cross-sectional momentum second-lowest at 20.118% despite the highest cost;
- 3 of 10 Sharpe pairs and 4 of 10 terminal-markout pairs reversed between
  no cost and the measured cost;
- strategy-specific high-minus-low clock gaps spanning 0.770--10.479 bp.

For targeted checks:

```bash
uv run pytest -q rewrite2/paper/expander/frozen_order_strategy_cost_v1_20260801/tests
uv run python verify_outputs.py
```

After reproduction, compile the camera-ready manuscript with:

```bash
cd rewrite2/paper
latexmk -r latexmkrc paper_en.tex
```

## Raw-to-analysis workflow

Researchers with independently licensed inputs can reconstruct the compact
inputs using the scripts in `scripts/`: replay L2 messages with the Rust engine,
build daily and portfolio summaries, construct prior-information ADV universes,
materialize the minute analysis mart, estimate cross-fitted clock surfaces, and
run scheduled-clock diagnostics. Every script exposes explicit input/output
arguments; `data/raw/`, `data/features/`, and `data/work/` are intentionally
untracked.

## Scope

- Cost is the expected immediate book-walk cost relative to the contemporaneous
  midpoint using displayed L2 depth.
- Clock comparisons are descriptive scheduling diagnostics, not causal effects
  or trading signals.
- The frozen-order experiment applies the same measurement rule to five fixed
  strategy systems and holds order identifiers, sides, notionals, and marks
  fixed across cost rules.
- Hidden liquidity, queue position, latency, fees, funding, replenishment,
  dynamic impact, and full implementation shortfall are outside the release.
