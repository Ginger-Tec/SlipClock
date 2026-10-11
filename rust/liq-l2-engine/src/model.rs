use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Bid,
    Ask,
}

impl Side {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "bid" | "bids" | "buy" => Some(Self::Bid),
            "ask" | "asks" | "sell" => Some(Self::Ask),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl fmt::Display for TradeSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buy => write!(f, "buy"),
            Self::Sell => write!(f, "sell"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RefPrice {
    Mid,
    Micro,
}

impl fmt::Display for RefPrice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mid => write!(f, "mid"),
            Self::Micro => write!(f, "micro"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Event {
    pub exchange_timestamp_us: u64,
    pub local_timestamp_us: u64,
    pub is_snapshot: bool,
    pub side: Side,
    pub price_atoms: i128,
    pub price_scale: i32,
    pub amount: f64,
    pub source_file_index: u32,
    pub source_row_index: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventGroup {
    pub local_timestamp_us: u64,
    pub events: Vec<L2Event>,
}

impl EventGroup {
    pub fn contains_snapshot(&self) -> bool {
        self.events.iter().any(|e| e.is_snapshot)
    }

    pub fn exchange_timestamp_us(&self) -> u64 {
        self.events
            .last()
            .map(|e| e.exchange_timestamp_us)
            .unwrap_or_default()
    }

    pub fn source_row_index(&self) -> u64 {
        self.events
            .last()
            .map(|e| e.source_row_index)
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub exchange: String,
    pub symbol: String,
    pub date: String,
    pub grid: Grid,
    pub engine_variant: String,
    pub sample_clock: String,
    pub output_layout: OutputLayout,
    pub notionals: Vec<f64>,
    pub refs: Vec<RefPrice>,
    pub strict: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            exchange: "binance-futures".to_string(),
            symbol: "ARBUSDT".to_string(),
            date: "2024-07-01".to_string(),
            grid: Grid::Minutes(1),
            engine_variant: "btree".to_string(),
            sample_clock: "local".to_string(),
            output_layout: OutputLayout::Full,
            notionals: vec![1_000.0, 10_000.0, 100_000.0],
            refs: vec![RefPrice::Mid, RefPrice::Micro],
            strict: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputLayout {
    Full,
    Compact,
}

impl OutputLayout {
    pub fn parse(raw: &str) -> anyhow::Result<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "full" => Ok(Self::Full),
            "compact" => Ok(Self::Compact),
            other => anyhow::bail!("unsupported output layout {other:?}; use full or compact"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Compact => "compact",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grid {
    Seconds(u64),
    Minutes(u64),
}

const MAX_GRID_STEP_US: u64 = 86_400 * 1_000_000;

impl Grid {
    pub fn parse(raw: &str) -> anyhow::Result<Self> {
        if let Some(v) = raw.strip_suffix('s') {
            let n = parse_positive_grid(v, "seconds")?;
            let grid = Self::Seconds(n);
            grid.checked_step_us()?;
            return Ok(grid);
        }
        if let Some(v) = raw.strip_suffix('m') {
            let n = parse_positive_grid(v, "minutes")?;
            let grid = Self::Minutes(n);
            grid.checked_step_us()?;
            return Ok(grid);
        }
        anyhow::bail!("unsupported grid {raw:?}; use e.g. 1s or 1m")
    }

    pub fn name(&self) -> String {
        match self {
            Self::Seconds(n) => format!("{n}s"),
            Self::Minutes(n) => format!("{n}m"),
        }
    }

    pub fn checked_step_us(&self) -> anyhow::Result<u64> {
        match self {
            Self::Seconds(n) => checked_positive_step(*n, 1_000_000, "seconds"),
            Self::Minutes(n) => checked_positive_step(*n, 60_000_000, "minutes"),
        }
    }

    pub fn step_us(&self) -> u64 {
        self.checked_step_us()
            .expect("Grid::parse or checked_step_us should reject zero/overflow grids")
    }
}

fn parse_positive_grid(raw: &str, unit: &str) -> anyhow::Result<u64> {
    let n: u64 = raw.parse()?;
    if n == 0 {
        anyhow::bail!("grid {unit} must be positive")
    }
    Ok(n)
}

fn checked_positive_step(n: u64, multiplier: u64, unit: &str) -> anyhow::Result<u64> {
    if n == 0 {
        anyhow::bail!("grid {unit} must be positive");
    }
    let step = n
        .checked_mul(multiplier)
        .ok_or_else(|| anyhow::anyhow!("grid {unit} overflow when converted to microseconds"))?;
    if step > MAX_GRID_STEP_US {
        anyhow::bail!("grid step cannot exceed one UTC day");
    }
    Ok(step)
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Bbo {
    pub bid_price_atoms: i128,
    pub bid_price: f64,
    pub bid_amount: f64,
    pub ask_price_atoms: i128,
    pub ask_price: f64,
    pub ask_amount: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct WalkResult {
    pub target_qty: f64,
    pub filled_qty: f64,
    pub filled_notional: f64,
    pub vwap: f64,
    pub cost_bps: f64,
    pub complete: bool,
    pub levels_used: u32,
    pub last_price: f64,
    pub shortfall_qty: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WalkKey {
    pub side: TradeSide,
    pub notional_key: String,
    pub reference: RefPrice,
}

impl WalkKey {
    pub fn new(side: TradeSide, notional: f64, reference: RefPrice) -> Self {
        let notional_key = if notional.fract().abs() < f64::EPSILON {
            format!("{:.0}", notional)
        } else {
            format!("{}", notional).replace('.', "p")
        };
        Self {
            side,
            notional_key,
            reference,
        }
    }

    pub fn prefix(&self) -> String {
        format!("{}_{}_{}", self.side, self.notional_key, self.reference)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureRow {
    pub exchange: String,
    pub symbol: String,
    pub date: String,
    pub grid: String,
    pub engine_variant: String,
    pub sample_clock: String,
    pub sample_local_timestamp_us: i64,
    pub last_local_timestamp_us: i64,
    pub last_exchange_timestamp_us: i64,
    pub source_file_count: u32,
    pub source_row_index: u64,
    pub bid_price: f64,
    pub bid_amount: f64,
    pub ask_price: f64,
    pub ask_amount: f64,
    pub mid_price: f64,
    pub micro_price: f64,
    pub spread_bps: f64,
    pub bbo_imbalance: f64,
    pub book_bid_levels: u32,
    pub book_ask_levels: u32,
    pub invalid_book_flag: bool,
    pub walks: std::collections::HashMap<WalkKey, WalkResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcStats {
    pub exchange: String,
    pub symbol: String,
    pub date: String,
    pub grid: String,
    pub engine_variant: String,
    pub sample_clock: String,
    pub output_layout: String,
    pub input_files: u64,
    pub input_rows: u64,
    pub accepted_rows: u64,
    pub record_batches: u64,
    pub row_groups: u64,
    pub snapshot_groups: u64,
    pub reset_count: u64,
    pub pre_snapshot_rows_skipped: u64,
    pub local_timestamp_groups: u64,
    pub max_group_rows: u64,
    pub non_monotonic_exchange_ts_count: u64,
    pub equal_exchange_ts_group_count: u64,
    pub non_monotonic_local_ts_count: u64,
    pub crossed_book_count: u64,
    pub empty_book_count: u64,
    pub unknown_side_count: u64,
    pub invalid_price_count: u64,
    pub invalid_amount_count: u64,
    pub nondivisible_tick_count: u64,
    pub samples_emitted: u64,
    pub samples_skipped: u64,
    pub samples_skipped_pre_snapshot: u64,
    pub samples_skipped_invalid_book: u64,
    pub runtime_ms: u128,
    pub decode_ms: u128,
    pub book_update_ms: u128,
    pub sampling_ms: u128,
    pub write_ms: u128,
    pub rows_per_sec: f64,
    pub peak_rss_bytes: Option<u64>,
    pub output_path: Option<PathBuf>,
    pub output_rows: u64,
    pub output_bytes: u64,
}

impl QcStats {
    pub fn new(config: &EngineConfig) -> Self {
        Self {
            exchange: config.exchange.clone(),
            symbol: config.symbol.clone(),
            date: config.date.clone(),
            grid: config.grid.name(),
            engine_variant: config.engine_variant.clone(),
            sample_clock: config.sample_clock.clone(),
            output_layout: config.output_layout.name().to_string(),
            input_files: 0,
            input_rows: 0,
            accepted_rows: 0,
            record_batches: 0,
            row_groups: 0,
            snapshot_groups: 0,
            reset_count: 0,
            pre_snapshot_rows_skipped: 0,
            local_timestamp_groups: 0,
            max_group_rows: 0,
            non_monotonic_exchange_ts_count: 0,
            equal_exchange_ts_group_count: 0,
            non_monotonic_local_ts_count: 0,
            crossed_book_count: 0,
            empty_book_count: 0,
            unknown_side_count: 0,
            invalid_price_count: 0,
            invalid_amount_count: 0,
            nondivisible_tick_count: 0,
            samples_emitted: 0,
            samples_skipped: 0,
            samples_skipped_pre_snapshot: 0,
            samples_skipped_invalid_book: 0,
            runtime_ms: 0,
            decode_ms: 0,
            book_update_ms: 0,
            sampling_ms: 0,
            write_ms: 0,
            rows_per_sec: 0.0,
            peak_rss_bytes: None,
            output_path: None,
            output_rows: 0,
            output_bytes: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineOutput {
    pub rows: Vec<FeatureRow>,
    pub qc: QcStats,
}

pub fn atoms_to_f64(atoms: i128, scale: i32) -> f64 {
    atoms as f64 / 10_f64.powi(scale)
}
