use crate::book::{BTreeBook, OrderBook, VecBook};
use crate::model::*;
use anyhow::{Result, bail};
use chrono::{NaiveDate, TimeZone, Utc};
use std::collections::HashMap;
use std::iter::Peekable;
use std::time::Instant;

pub struct EventGroupIter<I: Iterator<Item = L2Event>> {
    inner: Peekable<I>,
}

impl<I: Iterator<Item = L2Event>> EventGroupIter<I> {
    pub fn new(inner: I) -> Self {
        Self {
            inner: inner.peekable(),
        }
    }
}

impl<I: Iterator<Item = L2Event>> Iterator for EventGroupIter<I> {
    type Item = EventGroup;

    fn next(&mut self) -> Option<Self::Item> {
        let first = self.inner.next()?;
        let local_ts = first.local_timestamp_us;
        let mut events = vec![first];
        while matches!(self.inner.peek(), Some(next) if next.local_timestamp_us == local_ts) {
            events.push(self.inner.next().expect("peeked event must exist"));
        }
        Some(EventGroup {
            local_timestamp_us: local_ts,
            events,
        })
    }
}

pub fn run_events<I>(events: I, config: EngineConfig) -> Result<EngineOutput>
where
    I: IntoIterator<Item = L2Event>,
{
    let groups = EventGroupIter::new(events.into_iter());
    run_event_groups(groups, config)
}

pub fn run_event_groups<I>(groups: I, config: EngineConfig) -> Result<EngineOutput>
where
    I: IntoIterator<Item = EventGroup>,
{
    let mut runner = EngineRunner::new(config)?;
    for group in groups {
        runner.process_group(group)?;
    }
    runner.finish()
}

pub enum EngineRunner {
    BTree(BookEngineRunner<BTreeBook>),
    Vec(BookEngineRunner<VecBook>),
}

impl EngineRunner {
    pub fn new(config: EngineConfig) -> Result<Self> {
        match config.engine_variant.as_str() {
            "btree" => Ok(Self::BTree(BookEngineRunner::new(config)?)),
            "vec" | "sorted-vec" | "sorted_vec" => Ok(Self::Vec(BookEngineRunner::new(config)?)),
            other => bail!("unsupported engine_variant {other:?}; use btree or vec"),
        }
    }

    pub fn process_group(&mut self, group: EventGroup) -> Result<()> {
        match self {
            Self::BTree(runner) => runner.process_group(group),
            Self::Vec(runner) => runner.process_group(group),
        }
    }

    pub fn finish(self) -> Result<EngineOutput> {
        match self {
            Self::BTree(runner) => runner.finish(),
            Self::Vec(runner) => runner.finish(),
        }
    }
}

pub struct BookEngineRunner<B>
where
    B: OrderBook + Default,
{
    config: EngineConfig,
    qc: QcStats,
    book: B,
    rows: Vec<FeatureRow>,
    day_end: u64,
    grid_step_us: u64,
    next_sample: u64,
    snapshot_seen: bool,
    last_applied_local_ts: Option<u64>,
    last_exchange_ts: Option<u64>,
    last_group_exchange_ts: Option<u64>,
    last_group_row_index: u64,
    max_source_file_index: u32,
    book_update_ns: u128,
    sampling_ns: u128,
}

impl<B> BookEngineRunner<B>
where
    B: OrderBook + Default,
{
    fn new(config: EngineConfig) -> Result<Self> {
        let (day_start, day_end) = day_bounds_us(&config.date)?;
        let grid_step_us = config.grid.checked_step_us()?;
        Ok(Self {
            qc: QcStats::new(&config),
            config,
            book: B::default(),
            rows: Vec::new(),
            day_end,
            grid_step_us,
            next_sample: day_start,
            snapshot_seen: false,
            last_applied_local_ts: None,
            last_exchange_ts: None,
            last_group_exchange_ts: None,
            last_group_row_index: 0,
            max_source_file_index: 0,
            book_update_ns: 0,
            sampling_ns: 0,
        })
    }

    fn process_group(&mut self, group: EventGroup) -> Result<()> {
        self.qc.local_timestamp_groups += 1;
        self.qc.input_rows += group.events.len() as u64;
        self.qc.accepted_rows += group.events.len() as u64;
        self.qc.max_group_rows = self.qc.max_group_rows.max(group.events.len() as u64);

        if let Some(prev_local) = self.last_applied_local_ts {
            if group.local_timestamp_us < prev_local {
                self.qc.non_monotonic_local_ts_count += 1;
                if self.config.strict {
                    bail!(
                        "local_timestamp regressed: previous={prev_local}, current={}",
                        group.local_timestamp_us
                    );
                }
            }
            let sampling_started = Instant::now();
            emit_due_samples(
                &mut self.rows,
                &mut self.qc,
                &self.config,
                &self.book,
                &mut self.next_sample,
                prev_local,
                group.local_timestamp_us,
                self.last_exchange_ts.unwrap_or_default(),
                self.last_group_row_index,
                self.max_source_file_index.saturating_add(1),
                self.day_end,
                self.grid_step_us,
            );
            self.sampling_ns += sampling_started.elapsed().as_nanos();
        } else {
            while self.next_sample < group.local_timestamp_us && self.next_sample <= self.day_end {
                self.qc.samples_skipped_pre_snapshot += 1;
                self.qc.samples_skipped += 1;
                self.next_sample += self.grid_step_us;
            }
        }

        for event in &group.events {
            self.max_source_file_index = self.max_source_file_index.max(event.source_file_index);
        }

        for event in &group.events {
            if let Some(prev_ex) = self.last_exchange_ts {
                if event.exchange_timestamp_us < prev_ex {
                    self.qc.non_monotonic_exchange_ts_count += 1;
                } else if event.exchange_timestamp_us == prev_ex {
                    self.qc.equal_exchange_ts_group_count += 1;
                }
            }
            self.last_exchange_ts = Some(event.exchange_timestamp_us);
        }

        if !self.snapshot_seen && !group.contains_snapshot() {
            self.qc.pre_snapshot_rows_skipped += group.events.len() as u64;
            return Ok(());
        }

        let update_started = Instant::now();
        if group.contains_snapshot() {
            self.qc.snapshot_groups += 1;
            self.qc.reset_count += 1;
            self.book.reset();
            self.snapshot_seen = true;
        }

        for event in &group.events {
            self.book.update(
                event.side,
                event.price_atoms,
                event.price_scale,
                event.amount,
            );
        }
        self.book_update_ns += update_started.elapsed().as_nanos();

        self.last_applied_local_ts = Some(group.local_timestamp_us);
        self.last_group_exchange_ts = Some(group.exchange_timestamp_us());
        self.last_group_row_index = group.source_row_index();
        Ok(())
    }

    fn finish(mut self) -> Result<EngineOutput> {
        if let Some(prev_local) = self.last_applied_local_ts {
            let sampling_started = Instant::now();
            emit_due_samples(
                &mut self.rows,
                &mut self.qc,
                &self.config,
                &self.book,
                &mut self.next_sample,
                prev_local,
                self.day_end
                    .saturating_add(self.grid_step_us)
                    .saturating_add(1),
                self.last_group_exchange_ts.unwrap_or_default(),
                self.last_group_row_index,
                self.max_source_file_index.saturating_add(1),
                self.day_end,
                self.grid_step_us,
            );
            self.sampling_ns += sampling_started.elapsed().as_nanos();
        }

        self.qc.book_update_ms = self.book_update_ns / 1_000_000;
        self.qc.sampling_ms = self.sampling_ns / 1_000_000;
        self.qc.runtime_ms = self.qc.book_update_ms + self.qc.sampling_ms;
        self.qc.output_rows = self.rows.len() as u64;
        if self.qc.runtime_ms > 0 {
            self.qc.rows_per_sec = self.qc.input_rows as f64 / (self.qc.runtime_ms as f64 / 1000.0);
        }
        Ok(EngineOutput {
            rows: self.rows,
            qc: self.qc,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_due_samples<B>(
    rows: &mut Vec<FeatureRow>,
    qc: &mut QcStats,
    config: &EngineConfig,
    book: &B,
    next_sample: &mut u64,
    last_local_ts: u64,
    exclusive_next_local_ts: u64,
    last_exchange_ts: u64,
    source_row_index: u64,
    source_file_count: u32,
    day_end: u64,
    grid_step_us: u64,
) where
    B: OrderBook,
{
    while *next_sample < last_local_ts && *next_sample <= day_end {
        qc.samples_skipped_pre_snapshot += 1;
        qc.samples_skipped += 1;
        *next_sample += grid_step_us;
    }
    while *next_sample <= day_end
        && last_local_ts <= *next_sample
        && *next_sample < exclusive_next_local_ts
    {
        match sample_row(
            config,
            book,
            *next_sample,
            last_local_ts,
            last_exchange_ts,
            source_file_count,
            source_row_index,
        ) {
            SampleOutcome::Valid(row) => {
                qc.samples_emitted += 1;
                rows.push(*row);
            }
            SampleOutcome::Empty => {
                qc.empty_book_count += 1;
                qc.samples_skipped_invalid_book += 1;
                qc.samples_skipped += 1;
            }
            SampleOutcome::Crossed => {
                qc.crossed_book_count += 1;
                qc.samples_skipped_invalid_book += 1;
                qc.samples_skipped += 1;
            }
        }
        *next_sample += grid_step_us;
    }
}

fn sample_row<B>(
    config: &EngineConfig,
    book: &B,
    sample_local_ts: u64,
    last_local_ts: u64,
    last_exchange_ts: u64,
    source_file_count: u32,
    source_row_index: u64,
) -> SampleOutcome
where
    B: OrderBook,
{
    let Some(bbo) = book.bbo() else {
        return SampleOutcome::Empty;
    };
    if bbo.ask_price < bbo.bid_price {
        return SampleOutcome::Crossed;
    }
    if bbo.bid_price <= 0.0 || bbo.ask_price <= 0.0 {
        return SampleOutcome::Empty;
    }
    let mid = (bbo.bid_price + bbo.ask_price) / 2.0;
    let denom = bbo.bid_amount + bbo.ask_amount;
    if denom <= 0.0 || mid <= 0.0 {
        return SampleOutcome::Empty;
    }
    let micro = (bbo.ask_price * bbo.bid_amount + bbo.bid_price * bbo.ask_amount) / denom;
    let spread_bps = (bbo.ask_price - bbo.bid_price) / mid * 10_000.0;
    let bbo_imbalance = bbo.bid_amount / denom;
    let (book_bid_levels, book_ask_levels) = book.level_counts();
    let mut walks = HashMap::new();
    for &notional in &config.notionals {
        for &reference in &config.refs {
            let ref_price = match reference {
                RefPrice::Mid => mid,
                RefPrice::Micro => micro,
            };
            for side in [TradeSide::Buy, TradeSide::Sell] {
                walks.insert(
                    WalkKey::new(side, notional, reference),
                    book.walk(side, notional, ref_price),
                );
            }
        }
    }
    SampleOutcome::Valid(Box::new(FeatureRow {
        exchange: config.exchange.clone(),
        symbol: config.symbol.clone(),
        date: config.date.clone(),
        grid: config.grid.name(),
        engine_variant: config.engine_variant.clone(),
        sample_clock: config.sample_clock.clone(),
        sample_local_timestamp_us: sample_local_ts as i64,
        last_local_timestamp_us: last_local_ts as i64,
        last_exchange_timestamp_us: last_exchange_ts as i64,
        source_file_count,
        source_row_index,
        bid_price: bbo.bid_price,
        bid_amount: bbo.bid_amount,
        ask_price: bbo.ask_price,
        ask_amount: bbo.ask_amount,
        mid_price: mid,
        micro_price: micro,
        spread_bps,
        bbo_imbalance,
        book_bid_levels,
        book_ask_levels,
        invalid_book_flag: false,
        walks,
    }))
}

enum SampleOutcome {
    Valid(Box<FeatureRow>),
    Empty,
    Crossed,
}

pub fn day_bounds_us(date: &str) -> Result<(u64, u64)> {
    let day = NaiveDate::parse_from_str(date, "%Y-%m-%d")?;
    let start = Utc.from_utc_datetime(&day.and_hms_opt(0, 0, 0).unwrap());
    let start_us = start.timestamp_micros() as u64;
    let end_us = start_us + 86_400 * 1_000_000 - 1;
    Ok((start_us, end_us))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(
        local: u64,
        exchange: u64,
        snapshot: bool,
        side: Side,
        price: i128,
        amount: f64,
        row: u64,
    ) -> L2Event {
        L2Event {
            exchange_timestamp_us: exchange,
            local_timestamp_us: local,
            is_snapshot: snapshot,
            side,
            price_atoms: price,
            price_scale: 0,
            amount,
            source_file_index: 0,
            source_row_index: row,
        }
    }

    #[test]
    fn groups_equal_local_timestamp_atomically() {
        let events = vec![
            ev(10, 10, true, Side::Bid, 99, 1.0, 0),
            ev(10, 10, true, Side::Ask, 101, 1.0, 1),
            ev(20, 20, false, Side::Bid, 100, 1.0, 2),
            ev(20, 20, false, Side::Ask, 102, 1.0, 3),
        ];
        let groups: Vec<_> = EventGroupIter::new(events.into_iter()).collect();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].events.len(), 2);
        assert_eq!(groups[1].events.len(), 2);
    }

    #[test]
    fn skips_pre_snapshot_and_emits_after_complete_snapshot_group() {
        let (start, _) = day_bounds_us("2024-07-01").unwrap();
        let events = vec![
            ev(start + 1, start + 1, false, Side::Bid, 98, 1.0, 0),
            ev(
                start + 60_000_000,
                start + 60_000_000,
                true,
                Side::Bid,
                99,
                10.0,
                1,
            ),
            ev(
                start + 60_000_000,
                start + 60_000_000,
                true,
                Side::Ask,
                101,
                10.0,
                2,
            ),
            ev(
                start + 120_000_000,
                start + 120_000_000,
                false,
                Side::Bid,
                100,
                10.0,
                3,
            ),
        ];
        let config = EngineConfig {
            date: "2024-07-01".into(),
            grid: Grid::Minutes(1),
            ..EngineConfig::default()
        };
        let out = run_events(events, config).unwrap();
        assert_eq!(out.qc.pre_snapshot_rows_skipped, 1);
        assert_eq!(
            out.rows[0].sample_local_timestamp_us as u64,
            start + 60_000_000
        );
        assert_eq!(out.rows[0].bid_price, 99.0);
        assert_eq!(out.rows[0].ask_price, 101.0);
    }

    #[test]
    fn non_monotonic_exchange_timestamp_is_qc_not_default_error() {
        let (start, _) = day_bounds_us("2024-07-01").unwrap();
        let events = vec![
            ev(start + 1, start + 10, true, Side::Bid, 99, 10.0, 0),
            ev(start + 1, start + 10, true, Side::Ask, 101, 10.0, 1),
            ev(
                start + 60_000_000,
                start + 5,
                false,
                Side::Bid,
                100,
                10.0,
                2,
            ),
        ];
        let config = EngineConfig {
            date: "2024-07-01".into(),
            grid: Grid::Minutes(1),
            ..EngineConfig::default()
        };
        let out = run_events(events, config).unwrap();
        assert!(out.qc.non_monotonic_exchange_ts_count >= 1);
    }

    #[test]
    fn snapshot_resets_once_per_group() {
        let (start, _) = day_bounds_us("2024-07-01").unwrap();
        let events = vec![
            ev(start + 1, start + 1, true, Side::Bid, 99, 10.0, 0),
            ev(start + 1, start + 1, true, Side::Ask, 101, 10.0, 1),
            ev(
                start + 60_000_000,
                start + 60_000_000,
                true,
                Side::Bid,
                98,
                10.0,
                2,
            ),
            ev(
                start + 60_000_000,
                start + 60_000_000,
                true,
                Side::Ask,
                102,
                10.0,
                3,
            ),
        ];
        let config = EngineConfig {
            date: "2024-07-01".into(),
            grid: Grid::Minutes(1),
            ..EngineConfig::default()
        };
        let out = run_events(events, config).unwrap();
        assert_eq!(out.qc.reset_count, 2);
        let row = out
            .rows
            .iter()
            .find(|r| r.sample_local_timestamp_us as u64 == start + 60_000_000)
            .unwrap();
        assert_eq!(row.bid_price, 98.0);
        assert_eq!(row.ask_price, 102.0);
    }

    #[test]
    fn zero_grid_is_rejected_instead_of_spinning() {
        let config = EngineConfig {
            date: "2024-07-01".into(),
            grid: Grid::Seconds(0),
            ..EngineConfig::default()
        };
        let err = run_events(Vec::<L2Event>::new(), config).unwrap_err();
        assert!(err.to_string().contains("must be positive"));
        assert!(Grid::parse("0s").is_err());
        assert!(Grid::parse("0m").is_err());
    }

    #[test]
    fn day_scale_grid_is_the_maximum_supported_step() {
        assert_eq!(Grid::parse("86400s").unwrap().step_us(), 86_400_000_000);
        assert_eq!(Grid::parse("1440m").unwrap().step_us(), 86_400_000_000);
        assert!(Grid::parse("86401s").is_err());
        assert!(Grid::parse("1441m").is_err());
    }

    #[test]
    fn crossed_books_increment_crossed_counter() {
        let (start, _) = day_bounds_us("2024-07-01").unwrap();
        let events = vec![
            ev(
                start + 60_000_000,
                start + 60_000_000,
                true,
                Side::Bid,
                101,
                10.0,
                0,
            ),
            ev(
                start + 60_000_000,
                start + 60_000_000,
                true,
                Side::Ask,
                100,
                10.0,
                1,
            ),
            ev(
                start + 120_000_000,
                start + 120_000_000,
                false,
                Side::Bid,
                101,
                10.0,
                2,
            ),
        ];
        let config = EngineConfig {
            date: "2024-07-01".into(),
            grid: Grid::Minutes(1),
            ..EngineConfig::default()
        };
        let out = run_events(events, config).unwrap();
        assert_eq!(out.rows.len(), 0);
        assert!(out.qc.crossed_book_count >= 1);
        assert_eq!(out.qc.empty_book_count, 0);
        assert_eq!(
            out.qc.samples_skipped_invalid_book,
            out.qc.crossed_book_count
        );
    }
}
