use crate::model::{Bbo, Side, TradeSide, WalkResult, atoms_to_f64};
use std::collections::BTreeMap;

pub trait OrderBook {
    fn reset(&mut self);
    fn update(&mut self, side: Side, price_atoms: i128, price_scale: i32, amount: f64);
    fn bbo(&self) -> Option<Bbo>;
    fn level_counts(&self) -> (u32, u32);
    fn walk(&self, side: TradeSide, notional_usd: f64, ref_price: f64) -> WalkResult;
}

#[derive(Debug, Clone, Default)]
pub struct BTreeBook {
    bids: BTreeMap<i128, Level>,
    asks: BTreeMap<i128, Level>,
}

#[derive(Debug, Clone, Default)]
pub struct VecBook {
    bids: Vec<(i128, Level)>,
    asks: Vec<(i128, Level)>,
}

#[derive(Debug, Clone, Copy)]
struct Level {
    amount: f64,
    scale: i32,
}

impl BTreeBook {
    pub fn new() -> Self {
        Self::default()
    }
}

impl VecBook {
    pub fn new() -> Self {
        Self::default()
    }

    fn levels_mut(&mut self, side: Side) -> &mut Vec<(i128, Level)> {
        match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        }
    }
}

impl OrderBook for BTreeBook {
    fn reset(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    fn update(&mut self, side: Side, price_atoms: i128, price_scale: i32, amount: f64) {
        let book = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        if !amount.is_finite() || amount <= 0.0 {
            book.remove(&price_atoms);
        } else {
            book.insert(
                price_atoms,
                Level {
                    amount,
                    scale: price_scale,
                },
            );
        }
    }

    fn bbo(&self) -> Option<Bbo> {
        let (&bid_atoms, bid_level) = self.bids.iter().next_back()?;
        let (&ask_atoms, ask_level) = self.asks.iter().next()?;
        let bid_price = atoms_to_f64(bid_atoms, bid_level.scale);
        let ask_price = atoms_to_f64(ask_atoms, ask_level.scale);
        Some(Bbo {
            bid_price_atoms: bid_atoms,
            bid_price,
            bid_amount: bid_level.amount,
            ask_price_atoms: ask_atoms,
            ask_price,
            ask_amount: ask_level.amount,
        })
    }

    fn level_counts(&self) -> (u32, u32) {
        (self.bids.len() as u32, self.asks.len() as u32)
    }

    fn walk(&self, side: TradeSide, notional_usd: f64, ref_price: f64) -> WalkResult {
        match side {
            TradeSide::Buy => walk_levels(
                side,
                self.asks.iter().map(|(&price, &level)| (price, level)),
                notional_usd,
                ref_price,
            ),
            TradeSide::Sell => walk_levels(
                side,
                self.bids
                    .iter()
                    .rev()
                    .map(|(&price, &level)| (price, level)),
                notional_usd,
                ref_price,
            ),
        }
    }
}

impl OrderBook for VecBook {
    fn reset(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    fn update(&mut self, side: Side, price_atoms: i128, price_scale: i32, amount: f64) {
        let levels = self.levels_mut(side);
        match levels.binary_search_by_key(&price_atoms, |(price, _)| *price) {
            Ok(idx) if !amount.is_finite() || amount <= 0.0 => {
                levels.remove(idx);
            }
            Ok(idx) => {
                levels[idx].1 = Level {
                    amount,
                    scale: price_scale,
                };
            }
            Err(_) if !amount.is_finite() || amount <= 0.0 => {}
            Err(idx) => levels.insert(
                idx,
                (
                    price_atoms,
                    Level {
                        amount,
                        scale: price_scale,
                    },
                ),
            ),
        }
    }

    fn bbo(&self) -> Option<Bbo> {
        let (bid_atoms, bid_level) = self.bids.last()?;
        let (ask_atoms, ask_level) = self.asks.first()?;
        let bid_price = atoms_to_f64(*bid_atoms, bid_level.scale);
        let ask_price = atoms_to_f64(*ask_atoms, ask_level.scale);
        Some(Bbo {
            bid_price_atoms: *bid_atoms,
            bid_price,
            bid_amount: bid_level.amount,
            ask_price_atoms: *ask_atoms,
            ask_price,
            ask_amount: ask_level.amount,
        })
    }

    fn level_counts(&self) -> (u32, u32) {
        (self.bids.len() as u32, self.asks.len() as u32)
    }

    fn walk(&self, side: TradeSide, notional_usd: f64, ref_price: f64) -> WalkResult {
        match side {
            TradeSide::Buy => walk_levels(
                side,
                self.asks.iter().map(|(price, level)| (*price, *level)),
                notional_usd,
                ref_price,
            ),
            TradeSide::Sell => walk_levels(
                side,
                self.bids
                    .iter()
                    .rev()
                    .map(|(price, level)| (*price, *level)),
                notional_usd,
                ref_price,
            ),
        }
    }
}

fn walk_levels<I>(side: TradeSide, levels: I, notional_usd: f64, ref_price: f64) -> WalkResult
where
    I: IntoIterator<Item = (i128, Level)>,
{
    if !ref_price.is_finite() || ref_price <= 0.0 || notional_usd <= 0.0 {
        return WalkResult::default();
    }
    let target_qty = notional_usd / ref_price;
    let mut remaining = target_qty;
    let mut filled_qty = 0.0;
    let mut filled_notional = 0.0;
    let mut levels_used = 0u32;
    let mut last_price = f64::NAN;

    for (price_atoms, level) in levels {
        if remaining <= 1e-15 {
            break;
        }
        let price = atoms_to_f64(price_atoms, level.scale);
        let take = remaining.min(level.amount);
        if take <= 0.0 {
            continue;
        }
        filled_qty += take;
        filled_notional += take * price;
        remaining -= take;
        levels_used += 1;
        last_price = price;
    }

    if filled_qty <= 0.0 {
        return WalkResult {
            target_qty,
            shortfall_qty: target_qty,
            ..WalkResult::default()
        };
    }
    let vwap = filled_notional / filled_qty;
    let cost_bps = match side {
        TradeSide::Buy => (vwap / ref_price - 1.0) * 10_000.0,
        TradeSide::Sell => (1.0 - vwap / ref_price) * 10_000.0,
    };
    let complete = remaining <= target_qty * 1e-12 + 1e-12;
    WalkResult {
        target_qty,
        filled_qty,
        filled_notional,
        vwap,
        cost_bps,
        complete,
        levels_used,
        last_price,
        shortfall_qty: remaining.max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_best_bid_ask_and_walk<B: OrderBook + Default>() {
        let mut book = B::default();
        book.update(Side::Bid, 99_0000000000, 10, 10.0);
        book.update(Side::Bid, 98_0000000000, 10, 10.0);
        book.update(Side::Ask, 101_0000000000, 10, 5.0);
        book.update(Side::Ask, 102_0000000000, 10, 10.0);
        let bbo = book.bbo().unwrap();
        assert_eq!(bbo.bid_price, 99.0);
        assert_eq!(bbo.ask_price, 101.0);
        let buy = book.walk(TradeSide::Buy, 1_000.0, 100.0);
        assert!(buy.complete);
        assert_eq!(buy.levels_used, 2);
        let sell = book.walk(TradeSide::Sell, 1_000.0, 100.0);
        assert!(sell.complete);
        assert_eq!(sell.levels_used, 1);
    }

    #[test]
    fn btree_best_bid_ask_and_walk() {
        assert_best_bid_ask_and_walk::<BTreeBook>();
    }

    #[test]
    fn vec_best_bid_ask_and_walk() {
        assert_best_bid_ask_and_walk::<VecBook>();
    }

    fn assert_zero_amount_deletes<B: OrderBook + Default>() {
        let mut book = B::default();
        book.update(Side::Bid, 100, 0, 1.0);
        assert!(book.bbo().is_none());
        book.update(Side::Ask, 101, 0, 1.0);
        assert!(book.bbo().is_some());
        book.update(Side::Ask, 101, 0, 0.0);
        assert!(book.bbo().is_none());
    }

    #[test]
    fn btree_zero_amount_deletes() {
        assert_zero_amount_deletes::<BTreeBook>();
    }

    #[test]
    fn vec_zero_amount_deletes() {
        assert_zero_amount_deletes::<VecBook>();
    }
}
