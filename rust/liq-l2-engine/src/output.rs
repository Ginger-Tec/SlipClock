use crate::model::*;
use anyhow::Result;
use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub fn write_feature_parquet(
    path: &Path,
    rows: &[FeatureRow],
    config: &EngineConfig,
) -> Result<u64> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let batch = rows_to_batch(rows, config)?;
    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_row_count(Some(50_000))
        .build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(std::fs::metadata(path)?.len())
}

pub fn write_qc_json(path: &Path, qc: &QcStats) -> Result<u64> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(qc)?)?;
    std::fs::rename(tmp, path)?;
    Ok(std::fs::metadata(path)?.len())
}

pub fn rows_to_batch(rows: &[FeatureRow], config: &EngineConfig) -> Result<RecordBatch> {
    let mut fields: Vec<Field> = vec![
        Field::new("sample_local_timestamp_us", DataType::Int64, false),
        Field::new("last_local_timestamp_us", DataType::Int64, false),
        Field::new("last_exchange_timestamp_us", DataType::Int64, false),
        Field::new("source_file_count", DataType::UInt32, false),
        Field::new("source_row_index", DataType::UInt64, false),
        Field::new("bid_price", DataType::Float64, false),
        Field::new("bid_amount", DataType::Float64, false),
        Field::new("ask_price", DataType::Float64, false),
        Field::new("ask_amount", DataType::Float64, false),
        Field::new("mid_price", DataType::Float64, false),
        Field::new("micro_price", DataType::Float64, false),
        Field::new("spread_bps", DataType::Float64, false),
        Field::new("bbo_imbalance", DataType::Float64, false),
        Field::new("book_bid_levels", DataType::UInt32, false),
        Field::new("book_ask_levels", DataType::UInt32, false),
        Field::new("invalid_book_flag", DataType::Boolean, false),
    ];
    let walk_keys = walk_keys(config);
    for key in &walk_keys {
        let p = key.prefix();
        match config.output_layout {
            OutputLayout::Full => fields.extend([
                Field::new(format!("{p}_target_qty"), DataType::Float64, false),
                Field::new(format!("{p}_filled_qty"), DataType::Float64, false),
                Field::new(format!("{p}_filled_notional"), DataType::Float64, false),
                Field::new(format!("{p}_vwap"), DataType::Float64, false),
                Field::new(format!("{p}_cost_bps"), DataType::Float64, false),
                Field::new(format!("{p}_complete"), DataType::Boolean, false),
                Field::new(format!("{p}_levels_used"), DataType::UInt32, false),
                Field::new(format!("{p}_last_price"), DataType::Float64, false),
                Field::new(format!("{p}_shortfall_qty"), DataType::Float64, false),
            ]),
            OutputLayout::Compact => fields.extend([
                Field::new(format!("{p}_cost_bps"), DataType::Float64, false),
                Field::new(format!("{p}_complete"), DataType::Boolean, false),
                Field::new(format!("{p}_levels_used"), DataType::UInt32, false),
            ]),
        }
    }

    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.sample_local_timestamp_us),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.last_local_timestamp_us),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.last_exchange_timestamp_us),
        )),
        Arc::new(UInt32Array::from_iter_values(
            rows.iter().map(|r| r.source_file_count),
        )),
        Arc::new(UInt64Array::from_iter_values(
            rows.iter().map(|r| r.source_row_index),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.bid_price),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.bid_amount),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.ask_price),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.ask_amount),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.mid_price),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.micro_price),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.spread_bps),
        )),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.bbo_imbalance),
        )),
        Arc::new(UInt32Array::from_iter_values(
            rows.iter().map(|r| r.book_bid_levels),
        )),
        Arc::new(UInt32Array::from_iter_values(
            rows.iter().map(|r| r.book_ask_levels),
        )),
        Arc::new(BooleanArray::from_iter(
            rows.iter().map(|r| Some(r.invalid_book_flag)),
        )),
    ];
    for key in &walk_keys {
        let vals: Vec<_> = rows
            .iter()
            .map(|r| r.walks.get(key).copied().unwrap_or_default())
            .collect();
        match config.output_layout {
            OutputLayout::Full => arrays.extend([
                Arc::new(Float64Array::from_iter_values(
                    vals.iter().map(|v| v.target_qty),
                )) as ArrayRef,
                Arc::new(Float64Array::from_iter_values(
                    vals.iter().map(|v| v.filled_qty),
                )) as ArrayRef,
                Arc::new(Float64Array::from_iter_values(
                    vals.iter().map(|v| v.filled_notional),
                )) as ArrayRef,
                Arc::new(Float64Array::from_iter_values(vals.iter().map(|v| v.vwap))) as ArrayRef,
                Arc::new(Float64Array::from_iter_values(
                    vals.iter().map(|v| v.cost_bps),
                )) as ArrayRef,
                Arc::new(BooleanArray::from_iter(
                    vals.iter().map(|v| Some(v.complete)),
                )) as ArrayRef,
                Arc::new(UInt32Array::from_iter_values(
                    vals.iter().map(|v| v.levels_used),
                )) as ArrayRef,
                Arc::new(Float64Array::from_iter_values(
                    vals.iter().map(|v| v.last_price),
                )) as ArrayRef,
                Arc::new(Float64Array::from_iter_values(
                    vals.iter().map(|v| v.shortfall_qty),
                )) as ArrayRef,
            ]),
            OutputLayout::Compact => arrays.extend([
                Arc::new(Float64Array::from_iter_values(
                    vals.iter().map(|v| v.cost_bps),
                )) as ArrayRef,
                Arc::new(BooleanArray::from_iter(
                    vals.iter().map(|v| Some(v.complete)),
                )) as ArrayRef,
                Arc::new(UInt32Array::from_iter_values(
                    vals.iter().map(|v| v.levels_used),
                )) as ArrayRef,
            ]),
        }
    }
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?)
}

pub fn walk_keys(config: &EngineConfig) -> Vec<WalkKey> {
    let mut keys = Vec::new();
    for &side in &[TradeSide::Buy, TradeSide::Sell] {
        for &notional in &config.notionals {
            for &reference in &config.refs {
                keys.push(WalkKey::new(side, notional, reference));
            }
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Grid, L2Event, Side, run_events};

    #[test]
    fn output_schema_contains_v0_columns() {
        let (start, _) = crate::engine::day_bounds_us("2024-07-01").unwrap();
        let events = vec![
            L2Event {
                exchange_timestamp_us: start + 1,
                local_timestamp_us: start + 1,
                is_snapshot: true,
                side: Side::Bid,
                price_atoms: 99,
                price_scale: 0,
                amount: 10.0,
                source_file_index: 0,
                source_row_index: 0,
            },
            L2Event {
                exchange_timestamp_us: start + 1,
                local_timestamp_us: start + 1,
                is_snapshot: true,
                side: Side::Ask,
                price_atoms: 101,
                price_scale: 0,
                amount: 10.0,
                source_file_index: 0,
                source_row_index: 1,
            },
        ];
        let config = EngineConfig {
            date: "2024-07-01".into(),
            grid: Grid::Minutes(1),
            ..EngineConfig::default()
        };
        let out = run_events(events, config.clone()).unwrap();
        let batch = rows_to_batch(&out.rows, &config).unwrap();
        let names: Vec<_> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert!(names.contains(&"sample_local_timestamp_us".to_string()));
        assert!(names.contains(&"spread_bps".to_string()));
        assert!(names.contains(&"buy_1000_mid_cost_bps".to_string()));
        assert!(names.contains(&"sell_100000_micro_complete".to_string()));
    }

    #[test]
    fn compact_layout_keeps_cost_gate_columns_and_drops_fill_detail() {
        let (start, _) = crate::engine::day_bounds_us("2024-07-01").unwrap();
        let events = vec![
            L2Event {
                exchange_timestamp_us: start + 60_000_000,
                local_timestamp_us: start + 60_000_000,
                is_snapshot: true,
                side: Side::Bid,
                price_atoms: 99,
                price_scale: 0,
                amount: 10.0,
                source_file_index: 0,
                source_row_index: 0,
            },
            L2Event {
                exchange_timestamp_us: start + 60_000_000,
                local_timestamp_us: start + 60_000_000,
                is_snapshot: true,
                side: Side::Ask,
                price_atoms: 101,
                price_scale: 0,
                amount: 10.0,
                source_file_index: 0,
                source_row_index: 1,
            },
        ];
        let config = EngineConfig {
            date: "2024-07-01".into(),
            grid: Grid::Minutes(1),
            output_layout: OutputLayout::Compact,
            ..EngineConfig::default()
        };
        let out = run_events(events, config.clone()).unwrap();
        let batch = rows_to_batch(&out.rows, &config).unwrap();
        let names: Vec<_> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert!(names.contains(&"buy_1000_mid_cost_bps".to_string()));
        assert!(names.contains(&"buy_1000_mid_complete".to_string()));
        assert!(names.contains(&"buy_1000_mid_levels_used".to_string()));
        assert!(!names.contains(&"buy_1000_mid_target_qty".to_string()));
        assert!(!names.contains(&"buy_1000_mid_vwap".to_string()));
    }
}
