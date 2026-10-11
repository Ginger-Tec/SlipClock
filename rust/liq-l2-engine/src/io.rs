use crate::model::{EventGroup, L2Event, Side, atoms_to_f64};
use anyhow::{Context, Result, anyhow, bail};
use arrow::array::{
    Array, BooleanArray, Decimal128Array, Float64Array, Int64Array, StringArray, UInt64Array,
};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::cmp::Ordering;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

#[derive(Debug, Clone, Default)]
pub struct ReadStats {
    pub input_files: u64,
    pub record_batches: u64,
    pub row_groups: u64,
    pub input_rows: u64,
    pub accepted_rows: u64,
    pub unknown_side_count: u64,
    pub invalid_price_count: u64,
    pub invalid_amount_count: u64,
    pub decode_ms: u128,
    pub consumer_ms: u128,
}

pub fn read_events_from_parquet_dir(
    path: &Path,
    batch_size: usize,
) -> Result<(Vec<L2Event>, ReadStats)> {
    let mut files = parquet_files(path)?;
    files.sort();
    let mut stats = ReadStats::default();
    let mut events = Vec::new();
    for (file_index, file_path) in files.iter().enumerate() {
        let file = File::open(file_path)
            .with_context(|| format!("open parquet file {}", file_path.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("build parquet reader {}", file_path.display()))?;
        stats.row_groups += builder.metadata().num_row_groups() as u64;
        let schema_desc = builder.metadata().file_metadata().schema_descr_ptr();
        let indices = required_leaf_indices(
            builder
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>()
                .as_slice(),
        )?;
        let projection = ProjectionMask::leaves(&schema_desc, indices);
        let mut reader = builder
            .with_projection(projection)
            .with_batch_size(batch_size)
            .build()?;
        let mut source_row_index = 0u64;
        while let Some(batch) = reader.next().transpose()? {
            stats.record_batches += 1;
            let cols = ProjectedColumns::from_batch(&batch)?;
            for i in 0..batch.num_rows() {
                let raw_row_index = source_row_index;
                source_row_index += 1;
                stats.input_rows += 1;
                if let Some(event) = parse_event(
                    &cols,
                    i,
                    file_path,
                    file_index as u32,
                    raw_row_index,
                    &mut stats,
                )? {
                    events.push(event);
                    stats.accepted_rows += 1;
                }
            }
        }
        stats.input_files += 1;
    }
    events.sort_by_key(|e| {
        (
            e.local_timestamp_us,
            e.source_file_index,
            e.source_row_index,
        )
    });
    Ok((events, stats))
}

pub fn stream_event_groups_from_parquet_dir<F>(
    path: &Path,
    batch_size: usize,
    mut on_group: F,
) -> Result<ReadStats>
where
    F: FnMut(EventGroup) -> Result<()>,
{
    let started = Instant::now();
    let mut consumer_ns = 0u128;
    let mut files = parquet_files(path)?;
    files.sort();
    let mut stats = ReadStats::default();
    let mut pending_group: Option<EventGroup> = None;

    for (file_index, file_path) in files.iter().enumerate() {
        let file = File::open(file_path)
            .with_context(|| format!("open parquet file {}", file_path.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("build parquet reader {}", file_path.display()))?;
        stats.row_groups += builder.metadata().num_row_groups() as u64;
        let schema_desc = builder.metadata().file_metadata().schema_descr_ptr();
        let indices = required_leaf_indices(
            builder
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>()
                .as_slice(),
        )?;
        let projection = ProjectionMask::leaves(&schema_desc, indices);
        let mut reader = builder
            .with_projection(projection)
            .with_batch_size(batch_size)
            .build()?;
        let mut source_row_index = 0u64;
        while let Some(batch) = reader.next().transpose()? {
            stats.record_batches += 1;
            let cols = ProjectedColumns::from_batch(&batch)?;
            for i in 0..batch.num_rows() {
                let raw_row_index = source_row_index;
                source_row_index += 1;
                stats.input_rows += 1;
                let Some(event) = parse_event(
                    &cols,
                    i,
                    file_path,
                    file_index as u32,
                    raw_row_index,
                    &mut stats,
                )?
                else {
                    continue;
                };
                stats.accepted_rows += 1;

                match pending_group.as_mut() {
                    None => {
                        pending_group = Some(EventGroup {
                            local_timestamp_us: event.local_timestamp_us,
                            events: vec![event],
                        });
                    }
                    Some(group) => match event.local_timestamp_us.cmp(&group.local_timestamp_us) {
                        Ordering::Equal => group.events.push(event),
                        Ordering::Greater => {
                            let finished = pending_group
                                .replace(EventGroup {
                                    local_timestamp_us: event.local_timestamp_us,
                                    events: vec![event],
                                })
                                .expect("pending group must exist before replace");
                            let consumer_started = Instant::now();
                            on_group(finished)?;
                            consumer_ns += consumer_started.elapsed().as_nanos();
                        }
                        Ordering::Less => {
                            bail!(
                                "streaming parquet input is not sorted by local_timestamp: previous_group={}, current={} in {} row {raw_row_index}; normalize/sort before streaming",
                                group.local_timestamp_us,
                                event.local_timestamp_us,
                                file_path.display(),
                            );
                        }
                    },
                }
            }
        }
        stats.input_files += 1;
    }

    if let Some(group) = pending_group {
        let consumer_started = Instant::now();
        on_group(group)?;
        consumer_ns += consumer_started.elapsed().as_nanos();
    }
    stats.consumer_ms = consumer_ns / 1_000_000;
    stats.decode_ms = started
        .elapsed()
        .as_millis()
        .saturating_sub(stats.consumer_ms);
    Ok(stats)
}

fn parquet_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.exists() {
        bail!("input path does not exist: {}", path.display());
    }
    let files = WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "parquet"))
        .collect::<Vec<_>>();
    if files.is_empty() {
        bail!("no parquet files found under {}", path.display());
    }
    Ok(files)
}

fn required_leaf_indices(field_names: &[&str]) -> Result<Vec<usize>> {
    let required = [
        "timestamp",
        "local_timestamp",
        "is_snapshot",
        "side",
        "price",
        "amount",
    ];
    let mut out = Vec::with_capacity(required.len());
    for name in required {
        let idx = field_names
            .iter()
            .position(|candidate| *candidate == name)
            .ok_or_else(|| {
                anyhow!("required parquet column {name:?} not found; available={field_names:?}")
            })?;
        out.push(idx);
    }
    Ok(out)
}

fn parse_event(
    cols: &ProjectedColumns<'_>,
    i: usize,
    file_path: &Path,
    file_index: u32,
    raw_row_index: u64,
    stats: &mut ReadStats,
) -> Result<Option<L2Event>> {
    let side_raw = cols.side.value(i);
    let Some(side) = Side::parse(side_raw) else {
        stats.unknown_side_count += 1;
        return Ok(None);
    };
    let (price_atoms, price_scale) = match decimal_atoms_scale(cols.price, i) {
        Ok(v) => v,
        Err(err) => {
            stats.invalid_price_count += 1;
            if stats.invalid_price_count <= 5 {
                eprintln!(
                    "skipping row with invalid price in {} row {raw_row_index}: {err}",
                    file_path.display()
                );
            }
            return Ok(None);
        }
    };
    let amount = match numeric_value(cols.amount, i) {
        Ok(v) => v,
        Err(err) => {
            stats.invalid_amount_count += 1;
            if stats.invalid_amount_count <= 5 {
                eprintln!(
                    "skipping row with invalid amount in {} row {raw_row_index}: {err}",
                    file_path.display()
                );
            }
            return Ok(None);
        }
    };
    Ok(Some(L2Event {
        exchange_timestamp_us: cols.timestamp.value(i),
        local_timestamp_us: cols.local_timestamp.value(i),
        is_snapshot: cols.is_snapshot.value(i),
        side,
        price_atoms,
        price_scale,
        amount,
        source_file_index: file_index,
        source_row_index: raw_row_index,
    }))
}

struct ProjectedColumns<'a> {
    timestamp: &'a UInt64Array,
    local_timestamp: &'a UInt64Array,
    is_snapshot: &'a BooleanArray,
    side: StringArray,
    price: &'a dyn Array,
    amount: &'a dyn Array,
}

impl<'a> ProjectedColumns<'a> {
    fn from_batch(batch: &'a RecordBatch) -> Result<Self> {
        let timestamp = required_col(batch, "timestamp")?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| anyhow!("timestamp is not UInt64"))?;
        let local_timestamp = required_col(batch, "local_timestamp")?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| anyhow!("local_timestamp is not UInt64"))?;
        let is_snapshot = required_col(batch, "is_snapshot")?
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| anyhow!("is_snapshot is not Boolean"))?;
        let side_cast =
            cast(required_col(batch, "side")?, &DataType::Utf8).context("cast side to Utf8")?;
        let side = side_cast
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("side cast to Utf8 did not produce StringArray"))?
            .clone();
        let price = required_col(batch, "price")?;
        let amount = required_col(batch, "amount")?;
        Ok(Self {
            timestamp,
            local_timestamp,
            is_snapshot,
            side,
            price,
            amount,
        })
    }
}

fn required_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array> {
    let idx = batch
        .schema()
        .fields()
        .iter()
        .position(|candidate| candidate.name() == name)
        .ok_or_else(|| {
            let schema = batch.schema();
            let available = schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>();
            anyhow!("required projected column {name:?} not found; available={available:?}")
        })?;
    Ok(batch.column(idx).as_ref())
}

fn decimal_atoms_scale(array: &dyn Array, i: usize) -> Result<(i128, i32)> {
    if array.is_null(i) {
        bail!("null price");
    }
    match array.data_type() {
        DataType::Decimal128(_, scale) => {
            let dec = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| anyhow!("decimal downcast failed"))?;
            Ok((dec.value(i), (*scale).into()))
        }
        DataType::Float64 => {
            let v = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(i);
            if !v.is_finite() {
                bail!("non-finite float price");
            }
            Ok(((v * 1e10).round() as i128, 10))
        }
        DataType::Int64 => {
            let v = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(i);
            Ok((v as i128, 0))
        }
        other => bail!("unsupported price type: {other:?}"),
    }
}

fn numeric_value(array: &dyn Array, i: usize) -> Result<f64> {
    if array.is_null(i) {
        bail!("null numeric value");
    }
    match array.data_type() {
        DataType::Decimal128(_, scale) => {
            let dec = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| anyhow!("decimal downcast failed"))?;
            Ok(atoms_to_f64(dec.value(i), (*scale).into()))
        }
        DataType::Float64 => {
            let v = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(i);
            if !v.is_finite() {
                bail!("non-finite float amount");
            }
            Ok(v)
        }
        DataType::Int64 => Ok(array
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(i) as f64),
        other => bail!("unsupported numeric amount type: {other:?}"),
    }
}
