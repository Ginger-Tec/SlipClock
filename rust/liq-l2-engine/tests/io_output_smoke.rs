use anyhow::Result;
use arrow::array::{
    BooleanArray, Decimal128Array, Float64Array, StringArray, StringDictionaryBuilder, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::record_batch::RecordBatch;
use liq_l2_engine::engine::day_bounds_us;
use liq_l2_engine::io::{read_events_from_parquet_dir, stream_event_groups_from_parquet_dir};
use liq_l2_engine::output::{write_feature_parquet, write_qc_json};
use liq_l2_engine::{EngineConfig, Grid, run_events};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::sync::Arc;

#[test]
fn reads_tardis_like_decimal_dictionary_fixture_and_writes_outputs() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let input_path = tmp.path().join("20240701.ARBUSDT.parquet");
    write_tardis_like_fixture(&input_path)?;

    let (events, read_stats) = read_events_from_parquet_dir(tmp.path(), 2)?;
    assert_eq!(read_stats.input_files, 1);
    assert!(read_stats.record_batches >= 1);
    assert_eq!(read_stats.input_rows, 4);
    assert_eq!(read_stats.accepted_rows, 4);
    assert_eq!(events.len(), 4);
    assert_eq!(events[0].local_timestamp_us, events[1].local_timestamp_us);
    assert_eq!(events[0].price_scale, 10);

    let config = EngineConfig {
        symbol: "ARBUSDT".to_string(),
        date: "2024-07-01".to_string(),
        grid: Grid::Minutes(1),
        ..EngineConfig::default()
    };
    let mut out = run_events(events, config.clone())?;
    out.qc.input_files = read_stats.input_files;
    out.qc.record_batches = read_stats.record_batches;
    out.qc.row_groups = read_stats.row_groups;

    let feature_path = tmp.path().join("features/slippage.parquet");
    let qc_path = tmp.path().join("features/slippage.qc.json");
    let feature_bytes = write_feature_parquet(&feature_path, &out.rows, &config)?;
    out.qc.output_bytes = feature_bytes;
    out.qc.output_path = Some(feature_path.clone());
    write_qc_json(&qc_path, &out.qc)?;

    assert!(feature_bytes > 0);
    assert!(qc_path.exists());

    let file = File::open(feature_path)?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let batch = reader.next().expect("one output batch")?;
    let names: Vec<_> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert!(names.contains(&"sample_local_timestamp_us".to_string()));
    assert!(names.contains(&"spread_bps".to_string()));
    assert!(names.contains(&"buy_1000_mid_cost_bps".to_string()));
    assert!(batch.num_rows() > 0);
    Ok(())
}

#[test]
fn preserves_source_order_within_equal_local_timestamp_groups() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let input_path = tmp.path().join("same-local.parquet");
    write_same_local_order_fixture(&input_path)?;

    let (events, read_stats) = read_events_from_parquet_dir(tmp.path(), 16)?;
    assert_eq!(read_stats.input_rows, 4);
    assert_eq!(read_stats.accepted_rows, 4);
    let bid_updates: Vec<_> = events
        .iter()
        .filter(|e| e.side == liq_l2_engine::Side::Bid)
        .map(|e| (e.exchange_timestamp_us, e.price_atoms))
        .collect();
    assert_eq!(
        bid_updates,
        vec![
            (200, 99_i128 * 10_000_000_000),
            (100, 100_i128 * 10_000_000_000)
        ]
    );

    let config = EngineConfig {
        symbol: "ORDER".to_string(),
        date: "2024-07-01".to_string(),
        grid: Grid::Minutes(1),
        ..EngineConfig::default()
    };
    let out = run_events(events, config)?;
    assert_eq!(out.rows[0].bid_price, 100.0);
    Ok(())
}

#[test]
fn rejects_non_finite_float_price_and_amount_rows() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let input_path = tmp.path().join("float-invalid.parquet");
    write_float_invalid_fixture(&input_path)?;

    let (events, read_stats) = read_events_from_parquet_dir(tmp.path(), 16)?;
    assert_eq!(read_stats.input_rows, 4);
    assert_eq!(read_stats.accepted_rows, 2);
    assert_eq!(read_stats.invalid_price_count, 1);
    assert_eq!(read_stats.invalid_amount_count, 1);
    assert_eq!(events.len(), 2);
    Ok(())
}

#[test]
fn streams_local_timestamp_groups_without_materializing_all_events() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let input_path = tmp.path().join("stream.parquet");
    write_tardis_like_fixture(&input_path)?;

    let mut groups = Vec::new();
    let read_stats = stream_event_groups_from_parquet_dir(tmp.path(), 2, |group| {
        groups.push(group);
        Ok(())
    })?;
    assert_eq!(read_stats.input_rows, 4);
    assert_eq!(read_stats.accepted_rows, 4);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].events.len(), 2);
    assert_eq!(groups[1].events.len(), 2);

    let config = EngineConfig {
        symbol: "ARBUSDT".to_string(),
        date: "2024-07-01".to_string(),
        grid: Grid::Minutes(1),
        ..EngineConfig::default()
    };
    let out = liq_l2_engine::engine::run_event_groups(groups, config)?;
    assert_eq!(out.rows.len(), 1_439);
    Ok(())
}

#[test]
fn streaming_rejects_unsorted_local_timestamps() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let input_path = tmp.path().join("unsorted-local.parquet");
    write_unsorted_local_fixture(&input_path)?;

    let err = stream_event_groups_from_parquet_dir(tmp.path(), 16, |_| Ok(()))
        .expect_err("streaming must fail closed on unsorted local timestamps");
    assert!(err.to_string().contains("not sorted by local_timestamp"));
    Ok(())
}

fn write_tardis_like_fixture(path: &std::path::Path) -> Result<()> {
    let (start, _) = day_bounds_us("2024-07-01")?;
    let timestamps = UInt64Array::from_iter_values([
        start + 60_000_000,
        start + 60_000_000,
        start + 120_000_000,
        start + 120_000_000,
    ]);
    let local_timestamps = UInt64Array::from_iter_values([
        start + 60_000_000,
        start + 60_000_000,
        start + 120_000_000,
        start + 120_000_000,
    ]);
    let snapshots = BooleanArray::from_iter([Some(true), Some(true), Some(false), Some(false)]);
    let mut side_builder = StringDictionaryBuilder::<Int32Type>::new();
    side_builder.append("bid")?;
    side_builder.append("ask")?;
    side_builder.append("bid")?;
    side_builder.append("ask")?;
    let side = side_builder.finish();
    let price = Decimal128Array::from_iter_values([
        99_i128 * 10_000_000_000,
        101_i128 * 10_000_000_000,
        100_i128 * 10_000_000_000,
        102_i128 * 10_000_000_000,
    ])
    .with_precision_and_scale(24, 10)?;
    let amount = Decimal128Array::from_iter_values([
        10_i128 * 10_000_000_000,
        10_i128 * 10_000_000_000,
        11_i128 * 10_000_000_000,
        12_i128 * 10_000_000_000,
    ])
    .with_precision_and_scale(24, 10)?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("exchange", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("local_timestamp", DataType::UInt64, false),
        Field::new("is_snapshot", DataType::Boolean, false),
        Field::new(
            "side",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("price", DataType::Decimal128(24, 10), false),
        Field::new("amount", DataType::Decimal128(24, 10), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(["binance-futures"; 4])),
            Arc::new(StringArray::from_iter_values(["ARBUSDT"; 4])),
            Arc::new(timestamps),
            Arc::new(local_timestamps),
            Arc::new(snapshots),
            Arc::new(side),
            Arc::new(price),
            Arc::new(amount),
        ],
    )?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_same_local_order_fixture(path: &std::path::Path) -> Result<()> {
    let (start, _) = day_bounds_us("2024-07-01")?;
    let local = start + 60_000_000;
    let timestamps = UInt64Array::from_iter_values([200, 300, 100, 400]);
    let local_timestamps = UInt64Array::from_iter_values([local, local, local, local]);
    let snapshots = BooleanArray::from_iter([Some(true), Some(true), Some(true), Some(true)]);
    let mut side_builder = StringDictionaryBuilder::<Int32Type>::new();
    side_builder.append("bid")?;
    side_builder.append("ask")?;
    side_builder.append("bid")?;
    side_builder.append("ask")?;
    let side = side_builder.finish();
    let price = Decimal128Array::from_iter_values([
        99_i128 * 10_000_000_000,
        101_i128 * 10_000_000_000,
        100_i128 * 10_000_000_000,
        102_i128 * 10_000_000_000,
    ])
    .with_precision_and_scale(24, 10)?;
    let amount = Decimal128Array::from_iter_values([
        10_i128 * 10_000_000_000,
        10_i128 * 10_000_000_000,
        10_i128 * 10_000_000_000,
        10_i128 * 10_000_000_000,
    ])
    .with_precision_and_scale(24, 10)?;
    write_l2_batch(
        path,
        timestamps,
        local_timestamps,
        snapshots,
        Arc::new(side),
        Arc::new(price),
        Arc::new(amount),
    )
}

fn write_float_invalid_fixture(path: &std::path::Path) -> Result<()> {
    let (start, _) = day_bounds_us("2024-07-01")?;
    let timestamps = UInt64Array::from_iter_values([start + 1, start + 2, start + 3, start + 4]);
    let local_timestamps =
        UInt64Array::from_iter_values([start + 1, start + 2, start + 3, start + 4]);
    let snapshots = BooleanArray::from_iter([Some(true), Some(true), Some(true), Some(true)]);
    let mut side_builder = StringDictionaryBuilder::<Int32Type>::new();
    side_builder.append("bid")?;
    side_builder.append("ask")?;
    side_builder.append("bid")?;
    side_builder.append("ask")?;
    let side = side_builder.finish();
    let price = Float64Array::from_iter_values([99.0, 101.0, f64::NAN, 102.0]);
    let amount = Float64Array::from_iter_values([10.0, f64::INFINITY, 10.0, 10.0]);
    write_l2_batch(
        path,
        timestamps,
        local_timestamps,
        snapshots,
        Arc::new(side),
        Arc::new(price),
        Arc::new(amount),
    )
}

fn write_unsorted_local_fixture(path: &std::path::Path) -> Result<()> {
    let (start, _) = day_bounds_us("2024-07-01")?;
    let timestamps = UInt64Array::from_iter_values([start + 1, start + 2, start + 3, start + 4]);
    let local_timestamps = UInt64Array::from_iter_values([
        start + 120_000_000,
        start + 120_000_000,
        start + 60_000_000,
        start + 60_000_000,
    ]);
    let snapshots = BooleanArray::from_iter([Some(true), Some(true), Some(false), Some(false)]);
    let mut side_builder = StringDictionaryBuilder::<Int32Type>::new();
    side_builder.append("bid")?;
    side_builder.append("ask")?;
    side_builder.append("bid")?;
    side_builder.append("ask")?;
    let side = side_builder.finish();
    let price = Decimal128Array::from_iter_values([
        99_i128 * 10_000_000_000,
        101_i128 * 10_000_000_000,
        100_i128 * 10_000_000_000,
        102_i128 * 10_000_000_000,
    ])
    .with_precision_and_scale(24, 10)?;
    let amount = Decimal128Array::from_iter_values([
        10_i128 * 10_000_000_000,
        10_i128 * 10_000_000_000,
        10_i128 * 10_000_000_000,
        10_i128 * 10_000_000_000,
    ])
    .with_precision_and_scale(24, 10)?;
    write_l2_batch(
        path,
        timestamps,
        local_timestamps,
        snapshots,
        Arc::new(side),
        Arc::new(price),
        Arc::new(amount),
    )
}

fn write_l2_batch(
    path: &std::path::Path,
    timestamps: UInt64Array,
    local_timestamps: UInt64Array,
    snapshots: BooleanArray,
    side: arrow::array::ArrayRef,
    price: arrow::array::ArrayRef,
    amount: arrow::array::ArrayRef,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("exchange", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("local_timestamp", DataType::UInt64, false),
        Field::new("is_snapshot", DataType::Boolean, false),
        Field::new(
            "side",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("price", price.data_type().clone(), false),
        Field::new("amount", amount.data_type().clone(), false),
    ]));
    let n = timestamps.len();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                "binance-futures",
                n,
            ))),
            Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                "TEST", n,
            ))),
            Arc::new(timestamps),
            Arc::new(local_timestamps),
            Arc::new(snapshots),
            side,
            price,
            amount,
        ],
    )?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}
