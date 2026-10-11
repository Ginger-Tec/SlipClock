use anyhow::{Context, Result, bail};
use chrono::NaiveDate;
use clap::{Parser, Subcommand};
use liq_l2_engine::engine::EngineRunner;
use liq_l2_engine::io::stream_event_groups_from_parquet_dir;
use liq_l2_engine::model::{EngineConfig, Grid, OutputLayout, RefPrice};
use liq_l2_engine::output::{write_feature_parquet, write_qc_json};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(version, about = "Generate Tardis incremental L2 slippage features.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Read Tardis incremental_book_L2 Parquet and emit slippage feature Parquet + QC JSON.
    Run(RunArgs),
    /// Process one symbol across an inclusive date range, writing one partition per date.
    RunRange(RunRangeArgs),
}

#[derive(Debug, Parser)]
struct RunArgs {
    /// Input Parquet file or directory containing one symbol/date partition.
    #[arg(long)]
    input: PathBuf,

    #[arg(long, default_value = "binance-futures")]
    exchange: String,

    #[arg(long)]
    symbol: String,

    /// UTC date partition, yyyy-mm-dd.
    #[arg(long)]
    date: String,

    /// Local-clock sampling grid, e.g. 1m or 1s.
    #[arg(long, default_value = "1m")]
    grid: String,

    /// G001 supports local timestamp sampling only.
    #[arg(long, default_value = "local")]
    sample_clock: String,

    /// Comma-separated USD notional sizes.
    #[arg(long, default_value = "1000,10000,100000")]
    notionals: String,

    /// Comma-separated reference prices: mid,micro.
    #[arg(long, default_value = "mid,micro")]
    refs: String,

    /// Output schema: full keeps fill diagnostics; compact keeps cost/complete/levels columns.
    #[arg(long, default_value = "full")]
    output_layout: String,

    /// Engine variant: btree or vec.
    #[arg(long, default_value = "btree")]
    engine: String,

    /// Output Parquet file path, or directory to receive slippage.parquet.
    #[arg(long)]
    out: PathBuf,

    /// Optional QC JSON path. Defaults beside output Parquet as *.qc.json.
    #[arg(long)]
    qc_out: Option<PathBuf>,

    /// Arrow record batch size for Parquet decoding.
    #[arg(long, default_value_t = 262_144)]
    batch_size: usize,

    /// Treat local_timestamp regressions as fatal.
    #[arg(long, default_value_t = false)]
    strict: bool,
}

#[derive(Debug, Parser)]
struct RunRangeArgs {
    /// Input root. Accepts either a symbol root containing date=YYYY-MM-DD partitions, or the Tardis parquet root.
    #[arg(long)]
    input_root: PathBuf,

    #[arg(long, default_value = "binance-futures")]
    exchange: String,

    #[arg(long)]
    symbol: String,

    /// Inclusive UTC start date, yyyy-mm-dd.
    #[arg(long)]
    start_date: String,

    /// Inclusive UTC end date, yyyy-mm-dd.
    #[arg(long)]
    end_date: String,

    /// Local-clock sampling grid. For current production sweeps this should stay at 1m.
    #[arg(long, default_value = "1m")]
    grid: String,

    /// Streaming path supports local timestamp sampling only.
    #[arg(long, default_value = "local")]
    sample_clock: String,

    /// Comma-separated USD notional sizes.
    #[arg(long, default_value = "1000,10000,100000")]
    notionals: String,

    /// Comma-separated reference prices: mid,micro.
    #[arg(long, default_value = "mid,micro")]
    refs: String,

    /// Output schema: full keeps fill diagnostics; compact keeps cost/complete/levels columns.
    #[arg(long, default_value = "compact")]
    output_layout: String,

    /// Engine variant: btree or vec.
    #[arg(long, default_value = "btree")]
    engine: String,

    /// Output root. Each day is written as exchange=/symbol=/date=/grid=/slippage.parquet.
    #[arg(long)]
    out_root: PathBuf,

    /// Arrow record batch size for Parquet decoding.
    #[arg(long, default_value_t = 262_144)]
    batch_size: usize,

    /// Treat local_timestamp regressions as fatal inside the engine. The streaming reader always fails closed on unsorted input.
    #[arg(long, default_value_t = false)]
    strict: bool,

    /// Skip missing date partitions instead of failing the whole range.
    #[arg(long, default_value_t = false)]
    skip_missing: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args),
        Command::RunRange(args) => run_range(args),
    }
}

fn run(args: RunArgs) -> Result<()> {
    if args.sample_clock != "local" {
        bail!(
            "G001 supports --sample-clock local only; got {:?}",
            args.sample_clock
        );
    }
    if !matches!(
        args.engine.as_str(),
        "btree" | "vec" | "sorted-vec" | "sorted_vec"
    ) {
        bail!("unsupported --engine {:?}; use btree or vec", args.engine);
    }
    if args.batch_size == 0 {
        bail!("--batch-size must be positive");
    }
    validate_hive_input_labels(&args.input, &args.exchange, &args.symbol, &args.date)?;

    let config = EngineConfig {
        exchange: args.exchange,
        symbol: args.symbol,
        date: args.date,
        grid: Grid::parse(&args.grid).context("parse --grid")?,
        engine_variant: args.engine,
        sample_clock: args.sample_clock,
        output_layout: OutputLayout::parse(&args.output_layout).context("parse --output-layout")?,
        notionals: parse_notionals(&args.notionals)?,
        refs: parse_refs(&args.refs)?,
        strict: args.strict,
    };

    let output_path = resolve_output_path(&args.out);
    let qc_path = args
        .qc_out
        .unwrap_or_else(|| output_path.with_extension("qc.json"));
    let summary = execute_one_day(&args.input, config, output_path, qc_path, args.batch_size)?;

    println!("{}", summary);
    Ok(())
}

fn run_range(args: RunRangeArgs) -> Result<()> {
    if args.sample_clock != "local" {
        bail!(
            "streaming range supports --sample-clock local only; got {:?}",
            args.sample_clock
        );
    }
    if !matches!(
        args.engine.as_str(),
        "btree" | "vec" | "sorted-vec" | "sorted_vec"
    ) {
        bail!("unsupported --engine {:?}; use btree or vec", args.engine);
    }
    if args.batch_size == 0 {
        bail!("--batch-size must be positive");
    }
    let start =
        NaiveDate::parse_from_str(&args.start_date, "%Y-%m-%d").context("parse --start-date")?;
    let end = NaiveDate::parse_from_str(&args.end_date, "%Y-%m-%d").context("parse --end-date")?;
    if end < start {
        bail!("--end-date must be >= --start-date");
    }
    let grid = Grid::parse(&args.grid).context("parse --grid")?;
    let output_layout =
        OutputLayout::parse(&args.output_layout).context("parse --output-layout")?;
    let notionals = parse_notionals(&args.notionals)?;
    let refs = parse_refs(&args.refs)?;

    let mut cases = Vec::new();
    let mut day = start;
    while day <= end {
        let date = day.format("%Y-%m-%d").to_string();
        let input = resolve_range_input_path(&args.input_root, &args.exchange, &args.symbol, &date);
        if !input.exists() {
            if args.skip_missing {
                cases.push(serde_json::json!({
                    "status": "skipped_missing",
                    "date": date,
                    "input": input,
                }));
                day = day.succ_opt().expect("valid next day in bounded range");
                continue;
            }
            bail!("missing input partition for {date}: {}", input.display());
        }
        validate_hive_input_labels(&input, &args.exchange, &args.symbol, &date)?;
        let config = EngineConfig {
            exchange: args.exchange.clone(),
            symbol: args.symbol.to_ascii_uppercase(),
            date: date.clone(),
            grid,
            engine_variant: args.engine.clone(),
            sample_clock: args.sample_clock.clone(),
            output_layout,
            notionals: notionals.clone(),
            refs: refs.clone(),
            strict: args.strict,
        };
        let output_path = partitioned_output_path(
            &args.out_root,
            &config.exchange,
            &config.symbol,
            &date,
            &config.grid.name(),
        );
        let qc_path = output_path.with_extension("qc.json");
        cases.push(execute_one_day(
            &input,
            config,
            output_path,
            qc_path,
            args.batch_size,
        )?);

        day = day.succ_opt().expect("valid next day in bounded range");
    }

    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "mode": "run-range",
            "symbol": args.symbol.to_ascii_uppercase(),
            "startDate": args.start_date,
            "endDate": args.end_date,
            "days": cases.len(),
            "cases": cases,
        })
    );
    Ok(())
}

fn execute_one_day(
    input: &Path,
    config: EngineConfig,
    output_path: PathBuf,
    qc_path: PathBuf,
    batch_size: usize,
) -> Result<serde_json::Value> {
    let mut runner = EngineRunner::new(config.clone())?;
    let read_stats = stream_event_groups_from_parquet_dir(input, batch_size, |group| {
        runner.process_group(group)
    })?;
    let mut output = runner.finish()?;
    output.qc.decode_ms = read_stats.decode_ms;
    output.qc.input_files = read_stats.input_files;
    output.qc.record_batches = read_stats.record_batches;
    output.qc.row_groups = read_stats.row_groups;
    output.qc.input_rows = read_stats.input_rows;
    output.qc.accepted_rows = read_stats.accepted_rows;
    output.qc.unknown_side_count = read_stats.unknown_side_count;
    output.qc.invalid_price_count = read_stats.invalid_price_count;
    output.qc.invalid_amount_count = read_stats.invalid_amount_count;
    if output.qc.runtime_ms > 0 {
        output.qc.rows_per_sec =
            output.qc.accepted_rows as f64 / (output.qc.runtime_ms as f64 / 1000.0);
    }

    let write_started = Instant::now();
    let output_bytes = write_feature_parquet(&output_path, &output.rows, &config)?;
    output.qc.write_ms = write_started.elapsed().as_millis();
    output.qc.output_path = Some(output_path.clone());
    output.qc.output_bytes = output_bytes;
    output.qc.output_rows = output.rows.len() as u64;
    write_qc_json(&qc_path, &output.qc)?;

    Ok(serde_json::json!({
        "status": "ok",
        "date": config.date,
        "inputFiles": output.qc.input_files,
        "inputRows": output.qc.input_rows,
        "acceptedRows": output.qc.accepted_rows,
        "outputRows": output.qc.output_rows,
        "outputPath": output_path,
        "qcPath": qc_path,
        "decodeMs": output.qc.decode_ms,
        "engineRuntimeMs": output.qc.runtime_ms,
        "bookUpdateMs": output.qc.book_update_ms,
        "samplingMs": output.qc.sampling_ms,
        "writeMs": output.qc.write_ms,
        "outputLayout": config.output_layout.name(),
        "outputBytes": output.qc.output_bytes,
    }))
}

fn resolve_output_path(out: &Path) -> PathBuf {
    if out.extension().is_some() {
        out.to_path_buf()
    } else {
        out.join("slippage.parquet")
    }
}

fn resolve_range_input_path(
    input_root: &Path,
    exchange: &str,
    symbol: &str,
    date: &str,
) -> PathBuf {
    let symbol = symbol.to_ascii_uppercase();
    let candidates = [
        input_root.join(format!("date={date}")),
        input_root
            .join(format!("exchange={exchange}"))
            .join("data_type=incremental_book_L2")
            .join(format!("symbol={symbol}"))
            .join(format!("date={date}")),
        input_root
            .join(format!("exchange={exchange}"))
            .join("data_type=incremental_l2")
            .join(format!("symbol={symbol}"))
            .join(format!("date={date}")),
        input_root
            .join(format!("symbol={symbol}"))
            .join(format!("date={date}")),
    ];
    candidates
        .iter()
        .find(|candidate| candidate.exists())
        .cloned()
        .unwrap_or_else(|| candidates[0].clone())
}

fn partitioned_output_path(
    out_root: &Path,
    exchange: &str,
    symbol: &str,
    date: &str,
    grid: &str,
) -> PathBuf {
    out_root
        .join(format!("exchange={exchange}"))
        .join(format!("symbol={}", symbol.to_ascii_uppercase()))
        .join(format!("date={date}"))
        .join(format!("grid={grid}"))
        .join("slippage.parquet")
}

fn validate_hive_input_labels(
    input: &Path,
    expected_exchange: &str,
    expected_symbol: &str,
    expected_date: &str,
) -> Result<()> {
    let expected_symbol = expected_symbol.to_ascii_uppercase();
    let expected = [
        ("exchange", expected_exchange.to_string()),
        ("symbol", expected_symbol),
        ("date", expected_date.to_string()),
    ];
    for component in input.components() {
        let raw = component.as_os_str().to_string_lossy();
        let Some((key, value)) = raw.split_once('=') else {
            continue;
        };
        for (expected_key, expected_value) in &expected {
            if key == *expected_key && value != expected_value {
                bail!(
                    "input Hive label {key}={value:?} does not match requested {key}={expected_value:?}"
                );
            }
        }
    }
    Ok(())
}

fn parse_notionals(raw: &str) -> Result<Vec<f64>> {
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            let value: f64 = s.parse().with_context(|| format!("parse notional {s:?}"))?;
            if !value.is_finite() || value <= 0.0 {
                bail!("notional must be positive finite: {s:?}");
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        bail!("at least one notional is required");
    }
    Ok(values)
}

fn parse_refs(raw: &str) -> Result<Vec<RefPrice>> {
    let refs = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| match s.to_ascii_lowercase().as_str() {
            "mid" => Ok(RefPrice::Mid),
            "micro" => Ok(RefPrice::Micro),
            other => bail!("unsupported reference {other:?}; use mid,micro"),
        })
        .collect::<Result<Vec<_>>>()?;
    if refs.is_empty() {
        bail!("at least one reference is required");
    }
    Ok(refs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_matching_hive_labels() {
        validate_hive_input_labels(
            Path::new("exchange=binance-futures/symbol=ARBUSDT/date=2024-07-01"),
            "binance-futures",
            "ARBUSDT",
            "2024-07-01",
        )
        .unwrap();
    }

    #[test]
    fn rejects_mismatched_symbol_label() {
        let err = validate_hive_input_labels(
            Path::new("exchange=binance-futures/symbol=BTCUSDT/date=2024-07-01"),
            "binance-futures",
            "ARBUSDT",
            "2024-07-01",
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not match requested"));
    }

    #[test]
    fn resolves_range_input_from_symbol_root() {
        let tmp = tempfile::tempdir().unwrap();
        let day = tmp.path().join("date=2024-07-01");
        std::fs::create_dir_all(&day).unwrap();
        let resolved =
            resolve_range_input_path(tmp.path(), "binance-futures", "arbusdt", "2024-07-01");
        assert_eq!(resolved, day);
    }

    #[test]
    fn builds_partitioned_output_path() {
        let path = partitioned_output_path(
            Path::new("features"),
            "binance-futures",
            "arbusdt",
            "2024-07-01",
            "1m",
        );
        assert_eq!(
            path,
            Path::new(
                "features/exchange=binance-futures/symbol=ARBUSDT/date=2024-07-01/grid=1m/slippage.parquet"
            )
        );
    }
}
