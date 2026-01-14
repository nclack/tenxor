use bytesize::ByteSize;
use clap::Parser;
use std::io::Write;
use std::thread::sleep;
use std::time::{Duration, Instant};
use tenxor::CudaWriter;

#[derive(Parser, Debug)]
#[command(name = "benchmark")]
#[command(about = "Benchmark CudaWriter streaming performance")]
struct Args {
    /// Total bytes to stream
    #[arg(long, default_value = "107374182400")] // 100 GB
    total_bytes: usize,

    /// Size of each write operation
    #[arg(long, default_value = "104857600")] // 100 MB
    write_size: usize,

    /// CudaWriter buffer capacity
    #[arg(long, default_value = "1073741824")] // 1 GB
    buffer_capacity: usize,

    /// CUDA device ID
    #[arg(long, default_value = "0")]
    device: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    tracing::info!(
        total = %ByteSize(args.total_bytes as u64),
        write_size = %ByteSize(args.write_size as u64),
        buffer_capacity = %ByteSize(args.buffer_capacity as u64),
        device = args.device,
        "Starting benchmark"
    );

    // Initialize CUDA context and stream
    let ctx = cudarc::driver::CudaContext::new(args.device)
        .map_err(|e| format!("Failed to create CUDA context: {:?}", e))?;
    let stream = ctx
        .new_stream()
        .map_err(|e| format!("Failed to create CUDA stream: {:?}", e))?;

    // Create CudaWriter
    let mut writer = CudaWriter::init(ctx.clone(), stream.clone(), args.buffer_capacity)
        .map_err(|e| format!("Failed to create CudaWriter: {:?}", e))?;

    // Pre-allocate zero buffer
    let write_buffer = vec![0u8; args.write_size];

    let num_writes = args.total_bytes / args.write_size;
    let mut write_latencies = Vec::with_capacity(num_writes);

    tracing::info!("Starting write loop with {} writes", num_writes);

    let start = Instant::now();
    let mut bytes_written = 0;

    while bytes_written < args.total_bytes {
        let write_start = Instant::now();
        let n = writer.write(&write_buffer)?;
        let write_duration = write_start.elapsed();

        if n > 0 {
            write_latencies.push(write_duration);
            bytes_written += n;

            if bytes_written % (10 * args.write_size) == 0 {
                let progress_pct = (bytes_written as f64 / args.total_bytes as f64) * 100.0;
                tracing::info!(
                    bytes_written = %ByteSize(bytes_written as u64),
                    progress = format!("{:.1}%", progress_pct),
                    "Progress"
                );
            }
        }
        sleep(Duration::from_millis(1));
    }

    let write_time = start.elapsed();

    tracing::info!("Write loop complete, flushing...");
    let flush_start = Instant::now();
    writer.flush()?;
    stream
        .synchronize()
        .map_err(|e| format!("Failed to synchronize stream: {:?}", e))?;
    let flush_time = flush_start.elapsed();

    let total_time = start.elapsed();

    // Calculate statistics
    write_latencies.sort();
    let p50 = write_latencies[write_latencies.len() / 2];
    let p95 = write_latencies[write_latencies.len() * 95 / 100];
    let p99 = write_latencies[write_latencies.len() * 99 / 100];
    let avg_write: std::time::Duration =
        write_latencies.iter().sum::<std::time::Duration>() / write_latencies.len() as u32;

    let throughput_gbs = (args.total_bytes as f64) / total_time.as_secs_f64() / 1e9;
    let pcie_efficiency = (throughput_gbs / 64.0) * 100.0; // Assuming 64 GB/s theoretical max

    // Print results
    println!("\n=== Benchmark Results ===");
    println!("Total bytes:        {}", ByteSize(args.total_bytes as u64));
    println!("Write size:         {}", ByteSize(args.write_size as u64));
    println!(
        "Buffer capacity:    {}",
        ByteSize(args.buffer_capacity as u64)
    );
    println!("Number of writes:   {}", num_writes);
    println!();
    println!("Total time:         {:.3} s", total_time.as_secs_f64());
    println!("Write time:         {:.3} s", write_time.as_secs_f64());
    println!("Flush time:         {:.3} s", flush_time.as_secs_f64());
    println!();
    println!("Throughput:         {:.2} GB/s", throughput_gbs);
    println!("PCIe efficiency:    {:.1}%", pcie_efficiency);
    println!();
    println!("Write latency (avg): {:?}", avg_write);
    println!("Write latency (p50): {:?}", p50);
    println!("Write latency (p95): {:?}", p95);
    println!("Write latency (p99): {:?}", p99);

    Ok(())
}
