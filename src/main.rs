use std::io::Write;
use tensorstream::CudaWriter;

pub fn thing() -> Result<(), std::io::Error> {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut w = CudaWriter::init(ctx, stream, 1 << 10).unwrap();

    let data = vec![0u8; 1024];
    w.write(&data)?;
    Ok(())
}
fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();
    println!("Hello, world!");
    thing().unwrap();
}
