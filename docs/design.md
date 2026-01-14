# Design Documentation

## Architecture Overview

### Tiled Tensor Transformation Pipeline

The core pipeline transforms D-dimensional tensors through tiling and
transposition for efficient parallel compression.

#### Mathematical Foundation

Given:
- Input tensor shape: $(s_{D-1}, \ldots, s_0)$
- Tile shape: $(n_{D-1}, \ldots, n_0)$
- Number of tiles per dimension: $t_d = \lceil s_d / n_d \rceil$
- Total tiles: $T = \prod_{d=0}^{D-1} t_d$

Transformation sequence:
1. **Lift**: Treat input as shape $(t_{D-1}, n_{D-1}, \ldots, t_0, n_0)$
2. **Transpose**: Reorganize to $(t_{D-1}, \ldots, t_0, n_{D-1}, \ldots, n_0)$
3. **Compress**: Apply zstd to each of the $T$ tiles in parallel

#### Streaming Architecture

```
Input stream (row-major bytes)
    ↓
Writer interface (std::io::Write)
    ↓
Double-buffered host pinned memory (~1GB each)
    ↓ (async H2D transfer)
GPU staging buffer
    ↓ (CUDA kernel)
Tiled/transposed GPU memory
    ↓ (nvcompBatchedZstdCompressAsync)
Compressed tiles in GPU memory
    ↓ (async D2H transfer)
Host output buffer
    ↓
Disk I/O (eventual zarr store)
```

**Performance considerations:**
- Double buffering: ~1GB buffers @ 2-10 GB/s input → ~200ms fill time
- Overlap CPU→GPU transfer with kernel execution
- Batch compress all tiles in parallel using nvcomp

## Data Structures

We're working with streamed arrays/tensors so they're never fully realized
in memory. So we don't need an container object for the array. We just need
the metadata describing the array.

### Dimension Metadata

```rust
struct Dimension {
    size_px: usize;       // Size of this dimension
    tile_size_px: usize;  // Tile size for this dimension
}
```
### Layout

Describes the tensor layout and tiling scheme. Maximum 64 dimensions supported
(natural limit: if all dimensions have size 2, max elements = $2^{64}$).
Dimensions with size ≤1 are elided.

```rust
struct Layout<const RANK: usize> {
    shape: [usize;RANK],
    strides: [usize; RANK];
}
```

## Data Types

Supported pixel types:
- `uint8_t` (u8)
- `uint16_t` (u16) - most common for target application
- `uint32_t` (u32)
- `float` (f32)
- `double` (f64)

## Writer Interface

Writers process incoming byte streams by implementing the `std::io::Write`
trait. They are responsible for consuming as much of the input slice passed
to the write call as possible, and returning the number of bytes consumed
back to the caller.

When the writer can't write (e.g. a buffer is full or something is busy) it
should not block the caller. Instead it should return that 0 bytes were
consumed. The caller should retry again later. 0 bytes returned does not
indicate that there was an error condition or that the stream was closed -
those cases should be handled as errors.

## Benchmarking strategy

We're interested in trying to observe how close we can possibly get to the
physical limits of the system we're benchmarking on. As a result, we often
want to see the *best-case performance* rather than the worst. For measuring
bandwidth, we try the same thing many times and report the minimum elapsed
time.

The distribution of elapsed times and latencies is still interesting.

## Future Considerations

- Sharding layer (mentioned but not yet designed)
- Zarr store backend integration
- Dynamic buffer sizing based on input rate
- Multiscale downsampling with median (order statistic) filtering


