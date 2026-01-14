use cudarc::{
    self,
    driver::{result::event::query, CudaContext, CudaEvent, CudaStream, DriverError},
    runtime::{result::RuntimeError, sys},
};
use std::{io::Write, sync::Arc};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaWriterError {
    #[error("CUDA runtime error: {0:?}")]
    Runtime(RuntimeError),
    #[error("CUDA driver error: {0:?}")]
    Driver(DriverError),
}

impl From<RuntimeError> for CudaWriterError {
    fn from(err: RuntimeError) -> Self {
        CudaWriterError::Runtime(err)
    }
}

impl From<DriverError> for CudaWriterError {
    fn from(err: DriverError) -> Self {
        CudaWriterError::Driver(err)
    }
}

/*
 CudaWriter
 Enables high-speed streaming of data to the gpu.
*/

struct HostBuffer {
    ptr: *mut u8,
    cap: usize,
    len: usize,
    state: BufferState,
    event: CudaEvent,
}

impl HostBuffer {
    #[tracing::instrument(skip(ctx), fields(capacity))]
    fn init(ctx: Arc<CudaContext>, capacity: usize) -> Result<Self, CudaWriterError> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        unsafe {
            let result = sys::cudaHostAlloc(
                &mut ptr as *mut *mut std::ffi::c_void,
                capacity,
                sys::cudaHostAllocWriteCombined,
            );

            if result != sys::cudaError_t::cudaSuccess {
                tracing::error!(?result, "cudaHostAlloc failed");
                return Err(RuntimeError(result).into());
            }
        }

        let event = ctx.new_event(Some(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        ))?;
        Ok(Self {
            ptr: ptr as *mut u8,
            cap: capacity,
            len: 0,
            state: BufferState::Filling,
            event,
        })
    }

    fn capacity(&self) -> usize {
        self.cap
    }

    fn available(&self) -> usize {
        self.capacity() - self.len
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                sys::cudaFreeHost(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

#[derive(Debug)]
enum BufferState {
    Filling,
    Flushing,
}

pub struct CudaWriter {
    stream: Arc<CudaStream>,
    host: [HostBuffer; 2],
    active: usize,
    device_dst: *mut u8,
}

impl CudaWriter {
    #[tracing::instrument(skip(device, stream), fields(capacity))]
    pub fn init(
        device: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        capacity: usize,
    ) -> Result<Self, CudaWriterError> {
        let mut device_ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        unsafe {
            let result = sys::cudaMallocAsync(
                &mut device_ptr as *mut *mut std::ffi::c_void,
                capacity,
                // SAFETY: CUstream and cudaStream_t are the same underlying type
                stream.cu_stream() as sys::cudaStream_t,
            );

            if result != sys::cudaError_t::cudaSuccess {
                tracing::error!(?result, "cudaMallocAsync failed");
                return Err(RuntimeError(result).into());
            }
        }

        tracing::info!(capacity, "Allocated device buffer");

        Ok(Self {
            stream: stream.clone(),
            host: [
                HostBuffer::init(device.clone(), capacity)?,
                HostBuffer::init(device.clone(), capacity)?,
            ],
            active: 0,
            device_dst: device_ptr as *mut u8,
        })
    }

    /// Non-blocking prefix write
    #[tracing::instrument(skip(self, src), fields(src_len = src.len(), active = self.active))]
    fn write(&mut self, src: &[u8]) -> Result<usize, CudaWriterError> {
        self.try_reclaim();

        let buf = &mut self.host[self.active];

        if !matches!(buf.state, BufferState::Filling) {
            return Ok(0);
        }

        let n = src.len().min(buf.available());
        let offset = buf.len;

        // Copy to write-combined host buffer
        unsafe {
            let dst = buf.ptr.add(offset);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n);
        }
        buf.len += n;

        if buf.len == buf.capacity() {
            tracing::debug!(buffer_idx = self.active, "Buffer full, flushing");
            self.flush_active()?;
        }

        Ok(n)
    }

    /// Polls flushing buffers to reclaim them
    fn try_reclaim(&mut self) {
        for (idx, buf) in self.host.iter_mut().enumerate() {
            if let BufferState::Flushing = buf.state {
                if unsafe { query(buf.event.cu_event()) }.is_ok() {
                    tracing::debug!(buffer_idx = idx, "Buffer reclaimed");
                    buf.state = BufferState::Filling;
                }
            }
        }
    }

    #[tracing::instrument(skip(self), fields(buffer_idx = self.active))]
    fn flush_active(&mut self) -> Result<(), CudaWriterError> {
        let idx = self.active;
        let len = self.host[idx].len;

        tracing::debug!(bytes = len, "Flushing buffer to device");

        // Use Driver API for async memory copy
        unsafe {
            use cudarc::driver::sys;
            let result = sys::cuMemcpyHtoDAsync_v2(
                self.device_dst as sys::CUdeviceptr,
                self.host[idx].ptr as *const std::ffi::c_void,
                len,
                self.stream.cu_stream(),
            );

            if result != sys::CUresult::CUDA_SUCCESS {
                tracing::error!(?result, "cuMemcpyHtoDAsync failed");
                return Err(DriverError(result).into());
            }
        }

        // Record the pre-allocated event to track completion
        self.host[idx].event.record(&self.stream)?;

        // Mark buffer as flushing and reset length
        self.host[idx].state = BufferState::Flushing;
        self.host[idx].len = 0;

        // Swap to the other buffer
        self.active = 1 - idx;
        tracing::debug!(new_active = self.active, "Swapped to other buffer");

        Ok(())
    }
}

impl Write for CudaWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        CudaWriter::write(self, buf).map_err(|e| std::io::Error::other(format!("{:?}", e)))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        tracing::debug!("Flushing all buffers");

        // Flush the active buffer if it has data
        let active_len = self.host[self.active].len;
        if active_len > 0 && matches!(self.host[self.active].state, BufferState::Filling) {
            tracing::debug!(active_buffer_len = active_len, "Flushing active buffer");
            self.flush_active()
                .map_err(|e| std::io::Error::other(format!("{:?}", e)))?;
        }

        // Wait for all buffers to complete flushing
        for (idx, buf) in self.host.iter().enumerate() {
            if let BufferState::Flushing = buf.state {
                tracing::debug!(buffer_idx = idx, "Synchronizing buffer");
                buf.event
                    .synchronize()
                    .map_err(|e| std::io::Error::other(format!("{:?}", e)))?;
            }
        }

        tracing::debug!("All buffers flushed");
        Ok(())
    }
}

impl Drop for CudaWriter {
    fn drop(&mut self) {
        if !self.device_dst.is_null() {
            tracing::debug!("Freeing device buffer");
            unsafe {
                sys::cudaFreeAsync(
                    self.device_dst as *mut std::ffi::c_void,
                    self.stream.cu_stream() as sys::cudaStream_t,
                );
            }
        }
    }
}
