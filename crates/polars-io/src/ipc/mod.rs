#[cfg(feature = "ipc")]
mod ipc_file;
#[cfg(feature = "cloud")]
mod ipc_reader_async;
#[cfg(feature = "ipc_streaming")]
mod ipc_stream;
#[cfg(feature = "ipc")]
mod mmap;
mod write;
#[cfg(all(feature = "async", feature = "ipc"))]
mod write_async;

#[cfg(feature = "ipc")]
pub use ipc_file::{IpcBatchedReader, IpcReader, IpcScanOptions};
#[cfg(feature = "cloud")]
pub use ipc_reader_async::*;
#[cfg(feature = "ipc_streaming")]
pub use ipc_stream::*;
#[cfg(feature = "ipc")]
pub use write::{IpcBatchedWriter, IpcCompression, IpcWriter, IpcWriterOptions};
