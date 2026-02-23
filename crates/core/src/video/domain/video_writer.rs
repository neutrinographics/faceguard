use std::path::Path;

use crate::shared::frame::Frame;
use crate::shared::video_metadata::VideoMetadata;

/// Writes processed frames to a video file.
///
/// Implementations handle encoding, container format, and optional
/// audio muxing from the source file.
pub trait VideoWriter: Send {
    /// Opens an output file for writing with the given metadata.
    fn open(
        &mut self,
        path: &Path,
        metadata: &VideoMetadata,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// Writes a single frame to the output.
    fn write(&mut self, frame: &Frame) -> Result<(), Box<dyn std::error::Error>>;

    /// Closes the writer, flushing remaining data. Audio muxing happens here.
    fn close(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}
