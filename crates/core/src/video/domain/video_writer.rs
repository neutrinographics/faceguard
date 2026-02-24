use std::path::Path;

use crate::shared::frame::Frame;
use crate::shared::video_metadata::VideoMetadata;

/// Abstracts video encoding so the pipeline can write output without
/// depending on a specific codec library.
pub trait VideoWriter: Send {
    fn open(
        &mut self,
        path: &Path,
        metadata: &VideoMetadata,
    ) -> Result<(), Box<dyn std::error::Error>>;

    fn write(&mut self, frame: &Frame) -> Result<(), Box<dyn std::error::Error>>;

    /// Audio muxing from the source file happens during close.
    fn close(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}
