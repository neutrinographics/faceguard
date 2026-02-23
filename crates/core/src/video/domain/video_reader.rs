use std::path::Path;

use crate::shared::frame::Frame;
use crate::shared::video_metadata::VideoMetadata;

/// Reads frames from a video or image source.
///
/// Implementations handle I/O details (codec, container format, etc.)
/// while the pipeline works with the abstract `Frame` and `VideoMetadata`
/// types.
pub trait VideoReader: Send {
    /// Opens a video or image file and returns its metadata.
    fn open(&mut self, path: &Path) -> Result<VideoMetadata, Box<dyn std::error::Error>>;

    /// Returns an iterator over frames in decode order.
    fn frames(
        &mut self,
    ) -> Box<dyn Iterator<Item = Result<Frame, Box<dyn std::error::Error>>> + '_>;

    /// Releases any resources held by the reader.
    fn close(&mut self);
}
