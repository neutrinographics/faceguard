use std::path::Path;

use crate::shared::frame::Frame;
use crate::shared::video_metadata::VideoMetadata;

/// Abstracts video/image decoding so the pipeline can process any media source
/// without depending on a specific codec library.
pub trait VideoReader: Send {
    fn open(&mut self, path: &Path) -> Result<VideoMetadata, Box<dyn std::error::Error>>;

    fn frames(
        &mut self,
    ) -> Box<dyn Iterator<Item = Result<Frame, Box<dyn std::error::Error>>> + '_>;

    fn close(&mut self);
}
