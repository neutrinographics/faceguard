use std::path::Path;

use crate::shared::frame::Frame;

/// Writes a single frame to an image file, with optional resizing for thumbnails.
pub trait ImageWriter: Send {
    fn write(
        &self,
        path: &Path,
        frame: &Frame,
        size: Option<(u32, u32)>,
    ) -> Result<(), Box<dyn std::error::Error>>;
}
