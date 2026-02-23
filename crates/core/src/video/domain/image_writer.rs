use std::path::Path;

use crate::shared::frame::Frame;

/// Writes a single frame to an image file.
pub trait ImageWriter: Send {
    /// Writes a frame to the given path, optionally resizing to the given dimensions.
    fn write(
        &self,
        path: &Path,
        frame: &Frame,
        size: Option<(u32, u32)>,
    ) -> Result<(), Box<dyn std::error::Error>>;
}
