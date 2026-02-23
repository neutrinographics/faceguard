use crate::shared::frame::Frame;
use crate::shared::region::Region;

/// Domain interface for applying blur to specified regions within a frame.
///
/// Implementations modify the frame in-place (`&mut Frame`) to avoid allocation.
pub trait FrameBlurrer: Send {
    fn blur(&self, frame: &mut Frame, regions: &[Region])
        -> Result<(), Box<dyn std::error::Error>>;
}
