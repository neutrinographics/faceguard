use crate::shared::frame::Frame;
use crate::shared::region::Region;

/// Domain interface for face detection.
///
/// Implementations may be stateful (e.g., tracking across frames),
/// hence `&mut self`.
pub trait FaceDetector: Send {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Region>, Box<dyn std::error::Error>>;
}
