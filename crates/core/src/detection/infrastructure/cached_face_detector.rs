use std::collections::HashMap;
use std::sync::Arc;

use crate::detection::domain::face_detector::FaceDetector;
use crate::shared::frame::Frame;
use crate::shared::region::Region;

/// Replays pre-computed detection results by frame index.
///
/// Used when a preview pass has already detected all faces — the blur pass
/// can reuse those exact regions, guaranteeing that track IDs match what
/// the user selected in the preview UI.
pub struct CachedFaceDetector {
    cache: Arc<HashMap<usize, Vec<Region>>>,
}

impl CachedFaceDetector {
    pub fn new(cache: Arc<HashMap<usize, Vec<Region>>>) -> Self {
        Self { cache }
    }
}

impl FaceDetector for CachedFaceDetector {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Region>, Box<dyn std::error::Error>> {
        Ok(self.cache.get(&frame.index()).cloned().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(index: usize) -> Frame {
        Frame::new(vec![0u8; 100 * 100 * 3], 100, 100, 3, index)
    }

    fn region(track_id: u32, x: i32) -> Region {
        Region {
            x,
            y: 20,
            width: 50,
            height: 50,
            track_id: Some(track_id),
            full_width: None,
            full_height: None,
            unclamped_x: None,
            unclamped_y: None,
        }
    }

    #[test]
    fn test_returns_cached_regions_for_known_frame() {
        let regions = vec![region(1, 10), region(2, 60)];
        let cache = Arc::new(HashMap::from([(0, regions.clone())]));
        let mut detector = CachedFaceDetector::new(cache);

        let result = detector.detect(&frame(0)).unwrap();

        assert_eq!(result, regions);
    }

    #[test]
    fn test_returns_empty_for_unknown_frame() {
        let cache = Arc::new(HashMap::from([(0, vec![region(1, 10)])]));
        let mut detector = CachedFaceDetector::new(cache);

        let result = detector.detect(&frame(5)).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_returns_different_regions_per_frame() {
        let cache = Arc::new(HashMap::from([
            (0, vec![region(1, 10)]),
            (1, vec![region(1, 20), region(2, 60)]),
            (2, vec![]),
        ]));
        let mut detector = CachedFaceDetector::new(cache);

        assert_eq!(detector.detect(&frame(0)).unwrap().len(), 1);
        assert_eq!(detector.detect(&frame(1)).unwrap().len(), 2);
        assert_eq!(detector.detect(&frame(2)).unwrap().len(), 0);
    }

    #[test]
    fn test_empty_cache_always_returns_empty() {
        let mut detector = CachedFaceDetector::new(Arc::new(HashMap::new()));

        assert!(detector.detect(&frame(0)).unwrap().is_empty());
        assert!(detector.detect(&frame(99)).unwrap().is_empty());
    }

    #[test]
    fn test_track_ids_are_preserved() {
        let cache = Arc::new(HashMap::from([(0, vec![region(42, 10), region(7, 60)])]));
        let mut detector = CachedFaceDetector::new(cache);

        let result = detector.detect(&frame(0)).unwrap();

        assert_eq!(result[0].track_id, Some(42));
        assert_eq!(result[1].track_id, Some(7));
    }
}
