use std::collections::HashMap;

use crate::detection::domain::face_detector::FaceDetector;
use crate::shared::frame::Frame;
use crate::shared::region::Region;

/// Decorator that runs detection every N frames, reusing results in between.
///
/// On skipped frames, region positions are linearly extrapolated from the
/// velocity observed between the two most recent real detections, preventing
/// the stutter that would result from simply repeating stale positions.
pub struct SkipFrameDetector {
    inner: Box<dyn FaceDetector>,
    skip_interval: usize,
    frame_count: usize,
    last_regions: Vec<Region>,
    /// Per-track velocity (dx, dy per frame) between last two real detections.
    velocity: HashMap<u32, (f64, f64)>,
    /// Per-track position at last real detection.
    prev_pos: HashMap<u32, (i32, i32)>,
    frames_since_detect: usize,
}

impl SkipFrameDetector {
    pub fn new(inner: Box<dyn FaceDetector>, skip_interval: usize) -> Result<Self, &'static str> {
        if skip_interval < 1 {
            return Err("skip_interval must be >= 1");
        }
        Ok(Self {
            inner,
            skip_interval,
            frame_count: 0,
            last_regions: Vec::new(),
            velocity: HashMap::new(),
            prev_pos: HashMap::new(),
            frames_since_detect: 0,
        })
    }

    fn update_velocity(&mut self, regions: &[Region]) {
        let mut new_pos: HashMap<u32, (i32, i32)> = HashMap::new();
        for r in regions {
            if let Some(tid) = r.track_id {
                new_pos.insert(tid, (r.x, r.y));
                if let Some(&(old_x, old_y)) = self.prev_pos.get(&tid) {
                    let dx = (r.x - old_x) as f64 / self.skip_interval as f64;
                    let dy = (r.y - old_y) as f64 / self.skip_interval as f64;
                    self.velocity.insert(tid, (dx, dy));
                }
            }
        }
        self.prev_pos = new_pos;
    }

    fn extrapolate(&self, regions: &[Region], steps: usize) -> Vec<Region> {
        regions
            .iter()
            .map(|r| {
                let vel = r.track_id.and_then(|tid| self.velocity.get(&tid));
                match vel {
                    None => r.clone(),
                    Some(&(dx, dy)) => {
                        let new_x = r.x + (dx * steps as f64) as i32;
                        let new_y = r.y + (dy * steps as f64) as i32;
                        let new_ux = r.unclamped_x.map(|ux| ux + (dx * steps as f64) as i32);
                        let new_uy = r.unclamped_y.map(|uy| uy + (dy * steps as f64) as i32);
                        Region {
                            x: new_x.max(0),
                            y: new_y.max(0),
                            width: r.width,
                            height: r.height,
                            track_id: r.track_id,
                            full_width: r.full_width,
                            full_height: r.full_height,
                            unclamped_x: new_ux,
                            unclamped_y: new_uy,
                        }
                    }
                }
            })
            .collect()
    }
}

impl FaceDetector for SkipFrameDetector {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Region>, Box<dyn std::error::Error>> {
        if self.frame_count % self.skip_interval == 0 {
            let new_regions = self.inner.detect(frame)?;
            self.update_velocity(&new_regions);
            self.last_regions = new_regions;
            self.frames_since_detect = 0;
        } else {
            self.frames_since_detect += 1;
        }
        self.frame_count += 1;

        if self.frames_since_detect == 0 {
            Ok(self.last_regions.clone())
        } else {
            Ok(self.extrapolate(&self.last_regions, self.frames_since_detect))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeDetector {
        results: Vec<Vec<Region>>,
        call_count: usize,
    }

    impl FakeDetector {
        fn new(results: Vec<Vec<Region>>) -> Self {
            Self {
                results,
                call_count: 0,
            }
        }
    }

    impl FaceDetector for FakeDetector {
        fn detect(&mut self, _frame: &Frame) -> Result<Vec<Region>, Box<dyn std::error::Error>> {
            let result = self.results[self.call_count % self.results.len()].clone();
            self.call_count += 1;
            Ok(result)
        }
    }

    fn frame(index: usize) -> Frame {
        Frame::new(vec![0u8; 100 * 100 * 3], 100, 100, 3, index)
    }

    fn region(track_id: u32, x: i32, y: i32) -> Region {
        Region {
            x,
            y,
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
    fn test_detect_interval_1_delegates_every_frame() {
        let inner = FakeDetector::new(vec![vec![region(1, 10, 20)]; 3]);
        let mut detector = SkipFrameDetector::new(Box::new(inner), 1).unwrap();

        for i in 0..3 {
            let r = detector.detect(&frame(i)).unwrap();
            assert_eq!(r.len(), 1);
        }
    }

    #[test]
    fn test_detect_interval_2_skips_alternate_frames() {
        let inner = FakeDetector::new(vec![vec![region(1, 10, 20)], vec![region(1, 30, 20)]]);
        let mut detector = SkipFrameDetector::new(Box::new(inner), 2).unwrap();

        let r0 = detector.detect(&frame(0)).unwrap();
        let r1 = detector.detect(&frame(1)).unwrap(); // skipped
        let r2 = detector.detect(&frame(2)).unwrap(); // real detection

        assert_eq!(r0.len(), 1);
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].track_id, Some(1));
        assert_eq!(r2.len(), 1);
    }

    #[test]
    fn test_no_regions_on_skipped_frame() {
        let inner = FakeDetector::new(vec![vec![]]);
        let mut detector = SkipFrameDetector::new(Box::new(inner), 2).unwrap();

        let r0 = detector.detect(&frame(0)).unwrap();
        let r1 = detector.detect(&frame(1)).unwrap();

        assert!(r0.is_empty());
        assert!(r1.is_empty());
    }

    #[test]
    fn test_skip_interval_0_errors() {
        let inner = FakeDetector::new(vec![vec![]]);
        let result = SkipFrameDetector::new(Box::new(inner), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_faces_appear_after_skip() {
        let inner = FakeDetector::new(vec![
            vec![region(1, 10, 20)],
            vec![region(1, 10, 20), region(2, 60, 20)],
        ]);
        let mut detector = SkipFrameDetector::new(Box::new(inner), 2).unwrap();

        let r0 = detector.detect(&frame(0)).unwrap();
        let r1 = detector.detect(&frame(1)).unwrap(); // skipped
        let r2 = detector.detect(&frame(2)).unwrap(); // real

        assert_eq!(r0.len(), 1);
        assert_eq!(r1.len(), 1);
        assert_eq!(r2.len(), 2);
    }

    #[test]
    fn test_extrapolation_moves_region_on_skipped_frame() {
        // Frame 0: face at x=10. Frame 2: face at x=20. Velocity = 5px/frame.
        // Frame 3 (skipped): should extrapolate x from 20 by +5 = 25.
        let inner = FakeDetector::new(vec![vec![region(1, 10, 20)], vec![region(1, 20, 20)]]);
        let mut detector = SkipFrameDetector::new(Box::new(inner), 2).unwrap();

        detector.detect(&frame(0)).unwrap(); // real: x=10
        detector.detect(&frame(1)).unwrap(); // skipped (no velocity yet)
        detector.detect(&frame(2)).unwrap(); // real: x=20, vel = (20-10)/2 = 5/frame
        let r3 = detector.detect(&frame(3)).unwrap(); // skipped: extrapolate from x=20 by +5

        assert_eq!(r3.len(), 1);
        assert_eq!(r3[0].x, 25);
    }

    #[test]
    fn test_extrapolation_with_y_movement() {
        let inner = FakeDetector::new(vec![vec![region(1, 10, 10)], vec![region(1, 20, 30)]]);
        let mut detector = SkipFrameDetector::new(Box::new(inner), 2).unwrap();

        detector.detect(&frame(0)).unwrap();
        detector.detect(&frame(1)).unwrap();
        detector.detect(&frame(2)).unwrap(); // vel = (5, 10) per frame
        let r3 = detector.detect(&frame(3)).unwrap();

        assert_eq!(r3[0].x, 25); // 20 + 5*1
        assert_eq!(r3[0].y, 40); // 30 + 10*1
    }

    #[test]
    fn test_no_extrapolation_without_track_id() {
        let untracked = Region {
            x: 10,
            y: 20,
            width: 50,
            height: 50,
            track_id: None,
            full_width: None,
            full_height: None,
            unclamped_x: None,
            unclamped_y: None,
        };
        let inner = FakeDetector::new(vec![vec![untracked]]);
        let mut detector = SkipFrameDetector::new(Box::new(inner), 2).unwrap();

        detector.detect(&frame(0)).unwrap();
        let r1 = detector.detect(&frame(1)).unwrap();

        assert_eq!(r1[0].x, 10);
        assert_eq!(r1[0].y, 20);
    }

    #[test]
    fn test_no_velocity_on_first_cycle_returns_static() {
        let inner = FakeDetector::new(vec![vec![region(1, 10, 20)]]);
        let mut detector = SkipFrameDetector::new(Box::new(inner), 2).unwrap();

        detector.detect(&frame(0)).unwrap(); // first real detection
        let r1 = detector.detect(&frame(1)).unwrap(); // skipped, no velocity yet

        assert_eq!(r1[0].x, 10); // unchanged
    }

    #[test]
    fn test_extrapolation_clamps_x_y_to_zero() {
        let inner = FakeDetector::new(vec![vec![region(1, 10, 10)], vec![region(1, 2, 2)]]);
        let mut detector = SkipFrameDetector::new(Box::new(inner), 2).unwrap();

        detector.detect(&frame(0)).unwrap();
        detector.detect(&frame(1)).unwrap();
        detector.detect(&frame(2)).unwrap(); // vel = (-4, -4) per frame
        let r3 = detector.detect(&frame(3)).unwrap(); // 2 + (-4) = -2, clamped to 0

        assert_eq!(r3[0].x, 0);
        assert_eq!(r3[0].y, 0);
    }

    #[test]
    fn test_extrapolation_skip_interval_3() {
        let inner = FakeDetector::new(vec![
            vec![region(1, 10, 20)],
            vec![region(1, 40, 20)], // delta_x = 30 over 3 frames = 10/frame
        ]);
        let mut detector = SkipFrameDetector::new(Box::new(inner), 3).unwrap();

        detector.detect(&frame(0)).unwrap(); // real
        detector.detect(&frame(1)).unwrap(); // skipped
        detector.detect(&frame(2)).unwrap(); // skipped
        detector.detect(&frame(3)).unwrap(); // real: x=40, vel=10/frame
        let r4 = detector.detect(&frame(4)).unwrap(); // skipped: 40 + 10*1 = 50
        let r5 = detector.detect(&frame(5)).unwrap(); // skipped: 40 + 10*2 = 60

        assert_eq!(r4[0].x, 50);
        assert_eq!(r5[0].x, 60);
    }
}
