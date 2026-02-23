use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::detection::domain::face_detector::FaceDetector;
use crate::shared::frame::Frame;
use crate::shared::region::Region;
use crate::shared::video_metadata::VideoMetadata;
use crate::video::domain::image_writer::ImageWriter;
use crate::video::domain::video_reader::VideoReader;

/// Preview thumbnail size in pixels.
const PREVIEW_SIZE: u32 = 256;

/// Return type for [`PreviewFacesUseCase::execute`].
pub type PreviewResult = (HashMap<u32, PathBuf>, HashMap<usize, Vec<Region>>);

/// Scans a video for faces and saves the best crop of each tracked identity.
///
/// Selects the largest detection per track ID (by area) as the representative
/// crop, giving downstream grouping and the UI the clearest possible thumbnail.
pub struct PreviewFacesUseCase {
    reader: Box<dyn VideoReader>,
    detector: Box<dyn FaceDetector>,
    image_writer: Box<dyn ImageWriter>,
    on_progress: Option<Box<dyn Fn(usize, usize) -> bool + Send>>,
}

impl PreviewFacesUseCase {
    pub fn new(
        reader: Box<dyn VideoReader>,
        detector: Box<dyn FaceDetector>,
        image_writer: Box<dyn ImageWriter>,
        on_progress: Option<Box<dyn Fn(usize, usize) -> bool + Send>>,
    ) -> Self {
        Self {
            reader,
            detector,
            image_writer,
            on_progress,
        }
    }

    /// Runs face preview scan and saves crops to disk.
    ///
    /// Returns `(crops_by_id, detection_cache)`:
    /// - `crops_by_id`: track_id → file path of saved thumbnail
    /// - `detection_cache`: frame_index → detected regions (for reuse in blur pass)
    pub fn execute(
        &mut self,
        metadata: &VideoMetadata,
        output_dir: &Path,
    ) -> Result<PreviewResult, Box<dyn std::error::Error>> {
        // track_id → (area, crop_frame)
        let mut best: HashMap<u32, (u32, Frame)> = HashMap::new();
        let mut detection_cache: HashMap<usize, Vec<Region>> = HashMap::new();

        let total_frames = metadata.total_frames;

        // Collect frames to avoid borrow conflict between reader and detector
        let all_frames: Vec<Frame> = self.reader.frames().collect::<Result<Vec<_>, _>>()?;

        for frame in &all_frames {
            let regions = self.detector.detect(frame)?;

            detection_cache.insert(frame.index(), regions.clone());

            for r in &regions {
                let Some(track_id) = r.track_id else {
                    continue;
                };
                let area = r.width as u32 * r.height as u32;
                if !best.contains_key(&track_id) || area > best[&track_id].0 {
                    let crop = square_crop(frame, r);
                    best.insert(track_id, (area, crop));
                }
            }

            self.report_progress(frame.index() + 1, total_frames)?;
        }

        self.reader.close();

        let mut result: HashMap<u32, PathBuf> = HashMap::new();
        let mut sorted_ids: Vec<u32> = best.keys().copied().collect();
        sorted_ids.sort();

        for track_id in sorted_ids {
            let (_, crop) = best.remove(&track_id).unwrap();
            let path = output_dir.join(format!("{track_id}.jpg"));
            self.image_writer
                .write(&path, &crop, Some((PREVIEW_SIZE, PREVIEW_SIZE)))?;
            result.insert(track_id, path);
        }

        Ok((result, detection_cache))
    }

    fn report_progress(
        &self,
        current: usize,
        total: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref callback) = self.on_progress {
            if !callback(current, total) {
                return Err("Cancelled".into());
            }
        }
        Ok(())
    }
}

/// Extracts a square crop centered on the region, clamped to frame bounds.
fn square_crop(frame: &Frame, region: &Region) -> Frame {
    let fw = frame.width() as i32;
    let fh = frame.height() as i32;

    let cx = region.x + region.width / 2;
    let cy = region.y + region.height / 2;
    let half = region.width.max(region.height) / 2;

    let x1 = (cx - half).max(0) as usize;
    let y1 = (cy - half).max(0) as usize;
    let x2 = (cx + half).min(fw) as usize;
    let y2 = (cy + half).min(fh) as usize;

    let crop_w = x2 - x1;
    let crop_h = y2 - y1;
    let channels = frame.channels() as usize;

    let src = frame.as_ndarray();
    let mut data = Vec::with_capacity(crop_w * crop_h * channels);

    for row in y1..y2 {
        for col in x1..x2 {
            for c in 0..channels {
                data.push(src[[row, col, c]]);
            }
        }
    }

    Frame::new(data, crop_w as u32, crop_h as u32, channels as u8, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // --- Stubs ---

    struct StubReader {
        frames: Vec<Frame>,
    }

    impl StubReader {
        fn new(frames: Vec<Frame>) -> Self {
            Self { frames }
        }
    }

    impl VideoReader for StubReader {
        fn open(&mut self, _path: &Path) -> Result<VideoMetadata, Box<dyn std::error::Error>> {
            Ok(metadata(100, 100, 0))
        }

        fn frames(
            &mut self,
        ) -> Box<dyn Iterator<Item = Result<Frame, Box<dyn std::error::Error>>> + '_> {
            Box::new(self.frames.drain(..).map(Ok))
        }

        fn close(&mut self) {}
    }

    struct StubDetector {
        results: HashMap<usize, Vec<Region>>,
    }

    impl FaceDetector for StubDetector {
        fn detect(&mut self, frame: &Frame) -> Result<Vec<Region>, Box<dyn std::error::Error>> {
            Ok(self
                .results
                .get(&frame.index())
                .cloned()
                .unwrap_or_default())
        }
    }

    #[allow(clippy::type_complexity)]
    struct StubImageWriter {
        written: Arc<Mutex<Vec<(PathBuf, Frame, Option<(u32, u32)>)>>>,
    }

    impl StubImageWriter {
        fn new() -> Self {
            Self {
                written: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl ImageWriter for StubImageWriter {
        fn write(
            &self,
            path: &Path,
            frame: &Frame,
            size: Option<(u32, u32)>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            self.written
                .lock()
                .unwrap()
                .push((path.to_path_buf(), frame.clone(), size));
            Ok(())
        }
    }

    // --- Helpers ---

    fn make_frame(index: usize, w: u32, h: u32) -> Frame {
        Frame::new(vec![128; (w * h * 3) as usize], w, h, 3, index)
    }

    fn metadata(w: u32, h: u32, total: usize) -> VideoMetadata {
        VideoMetadata {
            width: w,
            height: h,
            fps: 30.0,
            total_frames: total,
            codec: String::new(),
            source_path: None,
        }
    }

    fn region(x: i32, y: i32, w: i32, h: i32, track_id: Option<u32>) -> Region {
        Region {
            x,
            y,
            width: w,
            height: h,
            track_id,
            full_width: None,
            full_height: None,
            unclamped_x: None,
            unclamped_y: None,
        }
    }

    // --- Tests ---

    #[test]
    fn test_saves_one_crop_per_track_id() {
        let dir = tempfile::tempdir().unwrap();
        let img_writer = StubImageWriter::new();
        let written = img_writer.written.clone();

        let mut det_results = HashMap::new();
        det_results.insert(0, vec![region(10, 10, 20, 20, Some(1))]);
        det_results.insert(1, vec![region(10, 10, 20, 20, Some(2))]);

        let mut uc = PreviewFacesUseCase::new(
            Box::new(StubReader::new(vec![
                make_frame(0, 100, 100),
                make_frame(1, 100, 100),
            ])),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(img_writer),
            None,
        );

        let (crops, _cache) = uc.execute(&metadata(100, 100, 2), dir.path()).unwrap();

        assert_eq!(crops.len(), 2);
        assert!(crops.contains_key(&1));
        assert!(crops.contains_key(&2));

        let written = written.lock().unwrap();
        assert_eq!(written.len(), 2);
    }

    #[test]
    fn test_keeps_largest_crop_per_track_id() {
        let dir = tempfile::tempdir().unwrap();
        let img_writer = StubImageWriter::new();
        let written = img_writer.written.clone();

        let mut det_results = HashMap::new();
        // Frame 0: small detection for track 1
        det_results.insert(0, vec![region(10, 10, 10, 10, Some(1))]);
        // Frame 1: larger detection for same track 1
        det_results.insert(1, vec![region(10, 10, 30, 30, Some(1))]);

        let mut uc = PreviewFacesUseCase::new(
            Box::new(StubReader::new(vec![
                make_frame(0, 100, 100),
                make_frame(1, 100, 100),
            ])),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(img_writer),
            None,
        );

        let (crops, _) = uc.execute(&metadata(100, 100, 2), dir.path()).unwrap();

        assert_eq!(crops.len(), 1);

        // Should have written one crop with the larger size
        let written = written.lock().unwrap();
        assert_eq!(written.len(), 1);
        // The crop should be from the 30x30 region (square crop → 30x30)
        assert_eq!(written[0].1.width(), 30);
        assert_eq!(written[0].1.height(), 30);
    }

    #[test]
    fn test_detection_cache_populated() {
        let dir = tempfile::tempdir().unwrap();

        let mut det_results = HashMap::new();
        det_results.insert(0, vec![region(10, 10, 20, 20, Some(1))]);
        det_results.insert(1, vec![]);

        let mut uc = PreviewFacesUseCase::new(
            Box::new(StubReader::new(vec![
                make_frame(0, 100, 100),
                make_frame(1, 100, 100),
            ])),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(StubImageWriter::new()),
            None,
        );

        let (_, cache) = uc.execute(&metadata(100, 100, 2), dir.path()).unwrap();

        assert_eq!(cache.len(), 2);
        assert_eq!(cache[&0].len(), 1);
        assert_eq!(cache[&1].len(), 0);
    }

    #[test]
    fn test_none_track_id_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let img_writer = StubImageWriter::new();
        let written = img_writer.written.clone();

        let mut det_results = HashMap::new();
        det_results.insert(0, vec![region(10, 10, 20, 20, None)]);

        let mut uc = PreviewFacesUseCase::new(
            Box::new(StubReader::new(vec![make_frame(0, 100, 100)])),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(img_writer),
            None,
        );

        let (crops, _) = uc.execute(&metadata(100, 100, 1), dir.path()).unwrap();

        assert!(crops.is_empty());
        assert!(written.lock().unwrap().is_empty());
    }

    #[test]
    fn test_writes_with_preview_size() {
        let dir = tempfile::tempdir().unwrap();
        let img_writer = StubImageWriter::new();
        let written = img_writer.written.clone();

        let mut det_results = HashMap::new();
        det_results.insert(0, vec![region(10, 10, 20, 20, Some(1))]);

        let mut uc = PreviewFacesUseCase::new(
            Box::new(StubReader::new(vec![make_frame(0, 100, 100)])),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(img_writer),
            None,
        );

        uc.execute(&metadata(100, 100, 1), dir.path()).unwrap();

        let written = written.lock().unwrap();
        assert_eq!(written[0].2, Some((256, 256)));
    }

    #[test]
    fn test_output_paths_use_track_id() {
        let dir = tempfile::tempdir().unwrap();
        let img_writer = StubImageWriter::new();
        let written = img_writer.written.clone();

        let mut det_results = HashMap::new();
        det_results.insert(0, vec![region(10, 10, 20, 20, Some(42))]);

        let mut uc = PreviewFacesUseCase::new(
            Box::new(StubReader::new(vec![make_frame(0, 100, 100)])),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(img_writer),
            None,
        );

        let (crops, _) = uc.execute(&metadata(100, 100, 1), dir.path()).unwrap();

        assert_eq!(crops[&42], dir.path().join("42.jpg"));
        let written = written.lock().unwrap();
        assert_eq!(written[0].0, dir.path().join("42.jpg"));
    }

    #[test]
    fn test_cancel_via_progress() {
        let dir = tempfile::tempdir().unwrap();

        let mut det_results = HashMap::new();
        det_results.insert(0, vec![region(10, 10, 20, 20, Some(1))]);
        det_results.insert(1, vec![region(10, 10, 20, 20, Some(2))]);

        let mut uc = PreviewFacesUseCase::new(
            Box::new(StubReader::new(vec![
                make_frame(0, 100, 100),
                make_frame(1, 100, 100),
                make_frame(2, 100, 100),
            ])),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(StubImageWriter::new()),
            Some(Box::new(|current, _total| current < 2)), // cancel after 2
        );

        let result = uc.execute(&metadata(100, 100, 3), dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_video() {
        let dir = tempfile::tempdir().unwrap();
        let img_writer = StubImageWriter::new();
        let written = img_writer.written.clone();

        let mut uc = PreviewFacesUseCase::new(
            Box::new(StubReader::new(vec![])),
            Box::new(StubDetector {
                results: HashMap::new(),
            }),
            Box::new(img_writer),
            None,
        );

        let (crops, cache) = uc.execute(&metadata(100, 100, 0), dir.path()).unwrap();

        assert!(crops.is_empty());
        assert!(cache.is_empty());
        assert!(written.lock().unwrap().is_empty());
    }

    #[test]
    fn test_square_crop_basic() {
        // 10x10 frame, region at (2,2) size 4x4
        let frame = make_frame(0, 10, 10);
        let r = region(2, 2, 4, 4, Some(1));
        let crop = square_crop(&frame, &r);
        // cx=4, cy=4, half=2 → x1=2, y1=2, x2=6, y2=6 → 4x4
        assert_eq!(crop.width(), 4);
        assert_eq!(crop.height(), 4);
    }

    #[test]
    fn test_square_crop_clamps_to_frame() {
        // 10x10 frame, region near edge
        let frame = make_frame(0, 10, 10);
        let r = region(7, 7, 6, 6, Some(1));
        let crop = square_crop(&frame, &r);
        // cx=10, cy=10, half=3 → x1=7, y1=7, x2=10, y2=10 → 3x3
        assert_eq!(crop.width(), 3);
        assert_eq!(crop.height(), 3);
    }

    #[test]
    fn test_square_crop_rectangular_region_uses_max_dim() {
        // 100x100 frame, tall region 10x30
        let frame = make_frame(0, 100, 100);
        let r = region(40, 35, 10, 30, Some(1));
        let crop = square_crop(&frame, &r);
        // cx=45, cy=50, half=15 → x1=30, y1=35, x2=60, y2=65 → 30x30
        assert_eq!(crop.width(), 30);
        assert_eq!(crop.height(), 30);
    }
}
