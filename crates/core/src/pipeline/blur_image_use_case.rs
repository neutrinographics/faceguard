use std::collections::HashSet;
use std::path::Path;

use crate::blurring::domain::frame_blurrer::FrameBlurrer;
use crate::detection::domain::face_detector::FaceDetector;
use crate::pipeline::region_filter::filter_regions;
use crate::video::domain::image_writer::ImageWriter;
use crate::video::domain::video_reader::VideoReader;

/// Single-image blurring pipeline: read → detect → filter → blur → write.
pub struct BlurImageUseCase {
    reader: Box<dyn VideoReader>,
    image_writer: Box<dyn ImageWriter>,
    detector: Box<dyn FaceDetector>,
    blurrer: Box<dyn FrameBlurrer>,
    blur_ids: Option<HashSet<u32>>,
    exclude_ids: Option<HashSet<u32>>,
}

impl BlurImageUseCase {
    pub fn new(
        reader: Box<dyn VideoReader>,
        image_writer: Box<dyn ImageWriter>,
        detector: Box<dyn FaceDetector>,
        blurrer: Box<dyn FrameBlurrer>,
        blur_ids: Option<HashSet<u32>>,
        exclude_ids: Option<HashSet<u32>>,
    ) -> Self {
        Self {
            reader,
            image_writer,
            detector,
            blurrer,
            blur_ids,
            exclude_ids,
        }
    }

    /// Reads a single image, detects faces, filters, blurs, and writes output.
    pub fn execute(
        &mut self,
        input_path: &Path,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _metadata = self.reader.open(input_path)?;

        let mut frame = self.reader.frames().next().ok_or("No frames in image")??;
        self.reader.close();

        let regions = self.detector.detect(&frame)?;
        let filtered = filter_regions(&regions, self.blur_ids.as_ref(), self.exclude_ids.as_ref());

        self.blurrer.blur(&mut frame, &filtered)?;
        self.image_writer.write(output_path, &frame, None)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::frame::Frame;
    use crate::shared::region::Region;
    use crate::shared::video_metadata::VideoMetadata;
    use std::sync::{Arc, Mutex};

    // --- Stubs ---

    struct StubImageReader {
        frame: Option<Frame>,
    }

    impl StubImageReader {
        fn new(frame: Frame) -> Self {
            Self { frame: Some(frame) }
        }
    }

    impl VideoReader for StubImageReader {
        fn open(&mut self, _path: &Path) -> Result<VideoMetadata, Box<dyn std::error::Error>> {
            Ok(VideoMetadata {
                width: self.frame.as_ref().unwrap().width(),
                height: self.frame.as_ref().unwrap().height(),
                fps: 0.0,
                total_frames: 1,
                codec: String::new(),
                source_path: None,
            })
        }

        fn frames(
            &mut self,
        ) -> Box<dyn Iterator<Item = Result<Frame, Box<dyn std::error::Error>>> + '_> {
            Box::new(self.frame.take().into_iter().map(Ok))
        }

        fn close(&mut self) {
            self.frame = None;
        }
    }

    struct StubImageWriter {
        written: Arc<Mutex<Vec<(std::path::PathBuf, Frame)>>>,
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
            _size: Option<(u32, u32)>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            self.written
                .lock()
                .unwrap()
                .push((path.to_path_buf(), frame.clone()));
            Ok(())
        }
    }

    struct StubDetector {
        regions: Vec<Region>,
    }

    impl FaceDetector for StubDetector {
        fn detect(&mut self, _frame: &Frame) -> Result<Vec<Region>, Box<dyn std::error::Error>> {
            Ok(self.regions.clone())
        }
    }

    struct PassthroughBlurrer {
        calls: Arc<Mutex<Vec<Vec<Region>>>>,
    }

    impl PassthroughBlurrer {
        fn new() -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl FrameBlurrer for PassthroughBlurrer {
        fn blur(
            &self,
            _frame: &mut Frame,
            regions: &[Region],
        ) -> Result<(), Box<dyn std::error::Error>> {
            self.calls.lock().unwrap().push(regions.to_vec());
            Ok(())
        }
    }

    // --- Helpers ---

    fn make_frame(w: u32, h: u32) -> Frame {
        Frame::new(vec![128; (w * h * 3) as usize], w, h, 3, 0)
    }

    fn region_with_track(track_id: Option<u32>) -> Region {
        Region {
            x: 10,
            y: 10,
            width: 30,
            height: 30,
            track_id,
            full_width: None,
            full_height: None,
            unclamped_x: None,
            unclamped_y: None,
        }
    }

    // --- Tests ---

    #[test]
    fn test_passes_regions_to_blurrer() {
        let blurrer = PassthroughBlurrer::new();
        let calls = blurrer.calls.clone();

        let mut uc = BlurImageUseCase::new(
            Box::new(StubImageReader::new(make_frame(100, 100))),
            Box::new(StubImageWriter::new()),
            Box::new(StubDetector {
                regions: vec![region_with_track(Some(1))],
            }),
            Box::new(blurrer),
            None,
            None,
        );

        uc.execute(Path::new("in.png"), Path::new("out.png"))
            .unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), 1);
        assert_eq!(calls[0][0].track_id, Some(1));
    }

    #[test]
    fn test_output_dimensions_preserved() {
        let img_writer = StubImageWriter::new();
        let written = img_writer.written.clone();

        let mut uc = BlurImageUseCase::new(
            Box::new(StubImageReader::new(make_frame(200, 150))),
            Box::new(img_writer),
            Box::new(StubDetector { regions: vec![] }),
            Box::new(PassthroughBlurrer::new()),
            None,
            None,
        );

        uc.execute(Path::new("in.png"), Path::new("out.png"))
            .unwrap();

        let written = written.lock().unwrap();
        assert_eq!(written[0].1.width(), 200);
        assert_eq!(written[0].1.height(), 150);
    }

    #[test]
    fn test_blur_ids_filters_regions() {
        let blurrer = PassthroughBlurrer::new();
        let calls = blurrer.calls.clone();

        let mut uc = BlurImageUseCase::new(
            Box::new(StubImageReader::new(make_frame(100, 100))),
            Box::new(StubImageWriter::new()),
            Box::new(StubDetector {
                regions: vec![region_with_track(Some(1)), region_with_track(Some(2))],
            }),
            Box::new(blurrer),
            Some(HashSet::from([1])),
            None,
        );

        uc.execute(Path::new("in.png"), Path::new("out.png"))
            .unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls[0].len(), 1);
        assert_eq!(calls[0][0].track_id, Some(1));
    }

    #[test]
    fn test_exclude_ids_filters_regions() {
        let blurrer = PassthroughBlurrer::new();
        let calls = blurrer.calls.clone();

        let mut uc = BlurImageUseCase::new(
            Box::new(StubImageReader::new(make_frame(100, 100))),
            Box::new(StubImageWriter::new()),
            Box::new(StubDetector {
                regions: vec![region_with_track(Some(1)), region_with_track(Some(2))],
            }),
            Box::new(blurrer),
            None,
            Some(HashSet::from([2])),
        );

        uc.execute(Path::new("in.png"), Path::new("out.png"))
            .unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls[0].len(), 1);
        assert_eq!(calls[0][0].track_id, Some(1));
    }

    #[test]
    fn test_no_faces_still_writes_image() {
        let img_writer = StubImageWriter::new();
        let written = img_writer.written.clone();

        let mut uc = BlurImageUseCase::new(
            Box::new(StubImageReader::new(make_frame(100, 100))),
            Box::new(img_writer),
            Box::new(StubDetector { regions: vec![] }),
            Box::new(PassthroughBlurrer::new()),
            None,
            None,
        );

        uc.execute(Path::new("in.png"), Path::new("out.png"))
            .unwrap();

        assert_eq!(written.lock().unwrap().len(), 1);
    }
}
