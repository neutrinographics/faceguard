use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::blurring::domain::frame_blurrer::FrameBlurrer;
use crate::detection::domain::face_detector::FaceDetector;
use crate::detection::domain::region_merger::RegionMerger;
use crate::shared::video_metadata::VideoMetadata;
use crate::video::domain::video_reader::VideoReader;
use crate::video::domain::video_writer::VideoWriter;

use super::pipeline_executor::{PipelineConfig, PipelineExecutor};

const DEFAULT_LOOKAHEAD: usize = 5;

/// Orchestrates the full video blurring pipeline.
///
/// Wires domain components together and delegates execution to a
/// `PipelineExecutor`. This is a single-use struct: `execute` consumes
/// the owned components, so calling it twice will fail.
pub struct BlurFacesUseCase {
    reader: Option<Box<dyn VideoReader>>,
    writer: Option<Box<dyn VideoWriter>>,
    detector: Option<Box<dyn FaceDetector>>,
    blurrer: Option<Box<dyn FrameBlurrer>>,
    merger: Option<RegionMerger>,
    executor: Box<dyn PipelineExecutor>,
    lookahead: usize,
    blur_ids: Option<HashSet<u32>>,
    exclude_ids: Option<HashSet<u32>>,
    on_progress: Option<Box<dyn Fn(usize, usize) -> bool + Send>>,
    cancelled: Arc<AtomicBool>,
}

impl BlurFacesUseCase {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        reader: Box<dyn VideoReader>,
        writer: Box<dyn VideoWriter>,
        detector: Box<dyn FaceDetector>,
        blurrer: Box<dyn FrameBlurrer>,
        merger: RegionMerger,
        executor: Box<dyn PipelineExecutor>,
        lookahead: Option<usize>,
        blur_ids: Option<HashSet<u32>>,
        exclude_ids: Option<HashSet<u32>>,
        on_progress: Option<Box<dyn Fn(usize, usize) -> bool + Send>>,
        cancelled: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            reader: Some(reader),
            writer: Some(writer),
            detector: Some(detector),
            blurrer: Some(blurrer),
            merger: Some(merger),
            executor,
            lookahead: lookahead.unwrap_or(DEFAULT_LOOKAHEAD),
            blur_ids,
            exclude_ids,
            on_progress,
            cancelled: cancelled.unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
        }
    }

    pub fn execute(
        &mut self,
        metadata: &VideoMetadata,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = PipelineConfig {
            lookahead: self.lookahead,
            blur_ids: self.blur_ids.take(),
            exclude_ids: self.exclude_ids.take(),
            on_progress: self.on_progress.take(),
            cancelled: self.cancelled.clone(),
        };

        self.executor.execute(
            self.reader.take().ok_or("Pipeline already executed")?,
            self.writer.take().ok_or("Pipeline already executed")?,
            self.detector.take().ok_or("Pipeline already executed")?,
            self.blurrer.take().ok_or("Pipeline already executed")?,
            self.merger.take().ok_or("Pipeline already executed")?,
            metadata,
            output_path,
            config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::domain::region_merger::RegionMerger;
    use crate::pipeline::infrastructure::threaded_pipeline_executor::ThreadedPipelineExecutor;
    use crate::shared::frame::Frame;
    use crate::shared::region::Region;
    use crate::shared::video_metadata::VideoMetadata;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;

    // --- Stubs ---

    struct StubReader {
        frames: Vec<Frame>,
        closed: Arc<Mutex<bool>>,
    }

    impl StubReader {
        fn new(frames: Vec<Frame>) -> Self {
            Self {
                frames,
                closed: Arc::new(Mutex::new(false)),
            }
        }
    }

    impl VideoReader for StubReader {
        fn open(&mut self, _path: &Path) -> Result<VideoMetadata, Box<dyn std::error::Error>> {
            Ok(metadata(100, 100))
        }

        fn frames(
            &mut self,
        ) -> Box<dyn Iterator<Item = Result<Frame, Box<dyn std::error::Error>>> + '_> {
            Box::new(self.frames.drain(..).map(Ok))
        }

        fn close(&mut self) {
            *self.closed.lock().unwrap() = true;
        }
    }

    struct StubWriter {
        written: Arc<Mutex<Vec<Frame>>>,
        closed: Arc<Mutex<bool>>,
    }

    impl StubWriter {
        fn new() -> Self {
            Self {
                written: Arc::new(Mutex::new(Vec::new())),
                closed: Arc::new(Mutex::new(false)),
            }
        }
    }

    impl VideoWriter for StubWriter {
        fn open(
            &mut self,
            _path: &Path,
            _metadata: &VideoMetadata,
        ) -> Result<(), Box<dyn std::error::Error>> {
            Ok(())
        }

        fn write(&mut self, frame: &Frame) -> Result<(), Box<dyn std::error::Error>> {
            self.written.lock().unwrap().push(frame.clone());
            Ok(())
        }

        fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
            *self.closed.lock().unwrap() = true;
            Ok(())
        }
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
    struct PassthroughBlurrer {
        calls: Arc<Mutex<Vec<(usize, Vec<Region>)>>>,
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
            frame: &mut Frame,
            regions: &[Region],
        ) -> Result<(), Box<dyn std::error::Error>> {
            self.calls
                .lock()
                .unwrap()
                .push((frame.index(), regions.to_vec()));
            Ok(())
        }
    }

    struct FailingDetector;

    impl FaceDetector for FailingDetector {
        fn detect(&mut self, _frame: &Frame) -> Result<Vec<Region>, Box<dyn std::error::Error>> {
            Err("detector error".into())
        }
    }

    // --- Helpers ---

    fn make_frame(index: usize) -> Frame {
        Frame::new(vec![128; 100 * 100 * 3], 100, 100, 3, index)
    }

    fn make_frames(count: usize) -> Vec<Frame> {
        (0..count).map(make_frame).collect()
    }

    fn metadata(w: u32, h: u32) -> VideoMetadata {
        VideoMetadata {
            width: w,
            height: h,
            fps: 30.0,
            total_frames: 0,
            codec: String::new(),
            source_path: None,
            rotation: 0,
        }
    }

    fn region_at(x: i32, y: i32, track_id: Option<u32>) -> Region {
        Region {
            x,
            y,
            width: 20,
            height: 20,
            track_id,
            full_width: None,
            full_height: None,
            unclamped_x: None,
            unclamped_y: None,
        }
    }

    fn meta_with_count(count: usize) -> VideoMetadata {
        VideoMetadata {
            width: 100,
            height: 100,
            fps: 30.0,
            total_frames: count,
            codec: String::new(),
            source_path: None,
            rotation: 0,
        }
    }

    fn default_executor() -> Box<dyn PipelineExecutor> {
        Box::new(ThreadedPipelineExecutor::new())
    }

    // --- Tests ---

    #[test]
    fn test_processes_all_frames() {
        let writer = StubWriter::new();
        let written = writer.written.clone();

        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(make_frames(5))),
            Box::new(writer),
            Box::new(StubDetector {
                results: HashMap::new(),
            }),
            Box::new(PassthroughBlurrer::new()),
            RegionMerger::new(),
            default_executor(),
            Some(0),
            None,
            None,
            None,
            None,
        );

        uc.execute(&meta_with_count(5), Path::new("/tmp/out.mp4"))
            .unwrap();
        assert_eq!(written.lock().unwrap().len(), 5);
    }

    #[test]
    fn test_frames_written_in_order() {
        let writer = StubWriter::new();
        let written = writer.written.clone();

        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(make_frames(10))),
            Box::new(writer),
            Box::new(StubDetector {
                results: HashMap::new(),
            }),
            Box::new(PassthroughBlurrer::new()),
            RegionMerger::new(),
            default_executor(),
            Some(3),
            None,
            None,
            None,
            None,
        );

        uc.execute(&meta_with_count(10), Path::new("/tmp/out.mp4"))
            .unwrap();

        let written = written.lock().unwrap();
        assert_eq!(written.len(), 10);
        for (i, frame) in written.iter().enumerate() {
            assert_eq!(frame.index(), i);
        }
    }

    #[test]
    fn test_no_faces_still_outputs_all_frames() {
        let writer = StubWriter::new();
        let written = writer.written.clone();

        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(make_frames(3))),
            Box::new(writer),
            Box::new(StubDetector {
                results: HashMap::new(),
            }),
            Box::new(PassthroughBlurrer::new()),
            RegionMerger::new(),
            default_executor(),
            Some(2),
            None,
            None,
            None,
            None,
        );

        uc.execute(&meta_with_count(3), Path::new("/tmp/out.mp4"))
            .unwrap();
        assert_eq!(written.lock().unwrap().len(), 3);
    }

    #[test]
    fn test_empty_video() {
        let writer = StubWriter::new();
        let written = writer.written.clone();

        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(vec![])),
            Box::new(writer),
            Box::new(StubDetector {
                results: HashMap::new(),
            }),
            Box::new(PassthroughBlurrer::new()),
            RegionMerger::new(),
            default_executor(),
            Some(5),
            None,
            None,
            None,
            None,
        );

        uc.execute(&meta_with_count(0), Path::new("/tmp/out.mp4"))
            .unwrap();
        assert!(written.lock().unwrap().is_empty());
    }

    #[test]
    fn test_closes_reader_and_writer() {
        let reader = StubReader::new(make_frames(2));
        let reader_closed = reader.closed.clone();
        let writer = StubWriter::new();
        let writer_closed = writer.closed.clone();

        let mut uc = BlurFacesUseCase::new(
            Box::new(reader),
            Box::new(writer),
            Box::new(StubDetector {
                results: HashMap::new(),
            }),
            Box::new(PassthroughBlurrer::new()),
            RegionMerger::new(),
            default_executor(),
            Some(0),
            None,
            None,
            None,
            None,
        );

        uc.execute(&meta_with_count(2), Path::new("/tmp/out.mp4"))
            .unwrap();

        assert!(*reader_closed.lock().unwrap());
        assert!(*writer_closed.lock().unwrap());
    }

    #[test]
    fn test_lookahead_provides_future_regions_to_merger() {
        let blurrer = PassthroughBlurrer::new();
        let calls = blurrer.calls.clone();

        // Face only in frame 2
        let mut det_results = HashMap::new();
        det_results.insert(2, vec![region_at(30, 30, Some(1))]);

        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(make_frames(5))),
            Box::new(StubWriter::new()),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(blurrer),
            RegionMerger::new(),
            default_executor(),
            Some(3),
            None,
            None,
            None,
            None,
        );

        uc.execute(&meta_with_count(5), Path::new("/tmp/out.mp4"))
            .unwrap();

        let calls = calls.lock().unwrap();
        // Frame 0 should have merged with lookahead (frames 1,2,3)
        // Frame 2's region should appear in frame 0's blur via merger
        assert_eq!(calls.len(), 5);

        // Frame 0's merged regions should include the track from frame 2
        let frame_0_regions = &calls[0].1;
        assert!(
            !frame_0_regions.is_empty(),
            "Frame 0 should have regions from lookahead (frame 2 has a face)"
        );
    }

    #[test]
    fn test_lookahead_zero_no_future_regions() {
        let blurrer = PassthroughBlurrer::new();
        let calls = blurrer.calls.clone();

        // Face only in frame 2
        let mut det_results = HashMap::new();
        det_results.insert(2, vec![region_at(30, 30, Some(1))]);

        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(make_frames(5))),
            Box::new(StubWriter::new()),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(blurrer),
            RegionMerger::new(),
            default_executor(),
            Some(0), // no lookahead
            None,
            None,
            None,
            None,
        );

        uc.execute(&meta_with_count(5), Path::new("/tmp/out.mp4"))
            .unwrap();

        let calls = calls.lock().unwrap();
        // Frame 0 should have no regions (face is only in frame 2)
        assert!(calls[0].1.is_empty());
        // Frame 2 should have the face region
        assert!(!calls[2].1.is_empty());
    }

    #[test]
    fn test_blur_ids_only_blurs_specified_faces() {
        let blurrer = PassthroughBlurrer::new();
        let calls = blurrer.calls.clone();

        let mut det_results = HashMap::new();
        det_results.insert(
            0,
            vec![region_at(10, 10, Some(1)), region_at(50, 50, Some(2))],
        );

        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(make_frames(1))),
            Box::new(StubWriter::new()),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(blurrer),
            RegionMerger::new(),
            default_executor(),
            Some(0),
            Some(HashSet::from([1])), // only blur track 1
            None,
            None,
            None,
        );

        uc.execute(&meta_with_count(1), Path::new("/tmp/out.mp4"))
            .unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls[0].1.len(), 1);
        assert_eq!(calls[0].1[0].track_id, Some(1));
    }

    #[test]
    fn test_exclude_ids_skips_specified_faces() {
        let blurrer = PassthroughBlurrer::new();
        let calls = blurrer.calls.clone();

        let mut det_results = HashMap::new();
        det_results.insert(
            0,
            vec![region_at(10, 10, Some(1)), region_at(50, 50, Some(2))],
        );

        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(make_frames(1))),
            Box::new(StubWriter::new()),
            Box::new(StubDetector {
                results: det_results,
            }),
            Box::new(blurrer),
            RegionMerger::new(),
            default_executor(),
            Some(0),
            None,
            Some(HashSet::from([2])), // exclude track 2
            None,
            None,
        );

        uc.execute(&meta_with_count(1), Path::new("/tmp/out.mp4"))
            .unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls[0].1.len(), 1);
        assert_eq!(calls[0].1[0].track_id, Some(1));
    }

    #[test]
    fn test_cancel_via_on_progress() {
        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(make_frames(10))),
            Box::new(StubWriter::new()),
            Box::new(StubDetector {
                results: HashMap::new(),
            }),
            Box::new(PassthroughBlurrer::new()),
            RegionMerger::new(),
            default_executor(),
            Some(0),
            None,
            None,
            Some(Box::new(|current, _total| current < 3)), // cancel after 3
            None,
        );

        let result = uc.execute(&meta_with_count(10), Path::new("/tmp/out.mp4"));
        assert!(result.is_err());
    }

    #[test]
    fn test_on_progress_returning_true_continues() {
        let progress_calls = Arc::new(Mutex::new(Vec::new()));
        let progress_clone = progress_calls.clone();

        let writer = StubWriter::new();
        let written = writer.written.clone();

        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(make_frames(5))),
            Box::new(writer),
            Box::new(StubDetector {
                results: HashMap::new(),
            }),
            Box::new(PassthroughBlurrer::new()),
            RegionMerger::new(),
            default_executor(),
            Some(0),
            None,
            None,
            Some(Box::new(move |current, total| {
                progress_clone.lock().unwrap().push((current, total));
                true
            })),
            None,
        );

        uc.execute(&meta_with_count(5), Path::new("/tmp/out.mp4"))
            .unwrap();

        assert_eq!(written.lock().unwrap().len(), 5);
        assert_eq!(progress_calls.lock().unwrap().len(), 5);
    }

    #[test]
    fn test_cancellation_via_atomic_bool() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = cancelled.clone();

        let writer = StubWriter::new();
        let written = writer.written.clone();

        // Cancel after 2 frames via progress callback side-effect
        let count = Arc::new(Mutex::new(0usize));
        let count_clone = count.clone();

        let mut uc = BlurFacesUseCase::new(
            Box::new(StubReader::new(make_frames(10))),
            Box::new(writer),
            Box::new(StubDetector {
                results: HashMap::new(),
            }),
            Box::new(PassthroughBlurrer::new()),
            RegionMerger::new(),
            default_executor(),
            Some(0),
            None,
            None,
            Some(Box::new(move |_current, _total| {
                let mut c = count_clone.lock().unwrap();
                *c += 1;
                if *c >= 3 {
                    cancelled_clone.store(true, Ordering::Relaxed);
                }
                true
            })),
            Some(cancelled),
        );

        uc.execute(&meta_with_count(10), Path::new("/tmp/out.mp4"))
            .unwrap();

        // Should have stopped early
        assert!(written.lock().unwrap().len() < 10);
    }

    #[test]
    fn test_closes_on_detector_error() {
        let reader = StubReader::new(make_frames(3));
        let _reader_closed = reader.closed.clone();
        let writer = StubWriter::new();
        let _writer_closed = writer.closed.clone();

        let mut uc = BlurFacesUseCase::new(
            Box::new(reader),
            Box::new(writer),
            Box::new(FailingDetector),
            Box::new(PassthroughBlurrer::new()),
            RegionMerger::new(),
            default_executor(),
            Some(0),
            None,
            None,
            None,
            None,
        );

        let result = uc.execute(&meta_with_count(3), Path::new("/tmp/out.mp4"));
        assert!(result.is_err());
    }
}
