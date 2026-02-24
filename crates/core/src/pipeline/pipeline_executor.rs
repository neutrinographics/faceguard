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

/// Configuration for a pipeline execution run.
pub struct PipelineConfig {
    pub lookahead: usize,
    pub blur_ids: Option<HashSet<u32>>,
    pub exclude_ids: Option<HashSet<u32>>,
    pub on_progress: Option<Box<dyn Fn(usize, usize) -> bool + Send>>,
    pub cancelled: Arc<AtomicBool>,
}

/// Abstracts how the read → detect → blur → write pipeline is executed.
///
/// This is a port (application-layer interface). Infrastructure provides
/// concrete implementations (e.g. threaded, single-threaded).
pub trait PipelineExecutor: Send {
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        reader: Box<dyn VideoReader>,
        writer: Box<dyn VideoWriter>,
        detector: Box<dyn FaceDetector>,
        blurrer: Box<dyn FrameBlurrer>,
        merger: RegionMerger,
        metadata: &VideoMetadata,
        output_path: &Path,
        config: PipelineConfig,
    ) -> Result<(), Box<dyn std::error::Error>>;
}
