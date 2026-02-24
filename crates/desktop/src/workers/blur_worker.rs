use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{Receiver, Sender};

use video_blur_core::blurring::infrastructure::blurrer_factory;
use video_blur_core::blurring::infrastructure::gpu_context::GpuContext;
use video_blur_core::detection::domain::face_detector::FaceDetector;
use video_blur_core::detection::domain::face_region_builder::{FaceRegionBuilder, DEFAULT_PADDING};
use video_blur_core::detection::domain::region_merger::RegionMerger;
use video_blur_core::detection::domain::region_smoother::{RegionSmoother, DEFAULT_ALPHA};
use video_blur_core::detection::infrastructure::bytetrack_tracker::ByteTracker;
use video_blur_core::detection::infrastructure::cached_face_detector::CachedFaceDetector;
use video_blur_core::detection::infrastructure::onnx_yolo_detector::OnnxYoloDetector;
use video_blur_core::detection::infrastructure::skip_frame_detector::SkipFrameDetector;
use video_blur_core::pipeline::blur_faces_use_case::BlurFacesUseCase;
use video_blur_core::pipeline::blur_image_use_case::BlurImageUseCase;
use video_blur_core::pipeline::infrastructure::threaded_pipeline_executor::ThreadedPipelineExecutor;
use video_blur_core::shared::constants::{IMAGE_EXTENSIONS, TRACKER_MAX_LOST};
use video_blur_core::shared::region::Region;
use video_blur_core::video::domain::image_writer::ImageWriter;
use video_blur_core::video::domain::video_reader::VideoReader;
use video_blur_core::video::domain::video_writer::VideoWriter;
use video_blur_core::video::infrastructure::ffmpeg_reader::FfmpegReader;
use video_blur_core::video::infrastructure::ffmpeg_writer::FfmpegWriter;
use video_blur_core::video::infrastructure::image_file_reader::ImageFileReader;
use video_blur_core::video::infrastructure::image_file_writer::ImageFileWriter;

use super::model_cache::ModelCache;

#[derive(Debug, Clone)]
pub enum WorkerMessage {
    DownloadProgress(u64, u64),
    BlurProgress(usize, usize),
    Complete,
    Error(String),
    Cancelled,
}

pub struct BlurParams {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub blur_shape: crate::settings::BlurShape,
    pub confidence: u32,
    pub blur_strength: u32,
    pub lookahead: u32,
    pub detection_cache: Option<Arc<HashMap<usize, Vec<Region>>>>,
    pub blur_ids: Option<HashSet<u32>>,
    pub model_cache: Arc<ModelCache>,
    pub gpu_context: Option<Arc<GpuContext>>,
}

pub fn spawn(params: BlurParams) -> (Receiver<WorkerMessage>, Arc<AtomicBool>) {
    let (tx, rx) = crossbeam_channel::unbounded::<WorkerMessage>();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_clone = cancelled.clone();

    thread::spawn(move || {
        if let Err(e) = run_blur(&tx, &cancelled_clone, &params) {
            if cancelled_clone.load(Ordering::Relaxed) {
                let _ = tx.send(WorkerMessage::Cancelled);
            } else {
                let _ = tx.send(WorkerMessage::Error(e.to_string()));
            }
        }
    });

    (rx, cancelled)
}

fn run_blur(
    tx: &Sender<WorkerMessage>,
    cancelled: &Arc<AtomicBool>,
    params: &BlurParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = &params.input_path;
    let output = &params.output_path;
    let confidence = params.confidence as f64 / 100.0;

    let detector = build_detector(params, tx, cancelled, confidence)?;
    let blurrer = build_blurrer(params);

    if is_image(input) {
        blur_image(input, output, detector, blurrer, params)?;
    } else {
        blur_video(input, output, detector, blurrer, params, tx, cancelled)?;
    }

    if cancelled.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }

    let _ = tx.send(WorkerMessage::Complete);
    Ok(())
}

fn build_detector(
    params: &BlurParams,
    tx: &Sender<WorkerMessage>,
    cancelled: &Arc<AtomicBool>,
    confidence: f64,
) -> Result<Box<dyn FaceDetector>, Box<dyn std::error::Error>> {
    if let Some(ref cache) = params.detection_cache {
        return Ok(Box::new(CachedFaceDetector::new(Arc::clone(cache))));
    }

    let tx_dl = tx.clone();
    let model_path = params
        .model_cache
        .wait_for_yolo(
            &|dl, total| {
                let _ = tx_dl.send(WorkerMessage::DownloadProgress(dl, total));
            },
            cancelled,
        )
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    if cancelled.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }

    let smoother = RegionSmoother::new(DEFAULT_ALPHA);
    let region_builder = FaceRegionBuilder::new(DEFAULT_PADDING, Some(Box::new(smoother)));
    let tracker = ByteTracker::new(TRACKER_MAX_LOST);

    let det = match params.model_cache.get_yolo_session() {
        Some((session, input_size)) => {
            log::info!("Blur: using shared YOLO session");
            OnnxYoloDetector::from_shared_session(
                session,
                input_size,
                region_builder,
                tracker,
                confidence,
            )
        }
        None => {
            log::info!("Blur: building new YOLO session from path");
            OnnxYoloDetector::new(&model_path, region_builder, tracker, confidence)?
        }
    };

    Ok(Box::new(SkipFrameDetector::new(Box::new(det), 2)?))
}

fn build_blurrer(
    params: &BlurParams,
) -> Box<dyn video_blur_core::blurring::domain::frame_blurrer::FrameBlurrer> {
    let blur_shape = match params.blur_shape {
        crate::settings::BlurShape::Ellipse => blurrer_factory::BlurShape::Elliptical,
        crate::settings::BlurShape::Rect => blurrer_factory::BlurShape::Rectangular,
    };
    blurrer_factory::create_blurrer_with_context(
        blur_shape,
        params.blur_strength as usize,
        params.gpu_context.clone(),
    )
}

fn blur_image(
    input: &std::path::Path,
    output: &std::path::Path,
    detector: Box<dyn FaceDetector>,
    blurrer: Box<dyn video_blur_core::blurring::domain::frame_blurrer::FrameBlurrer>,
    params: &BlurParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader: Box<dyn VideoReader> = Box::new(ImageFileReader::new());
    let writer: Box<dyn ImageWriter> = Box::new(ImageFileWriter::new());
    let mut use_case = BlurImageUseCase::new(
        reader,
        writer,
        detector,
        blurrer,
        params.blur_ids.clone(),
        None,
    );
    use_case.execute(input, output)?;
    Ok(())
}

fn blur_video(
    input: &std::path::Path,
    output: &std::path::Path,
    detector: Box<dyn FaceDetector>,
    blurrer: Box<dyn video_blur_core::blurring::domain::frame_blurrer::FrameBlurrer>,
    params: &BlurParams,
    tx: &Sender<WorkerMessage>,
    cancelled: &Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader: Box<dyn VideoReader> = Box::new(FfmpegReader::new());
    let metadata = reader.open(input)?;
    let writer: Box<dyn VideoWriter> = Box::new(FfmpegWriter::new());
    let merger = RegionMerger::new();

    let _ = tx.send(WorkerMessage::BlurProgress(0, metadata.total_frames));

    let tx_progress = tx.clone();
    let cancelled_progress = cancelled.clone();
    let progress: Box<dyn Fn(usize, usize) -> bool + Send> = Box::new(move |current, total| {
        let _ = tx_progress.send(WorkerMessage::BlurProgress(current, total));
        !cancelled_progress.load(Ordering::Relaxed)
    });

    let mut use_case = BlurFacesUseCase::new(
        reader,
        writer,
        detector,
        blurrer,
        merger,
        Box::new(ThreadedPipelineExecutor::new()),
        Some(params.lookahead as usize),
        params.blur_ids.clone(),
        None,
        Some(progress),
        Some(cancelled.clone()),
    );
    use_case.execute(&metadata, output)?;
    Ok(())
}

fn is_image(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}
