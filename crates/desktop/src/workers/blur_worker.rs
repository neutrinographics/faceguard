use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{Receiver, Sender};

use video_blur_core::blurring::infrastructure::blurrer_factory;
use video_blur_core::detection::domain::face_region_builder::{FaceRegionBuilder, DEFAULT_PADDING};
use video_blur_core::detection::domain::region_merger::RegionMerger;
use video_blur_core::detection::domain::region_smoother::{RegionSmoother, DEFAULT_ALPHA};
use video_blur_core::detection::infrastructure::bytetrack_tracker::ByteTracker;
use video_blur_core::detection::infrastructure::cached_face_detector::CachedFaceDetector;
use video_blur_core::detection::infrastructure::model_resolver;
use video_blur_core::detection::infrastructure::onnx_blazeface_detector::OnnxBlazefaceDetector;
use video_blur_core::detection::infrastructure::onnx_yolo_detector::OnnxYoloDetector;
use video_blur_core::detection::infrastructure::skip_frame_detector::SkipFrameDetector;
use video_blur_core::pipeline::blur_faces_use_case::BlurFacesUseCase;
use video_blur_core::pipeline::blur_image_use_case::BlurImageUseCase;
use video_blur_core::shared::region::Region;
use video_blur_core::video::infrastructure::ffmpeg_reader::FfmpegReader;
use video_blur_core::video::infrastructure::ffmpeg_writer::FfmpegWriter;
use video_blur_core::video::infrastructure::image_file_reader::ImageFileReader;
use video_blur_core::video::infrastructure::image_file_writer::ImageFileWriter;

const YOLO_MODEL_NAME: &str = "yolo11n-pose_widerface.onnx";
const YOLO_MODEL_URL: &str =
    "https://github.com/da1nerd/video-blur/releases/download/v0.1.0/yolo11n-pose_widerface.onnx";

const BLAZEFACE_MODEL_NAME: &str = "blaze_face_short_range.onnx";
const BLAZEFACE_MODEL_URL: &str =
    "https://github.com/da1nerd/video-blur/releases/download/models-v1/blaze_face_short_range.onnx";

const TRACKER_MAX_LOST: usize = 30;
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];

/// Messages sent from the worker thread to the UI.
#[derive(Debug, Clone)]
pub enum WorkerMessage {
    DownloadProgress(u64, u64),
    BlurProgress(usize, usize),
    Complete,
    Error(String),
    Cancelled,
}

/// Parameters for a blur job.
#[derive(Clone)]
pub struct BlurParams {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub detector: crate::settings::Detector,
    pub blur_shape: crate::settings::BlurShape,
    pub confidence: u32,
    pub blur_strength: u32,
    pub lookahead: u32,
    /// Cached detection results from preview pass (skip re-detecting).
    pub detection_cache: Option<HashMap<usize, Vec<Region>>>,
    /// If set, only blur these face IDs. None = blur all.
    pub blur_ids: Option<HashSet<u32>>,
}

fn is_image(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Spawn a background blur worker. Returns the channel receiver and cancellation token.
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
    let image_mode = is_image(input);

    // Build detector: use cached if available, otherwise fresh
    let detector: Box<dyn video_blur_core::detection::domain::face_detector::FaceDetector> =
        if let Some(cache) = params.detection_cache.clone() {
            Box::new(CachedFaceDetector::new(cache))
        } else {
            // Resolve model
            let (model_name, model_url) = match params.detector {
                crate::settings::Detector::Mediapipe => (BLAZEFACE_MODEL_NAME, BLAZEFACE_MODEL_URL),
                crate::settings::Detector::Yolo => (YOLO_MODEL_NAME, YOLO_MODEL_URL),
            };

            let tx_dl = tx.clone();
            let model_path = model_resolver::resolve(
                model_name,
                model_url,
                None,
                Some(Box::new(move |downloaded, total| {
                    let _ = tx_dl.send(WorkerMessage::DownloadProgress(downloaded, total));
                })),
            )?;

            if cancelled.load(Ordering::Relaxed) {
                return Err("Cancelled".into());
            }

            match params.detector {
                crate::settings::Detector::Mediapipe => {
                    let det = OnnxBlazefaceDetector::new(&model_path, confidence, 30.0)?;
                    Box::new(det)
                }
                crate::settings::Detector::Yolo => {
                    let smoother = RegionSmoother::new(DEFAULT_ALPHA);
                    let region_builder =
                        FaceRegionBuilder::new(DEFAULT_PADDING, Some(Box::new(smoother)));
                    let tracker = ByteTracker::new(TRACKER_MAX_LOST);
                    let det =
                        OnnxYoloDetector::new(&model_path, region_builder, tracker, confidence)?;
                    Box::new(SkipFrameDetector::new(Box::new(det), 2)?)
                }
            }
        };

    // Build blurrer
    let blur_shape = match params.blur_shape {
        crate::settings::BlurShape::Ellipse => blurrer_factory::BlurShape::Elliptical,
        crate::settings::BlurShape::Rect => blurrer_factory::BlurShape::Rectangular,
    };
    let blurrer = blurrer_factory::create_blurrer(blur_shape, params.blur_strength as usize);

    if image_mode {
        // Image mode — no progress needed
        let reader: Box<dyn video_blur_core::video::domain::video_reader::VideoReader> =
            Box::new(ImageFileReader::new());
        let image_writer: Box<dyn video_blur_core::video::domain::image_writer::ImageWriter> =
            Box::new(ImageFileWriter::new());

        let mut use_case = BlurImageUseCase::new(
            reader,
            image_writer,
            detector,
            blurrer,
            params.blur_ids.clone(),
            None,
        );
        use_case.execute(input, output)?;
    } else {
        // Video mode
        let mut reader: Box<dyn video_blur_core::video::domain::video_reader::VideoReader> =
            Box::new(FfmpegReader::new());
        let metadata = reader.open(input)?;
        let writer: Box<dyn video_blur_core::video::domain::video_writer::VideoWriter> =
            Box::new(FfmpegWriter::new());
        let merger = RegionMerger::new();

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
            Some(params.lookahead as usize),
            params.blur_ids.clone(),
            None,
            Some(progress),
            Some(cancelled.clone()),
        );
        use_case.execute(&metadata, output)?;
    }

    if cancelled.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }

    let _ = tx.send(WorkerMessage::Complete);
    Ok(())
}
