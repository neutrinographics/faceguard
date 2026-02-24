use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{Receiver, Sender};

use video_blur_core::detection::domain::face_grouper::FaceGrouper;
use video_blur_core::detection::domain::face_region_builder::{FaceRegionBuilder, DEFAULT_PADDING};
use video_blur_core::detection::domain::region_smoother::{RegionSmoother, DEFAULT_ALPHA};
use video_blur_core::detection::infrastructure::bytetrack_tracker::ByteTracker;
use video_blur_core::detection::infrastructure::histogram_face_grouper::HistogramFaceGrouper;
use video_blur_core::detection::infrastructure::model_resolver;
use video_blur_core::detection::infrastructure::onnx_yolo_detector::OnnxYoloDetector;
use video_blur_core::detection::infrastructure::skip_frame_detector::SkipFrameDetector;
use video_blur_core::pipeline::preview_faces_use_case::PreviewFacesUseCase;
use video_blur_core::shared::constants::{
    EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_URL, IMAGE_EXTENSIONS, TRACKER_MAX_LOST,
    YOLO_MODEL_NAME, YOLO_MODEL_URL,
};
use video_blur_core::shared::region::Region;
use video_blur_core::video::infrastructure::ffmpeg_reader::FfmpegReader;
use video_blur_core::video::infrastructure::image_file_reader::ImageFileReader;
use video_blur_core::video::infrastructure::image_file_writer::ImageFileWriter;

/// Messages sent from the preview worker thread to the UI.
#[derive(Debug, Clone)]
pub enum PreviewMessage {
    DownloadProgress(u64, u64),
    ScanProgress(usize, usize),
    Complete(PreviewResult),
    Error(String),
    Cancelled,
}

/// Result of a preview scan.
#[derive(Debug, Clone)]
pub struct PreviewResult {
    /// track_id → path to saved thumbnail JPEG
    pub crops: HashMap<u32, PathBuf>,
    /// Groups of track_ids representing the same person
    pub groups: Vec<Vec<u32>>,
    /// Cached detection results: frame_index → regions
    pub detection_cache: HashMap<usize, Vec<Region>>,
    /// Temporary directory containing crop files (keep alive)
    #[allow(dead_code)]
    pub temp_dir: PathBuf,
}

/// Parameters for a preview job.
#[derive(Clone)]
pub struct PreviewParams {
    pub input_path: PathBuf,
    pub confidence: u32,
}

/// Spawn a background preview worker. Returns the channel receiver and cancellation token.
pub fn spawn(params: PreviewParams) -> (Receiver<PreviewMessage>, Arc<AtomicBool>) {
    let (tx, rx) = crossbeam_channel::unbounded::<PreviewMessage>();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_clone = cancelled.clone();

    thread::spawn(move || {
        if let Err(e) = run_preview(&tx, &cancelled_clone, &params) {
            if cancelled_clone.load(Ordering::Relaxed) {
                let _ = tx.send(PreviewMessage::Cancelled);
            } else {
                let _ = tx.send(PreviewMessage::Error(e.to_string()));
            }
        }
    });

    (rx, cancelled)
}

fn run_preview(
    tx: &Sender<PreviewMessage>,
    cancelled: &Arc<AtomicBool>,
    params: &PreviewParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = &params.input_path;
    let confidence = params.confidence as f64 / 100.0;

    // Resolve detector model
    let tx_dl = tx.clone();
    let model_path = model_resolver::resolve(
        YOLO_MODEL_NAME,
        YOLO_MODEL_URL,
        None,
        Some(Box::new(move |downloaded, total| {
            let _ = tx_dl.send(PreviewMessage::DownloadProgress(downloaded, total));
        })),
    )?;

    if cancelled.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }

    // Resolve embedding model for grouping (try to download, fall back to histogram)
    let tx_dl2 = tx.clone();
    let embedding_path = model_resolver::resolve(
        EMBEDDING_MODEL_NAME,
        EMBEDDING_MODEL_URL,
        None,
        Some(Box::new(move |downloaded, total| {
            let _ = tx_dl2.send(PreviewMessage::DownloadProgress(downloaded, total));
        })),
    );

    if cancelled.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }

    // Build detector
    let smoother = RegionSmoother::new(DEFAULT_ALPHA);
    let region_builder = FaceRegionBuilder::new(DEFAULT_PADDING, Some(Box::new(smoother)));
    let tracker = ByteTracker::new(TRACKER_MAX_LOST);
    let det = OnnxYoloDetector::new(&model_path, region_builder, tracker, confidence)?;
    let detector: Box<dyn video_blur_core::detection::domain::face_detector::FaceDetector> =
        Box::new(SkipFrameDetector::new(Box::new(det), 2)?);

    // Open reader (image or video)
    let is_image = input
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false);

    let mut reader: Box<dyn video_blur_core::video::domain::video_reader::VideoReader> = if is_image
    {
        Box::new(ImageFileReader::new())
    } else {
        Box::new(FfmpegReader::new())
    };
    let metadata = reader.open(input)?;

    // Create temp directory for crop images
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.path().to_path_buf();

    let image_writer: Box<dyn video_blur_core::video::domain::image_writer::ImageWriter> =
        Box::new(ImageFileWriter::new());

    let tx_progress = tx.clone();
    let cancelled_progress = cancelled.clone();
    let progress: Box<dyn Fn(usize, usize) -> bool + Send> = Box::new(move |current, total| {
        let _ = tx_progress.send(PreviewMessage::ScanProgress(current, total));
        !cancelled_progress.load(Ordering::Relaxed)
    });

    let mut use_case = PreviewFacesUseCase::new(reader, detector, image_writer, Some(progress));
    let (crops, detection_cache) = use_case.execute(&metadata, &temp_path)?;

    if cancelled.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }

    // Group faces
    let embedding_result = embedding_path.map_err(|e| -> Box<dyn std::error::Error> { e.into() });
    let groups = group_faces(&crops, &embedding_result)?;

    // Keep temp_dir alive by leaking it (we store the path, cleanup happens on new input)
    std::mem::forget(temp_dir);

    let _ = tx.send(PreviewMessage::Complete(PreviewResult {
        crops,
        groups,
        detection_cache,
        temp_dir: temp_path,
    }));

    Ok(())
}

/// Group faces by identity using embedding model (preferred) or histogram fallback.
fn group_faces(
    crops: &HashMap<u32, PathBuf>,
    embedding_path: &Result<PathBuf, Box<dyn std::error::Error>>,
) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
    if crops.is_empty() {
        return Ok(vec![]);
    }

    // Load crop image data
    let mut crop_data: Vec<(u32, Vec<u8>, u32, u32)> = Vec::new();
    for (&track_id, path) in crops {
        let img = image::open(path)?.to_rgb8();
        let (w, h) = img.dimensions();
        crop_data.push((track_id, img.into_raw(), w, h));
    }

    // Build refs for the trait call
    let crop_refs: Vec<(u32, &[u8], u32, u32)> = crop_data
        .iter()
        .map(|(id, data, w, h)| (*id, data.as_slice(), *w, *h))
        .collect();

    // Try embedding grouper first, fall back to histogram
    if let Ok(ref model_path) = embedding_path {
        use video_blur_core::detection::infrastructure::embedding_face_grouper::EmbeddingFaceGrouper;
        match EmbeddingFaceGrouper::new(
            model_path,
            video_blur_core::detection::infrastructure::embedding_face_grouper::DEFAULT_THRESHOLD,
        ) {
            Ok(grouper) => match grouper.group(&crop_refs) {
                Ok(groups) => return Ok(groups),
                Err(e) => {
                    log::warn!("Embedding grouper failed, falling back to histogram: {e}");
                }
            },
            Err(e) => {
                log::warn!("Failed to load embedding model, falling back to histogram: {e}");
            }
        }
    }

    // Histogram fallback
    let grouper = HistogramFaceGrouper::default();
    grouper.group(&crop_refs)
}
