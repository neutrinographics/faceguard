use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{Receiver, Sender};

use faceguard_core::detection::domain::face_detector::FaceDetector;
use faceguard_core::detection::domain::face_grouper::FaceGrouper;
use faceguard_core::detection::domain::face_region_builder::FaceRegionBuilder;
use faceguard_core::detection::domain::region_smoother::{RegionSmoother, DEFAULT_ALPHA};
use faceguard_core::detection::infrastructure::bytetrack_tracker::ByteTracker;
use faceguard_core::detection::infrastructure::histogram_face_grouper::HistogramFaceGrouper;
use faceguard_core::detection::infrastructure::onnx_yolo_detector::OnnxYoloDetector;
use faceguard_core::detection::infrastructure::skip_frame_detector::SkipFrameDetector;
use faceguard_core::pipeline::preview_faces_use_case::PreviewFacesUseCase;
use faceguard_core::shared::constants::{IMAGE_EXTENSIONS, TRACKER_MAX_LOST};
use faceguard_core::shared::region::Region;
use faceguard_core::video::domain::image_writer::ImageWriter;
use faceguard_core::video::domain::video_reader::VideoReader;
use faceguard_core::video::infrastructure::ffmpeg_reader::FfmpegReader;
use faceguard_core::video::infrastructure::image_file_reader::ImageFileReader;
use faceguard_core::video::infrastructure::image_file_writer::ImageFileWriter;

use super::model_cache::ModelCache;

pub enum PreviewMessage {
    DownloadProgress(u64, u64),
    ScanProgress(usize, usize),
    Complete(PreviewResult),
    Error(String),
    Cancelled,
}

pub struct PreviewResult {
    pub crops: HashMap<u32, PathBuf>,
    pub groups: Vec<Vec<u32>>,
    pub detection_cache: HashMap<usize, Vec<Region>>,
    pub temp_dir: tempfile::TempDir,
}

pub struct PreviewParams {
    pub input_path: PathBuf,
    pub confidence: u32,
    pub blur_coverage: u32,
    pub center_offset: i32,
    pub model_cache: Arc<ModelCache>,
}

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

    let detector = build_detector(params, tx, cancelled, confidence)?;
    let embedding_path = wait_for_embedding(params, tx, cancelled);

    if cancelled.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }

    let mut reader: Box<dyn VideoReader> = if is_image(input) {
        Box::new(ImageFileReader::new())
    } else {
        Box::new(FfmpegReader::new())
    };
    let metadata = reader.open(input)?;

    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.path().to_path_buf();
    let image_writer: Box<dyn ImageWriter> = Box::new(ImageFileWriter::new());

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

    let embedding_result = embedding_path.map_err(|e| -> Box<dyn std::error::Error> { e.into() });
    let groups = group_faces(&crops, &embedding_result)?;

    let _ = tx.send(PreviewMessage::Complete(PreviewResult {
        crops,
        groups,
        detection_cache,
        temp_dir,
    }));

    Ok(())
}

fn build_detector(
    params: &PreviewParams,
    tx: &Sender<PreviewMessage>,
    cancelled: &Arc<AtomicBool>,
    confidence: f64,
) -> Result<Box<dyn FaceDetector>, Box<dyn std::error::Error>> {
    let tx_dl = tx.clone();
    let model_path = params
        .model_cache
        .wait_for_yolo(
            &|dl, total| {
                let _ = tx_dl.send(PreviewMessage::DownloadProgress(dl, total));
            },
            cancelled,
        )
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    if cancelled.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }

    let padding = params.blur_coverage as f64 / 100.0;
    let center_offset = params.center_offset as f64 / 100.0;
    let smoother = RegionSmoother::new(DEFAULT_ALPHA);
    let region_builder = FaceRegionBuilder::new(padding, center_offset, Some(Box::new(smoother)));
    let tracker = ByteTracker::new(TRACKER_MAX_LOST);

    let det = match params.model_cache.get_yolo_session() {
        Some((session, input_size)) => OnnxYoloDetector::from_shared_session(
            session,
            input_size,
            region_builder,
            tracker,
            confidence,
        ),
        None => OnnxYoloDetector::new(&model_path, region_builder, tracker, confidence)?,
    };

    Ok(Box::new(SkipFrameDetector::new(Box::new(det), 2)?))
}

fn wait_for_embedding(
    params: &PreviewParams,
    tx: &Sender<PreviewMessage>,
    cancelled: &Arc<AtomicBool>,
) -> Result<PathBuf, String> {
    let tx_dl = tx.clone();
    params.model_cache.wait_for_embedding(
        &|dl, total| {
            let _ = tx_dl.send(PreviewMessage::DownloadProgress(dl, total));
        },
        cancelled,
    )
}

/// Try embedding-based grouping first; fall back to histogram on any failure.
fn group_faces(
    crops: &HashMap<u32, PathBuf>,
    embedding_path: &Result<PathBuf, Box<dyn std::error::Error>>,
) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
    if crops.is_empty() {
        return Ok(vec![]);
    }

    let mut crop_data: Vec<(u32, Vec<u8>, u32, u32)> = Vec::new();
    for (&track_id, path) in crops {
        let img = image::open(path)?.to_rgb8();
        let (w, h) = img.dimensions();
        crop_data.push((track_id, img.into_raw(), w, h));
    }

    let crop_refs: Vec<(u32, &[u8], u32, u32)> = crop_data
        .iter()
        .map(|(id, data, w, h)| (*id, data.as_slice(), *w, *h))
        .collect();

    if let Ok(ref model_path) = embedding_path {
        use faceguard_core::detection::infrastructure::embedding_face_grouper::EmbeddingFaceGrouper;
        match EmbeddingFaceGrouper::new(
            model_path,
            faceguard_core::detection::infrastructure::embedding_face_grouper::DEFAULT_THRESHOLD,
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

    let grouper = HistogramFaceGrouper::default();
    grouper.group(&crop_refs)
}

fn is_image(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}
