use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;

use video_blur_core::blurring::infrastructure::blurrer_factory::{create_blurrer, BlurShape};
use video_blur_core::detection::domain::face_region_builder::{FaceRegionBuilder, DEFAULT_PADDING};
use video_blur_core::detection::domain::region_merger::RegionMerger;
use video_blur_core::detection::domain::region_smoother::{RegionSmoother, DEFAULT_ALPHA};
use video_blur_core::detection::infrastructure::bytetrack_tracker::ByteTracker;
use video_blur_core::detection::infrastructure::model_resolver;
use video_blur_core::detection::infrastructure::onnx_blazeface_detector::OnnxBlazefaceDetector;
use video_blur_core::detection::infrastructure::onnx_yolo_detector::OnnxYoloDetector;
use video_blur_core::detection::infrastructure::skip_frame_detector::SkipFrameDetector;
use video_blur_core::pipeline::blur_faces_use_case::BlurFacesUseCase;
use video_blur_core::pipeline::blur_image_use_case::BlurImageUseCase;
use video_blur_core::pipeline::preview_faces_use_case::PreviewFacesUseCase;
use video_blur_core::video::infrastructure::ffmpeg_reader::FfmpegReader;
use video_blur_core::video::infrastructure::ffmpeg_writer::FfmpegWriter;
use video_blur_core::video::infrastructure::image_file_reader::ImageFileReader;
use video_blur_core::video::infrastructure::image_file_writer::ImageFileWriter;

// ---------------------------------------------------------------------------
// Model constants
// ---------------------------------------------------------------------------

const YOLO_MODEL_NAME: &str = "yolo11n-face.onnx";
const YOLO_MODEL_URL: &str =
    "https://github.com/da1nerd/video-blur/releases/download/models-v1/yolo11n-face.onnx";

const BLAZEFACE_MODEL_NAME: &str = "blaze_face_short_range.onnx";
const BLAZEFACE_MODEL_URL: &str =
    "https://github.com/da1nerd/video-blur/releases/download/models-v1/blaze_face_short_range.onnx";

/// Max frames a track can be lost before removal (~1 second at 30 fps).
const TRACKER_MAX_LOST: usize = 30;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// Face detection and blurring for videos and images.
#[derive(Parser)]
#[command(name = "video-blur")]
struct Cli {
    /// Input video or image file.
    input: PathBuf,

    /// Output file (required unless --preview is used).
    output: Option<PathBuf>,

    /// Face detection confidence threshold (0.0-1.0).
    #[arg(long, default_value = "0.5")]
    confidence: f64,

    /// Gaussian blur kernel size (must be odd).
    #[arg(long, default_value = "201")]
    blur_strength: usize,

    /// Blur shape: ellipse or rect.
    #[arg(long, default_value = "ellipse")]
    blur_shape: String,

    /// Face detector backend: yolo or mediapipe.
    #[arg(long, default_value = "yolo")]
    detector: String,

    /// Frames to look ahead for early face blur.
    #[arg(long, default_value = "10")]
    lookahead: usize,

    /// Run detection every Nth frame (1 = every frame).
    #[arg(long, default_value = "2")]
    skip_frames: usize,

    /// Save face crops to directory instead of blurring.
    #[arg(long)]
    preview: Option<PathBuf>,

    /// Only blur these track IDs (comma-separated).
    #[arg(long, value_delimiter = ',')]
    blur_ids: Option<Vec<u32>>,

    /// Blur all faces except these track IDs (comma-separated).
    #[arg(long, value_delimiter = ',')]
    exclude_ids: Option<Vec<u32>>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn validate(cli: &Cli) -> Result<(), String> {
    if !cli.input.exists() {
        return Err(format!("Input file not found: {}", cli.input.display()));
    }
    if cli.blur_ids.is_some() && cli.exclude_ids.is_some() {
        return Err("--blur-ids and --exclude-ids are mutually exclusive".into());
    }
    if cli.preview.is_none() && cli.output.is_none() {
        return Err("Output file is required unless --preview is used".into());
    }
    if cli.blur_strength == 0 || cli.blur_strength % 2 == 0 {
        return Err(format!(
            "Blur strength must be a positive odd integer, got {}",
            cli.blur_strength
        ));
    }
    if !(0.0..=1.0).contains(&cli.confidence) {
        return Err(format!(
            "Confidence must be between 0.0 and 1.0, got {}",
            cli.confidence
        ));
    }
    if cli.detector != "yolo" && cli.detector != "mediapipe" {
        return Err(format!(
            "Detector must be 'yolo' or 'mediapipe', got '{}'",
            cli.detector
        ));
    }
    if cli.blur_shape != "ellipse" && cli.blur_shape != "rect" {
        return Err(format!(
            "Blur shape must be 'ellipse' or 'rect', got '{}'",
            cli.blur_shape
        ));
    }
    Ok(())
}

fn download_progress(downloaded: u64, total: u64) {
    if total > 0 {
        let pct = (downloaded as f64 / total as f64 * 100.0) as u32;
        eprint!("\rDownloading model... {pct}%");
    } else {
        eprint!("\rDownloading model... {downloaded} bytes");
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    validate(&cli).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    let input = &cli.input;
    let image_mode = is_image(input);

    // Resolve model
    let (model_name, model_url) = if cli.detector == "mediapipe" {
        (BLAZEFACE_MODEL_NAME, BLAZEFACE_MODEL_URL)
    } else {
        (YOLO_MODEL_NAME, YOLO_MODEL_URL)
    };

    log::info!("Resolving model: {model_name}");
    let model_path = model_resolver::resolve(
        model_name,
        model_url,
        None,
        Some(Box::new(download_progress)),
    )?;
    eprintln!(); // newline after download progress

    // Build detector
    let detector: Box<dyn video_blur_core::detection::domain::face_detector::FaceDetector> =
        if cli.detector == "mediapipe" {
            let det = OnnxBlazefaceDetector::new(&model_path, cli.confidence, 30.0)?;
            Box::new(det)
        } else {
            let smoother = RegionSmoother::new(DEFAULT_ALPHA);
            let region_builder = FaceRegionBuilder::new(DEFAULT_PADDING, Some(Box::new(smoother)));
            let tracker = ByteTracker::new(TRACKER_MAX_LOST);
            let det = OnnxYoloDetector::new(&model_path, region_builder, tracker, cli.confidence)?;
            Box::new(det)
        };

    // Wrap in skip-frame decorator if needed
    let detector: Box<dyn video_blur_core::detection::domain::face_detector::FaceDetector> =
        if cli.skip_frames > 1 {
            Box::new(SkipFrameDetector::new(detector, cli.skip_frames)?)
        } else {
            detector
        };

    // Build blurrer
    let blur_shape = if cli.blur_shape == "rect" {
        BlurShape::Rectangular
    } else {
        BlurShape::Elliptical
    };
    let blurrer = create_blurrer(blur_shape, cli.blur_strength);

    // Parse ID filters
    let blur_ids: Option<HashSet<u32>> = cli.blur_ids.map(|ids| ids.into_iter().collect());
    let exclude_ids: Option<HashSet<u32>> = cli.exclude_ids.map(|ids| ids.into_iter().collect());

    // Branch: preview / image / video
    if let Some(ref preview_dir) = cli.preview {
        // Preview mode
        let mut reader: Box<dyn video_blur_core::video::domain::video_reader::VideoReader> =
            if image_mode {
                Box::new(ImageFileReader::new())
            } else {
                Box::new(FfmpegReader::new())
            };
        let metadata = reader.open(input)?;
        let image_writer = Box::new(ImageFileWriter::new());

        let progress: Box<dyn Fn(usize, usize) -> bool + Send> = Box::new(|current, total| {
            eprint!("\rScanning frame {current}/{total}");
            true
        });

        let mut use_case = PreviewFacesUseCase::new(reader, detector, image_writer, Some(progress));
        let (crops, _cache) = use_case.execute(&metadata, preview_dir)?;
        eprintln!();
        log::info!(
            "Saved {} face crops to {}",
            crops.len(),
            preview_dir.display()
        );
    } else if image_mode {
        // Image mode
        let reader: Box<dyn video_blur_core::video::domain::video_reader::VideoReader> =
            Box::new(ImageFileReader::new());
        let image_writer: Box<dyn video_blur_core::video::domain::image_writer::ImageWriter> =
            Box::new(ImageFileWriter::new());
        let output = cli.output.as_ref().unwrap();

        let mut use_case = BlurImageUseCase::new(
            reader,
            image_writer,
            detector,
            blurrer,
            blur_ids,
            exclude_ids,
        );
        use_case.execute(input, output)?;
        log::info!("Output written to {}", output.display());
    } else {
        // Video mode
        let mut reader: Box<dyn video_blur_core::video::domain::video_reader::VideoReader> =
            Box::new(FfmpegReader::new());
        let metadata = reader.open(input)?;
        let writer: Box<dyn video_blur_core::video::domain::video_writer::VideoWriter> =
            Box::new(FfmpegWriter::new());
        let merger = RegionMerger::new();
        let output = cli.output.as_ref().unwrap();

        let total = metadata.total_frames;
        let progress: Box<dyn Fn(usize, usize) -> bool + Send> = Box::new(move |current, _| {
            eprint!("\rProcessing frame {current}/{total}");
            true
        });

        let mut use_case = BlurFacesUseCase::new(
            reader,
            writer,
            detector,
            blurrer,
            merger,
            Some(cli.lookahead),
            blur_ids,
            exclude_ids,
            Some(progress),
            None,
        );
        use_case.execute(&metadata, output)?;
        eprintln!();
        log::info!("Output written to {}", output.display());
    }

    Ok(())
}

fn main() {
    env_logger::init();

    if let Err(e) = run() {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
