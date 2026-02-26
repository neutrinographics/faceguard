use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;

use faceguard_core::blurring::domain::frame_blurrer::FrameBlurrer;
use faceguard_core::blurring::infrastructure::blurrer_factory::{create_blurrer, BlurShape};
use faceguard_core::detection::domain::face_detector::FaceDetector;
use faceguard_core::detection::domain::face_region_builder::{FaceRegionBuilder, DEFAULT_PADDING};
use faceguard_core::detection::domain::region_merger::RegionMerger;
use faceguard_core::detection::domain::region_smoother::{RegionSmoother, DEFAULT_ALPHA};
use faceguard_core::detection::infrastructure::bytetrack_tracker::ByteTracker;
use faceguard_core::detection::infrastructure::model_resolver;
use faceguard_core::detection::infrastructure::onnx_yolo_detector::OnnxYoloDetector;
use faceguard_core::detection::infrastructure::skip_frame_detector::SkipFrameDetector;
use faceguard_core::pipeline::blur_faces_use_case::BlurFacesUseCase;
use faceguard_core::pipeline::blur_image_use_case::BlurImageUseCase;
use faceguard_core::pipeline::infrastructure::threaded_pipeline_executor::ThreadedPipelineExecutor;
use faceguard_core::pipeline::preview_faces_use_case::PreviewFacesUseCase;
use faceguard_core::shared::constants::{
    IMAGE_EXTENSIONS, TRACKER_MAX_LOST, YOLO_MODEL_NAME, YOLO_MODEL_URL,
};
use faceguard_core::video::domain::image_writer::ImageWriter;
use faceguard_core::video::domain::video_reader::VideoReader;
use faceguard_core::video::domain::video_writer::VideoWriter;
use faceguard_core::video::infrastructure::ffmpeg_reader::FfmpegReader;
use faceguard_core::video::infrastructure::ffmpeg_writer::FfmpegWriter;
use faceguard_core::video::infrastructure::image_file_reader::ImageFileReader;
use faceguard_core::video::infrastructure::image_file_writer::ImageFileWriter;

/// Face detection and blurring for videos and images.
#[derive(Parser)]
#[command(name = "faceguard")]
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

    /// H.264 CRF quality (0=lossless, 51=worst, default 18).
    #[arg(long)]
    quality: Option<u32>,
}

fn main() {
    env_logger::init();

    if let Err(e) = run() {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    validate(&cli)?;

    let detector = build_detector(&cli)?;
    let blurrer = create_blurrer(parse_blur_shape(&cli.blur_shape), cli.blur_strength);
    let input = cli.input;
    let output = cli.output;
    let lookahead = cli.lookahead;
    let blur_ids = to_id_set(cli.blur_ids);
    let exclude_ids = to_id_set(cli.exclude_ids);
    let quality = cli.quality;

    if let Some(preview_dir) = cli.preview {
        run_preview(&input, &preview_dir, detector)?;
    } else if is_image(&input) {
        run_image_blur(
            &input,
            output.as_ref().unwrap(),
            detector,
            blurrer,
            blur_ids,
            exclude_ids,
        )?;
    } else {
        run_video_blur(
            &input,
            output.as_ref().unwrap(),
            lookahead,
            detector,
            blurrer,
            blur_ids,
            exclude_ids,
            quality,
        )?;
    }

    Ok(())
}

fn run_preview(
    input: &Path,
    preview_dir: &Path,
    detector: Box<dyn FaceDetector>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = open_reader(input);
    let metadata = reader.open(input)?;
    let image_writer: Box<dyn ImageWriter> = Box::new(ImageFileWriter::new());

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
    Ok(())
}

fn run_image_blur(
    input: &Path,
    output: &Path,
    detector: Box<dyn FaceDetector>,
    blurrer: Box<dyn FrameBlurrer>,
    blur_ids: Option<HashSet<u32>>,
    exclude_ids: Option<HashSet<u32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader: Box<dyn VideoReader> = Box::new(ImageFileReader::new());
    let image_writer: Box<dyn ImageWriter> = Box::new(ImageFileWriter::new());

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
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_video_blur(
    input: &Path,
    output: &Path,
    lookahead: usize,
    detector: Box<dyn FaceDetector>,
    blurrer: Box<dyn FrameBlurrer>,
    blur_ids: Option<HashSet<u32>>,
    exclude_ids: Option<HashSet<u32>>,
    quality: Option<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader: Box<dyn VideoReader> = Box::new(FfmpegReader::new());
    let metadata = reader.open(input)?;
    let ffmpeg_writer = match quality {
        Some(crf) => FfmpegWriter::new().with_crf(crf),
        None => FfmpegWriter::new(),
    };
    let writer: Box<dyn VideoWriter> = Box::new(ffmpeg_writer);

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
        RegionMerger::new(),
        Box::new(ThreadedPipelineExecutor::new()),
        Some(lookahead),
        blur_ids,
        exclude_ids,
        Some(progress),
        None,
    );
    use_case.execute(&metadata, output)?;
    eprintln!();
    log::info!("Output written to {}", output.display());
    Ok(())
}

fn build_detector(cli: &Cli) -> Result<Box<dyn FaceDetector>, Box<dyn std::error::Error>> {
    log::info!("Resolving model: {YOLO_MODEL_NAME}");
    let model_path = model_resolver::resolve(
        YOLO_MODEL_NAME,
        YOLO_MODEL_URL,
        None,
        Some(Box::new(download_progress)),
    )?;
    eprintln!();

    let smoother = RegionSmoother::new(DEFAULT_ALPHA);
    let region_builder = FaceRegionBuilder::new(DEFAULT_PADDING, Some(Box::new(smoother)));
    let tracker = ByteTracker::new(TRACKER_MAX_LOST);
    let base: Box<dyn FaceDetector> = Box::new(OnnxYoloDetector::new(
        &model_path,
        region_builder,
        tracker,
        cli.confidence,
    )?);

    if cli.skip_frames > 1 {
        Ok(Box::new(SkipFrameDetector::new(base, cli.skip_frames)?))
    } else {
        Ok(base)
    }
}

fn validate(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if !cli.input.exists() {
        return Err(format!("Input file not found: {}", cli.input.display()).into());
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
        )
        .into());
    }
    if !(0.0..=1.0).contains(&cli.confidence) {
        return Err(format!(
            "Confidence must be between 0.0 and 1.0, got {}",
            cli.confidence
        )
        .into());
    }
    if let Some(q) = cli.quality {
        if q > 51 {
            return Err(format!("Quality must be between 0 and 51, got {q}").into());
        }
    }
    if cli.blur_shape != "ellipse" && cli.blur_shape != "rect" {
        return Err(format!(
            "Blur shape must be 'ellipse' or 'rect', got '{}'",
            cli.blur_shape
        )
        .into());
    }
    Ok(())
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn open_reader(input: &Path) -> Box<dyn VideoReader> {
    if is_image(input) {
        Box::new(ImageFileReader::new())
    } else {
        Box::new(FfmpegReader::new())
    }
}

fn parse_blur_shape(shape: &str) -> BlurShape {
    if shape == "rect" {
        BlurShape::Rectangular
    } else {
        BlurShape::Elliptical
    }
}

fn to_id_set(ids: Option<Vec<u32>>) -> Option<HashSet<u32>> {
    ids.map(|v| v.into_iter().collect())
}

fn download_progress(downloaded: u64, total: u64) {
    if total > 0 {
        let pct = (downloaded as f64 / total as f64 * 100.0) as u32;
        eprint!("\rDownloading face detection model... {pct}%");
    } else {
        eprint!("\rDownloading face detection model... {downloaded} bytes");
    }
}
