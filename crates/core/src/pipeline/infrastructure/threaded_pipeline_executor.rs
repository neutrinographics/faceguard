use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::Ordering;

use crate::blurring::domain::frame_blurrer::FrameBlurrer;
use crate::detection::domain::face_detector::FaceDetector;
use crate::detection::domain::region_merger::RegionMerger;
use crate::pipeline::pipeline_executor::{PipelineConfig, PipelineExecutor};
use crate::shared::frame::Frame;
use crate::shared::region::Region;
use crate::shared::video_metadata::VideoMetadata;
use crate::video::domain::video_reader::VideoReader;
use crate::video::domain::video_writer::VideoWriter;

const DEFAULT_CHANNEL_CAPACITY: usize = 8;

type SendError = Box<dyn std::error::Error + Send + Sync>;

/// Executes the blur pipeline with dedicated threads for I/O and detection.
///
/// Layout: `reader → detect → main [buffer/merge/blur] → writer`
///
/// Detection and I/O run concurrently so they overlap, improving throughput
/// when detection is the bottleneck.
pub struct ThreadedPipelineExecutor {
    channel_capacity: usize,
}

impl ThreadedPipelineExecutor {
    pub fn new() -> Self {
        Self {
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
        }
    }
}

impl Default for ThreadedPipelineExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineExecutor for ThreadedPipelineExecutor {
    fn execute(
        &self,
        reader: Box<dyn VideoReader>,
        mut writer: Box<dyn VideoWriter>,
        detector: Box<dyn FaceDetector>,
        blurrer: Box<dyn FrameBlurrer>,
        merger: RegionMerger,
        metadata: &VideoMetadata,
        output_path: &Path,
        config: PipelineConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let frame_w = metadata.width;
        let frame_h = metadata.height;
        let total_frames = metadata.total_frames;
        let cap = self.channel_capacity;

        writer.open(output_path, metadata)?;

        let (frame_tx, frame_rx) =
            crossbeam_channel::bounded::<Result<Frame, SendError>>(cap);
        let (detected_tx, detected_rx) =
            crossbeam_channel::bounded::<Result<(Frame, Vec<Region>), SendError>>(cap);
        let (write_tx, write_rx) = crossbeam_channel::bounded::<Frame>(cap);

        let reader_handle = spawn_reader(reader, frame_tx, config.cancelled.clone());
        let detect_handle = spawn_detector(
            detector,
            frame_rx,
            detected_tx,
            config.cancelled.clone(),
            config.blur_ids.clone(),
            config.exclude_ids.clone(),
        );
        let writer_handle = spawn_writer(writer, write_rx);

        let main_error = run_main_loop(
            detected_rx,
            &write_tx,
            &merger,
            &*blurrer,
            frame_w,
            frame_h,
            total_frames,
            &config,
        );

        drop(write_tx);

        join_threads(reader_handle, detect_handle, writer_handle, main_error)
    }
}

fn spawn_reader(
    mut reader: Box<dyn VideoReader>,
    frame_tx: crossbeam_channel::Sender<Result<Frame, SendError>>,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> std::thread::JoinHandle<Box<dyn VideoReader>> {
    std::thread::spawn(move || {
        for frame_result in reader.frames() {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }
            let mapped = frame_result.map_err(|e| -> SendError { e.to_string().into() });
            if frame_tx.send(mapped).is_err() {
                break;
            }
        }
        reader.close();
        reader
    })
}

fn spawn_detector(
    mut detector: Box<dyn FaceDetector>,
    frame_rx: crossbeam_channel::Receiver<Result<Frame, SendError>>,
    detected_tx: crossbeam_channel::Sender<Result<(Frame, Vec<Region>), SendError>>,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    blur_ids: Option<std::collections::HashSet<u32>>,
    exclude_ids: Option<std::collections::HashSet<u32>>,
) -> std::thread::JoinHandle<Box<dyn FaceDetector>> {
    std::thread::spawn(move || {
        for frame_result in frame_rx {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }

            let result = match frame_result {
                Ok(frame) => match detector.detect(&frame) {
                    Ok(regions) => {
                        let filtered =
                            Region::filter(&regions, blur_ids.as_ref(), exclude_ids.as_ref());
                        Ok((frame, filtered))
                    }
                    Err(e) => Err(e.to_string().into()),
                },
                Err(e) => Err(e),
            };

            if detected_tx.send(result).is_err() {
                break;
            }
        }
        detector
    })
}

fn spawn_writer(
    mut writer: Box<dyn VideoWriter>,
    write_rx: crossbeam_channel::Receiver<Frame>,
) -> std::thread::JoinHandle<Result<Box<dyn VideoWriter>, SendError>> {
    std::thread::spawn(move || {
        for frame in write_rx {
            writer
                .write(&frame)
                .map_err(|e| -> SendError { e.to_string().into() })?;
        }
        Ok(writer)
    })
}

/// Runs the main thread loop: receive detected frames, buffer for lookahead,
/// merge regions, blur, and send to writer.
#[allow(clippy::too_many_arguments)]
fn run_main_loop(
    detected_rx: crossbeam_channel::Receiver<Result<(Frame, Vec<Region>), SendError>>,
    write_tx: &crossbeam_channel::Sender<Frame>,
    merger: &RegionMerger,
    blurrer: &dyn FrameBlurrer,
    frame_w: u32,
    frame_h: u32,
    total_frames: usize,
    config: &PipelineConfig,
) -> Option<Box<dyn std::error::Error>> {
    let mut buffer: VecDeque<(Frame, Vec<Region>)> = VecDeque::new();
    let mut frames_processed: usize = 0;

    for detected_result in detected_rx {
        if config.cancelled.load(Ordering::Relaxed) {
            break;
        }

        let (frame, filtered) = match detected_result {
            Ok(pair) => pair,
            Err(e) => return Some(e.to_string().into()),
        };

        buffer.push_back((frame, filtered));

        if buffer.len() > config.lookahead {
            if let Err(e) = flush_oldest(
                &mut buffer,
                merger,
                blurrer,
                frame_w,
                frame_h,
                write_tx,
                &mut frames_processed,
                total_frames,
                config,
            ) {
                return Some(e);
            }
        }
    }

    while !buffer.is_empty() {
        if config.cancelled.load(Ordering::Relaxed) {
            break;
        }
        if let Err(e) = flush_oldest(
            &mut buffer,
            merger,
            blurrer,
            frame_w,
            frame_h,
            write_tx,
            &mut frames_processed,
            total_frames,
            config,
        ) {
            return Some(e);
        }
    }

    None
}

/// Joins all pipeline threads and coalesces the first error encountered.
fn join_threads(
    reader_handle: std::thread::JoinHandle<Box<dyn VideoReader>>,
    detect_handle: std::thread::JoinHandle<Box<dyn FaceDetector>>,
    writer_handle: std::thread::JoinHandle<Result<Box<dyn VideoWriter>, SendError>>,
    mut first_error: Option<Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    fn set_if_none(slot: &mut Option<Box<dyn std::error::Error>>, err: Box<dyn std::error::Error>) {
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    match reader_handle.join() {
        Ok(mut r) => r.close(),
        Err(_) => set_if_none(&mut first_error, "Reader thread panicked".into()),
    }

    if detect_handle.join().is_err() {
        set_if_none(&mut first_error, "Detect thread panicked".into());
    }

    match writer_handle.join() {
        Ok(Ok(mut w)) => {
            if let Err(e) = w.close() {
                set_if_none(&mut first_error, e);
            }
        }
        Ok(Err(e)) => set_if_none(&mut first_error, e.to_string().into()),
        Err(_) => set_if_none(&mut first_error, "Writer thread panicked".into()),
    }

    match first_error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
fn flush_oldest(
    buffer: &mut VecDeque<(Frame, Vec<Region>)>,
    merger: &RegionMerger,
    blurrer: &dyn FrameBlurrer,
    frame_w: u32,
    frame_h: u32,
    write_tx: &crossbeam_channel::Sender<Frame>,
    frames_processed: &mut usize,
    total_frames: usize,
    config: &PipelineConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mut frame, own_regions) = buffer.pop_front().unwrap();

    let lookahead_regions: Vec<&[Region]> = buffer
        .iter()
        .map(|(_, regions)| regions.as_slice())
        .collect();

    let merged = merger.merge(&own_regions, &lookahead_regions, frame_w, frame_h);
    blurrer.blur(&mut frame, &merged)?;

    write_tx
        .send(frame)
        .map_err(|_| "Writer channel closed unexpectedly")?;

    *frames_processed += 1;

    if let Some(ref callback) = config.on_progress {
        if !callback(*frames_processed, total_frames) {
            return Err("Cancelled".into());
        }
    }

    Ok(())
}
