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

/// Default bounded channel capacity for inter-thread communication.
const DEFAULT_CHANNEL_CAPACITY: usize = 8;

/// Executes the blur pipeline using dedicated threads for I/O, detection, and blurring.
///
/// Pipeline layout:
/// ```text
/// reader_thread → detect_thread → main [buffer/merge/blur] → writer_thread
/// ```
///
/// Detection and blur run on separate threads so they overlap, improving
/// throughput when detection is the bottleneck.
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
        mut reader: Box<dyn VideoReader>,
        mut writer: Box<dyn VideoWriter>,
        mut detector: Box<dyn FaceDetector>,
        blurrer: Box<dyn FrameBlurrer>,
        merger: RegionMerger,
        metadata: &VideoMetadata,
        output_path: &Path,
        config: PipelineConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let frame_w = metadata.width;
        let frame_h = metadata.height;
        let total_frames = metadata.total_frames;

        writer.open(output_path, metadata)?;

        let cap = self.channel_capacity;

        // --- Reader thread ---
        let (frame_tx, frame_rx) = crossbeam_channel::bounded::<
            Result<Frame, Box<dyn std::error::Error + Send + Sync>>,
        >(cap);

        let cancelled_reader = config.cancelled.clone();

        let reader_handle = std::thread::spawn(move || {
            let frame_iter = reader.frames();
            for frame_result in frame_iter {
                if cancelled_reader.load(Ordering::Relaxed) {
                    break;
                }
                let mapped =
                    frame_result.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        e.to_string().into()
                    });
                if frame_tx.send(mapped).is_err() {
                    break;
                }
            }
            reader.close();
            reader
        });

        // --- Detect thread ---
        type DetectedItem = Result<(Frame, Vec<Region>), Box<dyn std::error::Error + Send + Sync>>;
        let (detected_tx, detected_rx) = crossbeam_channel::bounded::<DetectedItem>(cap);

        let cancelled_detect = config.cancelled.clone();
        let blur_ids = config.blur_ids.clone();
        let exclude_ids = config.exclude_ids.clone();

        let detect_handle = std::thread::spawn(move || {
            for frame_result in frame_rx {
                if cancelled_detect.load(Ordering::Relaxed) {
                    break;
                }

                let result = match frame_result {
                    Ok(frame) => match detector.detect(&frame) {
                        Ok(regions) => {
                            let filtered = Region::filter(
                                &regions,
                                blur_ids.as_ref(),
                                exclude_ids.as_ref(),
                            );
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
        });

        // --- Writer thread ---
        let (write_tx, write_rx) = crossbeam_channel::bounded::<Frame>(cap);

        let writer_handle = std::thread::spawn(
            move || -> Result<Box<dyn VideoWriter>, Box<dyn std::error::Error + Send + Sync>> {
                for frame in write_rx {
                    writer.write(&frame).map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() },
                    )?;
                }
                Ok(writer)
            },
        );

        // --- Main thread: buffer → merge → blur ---
        let mut buffer: VecDeque<(Frame, Vec<Region>)> = VecDeque::new();
        let mut frames_processed: usize = 0;
        let mut main_error: Option<Box<dyn std::error::Error>> = None;

        for detected_result in detected_rx {
            if config.cancelled.load(Ordering::Relaxed) {
                break;
            }

            let (frame, filtered) = match detected_result {
                Ok(pair) => pair,
                Err(e) => {
                    main_error = Some(e.to_string().into());
                    break;
                }
            };

            buffer.push_back((frame, filtered));

            if buffer.len() > config.lookahead {
                if let Err(e) = flush_oldest(
                    &mut buffer,
                    &merger,
                    &*blurrer,
                    frame_w,
                    frame_h,
                    &write_tx,
                    &mut frames_processed,
                    total_frames,
                    &config,
                ) {
                    main_error = Some(e);
                    break;
                }
            }
        }

        // Flush remaining buffered frames
        if main_error.is_none() {
            while !buffer.is_empty() {
                if config.cancelled.load(Ordering::Relaxed) {
                    break;
                }
                if let Err(e) = flush_oldest(
                    &mut buffer,
                    &merger,
                    &*blurrer,
                    frame_w,
                    frame_h,
                    &write_tx,
                    &mut frames_processed,
                    total_frames,
                    &config,
                ) {
                    main_error = Some(e);
                    break;
                }
            }
        }

        // Signal writer thread to finish
        drop(write_tx);

        // Join reader thread
        match reader_handle.join() {
            Ok(mut r) => r.close(),
            Err(_) => {
                if main_error.is_none() {
                    main_error = Some("Reader thread panicked".into());
                }
            }
        }

        // Join detect thread
        if detect_handle.join().is_err() && main_error.is_none() {
            main_error = Some("Detect thread panicked".into());
        }

        // Join writer thread, close it
        match writer_handle.join() {
            Ok(Ok(mut w)) => {
                if let Err(e) = w.close() {
                    if main_error.is_none() {
                        main_error = Some(e);
                    }
                }
            }
            Ok(Err(e)) => {
                if main_error.is_none() {
                    main_error = Some(e.to_string().into());
                }
            }
            Err(_) => {
                if main_error.is_none() {
                    main_error = Some("Writer thread panicked".into());
                }
            }
        }

        if let Some(e) = main_error {
            return Err(e);
        }

        Ok(())
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

    let lookahead_regions: Vec<&[Region]> =
        buffer.iter().map(|(_, regions)| regions.as_slice()).collect();

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
