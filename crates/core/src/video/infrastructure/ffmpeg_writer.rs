use std::path::{Path, PathBuf};

use crate::shared::frame::Frame;
use crate::shared::video_metadata::VideoMetadata;
use crate::video::domain::video_writer::VideoWriter;

pub const DEFAULT_CRF: u32 = 18;

/// Encodes video frames via ffmpeg-next with built-in audio muxing.
///
/// When the source video has an audio stream, it is copied directly
/// to the output — no separate ffmpeg binary or temp file needed.
pub struct FfmpegWriter {
    output_path: Option<PathBuf>,
    source_path: Option<PathBuf>,
    octx: Option<ffmpeg_next::format::context::Output>,
    encoder: Option<ffmpeg_next::codec::encoder::video::Encoder>,
    scaler: Option<ffmpeg_next::software::scaling::Context>,
    width: u32,
    height: u32,
    fps: i32,
    crf: u32,
    frame_count: usize,
    video_stream_index: usize,
    audio_source_stream_idx: Option<usize>,
    audio_output_stream_idx: Option<usize>,
    audio_source_time_base: Option<ffmpeg_next::Rational>,
    pub(crate) skip_audio_passthrough: bool,
}

// Safety: FfmpegWriter is only used from a single thread at a time.
// The raw pointers inside ffmpeg types are not shared across threads.
unsafe impl Send for FfmpegWriter {}

impl FfmpegWriter {
    pub fn new() -> Self {
        Self {
            output_path: None,
            source_path: None,
            octx: None,
            encoder: None,
            scaler: None,
            width: 0,
            height: 0,
            fps: 30,
            crf: DEFAULT_CRF,
            frame_count: 0,
            video_stream_index: 0,
            audio_source_stream_idx: None,
            audio_output_stream_idx: None,
            audio_source_time_base: None,
            skip_audio_passthrough: false,
        }
    }

    pub fn with_crf(mut self, crf: u32) -> Self {
        self.crf = crf;
        self
    }

    pub fn set_skip_audio_passthrough(&mut self, skip: bool) {
        self.skip_audio_passthrough = skip;
    }
}

impl Default for FfmpegWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoWriter for FfmpegWriter {
    fn open(
        &mut self,
        path: &Path,
        metadata: &VideoMetadata,
    ) -> Result<(), Box<dyn std::error::Error>> {
        ffmpeg_next::init()?;

        self.width = metadata.width;
        self.height = metadata.height;
        self.fps = sanitize_fps(metadata.fps);
        self.output_path = Some(path.to_path_buf());
        self.source_path = metadata.source_path.clone();

        let mut octx = ffmpeg_next::format::output(path)?;

        let encoder = create_video_encoder(&mut octx, metadata, self.fps, self.crf)?;

        self.video_stream_index = 0;

        let (audio_src, audio_ost, audio_tb) = if self.skip_audio_passthrough {
            (None, None, None)
        } else {
            setup_audio_passthrough(&mut octx, metadata)?
        };
        self.audio_source_stream_idx = audio_src;
        self.audio_output_stream_idx = audio_ost;
        self.audio_source_time_base = audio_tb;

        if metadata.rotation != 0 {
            set_stream_display_matrix(&mut octx, self.video_stream_index, metadata.rotation);
        }

        octx.write_header()?;

        let scaler = ffmpeg_next::software::scaling::Context::get(
            ffmpeg_next::format::Pixel::RGB24,
            metadata.width,
            metadata.height,
            ffmpeg_next::format::Pixel::YUV420P,
            metadata.width,
            metadata.height,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        )?;

        self.octx = Some(octx);
        self.encoder = Some(encoder);
        self.scaler = Some(scaler);
        self.frame_count = 0;

        Ok(())
    }

    fn write(&mut self, frame: &Frame) -> Result<(), Box<dyn std::error::Error>> {
        let encoder = self.encoder.as_mut().ok_or("FfmpegWriter: not opened")?;
        let scaler = self.scaler.as_mut().unwrap();
        let octx = self.octx.as_mut().unwrap();

        let rgb_frame = frame_to_rgb_video(frame, self.width, self.height);

        let mut yuv_frame = ffmpeg_next::util::frame::video::Video::empty();
        scaler.run(&rgb_frame, &mut yuv_frame)?;
        yuv_frame.set_pts(Some(self.frame_count as i64));

        encoder.send_frame(&yuv_frame)?;

        flush_packets(encoder, octx, self.video_stream_index, self.fps)?;

        self.frame_count += 1;
        Ok(())
    }

    fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut encoder) = self.encoder {
            let octx = self.octx.as_mut().unwrap();

            encoder.send_eof()?;
            flush_packets(encoder, octx, self.video_stream_index, self.fps)?;

            mux_audio_from_source(
                octx,
                self.audio_source_stream_idx,
                self.audio_output_stream_idx,
                self.audio_source_time_base,
                self.source_path.as_ref(),
            );

            octx.write_trailer()?;
        }

        self.reset();
        Ok(())
    }
}

impl FfmpegWriter {
    fn reset(&mut self) {
        self.octx = None;
        self.encoder = None;
        self.scaler = None;
        self.source_path = None;
        self.output_path = None;
        self.audio_source_stream_idx = None;
        self.audio_output_stream_idx = None;
        self.audio_source_time_base = None;
    }
}

/// Clamps fps to a positive integer, defaulting to 30 for invalid values.
fn sanitize_fps(fps: f64) -> i32 {
    let rounded = fps.round() as i32;
    if rounded <= 0 {
        30
    } else {
        rounded
    }
}

fn create_video_encoder(
    octx: &mut ffmpeg_next::format::context::Output,
    metadata: &VideoMetadata,
    fps: i32,
    crf: u32,
) -> Result<ffmpeg_next::codec::encoder::video::Encoder, Box<dyn std::error::Error>> {
    let global_header = octx
        .format()
        .flags()
        .contains(ffmpeg_next::format::Flags::GLOBAL_HEADER);

    // Prefer H.264 (libx264) for CRF support; fall back to MPEG4 with qscale
    let h264 = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::H264);
    let codec = h264
        .or_else(|| ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::MPEG4))
        .ok_or("No suitable video encoder found (tried H264, MPEG4)")?;
    let is_h264 = h264.is_some();

    let mut ost = octx.add_stream(Some(codec))?;

    let mut encoder_ctx = ffmpeg_next::codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()?;

    encoder_ctx.set_width(metadata.width);
    encoder_ctx.set_height(metadata.height);
    encoder_ctx.set_format(ffmpeg_next::format::Pixel::YUV420P);
    encoder_ctx.set_time_base(ffmpeg_next::Rational(1, fps));
    encoder_ctx.set_frame_rate(Some(ffmpeg_next::Rational(fps, 1)));

    if global_header {
        encoder_ctx.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
    }

    let mut opts = ffmpeg_next::Dictionary::new();
    if is_h264 {
        opts.set("preset", "medium");
        opts.set("crf", &crf.max(1).to_string());
    } else {
        // MPEG4 uses global_quality (qscale). Map CRF 1–51 to qscale 1–31.
        // FF_QP2LAMBDA = 128
        let qscale = (crf.clamp(1, 51) as f64 * 31.0 / 51.0).round().max(1.0) as i32;
        encoder_ctx.set_global_quality(qscale * 128);
    }
    let encoder = encoder_ctx.open_with(opts)?;
    ost.set_parameters(&encoder);

    Ok(encoder)
}

type AudioPassthroughInfo = (Option<usize>, Option<usize>, Option<ffmpeg_next::Rational>);

/// Adds an audio passthrough stream if the source video has audio.
/// Returns (source_stream_idx, output_stream_idx, source_time_base).
fn setup_audio_passthrough(
    octx: &mut ffmpeg_next::format::context::Output,
    metadata: &VideoMetadata,
) -> Result<AudioPassthroughInfo, Box<dyn std::error::Error>> {
    let Some(ref source_path) = metadata.source_path else {
        return Ok((None, None, None));
    };

    let Ok(ictx_source) = ffmpeg_next::format::input(source_path) else {
        return Ok((None, None, None));
    };

    let Some(audio_stream) = ictx_source.streams().best(ffmpeg_next::media::Type::Audio) else {
        return Ok((None, None, None));
    };

    let audio_idx = audio_stream.index();
    let audio_tb = audio_stream.time_base();
    let audio_params = audio_stream.parameters();

    let mut audio_ost =
        octx.add_stream(ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::None))?;
    audio_ost.set_parameters(audio_params);
    unsafe {
        (*audio_ost.parameters().as_mut_ptr()).codec_tag = 0;
    }
    let audio_ost_idx = audio_ost.index();

    Ok((Some(audio_idx), Some(audio_ost_idx), Some(audio_tb)))
}

/// Drains all pending encoded packets from the encoder into the output.
fn flush_packets(
    encoder: &mut ffmpeg_next::codec::encoder::video::Encoder,
    octx: &mut ffmpeg_next::format::context::Output,
    stream_index: usize,
    fps: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let ost_time_base = octx.stream(stream_index).unwrap().time_base();

    let mut encoded = ffmpeg_next::Packet::empty();
    while encoder.receive_packet(&mut encoded).is_ok() {
        encoded.set_stream(stream_index);
        encoded.rescale_ts(ffmpeg_next::Rational(1, fps), ost_time_base);
        encoded.write_interleaved(octx)?;
    }
    Ok(())
}

/// Copies audio packets from the source file into the output container.
fn mux_audio_from_source(
    octx: &mut ffmpeg_next::format::context::Output,
    audio_source_stream_idx: Option<usize>,
    audio_output_stream_idx: Option<usize>,
    audio_source_time_base: Option<ffmpeg_next::Rational>,
    source_path: Option<&PathBuf>,
) {
    let (Some(audio_src_idx), Some(audio_ost_idx), Some(audio_src_tb), Some(source_path)) = (
        audio_source_stream_idx,
        audio_output_stream_idx,
        audio_source_time_base,
        source_path,
    ) else {
        return;
    };

    let mut ictx = match ffmpeg_next::format::input(source_path) {
        Ok(ctx) => ctx,
        Err(e) => {
            log::warn!("Audio muxing failed: could not reopen source: {e}");
            return;
        }
    };

    let ost_audio_tb = octx.stream(audio_ost_idx).unwrap().time_base();
    for (stream, mut packet) in ictx.packets() {
        if stream.index() != audio_src_idx {
            continue;
        }
        packet.rescale_ts(audio_src_tb, ost_audio_tb);
        packet.set_position(-1);
        packet.set_stream(audio_ost_idx);
        if let Err(e) = packet.write_interleaved(octx) {
            log::warn!("Failed to write audio packet: {e}");
            break;
        }
    }
}

/// Sets a display matrix on an output stream to encode the given rotation angle.
///
/// Uses the raw FFmpeg C API (`av_stream_new_side_data`) because the
/// ffmpeg-next bindings don't expose stream-level side data writes.
/// The display matrix is a 3×3 transformation stored as 9 × i32 values:
/// the first 6 in 16.16 fixed-point, the last 3 in 2.30 fixed-point.
fn set_stream_display_matrix(
    octx: &mut ffmpeg_next::format::context::Output,
    stream_index: usize,
    rotation_degrees: i32,
) {
    use ffmpeg_next::sys::{av_stream_new_side_data, AVPacketSideDataType};

    let angle_rad = -(rotation_degrees as f64).to_radians();
    let cos_val = (angle_rad.cos() * 65536.0).round() as i32;
    let sin_val = (angle_rad.sin() * 65536.0).round() as i32;

    // Display matrix layout (row-major):
    //   [cos, sin, 0]
    //   [-sin, cos, 0]
    //   [0,    0,   1]  (2.30 fixed point for last row)
    let matrix: [i32; 9] = [cos_val, sin_val, 0, -sin_val, cos_val, 0, 0, 0, 0x40000000];

    unsafe {
        let stream_ptr = (*octx.as_mut_ptr()).streams.add(stream_index).read();
        let data_ptr = av_stream_new_side_data(
            stream_ptr,
            AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
            36,
        );
        if !data_ptr.is_null() {
            for (i, &val) in matrix.iter().enumerate() {
                std::ptr::copy_nonoverlapping(val.to_ne_bytes().as_ptr(), data_ptr.add(i * 4), 4);
            }
        }
    }
}

/// Converts a [`Frame`] into an ffmpeg RGB24 video frame, respecting stride.
fn frame_to_rgb_video(
    frame: &Frame,
    width: u32,
    height: u32,
) -> ffmpeg_next::util::frame::video::Video {
    let mut rgb_frame = ffmpeg_next::util::frame::video::Video::new(
        ffmpeg_next::format::Pixel::RGB24,
        width,
        height,
    );

    let stride = rgb_frame.stride(0);
    let dst = rgb_frame.data_mut(0);
    let src = frame.data();
    let row_bytes = width as usize * 3;

    for row in 0..height as usize {
        let src_start = row * row_bytes;
        let dst_start = row * stride;
        dst[dst_start..dst_start + row_bytes]
            .copy_from_slice(&src[src_start..src_start + row_bytes]);
    }

    rgb_frame
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::domain::video_reader::VideoReader;

    fn metadata(w: u32, h: u32, fps: f64) -> VideoMetadata {
        VideoMetadata {
            width: w,
            height: h,
            fps,
            total_frames: 0,
            codec: String::new(),
            source_path: None,
            rotation: 0,
        }
    }

    fn solid_frame(index: usize, w: u32, h: u32, value: u8) -> Frame {
        let data = vec![value; (w * h * 3) as usize];
        Frame::new(data, w, h, 3, index)
    }

    #[test]
    fn test_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.mp4");
        let meta = metadata(160, 120, 30.0);

        let mut writer = FfmpegWriter::new();
        writer.open(&path, &meta).unwrap();
        for i in 0..3 {
            writer.write(&solid_frame(i, 160, 120, 128)).unwrap();
        }
        writer.close().unwrap();

        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
    }

    #[test]
    fn test_written_video_has_correct_resolution() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.mp4");
        let meta = metadata(160, 120, 30.0);

        let mut writer = FfmpegWriter::new();
        writer.open(&path, &meta).unwrap();
        writer.write(&solid_frame(0, 160, 120, 128)).unwrap();
        writer.close().unwrap();

        ffmpeg_next::init().unwrap();
        let ictx = ffmpeg_next::format::input(&path).unwrap();
        let stream = ictx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .unwrap();
        let codec_ctx =
            ffmpeg_next::codec::context::Context::from_parameters(stream.parameters()).unwrap();
        let decoder = codec_ctx.decoder().video().unwrap();
        assert_eq!(decoder.width(), 160);
        assert_eq!(decoder.height(), 120);
    }

    #[test]
    fn test_write_without_open_returns_error() {
        let mut writer = FfmpegWriter::new();
        let result = writer.write(&solid_frame(0, 160, 120, 128));
        assert!(result.is_err());
    }

    #[test]
    fn test_close_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.mp4");
        let meta = metadata(160, 120, 30.0);

        let mut writer = FfmpegWriter::new();
        writer.open(&path, &meta).unwrap();
        writer.write(&solid_frame(0, 160, 120, 128)).unwrap();
        writer.close().unwrap();
        let _ = writer.close();
    }

    #[test]
    fn test_roundtrip_preserves_frames() {
        use crate::video::infrastructure::ffmpeg_reader::FfmpegReader;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.mp4");
        let meta = metadata(160, 120, 30.0);

        let mut writer = FfmpegWriter::new();
        writer.open(&path, &meta).unwrap();
        for i in 0..3 {
            writer.write(&solid_frame(i, 160, 120, 128)).unwrap();
        }
        writer.close().unwrap();

        let mut reader = FfmpegReader::new();
        let read_meta = reader.open(&path).unwrap();
        assert_eq!(read_meta.width, 160);
        assert_eq!(read_meta.height, 120);

        let frames: Vec<_> = reader.frames().map(|f| f.unwrap()).collect();
        assert_eq!(frames.len(), 3);

        let first = &frames[0];
        let avg: f64 =
            first.data().iter().map(|&b| b as f64).sum::<f64>() / first.data().len() as f64;
        assert!(
            (avg - 128.0).abs() < 40.0,
            "Average pixel value {avg} should be close to 128"
        );
    }

    #[test]
    fn test_skip_audio_passthrough_setter() {
        let mut writer = FfmpegWriter::new();
        assert!(!writer.skip_audio_passthrough);
        writer.set_skip_audio_passthrough(true);
        assert!(writer.skip_audio_passthrough);
    }
}
