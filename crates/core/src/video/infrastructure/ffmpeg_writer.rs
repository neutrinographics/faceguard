use std::path::{Path, PathBuf};

use crate::shared::frame::Frame;
use crate::shared::video_metadata::VideoMetadata;
use crate::video::domain::video_writer::VideoWriter;

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
    fps: f64,
    frame_count: usize,
    video_stream_index: usize,
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
            fps: 0.0,
            frame_count: 0,
            video_stream_index: 0,
        }
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
        self.fps = metadata.fps;
        self.output_path = Some(path.to_path_buf());
        self.source_path = metadata.source_path.clone();

        let mut octx = ffmpeg_next::format::output(path)?;

        let global_header = octx
            .format()
            .flags()
            .contains(ffmpeg_next::format::Flags::GLOBAL_HEADER);

        // Use MPEG4 as a widely compatible encoder
        let codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::MPEG4)
            .ok_or("MPEG4 encoder not found")?;

        let mut ost = octx.add_stream(Some(codec))?;

        let mut encoder_ctx = ffmpeg_next::codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;

        encoder_ctx.set_width(metadata.width);
        encoder_ctx.set_height(metadata.height);
        encoder_ctx.set_format(ffmpeg_next::format::Pixel::YUV420P);

        let fps_i = metadata.fps.round() as i32;
        let fps_i = if fps_i <= 0 { 30 } else { fps_i };

        encoder_ctx.set_time_base(ffmpeg_next::Rational(1, fps_i));
        encoder_ctx.set_frame_rate(Some(ffmpeg_next::Rational(fps_i, 1)));

        if global_header {
            encoder_ctx.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
        }

        let encoder = encoder_ctx.open_with(ffmpeg_next::Dictionary::new())?;
        ost.set_parameters(&encoder);

        self.video_stream_index = 0; // first stream

        octx.write_header()?;

        // Set up RGB -> YUV scaler
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

        // Create RGB frame from input data
        let mut rgb_frame = ffmpeg_next::util::frame::video::Video::new(
            ffmpeg_next::format::Pixel::RGB24,
            self.width,
            self.height,
        );

        let stride = rgb_frame.stride(0);
        let data = rgb_frame.data_mut(0);
        let src = frame.data();

        // Copy pixel data, respecting stride
        for row in 0..self.height as usize {
            let src_start = row * self.width as usize * 3;
            let dst_start = row * stride;
            data[dst_start..dst_start + self.width as usize * 3]
                .copy_from_slice(&src[src_start..src_start + self.width as usize * 3]);
        }

        // Convert RGB -> YUV
        let mut yuv_frame = ffmpeg_next::util::frame::video::Video::empty();
        scaler.run(&rgb_frame, &mut yuv_frame)?;
        yuv_frame.set_pts(Some(self.frame_count as i64));

        let fps_i = if self.fps.round() as i32 <= 0 {
            30
        } else {
            self.fps.round() as i32
        };

        encoder.send_frame(&yuv_frame)?;

        let ost_time_base = octx.stream(self.video_stream_index).unwrap().time_base();

        let mut encoded = ffmpeg_next::Packet::empty();
        while encoder.receive_packet(&mut encoded).is_ok() {
            encoded.set_stream(self.video_stream_index);
            encoded.rescale_ts(ffmpeg_next::Rational(1, fps_i), ost_time_base);
            encoded.write_interleaved(octx)?;
        }

        self.frame_count += 1;
        Ok(())
    }

    fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut encoder) = self.encoder {
            let fps_i = if self.fps.round() as i32 <= 0 {
                30
            } else {
                self.fps.round() as i32
            };

            let octx = self.octx.as_mut().unwrap();
            let ost_time_base = octx.stream(self.video_stream_index).unwrap().time_base();

            // Flush encoder
            encoder.send_eof()?;
            let mut encoded = ffmpeg_next::Packet::empty();
            while encoder.receive_packet(&mut encoded).is_ok() {
                encoded.set_stream(self.video_stream_index);
                encoded.rescale_ts(ffmpeg_next::Rational(1, fps_i), ost_time_base);
                encoded.write_interleaved(octx)?;
            }

            octx.write_trailer()?;
        }

        // Audio muxing: if source has audio, remux it into the output
        if let (Some(source_path), Some(output_path)) =
            (self.source_path.take(), self.output_path.take())
        {
            if let Err(e) = mux_audio(&source_path, &output_path) {
                log::warn!("Audio muxing failed: {e}");
            }
        }

        self.octx = None;
        self.encoder = None;
        self.scaler = None;

        Ok(())
    }
}

/// Copies audio from `source` into `video_output` by remuxing.
///
/// Creates a temp file with both video + audio, then replaces the original
/// output. Fails silently if no audio stream exists.
fn mux_audio(source: &Path, video_output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let ictx_source = ffmpeg_next::format::input(source)?;

    // Check if source has audio
    let has_audio = ictx_source
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .is_some();

    if !has_audio {
        return Ok(());
    }

    drop(ictx_source);

    // Re-open source and video-only output for remuxing
    let mut ictx_source = ffmpeg_next::format::input(source)?;
    let mut ictx_video = ffmpeg_next::format::input(video_output)?;

    let ext = video_output
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp4");
    let temp_path = video_output.with_extension(format!("_mux.{ext}"));

    let mut octx = ffmpeg_next::format::output(&temp_path)?;

    // Map video streams from video file
    let mut video_stream_map: Vec<isize> = vec![-1; ictx_video.nb_streams() as usize];
    let mut audio_stream_map: Vec<isize> = vec![-1; ictx_source.nb_streams() as usize];
    let mut ost_index: usize = 0;

    for (idx, stream) in ictx_video.streams().enumerate() {
        if stream.parameters().medium() == ffmpeg_next::media::Type::Video {
            let mut ost =
                octx.add_stream(ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::None))?;
            ost.set_parameters(stream.parameters());
            unsafe {
                (*ost.parameters().as_mut_ptr()).codec_tag = 0;
            }
            video_stream_map[idx] = ost_index as isize;
            ost_index += 1;
        }
    }

    // Map audio streams from source
    for (idx, stream) in ictx_source.streams().enumerate() {
        if stream.parameters().medium() == ffmpeg_next::media::Type::Audio {
            let mut ost =
                octx.add_stream(ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::None))?;
            ost.set_parameters(stream.parameters());
            unsafe {
                (*ost.parameters().as_mut_ptr()).codec_tag = 0;
            }
            audio_stream_map[idx] = ost_index as isize;
            ost_index += 1;
        }
    }

    octx.write_header()?;

    // Copy video packets
    let video_time_bases: Vec<_> = ictx_video.streams().map(|s| s.time_base()).collect();

    for (stream, mut packet) in ictx_video.packets() {
        let ist_idx = stream.index();
        let ost_idx = video_stream_map[ist_idx];
        if ost_idx < 0 {
            continue;
        }
        let ost_time_base = octx.stream(ost_idx as usize).unwrap().time_base();
        packet.rescale_ts(video_time_bases[ist_idx], ost_time_base);
        packet.set_position(-1);
        packet.set_stream(ost_idx as usize);
        packet.write_interleaved(&mut octx)?;
    }

    // Copy audio packets
    let audio_time_bases: Vec<_> = ictx_source.streams().map(|s| s.time_base()).collect();

    for (stream, mut packet) in ictx_source.packets() {
        let ist_idx = stream.index();
        let ost_idx = audio_stream_map[ist_idx];
        if ost_idx < 0 {
            continue;
        }
        let ost_time_base = octx.stream(ost_idx as usize).unwrap().time_base();
        packet.rescale_ts(audio_time_bases[ist_idx], ost_time_base);
        packet.set_position(-1);
        packet.set_stream(ost_idx as usize);
        packet.write_interleaved(&mut octx)?;
    }

    octx.write_trailer()?;

    // Replace original output with muxed version
    std::fs::rename(&temp_path, video_output)?;

    Ok(())
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

        // Read back and verify
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
        // Second close should not panic
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

        // Codec is lossy, but the overall brightness should be close
        let first = &frames[0];
        let avg: f64 =
            first.data().iter().map(|&b| b as f64).sum::<f64>() / first.data().len() as f64;
        assert!(
            (avg - 128.0).abs() < 40.0,
            "Average pixel value {avg} should be close to 128"
        );
    }
}
