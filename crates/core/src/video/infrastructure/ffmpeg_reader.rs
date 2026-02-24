use std::path::Path;

use crate::shared::frame::Frame;
use crate::shared::video_metadata::VideoMetadata;
use crate::video::domain::video_reader::VideoReader;

/// Decodes video frames via ffmpeg-next (libavformat + libavcodec).
///
/// Converts each decoded frame to RGB24 and wraps it in a [`Frame`].
pub struct FfmpegReader {
    input_ctx: Option<ffmpeg_next::format::context::Input>,
    video_stream_index: usize,
    metadata: Option<VideoMetadata>,
}

// Safety: FfmpegReader is only used from a single thread at a time.
// The raw pointers inside ffmpeg types are not shared across threads.
unsafe impl Send for FfmpegReader {}

impl FfmpegReader {
    pub fn new() -> Self {
        Self {
            input_ctx: None,
            video_stream_index: 0,
            metadata: None,
        }
    }
}

impl Default for FfmpegReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Lazy iterator that decodes video frames one at a time, avoiding the need
/// to buffer the entire video in memory before processing begins.
struct FfmpegFrameIter<'a> {
    ictx: &'a mut ffmpeg_next::format::context::Input,
    decoder: ffmpeg_next::decoder::Video,
    scaler: ffmpeg_next::software::scaling::Context,
    width: u32,
    height: u32,
    video_stream_index: usize,
    frame_index: usize,
    flushing: bool,
    done: bool,
}

impl<'a> FfmpegFrameIter<'a> {
    /// Try to receive a decoded frame from the decoder, scale it to RGB24,
    /// and return it as a `Frame`.
    fn try_receive(&mut self) -> Option<Result<Frame, Box<dyn std::error::Error>>> {
        let mut decoded = ffmpeg_next::util::frame::video::Video::empty();
        if self.decoder.receive_frame(&mut decoded).is_ok() {
            let mut rgb_frame = ffmpeg_next::util::frame::video::Video::empty();
            if let Err(e) = self.scaler.run(&decoded, &mut rgb_frame) {
                return Some(Err(Box::new(e)));
            }

            let stride = rgb_frame.stride(0);
            let data = rgb_frame.data(0);
            let width = self.width as usize;
            let height = self.height as usize;

            let mut pixels = Vec::with_capacity(width * height * 3);
            for row in 0..height {
                let row_start = row * stride;
                let row_end = row_start + width * 3;
                pixels.extend_from_slice(&data[row_start..row_end]);
            }

            let frame = Frame::new(pixels, self.width, self.height, 3, self.frame_index);
            self.frame_index += 1;
            Some(Ok(frame))
        } else {
            None
        }
    }
}

impl<'a> Iterator for FfmpegFrameIter<'a> {
    type Item = Result<Frame, Box<dyn std::error::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // First, try to receive any buffered decoded frames
        if let Some(result) = self.try_receive() {
            return Some(result);
        }

        if self.flushing {
            // Already sent EOF, no more frames
            self.done = true;
            return None;
        }

        // Feed packets until we get a decoded frame
        loop {
            // Read next packet from the demuxer
            let Some((stream, packet)) = self.ictx.packets().next() else {
                // No more packets — flush the decoder
                let _ = self.decoder.send_eof();
                self.flushing = true;
                // Drain any remaining buffered frames
                if let Some(result) = self.try_receive() {
                    return Some(result);
                }
                self.done = true;
                return None;
            };

            if stream.index() != self.video_stream_index {
                continue;
            }

            if self.decoder.send_packet(&packet).is_err() {
                continue;
            }

            if let Some(result) = self.try_receive() {
                return Some(result);
            }
        }
    }
}

impl VideoReader for FfmpegReader {
    fn open(&mut self, path: &Path) -> Result<VideoMetadata, Box<dyn std::error::Error>> {
        ffmpeg_next::init()?;

        let ictx = ffmpeg_next::format::input(path)?;

        let stream = ictx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or("No video stream found")?;

        let video_stream_index = stream.index();

        // Extract metadata from stream + codec
        let codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())?;
        let decoder = codec_ctx.decoder().video()?;

        let width = decoder.width();
        let height = decoder.height();

        let rate = stream.rate();
        let fps = if rate.denominator() != 0 {
            rate.numerator() as f64 / rate.denominator() as f64
        } else {
            0.0
        };

        let total_frames = stream.frames() as usize;

        let codec_name = decoder
            .codec()
            .map(|c| c.name().to_string())
            .unwrap_or_default();

        let metadata = VideoMetadata {
            width,
            height,
            fps,
            total_frames,
            codec: codec_name,
            source_path: Some(path.to_path_buf()),
        };

        self.video_stream_index = video_stream_index;
        self.metadata = Some(metadata.clone());
        self.input_ctx = Some(ictx);

        Ok(metadata)
    }

    fn frames(
        &mut self,
    ) -> Box<dyn Iterator<Item = Result<Frame, Box<dyn std::error::Error>>> + '_> {
        let Some(ictx) = self.input_ctx.as_mut() else {
            return Box::new(std::iter::once(Err("FfmpegReader: not opened".into())));
        };

        let stream = ictx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .unwrap();
        let codec_ctx =
            ffmpeg_next::codec::context::Context::from_parameters(stream.parameters()).unwrap();
        let decoder = codec_ctx.decoder().video().unwrap();

        let width = decoder.width();
        let height = decoder.height();

        let scaler = ffmpeg_next::software::scaling::Context::get(
            decoder.format(),
            width,
            height,
            ffmpeg_next::format::Pixel::RGB24,
            width,
            height,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        )
        .unwrap();

        let video_stream_index = self.video_stream_index;

        Box::new(FfmpegFrameIter {
            ictx,
            decoder,
            scaler,
            width,
            height,
            video_stream_index,
            frame_index: 0,
            flushing: false,
            done: false,
        })
    }

    fn close(&mut self) {
        self.input_ctx = None;
        self.metadata = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Creates a minimal test video using ffmpeg-next.
    fn create_test_video(path: &Path, num_frames: usize, width: u32, height: u32, fps: f64) {
        ffmpeg_next::init().unwrap();

        let mut octx = ffmpeg_next::format::output(path).unwrap();

        let global_header = octx
            .format()
            .flags()
            .contains(ffmpeg_next::format::Flags::GLOBAL_HEADER);

        let codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::MPEG4).unwrap();
        let mut ost = octx.add_stream(Some(codec)).unwrap();

        let mut encoder_ctx = ffmpeg_next::codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .unwrap();

        encoder_ctx.set_width(width);
        encoder_ctx.set_height(height);
        encoder_ctx.set_format(ffmpeg_next::format::Pixel::YUV420P);
        encoder_ctx.set_time_base(ffmpeg_next::Rational(1, fps as i32));
        encoder_ctx.set_frame_rate(Some(ffmpeg_next::Rational(fps as i32, 1)));

        if global_header {
            encoder_ctx.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
        }

        let mut encoder = encoder_ctx
            .open_with(ffmpeg_next::Dictionary::new())
            .unwrap();
        ost.set_parameters(&encoder);

        octx.write_header().unwrap();

        let ost_time_base = octx.stream(0).unwrap().time_base();

        let mut scaler = ffmpeg_next::software::scaling::Context::get(
            ffmpeg_next::format::Pixel::RGB24,
            width,
            height,
            ffmpeg_next::format::Pixel::YUV420P,
            width,
            height,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        )
        .unwrap();

        for i in 0..num_frames {
            let mut rgb_frame = ffmpeg_next::util::frame::video::Video::new(
                ffmpeg_next::format::Pixel::RGB24,
                width,
                height,
            );
            let stride = rgb_frame.stride(0);
            let data = rgb_frame.data_mut(0);
            // Fill with a value that changes per frame
            let value = ((i * 40) % 256) as u8;
            for row in 0..height as usize {
                for col in 0..width as usize {
                    let offset = row * stride + col * 3;
                    data[offset] = value;
                    data[offset + 1] = value;
                    data[offset + 2] = value;
                }
            }

            let mut yuv_frame = ffmpeg_next::util::frame::video::Video::empty();
            scaler.run(&rgb_frame, &mut yuv_frame).unwrap();
            yuv_frame.set_pts(Some(i as i64));

            encoder.send_frame(&yuv_frame).unwrap();

            let mut encoded = ffmpeg_next::Packet::empty();
            while encoder.receive_packet(&mut encoded).is_ok() {
                encoded.set_stream(0);
                encoded.rescale_ts(ffmpeg_next::Rational(1, fps as i32), ost_time_base);
                encoded.write_interleaved(&mut octx).unwrap();
            }
        }

        // Flush encoder
        encoder.send_eof().unwrap();
        let mut encoded = ffmpeg_next::Packet::empty();
        while encoder.receive_packet(&mut encoded).is_ok() {
            encoded.set_stream(0);
            encoded.rescale_ts(ffmpeg_next::Rational(1, fps as i32), ost_time_base);
            encoded.write_interleaved(&mut octx).unwrap();
        }

        octx.write_trailer().unwrap();
    }

    fn test_video_path(dir: &Path) -> PathBuf {
        dir.join("test.mp4")
    }

    #[test]
    fn test_open_returns_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = test_video_path(dir.path());
        create_test_video(&path, 5, 160, 120, 30.0);

        let mut reader = FfmpegReader::new();
        let meta = reader.open(&path).unwrap();
        assert_eq!(meta.width, 160);
        assert_eq!(meta.height, 120);
        assert!(meta.fps > 0.0);
        assert_eq!(meta.source_path, Some(path));
    }

    #[test]
    fn test_open_nonexistent_raises() {
        let mut reader = FfmpegReader::new();
        assert!(reader.open(Path::new("/nonexistent/test.mp4")).is_err());
    }

    #[test]
    fn test_frames_yields_correct_count() {
        let dir = tempfile::tempdir().unwrap();
        let path = test_video_path(dir.path());
        create_test_video(&path, 5, 160, 120, 30.0);

        let mut reader = FfmpegReader::new();
        reader.open(&path).unwrap();

        let frames: Vec<_> = reader.frames().collect();
        assert_eq!(frames.len(), 5);
        for f in &frames {
            assert!(f.is_ok());
        }
    }

    #[test]
    fn test_frames_have_sequential_indices() {
        let dir = tempfile::tempdir().unwrap();
        let path = test_video_path(dir.path());
        create_test_video(&path, 5, 160, 120, 30.0);

        let mut reader = FfmpegReader::new();
        reader.open(&path).unwrap();

        let frames: Vec<_> = reader.frames().map(|f| f.unwrap()).collect();
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.index(), i);
        }
    }

    #[test]
    fn test_frames_are_3_channel() {
        let dir = tempfile::tempdir().unwrap();
        let path = test_video_path(dir.path());
        create_test_video(&path, 5, 160, 120, 30.0);

        let mut reader = FfmpegReader::new();
        reader.open(&path).unwrap();

        let frame = reader.frames().next().unwrap().unwrap();
        assert_eq!(frame.channels(), 3);
        assert_eq!(frame.data().len(), (160 * 120 * 3) as usize);
    }

    #[test]
    fn test_frames_without_open_returns_error() {
        let mut reader = FfmpegReader::new();
        let result = reader.frames().next().unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn test_close_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = test_video_path(dir.path());
        create_test_video(&path, 1, 160, 120, 30.0);

        let mut reader = FfmpegReader::new();
        reader.open(&path).unwrap();
        reader.close();
        reader.close(); // should not panic
    }
}
