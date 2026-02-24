use std::path::Path;

use crate::shared::frame::Frame;
use crate::shared::video_metadata::VideoMetadata;
use crate::video::domain::video_reader::VideoReader;

/// Adapts a single image file to the [`VideoReader`] interface.
///
/// Treats the image as a one-frame video with `fps=0` and `total_frames=1`,
/// allowing the pipeline to process images and videos uniformly.
///
/// Uses ffmpeg for decoding, which is significantly faster than the pure-Rust
/// `image` crate for large images (e.g. 4032x3024 JPEG).
pub struct ImageFileReader {
    frame: Option<Frame>,
    metadata: Option<VideoMetadata>,
}

// Safety: ImageFileReader is only used from a single thread at a time.
// The raw pointers inside ffmpeg types are not shared across threads.
unsafe impl Send for ImageFileReader {}

impl ImageFileReader {
    pub fn new() -> Self {
        Self {
            frame: None,
            metadata: None,
        }
    }
}

impl Default for ImageFileReader {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoReader for ImageFileReader {
    fn open(&mut self, path: &Path) -> Result<VideoMetadata, Box<dyn std::error::Error>> {
        ffmpeg_next::init()?;

        let mut ictx = ffmpeg_next::format::input(path)?;

        let stream = ictx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or("No image data found")?;

        let codec_ctx =
            ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())?;
        let mut decoder = codec_ctx.decoder().video()?;

        let width = decoder.width();
        let height = decoder.height();

        // Decode the single frame
        let mut scaler = ffmpeg_next::software::scaling::Context::get(
            decoder.format(),
            width,
            height,
            ffmpeg_next::format::Pixel::RGB24,
            width,
            height,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        )?;

        let video_stream_index = stream.index();
        let mut decoded_frame = None;

        // Feed packets to decoder
        for (stream, packet) in ictx.packets() {
            if stream.index() != video_stream_index {
                continue;
            }
            decoder.send_packet(&packet)?;
            let mut video_frame = ffmpeg_next::util::frame::video::Video::empty();
            if decoder.receive_frame(&mut video_frame).is_ok() {
                let mut rgb_frame = ffmpeg_next::util::frame::video::Video::empty();
                scaler.run(&video_frame, &mut rgb_frame)?;

                let stride = rgb_frame.stride(0);
                let data = rgb_frame.data(0);

                let mut pixels = Vec::with_capacity((width * height * 3) as usize);
                for row in 0..height as usize {
                    let row_start = row * stride;
                    let row_end = row_start + (width as usize * 3);
                    pixels.extend_from_slice(&data[row_start..row_end]);
                }

                decoded_frame = Some(Frame::new(pixels, width, height, 3, 0));
                break;
            }
        }

        // Flush decoder if we haven't got a frame yet
        if decoded_frame.is_none() {
            let _ = decoder.send_eof();
            let mut video_frame = ffmpeg_next::util::frame::video::Video::empty();
            if decoder.receive_frame(&mut video_frame).is_ok() {
                let mut rgb_frame = ffmpeg_next::util::frame::video::Video::empty();
                scaler.run(&video_frame, &mut rgb_frame)?;

                let stride = rgb_frame.stride(0);
                let data = rgb_frame.data(0);

                let mut pixels = Vec::with_capacity((width * height * 3) as usize);
                for row in 0..height as usize {
                    let row_start = row * stride;
                    let row_end = row_start + (width as usize * 3);
                    pixels.extend_from_slice(&data[row_start..row_end]);
                }

                decoded_frame = Some(Frame::new(pixels, width, height, 3, 0));
            }
        }

        let frame = decoded_frame.ok_or("Failed to decode image")?;
        self.frame = Some(frame);

        let metadata = VideoMetadata {
            width,
            height,
            fps: 0.0,
            total_frames: 1,
            codec: String::new(),
            source_path: Some(path.to_path_buf()),
        };
        self.metadata = Some(metadata.clone());
        Ok(metadata)
    }

    fn frames(
        &mut self,
    ) -> Box<dyn Iterator<Item = Result<Frame, Box<dyn std::error::Error>>> + '_> {
        if self.frame.is_none() {
            return Box::new(std::iter::once(Err("ImageFileReader: not opened".into())));
        }
        Box::new(self.frame.take().into_iter().map(Ok))
    }

    fn close(&mut self) {
        self.frame = None;
        self.metadata = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write_test_image(dir: &Path, width: u32, height: u32) -> PathBuf {
        let path = dir.join("test.png");
        let mut img = image::RgbImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([50, 100, 200]);
        }
        img.save(&path).unwrap();
        path
    }

    #[test]
    fn test_open_returns_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_image(dir.path(), 100, 80);
        let mut reader = ImageFileReader::new();
        let meta = reader.open(&path).unwrap();
        assert_eq!(meta.width, 100);
        assert_eq!(meta.height, 80);
        assert_eq!(meta.fps, 0.0);
        assert_eq!(meta.total_frames, 1);
        assert_eq!(meta.codec, "");
        assert_eq!(meta.source_path, Some(path));
    }

    #[test]
    fn test_open_nonexistent_raises() {
        let mut reader = ImageFileReader::new();
        assert!(reader.open(Path::new("/nonexistent/test.png")).is_err());
    }

    #[test]
    fn test_frames_yields_single_frame() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_image(dir.path(), 100, 80);
        let mut reader = ImageFileReader::new();
        reader.open(&path).unwrap();

        let frames: Vec<_> = reader.frames().collect();
        assert_eq!(frames.len(), 1);
        let frame = frames.into_iter().next().unwrap().unwrap();
        assert_eq!(frame.index(), 0);
    }

    #[test]
    fn test_frame_is_rgb() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_image(dir.path(), 100, 80);
        let mut reader = ImageFileReader::new();
        reader.open(&path).unwrap();

        let frame = reader.frames().next().unwrap().unwrap();
        assert_eq!(frame.channels(), 3);
        // Check first pixel: R=50, G=100, B=200
        assert_eq!(frame.data()[0], 50);
        assert_eq!(frame.data()[1], 100);
        assert_eq!(frame.data()[2], 200);
    }

    #[test]
    fn test_frame_dimensions_match_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_image(dir.path(), 100, 80);
        let mut reader = ImageFileReader::new();
        let meta = reader.open(&path).unwrap();

        let frame = reader.frames().next().unwrap().unwrap();
        assert_eq!(frame.width(), meta.width);
        assert_eq!(frame.height(), meta.height);
    }

    #[test]
    fn test_frames_without_open_returns_error() {
        let mut reader = ImageFileReader::new();
        let result = reader.frames().next().unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn test_close_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_image(dir.path(), 100, 80);
        let mut reader = ImageFileReader::new();
        reader.open(&path).unwrap();
        reader.close();
        reader.close(); // should not panic
    }
}
