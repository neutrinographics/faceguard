use std::path::Path;

use crate::shared::frame::Frame;
use crate::shared::video_metadata::VideoMetadata;
use crate::video::domain::video_reader::VideoReader;

/// Adapts a single image file to the [`VideoReader`] interface.
///
/// Treats the image as a one-frame video with `fps=0` and `total_frames=1`,
/// allowing the pipeline to process images and videos uniformly.
pub struct ImageFileReader {
    frame: Option<Frame>,
    metadata: Option<VideoMetadata>,
}

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
        let img = image::open(path)?;
        let rgb = img.to_rgb8();
        let width = rgb.width();
        let height = rgb.height();
        let data = rgb.into_raw();

        self.frame = Some(Frame::new(data, width, height, 3, 0));
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
