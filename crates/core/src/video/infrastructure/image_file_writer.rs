use std::path::Path;

use crate::shared::frame::Frame;
use crate::video::domain::image_writer::ImageWriter;

/// Writes a single frame to an image file using the `image` crate.
///
/// Supports optional resizing for thumbnails.
pub struct ImageFileWriter;

impl ImageFileWriter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ImageFileWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageWriter for ImageFileWriter {
    fn write(
        &self,
        path: &Path,
        frame: &Frame,
        size: Option<(u32, u32)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure parent directory exists (infrastructure concern)
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let img = image::RgbImage::from_raw(frame.width(), frame.height(), frame.data().to_vec())
            .ok_or("Failed to create image from frame data")?;

        let img = if let Some((w, h)) = size {
            image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle)
        } else {
            img
        };

        img.save(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> Frame {
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..(width * height) {
            data.push(r);
            data.push(g);
            data.push(b);
        }
        Frame::new(data, width, height, 3, 0)
    }

    #[test]
    fn test_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.png");
        let frame = make_frame(100, 80, 50, 100, 200);
        let writer = ImageFileWriter::new();
        writer.write(&path, &frame, None).unwrap();
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
    }

    #[test]
    fn test_roundtrip_preserves_pixels() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.png");
        let frame = make_frame(50, 50, 50, 100, 200);
        let writer = ImageFileWriter::new();
        writer.write(&path, &frame, None).unwrap();

        // Read back and verify
        let img = image::open(&path).unwrap().to_rgb8();
        assert_eq!(img.width(), 50);
        assert_eq!(img.height(), 50);
        let pixel = img.get_pixel(0, 0);
        assert_eq!(pixel.0, [50, 100, 200]);
    }

    #[test]
    fn test_write_with_resize() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("thumb.png");
        let frame = make_frame(200, 200, 128, 128, 128);
        let writer = ImageFileWriter::new();
        writer.write(&path, &frame, Some((64, 64))).unwrap();

        let img = image::open(&path).unwrap();
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn test_write_invalid_path_returns_error() {
        let frame = make_frame(10, 10, 0, 0, 0);
        let writer = ImageFileWriter::new();
        assert!(writer
            .write(Path::new("/nonexistent/dir/out.png"), &frame, None)
            .is_err());
    }
}
