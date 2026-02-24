use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq)]
pub struct VideoMetadata {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub total_frames: usize,
    pub codec: String,
    pub source_path: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let meta = VideoMetadata {
            width: 1920,
            height: 1080,
            fps: 30.0,
            total_frames: 900,
            codec: "h264".to_string(),
            source_path: Some(PathBuf::from("/tmp/test.mp4")),
        };
        assert_eq!(meta.width, 1920);
        assert_eq!(meta.height, 1080);
        assert_eq!(meta.fps, 30.0);
        assert_eq!(meta.total_frames, 900);
        assert_eq!(meta.codec, "h264");
        assert_eq!(meta.source_path, Some(PathBuf::from("/tmp/test.mp4")));
    }

    #[test]
    fn test_clone_is_independent() {
        let meta = VideoMetadata {
            width: 640,
            height: 480,
            fps: 24.0,
            total_frames: 100,
            codec: "vp9".to_string(),
            source_path: None,
        };
        let cloned = meta.clone();
        assert_eq!(meta, cloned);
    }

    #[test]
    fn test_image_metadata() {
        // Images represented as single-frame video with fps=0
        let meta = VideoMetadata {
            width: 800,
            height: 600,
            fps: 0.0,
            total_frames: 1,
            codec: "png".to_string(),
            source_path: None,
        };
        assert_eq!(meta.total_frames, 1);
        assert_eq!(meta.fps, 0.0);
    }
}
