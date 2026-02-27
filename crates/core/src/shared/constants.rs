pub const YOLO_MODEL_NAME: &str = "yolo11n-pose_widerface.onnx";
pub const YOLO_MODEL_URL: &str =
    "https://github.com/neutrinographics/faceguard/releases/download/v0.1.0/yolo11n-pose_widerface.onnx";

pub const EMBEDDING_MODEL_NAME: &str = "w600k_r50.onnx";
pub const EMBEDDING_MODEL_URL: &str =
    "https://github.com/neutrinographics/faceguard/releases/download/v0.1.0/w600k_r50.onnx";

/// Max frames a track can be lost before removal (~1 second at 30 fps).
pub const TRACKER_MAX_LOST: usize = 30;

pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];

pub const WHISPER_MODEL_FILENAME: &str = "ggml-tiny.en.bin";
pub const WHISPER_SAMPLE_RATE: u32 = 16000;
