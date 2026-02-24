# video-blur-cli

Command-line interface for face detection and blurring in videos and images.

## Purpose

Thin composition root that wires together `video-blur-core` domain traits with infrastructure implementations and exposes them via a clap-based CLI. Contains no business logic — all detection, blurring, and pipeline orchestration lives in the core crate.

## Usage

```bash
# Blur all faces in a video
video-blur input.mp4 output.mp4

# Blur all faces in an image
video-blur photo.jpg blurred.jpg

# Adjust detection and blur parameters
video-blur input.mp4 output.mp4 --confidence 0.6 --blur-strength 151 --blur-shape rect

# Skip-frame detection for faster processing (detect every 3rd frame)
video-blur input.mp4 output.mp4 --skip-frames 3

# Preview mode: scan for faces and save thumbnails
video-blur input.mp4 --preview faces/

# Selective blurring after preview (filenames in faces/ are track IDs)
video-blur input.mp4 output.mp4 --blur-ids 1,3
video-blur input.mp4 output.mp4 --exclude-ids 2
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `<input>` | required | Input video or image file |
| `<output>` | required* | Output file (*optional when `--preview` is used) |
| `--confidence` | 0.5 | Face detection confidence threshold (0.0–1.0) |
| `--blur-strength` | 201 | Gaussian kernel size (must be positive and odd) |
| `--blur-shape` | ellipse | Blur shape: `ellipse` or `rect` |
| `--lookahead` | 10 | Frames buffered ahead for smooth face transitions |
| `--skip-frames` | 2 | Run detection every Nth frame (1 = every frame) |
| `--preview <dir>` | — | Save face crop thumbnails to directory instead of blurring |
| `--blur-ids` | — | Only blur these track IDs (comma-separated, mutually exclusive with `--exclude-ids`) |
| `--exclude-ids` | — | Blur all faces except these track IDs (comma-separated) |
| `--quality` | 18 | H.264 CRF quality (0=lossless, 51=worst) |

## Wiring

The CLI acts as the composition root, assembling the processing pipeline from core components:

- **Detection**: `OnnxYoloDetector` → optionally wrapped in `SkipFrameDetector` (when `--skip-frames > 1`). Region smoothing (`RegionSmoother`) and tracking (`ByteTracker`) are configured with domain defaults.
- **Blurring**: `blurrer_factory::create_blurrer()` auto-selects GPU or CPU backend based on hardware availability.
- **Video I/O**: `FfmpegReader`/`FfmpegWriter` for video, `ImageFileReader`/`ImageFileWriter` for images. Input type is detected by file extension.
- **Pipeline**: `BlurFacesUseCase` with `ThreadedPipelineExecutor` for video, `BlurImageUseCase` for images, `PreviewFacesUseCase` for preview mode.

## Model Resolution

ONNX models are resolved automatically on first run via `model_resolver`. The resolution order is:

1. User cache directory (`~/.cache/video-blur/` on Linux, `~/Library/Caches/video-blur/` on macOS)
2. Bundled path (for pre-packaged distributions)
3. Download from GitHub releases (with progress reporting to stderr)

## Design Decisions

- **`--blur-ids` and `--exclude-ids` are mutually exclusive** — Allowing both simultaneously would create ambiguous semantics. The core's `Region::filter` does support both (with `blur_ids` taking precedence), but the CLI enforces mutual exclusivity for user clarity.
- **Image detection via file extension** — Uses a static list of known image extensions (`IMAGE_EXTENSIONS`) rather than probing file headers. Simple, fast, and matches user expectations for a CLI tool.
- **Skip-frame default of 2** — Halves inference cost with negligible quality impact thanks to velocity-based extrapolation. Users processing high-motion content can set `--skip-frames 1` for full detection.
- **Progress on stderr** — Frame progress (`\r` overwrite) goes to stderr so stdout remains clean for piping.
