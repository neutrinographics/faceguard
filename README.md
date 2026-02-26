# Blur

Automatic face detection and blurring for videos and images. Detects faces using YOLO11, tracks them across frames with ByteTrack, and applies Gaussian blur with smooth transitions — all running locally with no cloud dependencies.

## Features

- **Video and image support** — MP4, MOV, AVI, MKV, JPEG, PNG, and more via FFmpeg
- **Persistent face tracking** — ByteTrack assigns stable IDs across frames so individual faces can be selectively blurred or excluded
- **Smooth transitions** — Lookahead buffering and edge-aware interpolation prevent pop-in; EMA smoothing eliminates jitter
- **GPU-accelerated blur** — wgpu compute shaders when available, CPU fallback otherwise
- **Elliptical or rectangular blur** — Elliptical shapes follow face contours naturally, extending off frame edges
- **Preview workflow** — Scan a video to identify faces, review thumbnails, then blur only the faces you choose
- **Skip-frame detection** — Run inference every Nth frame with velocity extrapolation to reduce cost without visible stutter
- **Desktop GUI** — iced-based application with real-time progress, face selection, and settings
- **CLI** — Scriptable command-line interface for batch processing

## Quick Start

### Prerequisites

- **Rust 1.75+**
- **FFmpeg 7 development libraries** — Required for video I/O
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev`
  - Or use the `static-ffmpeg` feature to build FFmpeg from source (slower initial build). Requires x264 development libraries:
    - macOS: `brew install x264`
    - Ubuntu/Debian: `apt install libx264-dev`

ONNX models are downloaded automatically on first run to `~/.cache/FaceGuard/`.

### Build

```bash
cargo build --release
```

### Run

**Blur all faces in a video:**
```bash
cargo run -p faceguard-cli --release -- input.mp4 output.mp4
```

**Blur all faces in an image:**
```bash
cargo run -p faceguard-cli --release -- photo.jpg blurred.jpg
```

**Preview faces first, then selectively blur:**
```bash
# Step 1: Scan and save face thumbnails
cargo run -p faceguard-cli --release -- input.mp4 --preview faces/

# Step 2: Review thumbnails in faces/ — filenames are track IDs
# Step 3: Blur only specific faces
cargo run -p faceguard-cli --release -- input.mp4 output.mp4 --blur-ids 1,3

# Or blur everyone except specific faces
cargo run -p faceguard-cli --release -- input.mp4 output.mp4 --exclude-ids 2
```

**Launch the desktop GUI:**
```bash
cargo run -p faceguard-desktop --release
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--confidence` | 0.5 | Detection confidence threshold (0.0-1.0) |
| `--blur-strength` | 201 | Gaussian kernel size (must be odd) |
| `--blur-shape` | ellipse | `ellipse` or `rect` |
| `--lookahead` | 10 | Frames buffered for smooth face transitions |
| `--skip-frames` | 2 | Run detection every Nth frame |
| `--preview <dir>` | — | Save face crops instead of blurring |
| `--blur-ids` | — | Only blur these track IDs (comma-separated) |
| `--exclude-ids` | — | Blur all faces except these (comma-separated) |
| `--quality` | 18 | H.264 CRF quality (0=lossless, 51=worst) |

## Project Structure

```
crates/
├── core/       Core library: detection, blurring, video I/O, pipeline orchestration
├── cli/        CLI binary (faceguard)
└── desktop/    Desktop GUI binary (faceguard-desktop)
```

The core crate is organized by feature slice (detection, blurring, video, pipeline), each with domain and infrastructure layers following Clean Architecture. See [`crates/core/README.md`](crates/core/README.md) for detailed architecture documentation.

## How It Works

1. **Read** — Decode video frames (or a single image) via FFmpeg, converting to RGB at the I/O boundary.
2. **Detect** — Run YOLO11-pose inference to find faces with 5-point landmarks. ByteTrack assigns persistent track IDs by correlating detections across frames.
3. **Filter** — Optionally include/exclude specific track IDs based on user selection.
4. **Merge** — Buffer upcoming frames and merge their detections with the current frame. New faces approaching from frame edges are interpolated inward for smooth reveal.
5. **Blur** — Apply separable Gaussian blur to each region, either as a rectangle or an ellipse that uses unclamped coordinates to extend naturally off-screen.
6. **Write** — Encode blurred frames back to video via FFmpeg, copying the audio stream from the source.

In video mode, these stages run on dedicated threads connected by bounded channels, overlapping I/O and inference for maximum throughput.

## Development

```bash
cargo test                       # Run all tests
cargo test -p faceguard-core    # Core library tests only
cargo clippy --all-targets       # Lint
cargo fmt --check                # Check formatting
```

## License

MIT
