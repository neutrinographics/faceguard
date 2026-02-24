# video-blur-core

Core library for face detection and blurring in video and image files.

## Purpose

Provides the domain logic and infrastructure adapters for a privacy-focused face blurring pipeline. Downstream crates (CLI, desktop GUI) act as composition roots that wire together domain traits with infrastructure implementations.

## Architecture

Organized as **feature slices** following Clean Architecture. Each slice contains a `domain/` layer (pure business logic, traits, entities) and an `infrastructure/` layer (external library integrations). A cross-cutting `shared/` module holds entities used across slices, and `pipeline/` is the application layer that orchestrates use cases.

```
src/
├── shared/          Cross-cutting domain entities (Frame, Region, VideoMetadata)
├── detection/       Face detection: YOLO inference, tracking, landmarks, grouping
├── blurring/        Frame blurring: Gaussian blur via CPU or GPU
├── video/           Video/image I/O: reading and writing via FFmpeg
└── pipeline/        Application layer: use case orchestration and threading
```

**Dependency rule**: `infrastructure` depends on `domain`, never the reverse. `pipeline` depends only on domain traits. `domain` has zero external crate dependencies (ndarray is permitted as a numerical primitive).

## Domain Model

### Frame
Contiguous RGB pixel buffer with width, height, channel count, and a sequence index. Format conversion (e.g., YUV to RGB) happens at I/O boundaries only. Blur operations mutate frames in-place (`&mut Frame`) to avoid allocation.

### Region
Immutable value object describing a rectangular blur target. Carries both **clamped** coordinates (visible area within frame bounds) and **unclamped** coordinates (the full pre-clip rectangle). This dual representation enables ellipses to slide naturally off frame edges instead of abruptly shrinking. Regions optionally carry a `track_id` for persistent identity across frames.

### VideoMetadata
Immutable descriptor of a video/image source: dimensions, FPS, frame count, codec, and source path. Images are represented as single-frame sources with `fps=0`.

## Domain Traits

| Trait | Slice | Purpose |
|-------|-------|---------|
| `FaceDetector` | detection | Detects faces in a frame, returns `Vec<Region>`. Stateful (`&mut self`) to support cross-frame tracking. |
| `FaceGrouper` | detection | Groups face crops by identity. Takes `(track_id, pixels, w, h)` tuples, returns groups of track IDs. |
| `RegionSmootherInterface` | detection | Temporal smoothing of region parameters via EMA to reduce jitter between frames. |
| `FrameBlurrer` | blurring | Applies blur to specified regions within a frame. Stateless (`&self`), mutates frame in-place. |
| `VideoReader` | video | Opens a video/image file and yields frames as an iterator. |
| `VideoWriter` | video | Writes processed frames to a video file, handles encoding and audio muxing on close. |
| `ImageWriter` | video | Writes a single frame to an image file with optional resize. |
| `PipelineExecutor` | pipeline | Abstracts how the read-detect-blur-write pipeline is executed (e.g., threaded vs single-threaded). |
| `PipelineLogger` | pipeline | Cross-cutting observer for progress, timing, and metrics during pipeline execution. |

## Use Cases

### BlurFacesUseCase
Full video pipeline: reads frames, detects faces, merges current detections with lookahead regions for smooth transitions, blurs, and writes output. Delegates execution to a `PipelineExecutor` for threading. Supports cancellation via `AtomicBool` and progress reporting via callback.

### BlurImageUseCase
Single-image pipeline: read, detect, filter by track ID, blur, write. No lookahead or threading needed.

### PreviewFacesUseCase
Scans a video for faces and saves the best crop (largest area) of each tracked identity as a thumbnail. Returns a detection cache that can be reused by `CachedFaceDetector` in the subsequent blur pass, guaranteeing track ID consistency between what the user previewed and what gets blurred.

## Key Algorithms

### Lookahead Region Merging (`RegionMerger`)
Buffers N future frames (default 5) and merges their detections with the current frame's. Faces appearing in lookahead but not yet in the current frame are interpolated toward the nearest frame edge, creating a smooth slide-in animation rather than a pop-in. Deduplication by track ID (current frame wins) and IoU prevents doubled regions.

### Temporal Smoothing (`RegionSmoother`)
Per-track EMA (alpha=0.6) on region center and half-dimensions. Reduces frame-to-frame jitter in bounding box positions without introducing perceptible lag.

### Profile-Aware Region Building (`FaceRegionBuilder`)
Converts YOLO bounding boxes and 5-point landmarks into blur regions. For profile (side-view) faces, the region center blends toward the bounding box center and the width expands toward the height, preventing partial face exposure. A minimum width ratio (0.8) ensures narrow detections still produce adequate coverage.

### Skip-Frame Detection (`SkipFrameDetector`)
Decorator that runs the inner detector every N frames, linearly extrapolating region positions on skipped frames using per-track velocity. Reduces inference cost proportionally while maintaining smooth region motion.

### Face Grouping
Two strategies for grouping detections that belong to the same person across a video:
- **Histogram-based** (`HistogramFaceGrouper`): HSV histogram correlation with union-find clustering. Fast, no model required.
- **Embedding-based** (`EmbeddingFaceGrouper`): ONNX face embedding model with cosine similarity. More accurate but requires a second model.

## Infrastructure Implementations

### Detection
- `OnnxYoloDetector` — YOLO11-pose via ONNX Runtime. Handles letterbox preprocessing, NMS, ByteTrack multi-object tracking, and landmark extraction.
- `CachedFaceDetector` — Replays pre-computed detections by frame index (from preview pass).
- `SkipFrameDetector` — Decorator that runs detection every N frames with velocity extrapolation.
- `HistogramFaceGrouper` / `EmbeddingFaceGrouper` — Two grouping strategies (see above).
- `model_resolver` — Resolves ONNX model files from cache or downloads them on first use.

### Blurring
- `CpuRectangularBlurrer` / `CpuEllipticalBlurrer` — Separable Gaussian blur on CPU.
- `GpuRectangularBlurrer` / `GpuEllipticalBlurrer` — wgpu compute shader blur with batched ROI processing.
- `blurrer_factory` — Probes for GPU at startup; falls back to CPU. Provides `create_blurrer()` and `gpu_available()`.

### Video
- `FfmpegReader` / `FfmpegWriter` — Video I/O via ffmpeg-next. Writer handles audio stream copy from source.
- `ImageFileReader` / `ImageFileWriter` — Single-image I/O via ffmpeg-next and the `image` crate.

### Pipeline
- `ThreadedPipelineExecutor` — Four-stage pipeline with dedicated threads for reading, detection, and writing. Main thread handles buffering, lookahead merging, and blurring. Uses bounded `crossbeam-channel` queues.

## Domain Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| IoU dedup threshold | 0.3 | `Region::deduplicate` — minimum IoU to consider regions duplicates |
| Landmark weights | [2, 2, 3, 1, 1] | `FaceLandmarks` — nose weighted 3x for stable centering |
| Padding | 0.4 | `FaceRegionBuilder` — symmetric padding around the face |
| Min width ratio | 0.8 | `FaceRegionBuilder` — prevents narrow detections |
| EMA alpha | 0.6 | `RegionSmoother` — higher = more responsive, lower = smoother |
| Edge fraction | 0.25 | `RegionMerger` — how close to edge before interpolation activates |
| Default lookahead | 5 | `BlurFacesUseCase` — frames buffered for slide-in animation |
| Channel capacity | 8 | `ThreadedPipelineExecutor` — bounded queue size between threads |
| Tracker max lost | 30 | `ByteTracker` — frames before a lost track is removed (~1s at 30fps) |
| Preview crop size | 256 | `PreviewFacesUseCase` — thumbnail output dimensions |
| Default confidence | 0.25 | `OnnxYoloDetector` — minimum detection confidence |
| NMS IoU threshold | 0.45 | `OnnxYoloDetector` — suppression threshold for overlapping boxes |

## Testing

```bash
cargo test -p video-blur-core                # All tests
cargo test -p video-blur-core -- region      # Tests matching "region"
cargo test -p video-blur-core -- --ignored   # Infrastructure tests requiring models/network
```

Domain tests use stub/fake trait implementations for isolation. Infrastructure tests that require ONNX models or network access are marked `#[ignore]`. `rstest` is used for parameterized tests and `approx` for float comparisons.
