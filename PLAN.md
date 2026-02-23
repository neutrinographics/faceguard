# Video Blur — Rust Rewrite Plan

## Overview

Full rewrite of the Video Blur application from Python to Rust, preserving all current functionality while achieving dramatic improvements in binary size (~800MB → ~80-120MB), performance (3-5x throughput), and startup time (3-5s → <0.5s). The architecture follows the same DDD, Clean Architecture, TDD, and screaming architecture principles as the original.

---

## Target Stack

| Concern | Crate | Why |
|---------|-------|-----|
| ML inference | `ort` | Runs all 3 models via ONNX Runtime. Single runtime replaces PyTorch + MediaPipe + onnxruntime |
| Object tracking | `jamtrack-rs` or `mot-rs` | Pure-Rust ByteTrack. Drop-in replacement for ultralytics tracker |
| Video I/O | `ffmpeg-next` | Decode/encode via libav. Replaces OpenCV VideoCapture/VideoWriter + ffmpeg binary for audio mux |
| Image processing | `image` + `imageproc` | Read/write images, resize for thumbnails. Lighter than OpenCV for image-only tasks |
| GPU blur | `wgpu` | Compute-shader Gaussian blur. Replaces CPU OpenCV GaussianBlur |
| CPU blur fallback | `ndarray` + manual Gaussian | For systems without GPU. Separable kernel for speed |
| Array math | `ndarray` | Replaces NumPy |
| GUI | `iced` | Elm-architecture, uses wgpu (shares GPU context with blur shaders) |
| CLI | `clap` | Argument parsing |
| Parallelism | `rayon` + `crossbeam` | Frame pipeline parallelism, channel-based producer/consumer |
| Testing | built-in `#[cfg(test)]` + `assert_approx_eq` | Parametrized via macros or `rstest` crate |
| Serialization | `serde` + `serde_json` | Settings persistence |
| HTTP downloads | `reqwest` (blocking or async) | Model downloads |
| Cross-platform paths | `dirs` | Cache/config directories |
| Logging | `tracing` | Structured logging with per-stage timing |

---

## Model Strategy

All three models converted to ONNX and run through `ort`:

| Model | Current Format | Action | Rust Runtime |
|-------|---------------|--------|-------------|
| `yolo11n-pose_widerface.pt` | PyTorch | Export to ONNX once (`model.export(format="onnx")`) | `ort` |
| `blaze_face_short_range.tflite` | TFLite | Convert to ONNX via `tf2onnx` or use `tract` for this one model | `ort` (or `tract`) |
| `w600k_r50.onnx` | ONNX | Use as-is | `ort` |

The YOLO export is a one-time step. The exported `.onnx` file ships with the app or downloads on first run, same as today.

---

## Project Structure

```
blur/
├── Cargo.toml                          # Workspace root
├── PLAN.md                             # This file
├── CLAUDE.md                           # AI assistant instructions
├── models/                             # Pre-converted ONNX models (git-ignored, downloaded)
├── shaders/                            # WGSL compute shaders
│   └── gaussian_blur.wgsl
├── crates/
│   ├── core/                           # Core library (equivalent to packages/core)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── shared/                 # Cross-cutting domain entities
│   │       │   ├── mod.rs
│   │       │   ├── frame.rs            # Frame entity
│   │       │   ├── region.rs           # Region entity (immutable struct)
│   │       │   ├── video_metadata.rs   # VideoMetadata entity
│   │       │   └── model_resolver.rs   # Model path resolution + download
│   │       ├── detection/              # Face detection feature slice
│   │       │   ├── mod.rs
│   │       │   ├── domain/
│   │       │   │   ├── mod.rs
│   │       │   │   ├── face_detector.rs         # Trait: FaceDetector
│   │       │   │   ├── face_landmarks.rs        # Entity: FaceLandmarks
│   │       │   │   ├── face_region_builder.rs   # Service: box+landmarks → Region
│   │       │   │   ├── region_smoother.rs       # Trait + EMA impl
│   │       │   │   ├── region_merger.rs         # Service: merge current + lookahead
│   │       │   │   └── face_grouper.rs          # Trait: FaceGrouper
│   │       │   └── infrastructure/
│   │       │       ├── mod.rs
│   │       │       ├── onnx_yolo_detector.rs     # YOLO via ort
│   │       │       ├── onnx_blazeface_detector.rs # BlazeFace via ort
│   │       │       ├── bytetrack_tracker.rs      # ByteTrack via jamtrack-rs
│   │       │       ├── skip_frame_detector.rs    # Decorator: run every Nth frame
│   │       │       ├── cached_face_detector.rs   # Replay pre-computed detections
│   │       │       ├── embedding_face_grouper.rs # ArcFace via ort
│   │       │       └── histogram_face_grouper.rs # HSV histogram comparison
│   │       ├── blurring/               # Frame blurring feature slice
│   │       │   ├── mod.rs
│   │       │   ├── domain/
│   │       │   │   ├── mod.rs
│   │       │   │   └── frame_blurrer.rs         # Trait: FrameBlurrer
│   │       │   └── infrastructure/
│   │       │       ├── mod.rs
│   │       │       ├── gpu_elliptical_blurrer.rs  # wgpu compute shader blur
│   │       │       ├── gpu_rectangular_blurrer.rs # wgpu compute shader blur
│   │       │       ├── cpu_elliptical_blurrer.rs  # Fallback: separable Gaussian
│   │       │       └── cpu_rectangular_blurrer.rs # Fallback: separable Gaussian
│   │       ├── video/                  # Video I/O feature slice
│   │       │   ├── mod.rs
│   │       │   ├── domain/
│   │       │   │   ├── mod.rs
│   │       │   │   ├── video_reader.rs          # Trait: VideoReader
│   │       │   │   ├── video_writer.rs          # Trait: VideoWriter
│   │       │   │   └── image_writer.rs          # Trait: ImageWriter
│   │       │   └── infrastructure/
│   │       │       ├── mod.rs
│   │       │       ├── ffmpeg_reader.rs          # ffmpeg-next decode
│   │       │       ├── ffmpeg_writer.rs          # ffmpeg-next encode (video+audio mux)
│   │       │       ├── image_file_reader.rs      # image crate imread
│   │       │       └── image_file_writer.rs      # image crate imwrite
│   │       └── pipeline/               # Application layer (use case orchestration)
│   │           ├── mod.rs
│   │           ├── blur_faces_use_case.rs       # Video blurring with lookahead
│   │           ├── blur_image_use_case.rs       # Single image blurring
│   │           ├── preview_faces_use_case.rs    # Face crop preview mode
│   │           ├── region_filter.rs             # blur_ids / exclude_ids filtering
│   │           └── pipeline_logger.rs           # Progress + timing reporting
│   ├── cli/                            # CLI binary (equivalent to core CLI)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs                 # CLI entry point + arg parsing
│   └── desktop/                        # Desktop GUI binary
│       ├── Cargo.toml
│       ├── assets/
│       │   ├── icon.png
│       │   └── icon.icns
│       └── src/
│           ├── main.rs                 # App entry point
│           ├── app.rs                  # Top-level iced Application
│           ├── theme.rs                # Dark/Light/System theme + high contrast
│           ├── settings.rs             # Persisted settings (serde + JSON file)
│           ├── tabs/
│           │   ├── mod.rs
│           │   ├── main_tab.rs         # File selection, preview, run
│           │   ├── settings_tab.rs     # Detector, blur, confidence, lookahead
│           │   ├── accessibility_tab.rs # Appearance, high contrast
│           │   ├── privacy_tab.rs      # Static text
│           │   └── about_tab.rs        # Branding, version
│           ├── widgets/
│           │   ├── mod.rs
│           │   ├── face_card.rs        # Single face thumbnail with selection
│           │   ├── face_group_card.rs  # Stacked group card with count badge
│           │   ├── faces_well.rs       # Flow-layout grid of face cards
│           │   ├── file_row.rs         # INPUT/OUTPUT file picker row
│           │   └── flow_layout.rs      # CSS-flexbox-style wrapping layout
│           └── workers/
│               ├── mod.rs
│               ├── preview_worker.rs   # Background face detection + grouping
│               └── blur_worker.rs      # Background blur processing
```

---

## Phase 1: Foundation & Shared Domain (Week 1-2)

### 1.1 Project Scaffolding
- [ ] Initialize Cargo workspace with `crates/core`, `crates/cli`, `crates/desktop`
- [ ] Set up `Cargo.toml` workspace dependencies (shared versions)
- [ ] Configure `clippy`, `rustfmt`, CI with GitHub Actions (test on Linux, macOS, Windows)
- [ ] Add `CLAUDE.md` with Rust-specific conventions

### 1.2 Shared Entities (TDD)

Write tests first, then implement for each:

**`frame.rs`**
- `Frame { data: Vec<u8>, width: u32, height: u32, channels: u8, index: usize }`
- Data stored as contiguous bytes (BGR or RGB — decide on RGB since `image` crate uses RGB)
- Method: `as_ndarray()` → view into data as 3D array via `ndarray`
- Tests: construction, dimension accessors, data integrity

**`region.rs`**
- Immutable struct (no `mut` methods, return new instances)
- Fields: `x, y, width, height, unclamped_x, unclamped_y, full_width, full_height, track_id: Option<u32>`
- Methods:
  - `iou(&self, other: &Region) -> f64`
  - `deduplicate(regions: &[Region], iou_threshold: f64) -> Vec<Region>`
  - `ellipse_center_in_roi() -> (f64, f64)`
  - `ellipse_axes() -> (f64, f64)`
- Tests: IoU calculation, deduplication, ellipse geometry at edges
- Use `#[derive(Clone, Debug, PartialEq)]`

**`video_metadata.rs`**
- `VideoMetadata { width: u32, height: u32, fps: f64, total_frames: usize, codec: String, source_path: PathBuf }`
- Immutable struct
- Tests: construction, accessor validation

**`model_resolver.rs`**
- Platform-aware cache directories via `dirs` crate
- Download with progress callback: `fn resolve(name, url, progress_cb) -> Result<PathBuf>`
- Check cache → check bundled → download
- Tests: cache hit, cache miss (mock HTTP or skip in CI)

---

## Phase 2: Detection Domain (Week 2-3)

### 2.1 Domain Traits & Entities (TDD)

**`face_detector.rs`** — Trait
```rust
pub trait FaceDetector: Send {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Region>>;
}
```

**`face_landmarks.rs`** — Entity
- 5-point landmarks: `[Point2D; 5]` where indices are: 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth
- `weighted_centroid() -> Point2D` (nose=3x, eyes=2x, mouth=1x)
- `profile_ratio() -> f64` (0=frontal, 1=full profile)
- Tests: centroid with various poses, profile ratio frontal/left/right/partial

**`face_region_builder.rs`** — Service
- `build(bbox, landmarks, frame_w, frame_h, smoother) -> Region`
- Profile-aware width: blend toward box height for profiles
- Center blending: shift toward box center for profiles (landmarks cluster on one side)
- Padding expansion (default 0.4)
- Minimum width constraint (80% of height)
- Clamped vs unclamped geometry
- Tests: frontal, left/right profile, strong profile, edge faces, small/large faces (parametrized)

**`region_smoother.rs`** — Trait + EMA Implementation
- Trait: `RegionSmootherInterface { fn smooth(&mut self, track_id, params) -> SmoothedParams }`
- EMA impl: alpha=0.6, per-track state
- Tests: convergence, multiple tracks, alpha edge cases (0, 1)

**`region_merger.rs`** — Service
- `merge(current: &[Region], lookahead: &[Vec<Region>], frame_w, frame_h) -> Vec<Region>`
- Dedup by track_id (current wins), then by IoU (threshold=0.3)
- Edge-aware interpolation: push lookahead regions toward nearest frame edge
- Tests: track_id dedup, IoU dedup, edge interpolation, empty inputs

**`face_grouper.rs`** — Trait
```rust
pub trait FaceGrouper: Send {
    fn group(&self, crops: &[(u32, &[u8], u32, u32)]) -> Result<Vec<Vec<u32>>>;
}
```

### 2.2 Detection Infrastructure (Week 3-4)

**`onnx_yolo_detector.rs`**
- Load YOLO ONNX model via `ort::Session`
- Pre-processing: letterbox resize to 640x640, normalize to [0,1], NCHW layout
- Post-processing: parse output tensor → boxes + confidences + keypoints
- Non-Maximum Suppression (NMS) — implement manually (~50 lines)
- Confidence threshold filtering
- GPU execution provider when available (CUDA, CoreML, DirectML)
- Tests: output parsing with known tensor data (mock session or use small test model)

**`bytetrack_tracker.rs`**
- Wrap `jamtrack-rs` ByteTrack
- Input: per-frame detections (boxes + scores)
- Output: tracked boxes with persistent IDs
- Integrate with YOLO detector: detect → track → extract landmarks → build regions
- Tests: ID persistence across frames, re-association after occlusion

**`onnx_blazeface_detector.rs`**
- Load BlazeFace ONNX model via `ort::Session`
- Pre-processing: resize to 128x128, normalize
- Post-processing: decode anchor boxes, NMS
- No landmarks, no tracking (matches current Python behavior)
- Tests: output parsing

**`skip_frame_detector.rs`** — Decorator
- Wraps any `FaceDetector`
- Runs real detection every N frames
- Linear velocity extrapolation on skipped frames
- Tests: skip interval, velocity calculation, extrapolation accuracy

**`cached_face_detector.rs`**
- Replays `HashMap<usize, Vec<Region>>` by frame index
- Tests: cache hit, cache miss returns empty

**`embedding_face_grouper.rs`**
- ArcFace ONNX model: resize crop to 112x112, normalize `(pixel - 127.5) / 127.5`
- L2-normalize output embedding
- Cosine similarity matrix → union-find clustering (threshold=0.4)
- Tests: identical images → same group, different → separate (model-dependent, skip if no model)

**`histogram_face_grouper.rs`**
- Convert crop to HSV, compute histogram, `compareHist` equivalent via manual correlation
- Union-find clustering (threshold=0.7)
- Tests: same color → grouped, different color → separate

---

## Phase 3: Blurring (Week 4-5)

### 3.1 Domain Trait (TDD)

**`frame_blurrer.rs`**
```rust
pub trait FrameBlurrer: Send {
    fn blur(&self, frame: &mut Frame, regions: &[Region]) -> Result<()>;
}
```
Note: `&mut Frame` instead of returning a new frame — avoid allocation.

### 3.2 GPU Blur via wgpu (Primary Path)

**`gaussian_blur.wgsl`** — Compute Shader
- Two-pass separable Gaussian blur (horizontal + vertical)
- Input: texture or storage buffer
- Uniforms: kernel_size, sigma, ROI bounds, ellipse center + axes (for masking)
- Ellipse mask applied in the vertical pass: blend blurred ↔ original based on ellipse SDF

**`gpu_elliptical_blurrer.rs`**
- Initialize wgpu device + queue (once, reuse across frames)
- Upload frame ROI to GPU buffer
- Dispatch horizontal blur → vertical blur + ellipse mask
- Read back result, composite into frame
- Handles multiple regions per frame (batch or sequential dispatch)
- Tests: known input → expected blurred output (run on CPU adapter in tests via wgpu's fallback)

**`gpu_rectangular_blurrer.rs`**
- Same as elliptical but skip the ellipse mask step
- Simpler shader variant

### 3.3 CPU Blur Fallback

**`cpu_elliptical_blurrer.rs`**
- Separable Gaussian kernel (precompute 1D kernel)
- Apply horizontal then vertical pass on ROI
- Ellipse mask via SDF: `((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1`
- Downscale-blur-upscale optimization (same as current Python: scale = kernel_size // 50)
- Tests: verify blur modifies pixels, ellipse boundary respected

**`cpu_rectangular_blurrer.rs`**
- Same separable blur, no mask
- Tests: ROI fully blurred

### 3.4 Blurrer Selection
- Auto-detect GPU availability at startup via `wgpu::Instance::request_adapter()`
- Fall back to CPU if no adapter
- Log which backend is in use

---

## Phase 4: Video I/O (Week 5-6)

### 4.1 Domain Traits (TDD)

**`video_reader.rs`**
```rust
pub trait VideoReader: Send {
    fn open(&mut self, path: &Path) -> Result<VideoMetadata>;
    fn frames(&mut self) -> Box<dyn Iterator<Item = Result<Frame>> + '_>;
    fn close(&mut self);
}
```

**`video_writer.rs`**
```rust
pub trait VideoWriter: Send {
    fn open(&mut self, path: &Path, metadata: &VideoMetadata) -> Result<()>;
    fn write(&mut self, frame: &Frame) -> Result<()>;
    fn close(&mut self) -> Result<()>;  // Audio mux happens here
}
```

**`image_writer.rs`**
```rust
pub trait ImageWriter: Send {
    fn write(&self, path: &Path, frame: &Frame, size: Option<(u32, u32)>) -> Result<()>;
}
```

### 4.2 Infrastructure (TDD)

**`ffmpeg_reader.rs`**
- Decode via `ffmpeg-next` (libavformat + libavcodec)
- Seek to frame, decode to RGB, wrap in Frame
- Extract metadata: width, height, fps, frame count, codec
- Tests: create test video in-memory, read back frames, verify metadata

**`ffmpeg_writer.rs`**
- Encode via `ffmpeg-next`
- **Audio mux built-in**: copy audio stream from source during encoding — no separate ffmpeg binary needed, no temp file, no post-processing step
- This is a significant simplification over the Python version
- Tests: write frames, verify output is valid video, verify audio preservation

**`image_file_reader.rs`**
- Load image via `image` crate, adapt to VideoReader trait (1 frame, fps=0)
- Tests: read test image, verify dimensions and pixel data

**`image_file_writer.rs`**
- Write via `image` crate
- Optional resize for thumbnails
- Tests: write and read back, verify dimensions

---

## Phase 5: Pipeline / Application Layer (Week 6-7)

### 5.1 Use Cases (TDD — all tested with stub implementations of traits)

**`blur_faces_use_case.rs`**
- Same architecture as Python: read → detect → buffer → merge → blur → write
- Threading via `crossbeam` channels:
  - Reader thread → `crossbeam::channel` → main detection/blur thread → channel → writer thread
  - Bounded channels (capacity 4) for backpressure
- Lookahead deque buffer
- Progress callback: `Fn(usize, usize)` (current_frame, total_frames)
- Cancellation via `AtomicBool`
- Tests (using stubs):
  - Processes all frames
  - Frames written in order
  - Reader and writer closed (including on error)
  - Lookahead provides future regions to merger
  - Progress callback fires
  - Cancellation stops processing
  - blur_ids / exclude_ids filtering

**`blur_image_use_case.rs`**
- Single frame: read → detect → filter → blur → write
- Tests: regions filtered, output dimensions preserved

**`preview_faces_use_case.rs`**
- Scan video, cache detections by frame index
- Keep largest crop per track_id
- Save 256x256 thumbnails
- Return `(HashMap<u32, PathBuf>, DetectionCache)`
- Tests: per-track crop saving, largest selection, progress callback

**`region_filter.rs`**
- `filter(regions, blur_ids, exclude_ids) -> Vec<Region>`
- blur_ids takes precedence
- Tests: both modes, empty sets, overlap

**`pipeline_logger.rs`**
- Per-stage timing via `tracing` spans
- Summary: avg ms per stage, throughput fps
- Tests: progress throttling, metrics averaging

---

## Phase 6: CLI Binary (Week 7)

**`crates/cli/src/main.rs`**

Mirror the current CLI exactly:

```
video-blur INPUT [OUTPUT] [OPTIONS]

Options:
  --confidence <FLOAT>       Face detection confidence threshold (0.0-1.0, default: 0.5)
  --blur-strength <INT>      Gaussian blur kernel size (must be odd, default: 201)
  --blur-shape <SHAPE>       ellipse | rect (default: ellipse)
  --detector <DETECTOR>      yolo | mediapipe (default: yolo)
  --lookahead <INT>          Frames to look ahead (default: 10)
  --skip-frames <N>          Run detection every Nth frame (default: 2)
  --preview <DIR>            Save face crops instead of blurring
  --blur-ids <IDS>           Only blur these track IDs (comma-separated)
  --exclude-ids <IDS>        Blur all except these track IDs
```

- Use `clap` derive macros for argument parsing
- Validate `--blur-ids` and `--exclude-ids` are mutually exclusive
- Validate `--blur-strength` is odd
- Auto-detect GPU (wgpu adapter → ort execution provider)
- Wire together infrastructure implementations and run use case
- Exit codes: 0 success, 1 error

---

## Phase 7: Desktop GUI (Week 7-9)

### 7.1 Application Shell

**`app.rs`** — iced `Application`
- Window title: "Video Blur — Neutrino Graphics"
- Minimum size: 520x360
- Tab navigation: Main, Settings, Accessibility, Privacy, About
- Message enum for all UI events
- Subscription for worker progress (via `iced::subscription` wrapping `crossbeam` channels)

### 7.2 Theme System

**`theme.rs`**
- Dark, Light, System modes
- High contrast toggle (larger text, increased contrast)
- Accent color for selection states
- Map to `iced::Theme` custom palette

### 7.3 Settings Persistence

**`settings.rs`**
- JSON file in platform config directory
- Fields: detector, confidence, blur_strength, blur_shape, lookahead, appearance, high_contrast
- Load on startup, save on change
- Restore defaults function

### 7.4 Tabs (mirror current UI exactly)

**`main_tab.rs`**
- Input file row: "INPUT" label + filename + Browse button
- Output file row: "OUTPUT" label + filename + Browse button (disabled until input selected)
- Auto-generate default output: `{stem}_blurred{ext}`
- Description: "Blur faces in videos and photos. Select a file to get started."
- Preview button: "Choose which faces to blur..."
- Faces well (flow layout of face cards with selection)
- Face count: "{selected} of {total} faces/groups selected"
- Group similar faces checkbox
- Progress bar with ETA: "X% — YmZZs remaining"
- Download progress: "Filename... X.X / Y.Y MB (Z%)"
- Run button / Cancel button (swap visibility during processing)
- File dialogs via `rfd` crate (native file dialogs)
- Completion dialog: "Processing complete!" with "Show in Folder" action

**`settings_tab.rs`**
- Detector combo: yolo / mediapipe with description
- Blur shape combo: ellipse / rect with description
- Confidence slider: 10-100, display as "0.XX"
- Blur strength slider: 51-401 step 2, enforce odd
- Lookahead slider: 0-30
- Restore Defaults button
- All values auto-persist

**`accessibility_tab.rs`**
- Appearance combo: System / Dark / Light
- High contrast checkbox
- Auto-persist and apply immediately

**`privacy_tab.rs`**
- Static text: "Your data stays on your device" section
- Static text: "Blurring is permanent" section
- Scrollable

**`about_tab.rs`**
- "Video Blur" app name
- Version from `Cargo.toml` (via `env!("CARGO_PKG_VERSION")`)
- "Made by Neutrino Graphics LLC"
- Tagline text
- "Visit neutrinographics.com" button
- Footer text

### 7.5 Custom Widgets

**`face_card.rs`**
- 76x76px thumbnail
- Selected state: accent border + filled check circle with checkmark
- Deselected state: semi-transparent overlay + unfilled check circle
- Hover: border brightens
- Click to toggle

**`face_group_card.rs`**
- Stacked appearance (up to 2 layers, 3px offset)
- Count badge "x{N}" bottom-right
- Same selection states as face_card
- Click toggles all track_ids in group

**`faces_well.rs`**
- Flow layout container (wrapping left-to-right)
- Header with group checkbox + count label
- Separator line (visible when groups exist)
- Scrollable body

**`file_row.rs`**
- Reusable file picker: label + filename display + browse button
- Empty state: "No file selected"

**`flow_layout.rs`**
- Custom iced layout: items flow left-to-right, wrap on overflow
- 8px spacing, 10px margins
- Calculate height-for-width

### 7.6 Background Workers

**`preview_worker.rs`**
- Run in `tokio::task::spawn_blocking` or `std::thread`
- Send progress via `crossbeam::channel` → iced subscription
- Download models → detect faces → group → save crops
- Cancellation via `AtomicBool`

**`blur_worker.rs`**
- Same threading model
- Reuse detection cache from preview when available
- Progress + ETA reporting

### 7.7 Footer
- "neutrinographics.com" link (center-aligned, opens browser via `open` crate)

---

## Phase 8: Cross-Platform Packaging (Week 9-10)

### 8.1 Build Configuration

**`Cargo.toml` (workspace)**
```toml
[profile.release]
lto = true
codegen-units = 1
strip = true
opt-level = "s"          # Optimize for size (or "z" for maximum compression)
panic = "abort"          # Smaller binary, no unwinding
```

### 8.2 Platform Builds

**macOS**
- `cargo build --release` → universal binary (x86_64 + aarch64 via `cargo-lipo` or separate builds)
- Bundle as `.app` using `cargo-bundle` or manual Info.plist + directory structure
- Codesign + notarize for distribution
- `.dmg` creation via `create-dmg`
- Include `icon.icns`
- Bundle identifier: `com.da1nerd.videoblur`

**Windows**
- Cross-compile via `x86_64-pc-windows-msvc` target (or build on Windows CI)
- `.msi` installer via `cargo-wix` or NSIS
- Include app icon as `.ico`
- ONNX Runtime ships as `.dll` alongside binary

**Linux**
- `cargo build --release` → static binary (musl libc for maximum portability)
- `.AppImage` via `linuxdeploy` or `.deb` via `cargo-deb`
- Bundle ONNX Runtime `.so` or statically link

### 8.3 CI/CD (GitHub Actions)

```yaml
matrix:
  os: [ubuntu-latest, macos-latest, windows-latest]
steps:
  - cargo test
  - cargo build --release
  - package (platform-specific)
  - upload artifact
```

### 8.4 Model Distribution
- Models are NOT embedded in binary (too large)
- Downloaded on first run to platform cache directory
- Optional: ship alongside binary in installer for offline use
- Model URLs defined as constants in `model_resolver.rs`

---

## Phase 9: Integration Testing & Polish (Week 10-11)

### 9.1 End-to-End Tests
- CLI: process a short test video, verify output exists and has expected frame count
- CLI: process a test image, verify output dimensions
- CLI: `--preview` mode, verify crop files created
- Desktop: (manual testing — iced doesn't have a mature test harness for UI)

### 9.2 Performance Benchmarking
- Benchmark: YOLO inference time per frame (ort vs Python ultralytics)
- Benchmark: blur time per frame (wgpu vs CPU)
- Benchmark: end-to-end throughput on 1080p 30fps test video
- Use `criterion` crate for reproducible benchmarks

### 9.3 Binary Size Audit
- `cargo bloat` to identify largest dependencies
- Strip unused ONNX Runtime execution providers
- Ensure wgpu only links needed backends per platform

### 9.4 Polish
- Error messages: user-friendly, not panics
- Graceful degradation: GPU blur → CPU blur, YOLO → BlazeFace
- Ctrl+C handling in CLI
- Window close handling in desktop (cancel active workers)

---

## Testing Strategy

### Approach
Mirrors the Python test suite's patterns, adapted to Rust idioms:

1. **Domain logic tested in isolation** via trait objects / stub implementations
2. **Parametrized tests** via `rstest` crate (equivalent to `@pytest.mark.parametrize`)
3. **Stub/fake implementations** of all traits (no mocking library needed)
4. **Approximate float assertions** via `assert_approx_eq` or `f64::abs(a - b) < epsilon`
5. **Infrastructure tests** that touch real ONNX models marked with `#[ignore]` (run separately in CI)

### Test Stubs (defined in each test module)
```rust
struct StubReader { frames: Vec<Frame>, opened: bool, closed: bool }
struct StubWriter { written: Vec<Frame>, opened: bool, closed: bool }
struct StubDetector { results: HashMap<usize, Vec<Region>> }
struct PassthroughBlurrer { calls: Vec<(usize, Vec<Region>)> }
struct FailingDetector;
```

### Coverage Targets
- Domain layer: 100% (all entities, services, traits)
- Infrastructure: integration tests where feasible, `#[ignore]` for model-dependent
- Pipeline: all use cases tested with stubs
- CLI: argument parsing tests
- Desktop: manual testing + screenshot comparison

---

## Key Design Decisions

### 1. RGB everywhere (not BGR)
The Python codebase uses BGR (OpenCV default). The Rust `image` crate and most Rust ecosystem uses RGB. Standardize on RGB internally, convert at I/O boundaries only.

### 2. `&mut Frame` for blur (zero-copy)
Blur modifies frame data in-place instead of returning a new frame. Avoids allocating a full frame copy per region per frame. Rust's borrow checker ensures safety.

### 3. Audio mux in ffmpeg-next (not separate binary)
The Python version shells out to ffmpeg for audio. With `ffmpeg-next`, audio stream copying is built into the writer. No temp files, no subprocess, no bundled ffmpeg binary. This alone saves ~80MB from the bundle.

### 4. wgpu for GUI + blur (shared GPU context)
Using `iced` (wgpu-based GUI) means the blur compute shaders can share the GPU device. No context switching between GUI rendering and blur computation.

### 5. Channels over threads for pipeline
Instead of `ThreadedFrameIterator`/`ThreadedFrameWriter` classes, use `crossbeam::channel` bounded channels. Reader thread → channel → processing thread → channel → writer thread. Same architecture, more idiomatic Rust.

### 6. Feature flags for optional components
```toml
[features]
default = ["gpu-blur", "yolo", "blazeface"]
gpu-blur = ["wgpu"]
yolo = []          # Always available (just ort)
blazeface = []     # Always available (just ort)
arcface = []       # Optional face grouping model
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| `jamtrack-rs` too immature | ByteTrack is ~200 lines of algorithm. Port manually if needed. |
| YOLO ONNX export mismatches | Test exported model against Python outputs before starting Rust inference code |
| `ffmpeg-next` build complexity | Use vendored feature flag. Fallback: shell out to system ffmpeg like Python does |
| wgpu compute shader debugging | Develop CPU blur first, GPU second. CPU fallback always available |
| iced flow layout missing | Implement custom layout (iced supports this). Port FlowLayout algorithm from Python (~60 lines) |
| ONNX Runtime linking on all platforms | Use `ort`'s download feature which auto-fetches pre-built ORT binaries at build time |

---

## Estimated Binary Size

| Component | Size |
|-----------|------|
| Rust binary (CLI or GUI, stripped, LTO) | ~5-10MB |
| ONNX Runtime shared lib | ~25-30MB |
| wgpu/GPU drivers | Linked to system (0MB) |
| iced GUI | Compiled in (~3-5MB) |
| Total binary | **~35-45MB** |
| + Models (downloaded separately) | +6MB (YOLO+BlazeFace) or +180MB (with ArcFace) |
| **Total distributed package** | **~40-50MB** (without ArcFace) |

Compared to current: **~800MB → ~45MB** (16x reduction).

---

## Appendix A: Domain Logic Specification

Every constant, formula, threshold, and behavioral rule that must be preserved exactly in the Rust rewrite. This is the contract — tests should encode these values.

### A.1 Region

**Fields** (all integer):
- `x, y, width, height` — clamped to visible frame area
- `unclamped_x, unclamped_y, full_width, full_height` — pre-clamp values (all `Option<i32>`)
- `track_id: Option<u32>`

**`ellipse_center_in_roi()`:**
```
fw = full_width.unwrap_or(width)
fh = full_height.unwrap_or(height)
ux = unclamped_x.unwrap_or(x)
uy = unclamped_y.unwrap_or(y)
center_x = fw / 2 - (x - ux)
center_y = fh / 2 - (y - uy)
```

**`ellipse_axes()`:**
```
fw = full_width.unwrap_or(width)
fh = full_height.unwrap_or(height)
semi_axis_x = fw / 2
semi_axis_y = fh / 2
```

**`iou(other)`:**
```
inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
if inter == 0: return 0.0
return inter / (area_a + area_b - inter)
```

**`deduplicate(regions, iou_threshold=0.3)`:**
- Greedy: iterate regions, keep only if IoU <= threshold with ALL previously-kept regions
- Default threshold: **0.3**

### A.2 FaceLandmarks

**Landmark indices:**
```
LEFT_EYE = 0, RIGHT_EYE = 1, NOSE = 2, LEFT_MOUTH = 3, RIGHT_MOUTH = 4
```

**Weights:** `[2.0, 2.0, 3.0, 1.0, 1.0]` (eyes=2x, nose=3x, mouth=1x)

**`center()` — weighted centroid:**
```
visible = points where x > 0
weights_filtered = WEIGHTS[visible]
cx = sum(visible_x * weight) / sum(weights_filtered)
cy = sum(visible_y * weight) / sum(weights_filtered)
```
- Raises error if no visible landmarks
- Visibility check: `point.x > 0`

**`profile_ratio()`:**
```
if nose.x <= 0 or left_eye.x <= 0 or right_eye.x <= 0: return 0.0
eye_mid_x = (left_eye.x + right_eye.x) / 2
eye_span = abs(right_eye.x - left_eye.x)
if eye_span <= 0: return 0.0
return min(abs(nose.x - eye_mid_x) / eye_span, 1.0)
```
- Range: [0.0, 1.0] — 0=frontal, 1=full profile

**`has_visible()`:** `any(point.x > 0)`

### A.3 FaceRegionBuilder

**Constants:**
- Default `padding = 0.4`
- Minimum width ratio: `0.8` (effective_w >= box_h * 0.8)

**Center blending (profile-aware):**
```
box_cx = (box.x1 + box.x2) / 2
box_cy = (box.y1 + box.y2) / 2
face_cx, face_cy = landmarks.center()
cx = face_cx + (box_cx - face_cx) * profile_ratio
cy = face_cy + (box_cy - face_cy) * profile_ratio
```
- When profile_ratio=0 (frontal): uses landmark center
- When profile_ratio=1 (profile): uses box center

**Size computation (profile-aware):**
```
box_w = box.x2 - box.x1
box_h = box.y2 - box.y1
effective_w = box_w + (box_h - box_w) * profile_ratio
effective_w = max(effective_w, box_h * 0.8)
half_w = effective_w * (1 + padding) / 2
half_h = box_h * (1 + padding) / 2
```

**Clamping to Region:**
```
unclamped_x = int(cx - half_w)
unclamped_y = int(cy - half_h)
full_w = int(half_w * 2)
full_h = int(half_h * 2)
x = max(unclamped_x, 0)
y = max(unclamped_y, 0)
w = max(int(min(cx + half_w, frame_w)) - x, 0)
h = max(int(min(cy + half_h, frame_h)) - y, 0)
```

### A.4 RegionSmoother (EMA)

**Constant:** `alpha = 0.6`

**Formula:** `ema[t] = alpha * current + (1 - alpha) * ema[t-1]`

**Rules:**
- Per-track_id state (HashMap)
- First observation for a track_id: `ema = current` (no blending)
- `track_id = None`: bypass smoothing, return params unchanged
- Params format: `[cx, cy, half_w, half_h]`

### A.5 RegionMerger

**Constant:** `EDGE_FRACTION = 0.25`

**Merge algorithm:**
1. Seed `seen_ids` with all track_ids from `current` regions
2. Start `result` with all `current` regions
3. For each `(idx, future_regions)` in `lookahead`:
   - For each region `r` in `future_regions`:
     - If `r.track_id` is Some and NOT in `seen_ids`: add to `seen_ids`, interpolate, append to result
     - If `r.track_id` is Some and IN `seen_ids`: skip (current wins)
     - If `r.track_id` is None: always append (no dedup)
4. Apply `Region::deduplicate(result)` with IoU threshold 0.3

**Edge interpolation (`_interpolate`):**
```
t = (idx + 1) / (total + 1)          // strength increases with distance

cx = region.x + region.width / 2
cy = region.y + region.height / 2

d_left = cx
d_right = frame_w - cx
d_top = cy
d_bottom = frame_h - cy
min_dist = min(d_left, d_right, d_top, d_bottom)

// Threshold depends on which edge is nearest
threshold = if nearest is left/right { frame_w * 0.25 } else { frame_h * 0.25 }

if min_dist > threshold: return region unchanged

// Push toward nearest edge
dx, dy = match nearest {
    left   => (-d_left * t, 0),
    right  => (d_right * t, 0),
    top    => (0, -d_top * t),
    bottom => (0, d_bottom * t),
}

new_x = max(region.x + dx, 0)
new_y = max(region.y + dy, 0)
// unclamped_x/y also shifted by dx/dy (if present)
```

### A.6 Region Filter

**Priority rule:** `blur_ids` takes absolute precedence over `exclude_ids`
```
if blur_ids is Some: keep only regions where track_id IN blur_ids
else if exclude_ids is Some: keep only regions where track_id NOT IN exclude_ids
else: keep all
```
**Edge case:** Regions with `track_id = None`:
- With `blur_ids` set: excluded (None not in any set)
- With `exclude_ids` set: included (None not in exclusion set)

### A.7 BlurFacesUseCase

**Default lookahead:** 5 frames
**Thread queue capacity:** 4 (bounded channel)

**Buffer algorithm:**
```
for frame in reader:
    regions = detect(frame)
    filtered = filter_regions(regions, blur_ids, exclude_ids)
    buffer.push_back((frame, filtered))
    if buffer.len() > lookahead:
        flush_oldest()
while !buffer.is_empty():
    flush_oldest()
```

**Flush:**
```
(frame, own_regions) = buffer.pop_front()
lookahead_regions = buffer.iter().map(|(_, r)| r).collect()
merged = merger.merge(own_regions, lookahead_regions, frame_w, frame_h)
blurred = blurrer.blur(frame, merged)
writer.write(blurred)
```

**Cancellation:** Progress callback returns `false` → raise cancellation error

### A.8 PreviewFacesUseCase

**Crop selection:** Largest detection per track_id by area (`width * height`)
**Preview size:** 256x256 thumbnails
**Square crop algorithm:**
```
cx = region.x + region.width / 2
cy = region.y + region.height / 2
half = max(region.width, region.height) / 2
x1 = max(cx - half, 0)
y1 = max(cy - half, 0)
x2 = min(cx + half, frame_w)
y2 = min(cy + half, frame_h)
crop = frame[y1..y2, x1..x2]
```

### A.9 Infrastructure Constants (for reference)

| Constant | Value | Location |
|----------|-------|----------|
| YOLO input size | 640x640 | onnx_yolo_detector |
| YOLO default confidence | 0.25 | onnx_yolo_detector |
| BlazeFace input size | 128x128 | onnx_blazeface_detector |
| BlazeFace default confidence | 0.5 | onnx_blazeface_detector |
| ArcFace input size | 112x112 | embedding_face_grouper |
| ArcFace normalization | (pixel - 127.5) / 127.5 | embedding_face_grouper |
| ArcFace similarity threshold | 0.4 | embedding_face_grouper |
| Histogram similarity threshold | 0.7 | histogram_face_grouper |
| SkipFrameDetector default interval | 2 | skip_frame_detector |
| Blur kernel size default | 201 | cli default |
| Downscale factor | kernel_size // 50 | cpu_elliptical_blurrer |
| Downscaled kernel | kernel_size // scale (forced odd) | cpu_elliptical_blurrer |
| Face card size | 76x76px | desktop face_card |
| Group card stack offset | 3px | desktop face_group_card |
| Flow layout spacing | 8px | desktop flow_layout |
| Flow layout margin | 10px | desktop flow_layout |
