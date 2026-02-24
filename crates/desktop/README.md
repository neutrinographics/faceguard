# video-blur-desktop

Desktop GUI for face detection and blurring in videos and images, built with [iced](https://iced.rs/).

## Purpose

Composition root that wires together `video-blur-core` domain traits with infrastructure implementations and exposes them through an iced-based graphical interface. Contains no business logic — all detection, blurring, and pipeline orchestration lives in the core crate. The desktop crate handles UI state management, background worker coordination, and user interaction.

## Architecture

The app follows iced's **Elm architecture** (Model → Message → Update → View):

```
src/
├── main.rs              App entry point, window configuration (560×440)
├── app.rs               Top-level App struct, Message enum, update/view/subscription
├── settings.rs          Persistent user preferences (JSON to platform config dir)
├── theme.rs             4 color palettes with system theme detection
├── tabs/
│   ├── main_tab.rs      Blur tab: file selection, progress, face thumbnails
│   ├── settings_tab.rs  Settings tab: blur shape, intensity, sensitivity, appearance
│   └── about_tab.rs     About tab: version, privacy statement, permanence warning
├── workers/
│   ├── mod.rs           Worker module exports
│   ├── preview_worker.rs  Background thread for face scanning + grouping
│   ├── blur_worker.rs     Background thread for blur processing
│   └── model_cache.rs     Startup model resolution + ONNX session pre-building
└── widgets/
    └── faces_well.rs    Face thumbnail grid with selection and grouping
```

## Processing States

The app transitions through a linear state machine:

```
Idle → Preparing → Downloading → Scanning → Previewed → Blurring → Complete
                                                                  ↘ Error
```

- **Idle**: No file loaded, waiting for user to select input
- **Preparing**: Input file selected, resolving models
- **Downloading**: ONNX models downloading (with progress)
- **Scanning**: Running face detection across all frames (with frame progress)
- **Previewed**: Faces displayed as thumbnails for selection
- **Blurring**: Applying blur to selected faces (with frame progress)
- **Complete**: Output file written, ready for next job
- **Error**: Recoverable error state with message

## Background Workers

All heavy computation runs on background threads to keep the UI responsive:

- **ModelCache**: Resolves model paths and pre-builds ONNX sessions at startup using a dedicated thread. Callers wait on a `Condvar` until the session is ready. This eliminates cold-start latency on the first blur job.
- **PreviewWorker**: Builds `OnnxYoloDetector` (wrapped in `SkipFrameDetector`), runs `PreviewFacesUseCase`, then groups faces by identity. Prefers the embedding-based grouper when an embedding model is available; falls back to histogram-based grouping.
- **BlurWorker**: Runs `BlurFacesUseCase` via `ThreadedPipelineExecutor`. Reuses the detection cache from the preview scan via `CachedFaceDetector` to avoid redundant inference. Shares a GPU context (`wgpu` device/queue) across jobs.

Workers communicate results back to the UI thread via `crossbeam-channel` senders, which the iced subscription polls.

## Face Selection

`FacesWellState` manages the face thumbnail grid:

- Faces are displayed individually or grouped by identity (when grouping is available)
- Users click thumbnails to toggle selection — selected faces will be blurred
- All faces start selected by default
- Group headers allow selecting/deselecting all faces in an identity cluster
- Thumbnail images are stored in a RAII-managed temp directory that cleans up automatically

## Settings Persistence

User preferences are stored as JSON in the platform config directory:
- macOS: `~/Library/Application Support/video-blur/settings.json`
- Linux: `~/.config/video-blur/settings.json`

Configurable options: blur shape (ellipse/rect), blur intensity, detection sensitivity, lookahead frames, appearance (system/dark/light), high contrast mode, and font scale.

## Theming

Four built-in palettes: dark, light, high-contrast dark, and high-contrast light. System theme is detected on macOS via `defaults read -g AppleInterfaceStyle`. The high-contrast variants increase text/border contrast for accessibility.

## Design Decisions

- **GPU context reuse** — A single `wgpu` device/queue pair is created once and shared across blur jobs, avoiding repeated GPU initialization overhead.
- **Detection cache reuse** — When blurring after a preview scan, the app wraps detections in `CachedFaceDetector` so the blur pipeline replays cached results instead of re-running inference. This makes the blur step nearly instant for detection.
- **RAII temp directories** — Face thumbnails are written to a `tempfile::TempDir` owned by `FacesWellState`. When the state is dropped (new file loaded or app closed), the directory and all thumbnails are cleaned up automatically.
- **Cooperative cancellation** — Workers check an `AtomicBool` cancel flag between frames, allowing the user to abort long-running scans or blur jobs without killing threads.
- **Model pre-loading** — `ModelCache` begins resolving and building ONNX sessions immediately at app startup, hiding model download and initialization latency behind the time the user spends selecting a file.
