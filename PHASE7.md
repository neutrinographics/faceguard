# Phase 7: Desktop GUI

## Context

Phases 1-6 are complete (269 tests, CLI working). The desktop crate (`crates/desktop/`) has only a stub `main.rs`. This phase builds a full GUI app using **iced 0.14** that replicates the Python PySide6 desktop app. The Python reference is at `/Users/joel/git/da1nerd/video-blur/packages/desktop/src/video_blur_desktop/` (13 files, ~2,500 lines).

This is large — broken into 9 sub-phases, each independently buildable.

## Dependencies to Add

**Workspace `Cargo.toml`** — new deps:
```toml
iced = { version = "0.14", features = ["image"] }
rfd = "0.15"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
open = "5"
```

**`crates/desktop/Cargo.toml`**:
```toml
[dependencies]
video-blur-core = { workspace = true }
iced = { workspace = true }
rfd = "0.15"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
dirs = { workspace = true }
open = "5"
log = { workspace = true }
env_logger = { workspace = true }
crossbeam-channel = { workspace = true }
image = { workspace = true }
```

## File Structure

```
crates/desktop/src/
├── main.rs              — Entry point
├── app.rs               — App state, Message enum, update(), view(), subscription()
├── theme.rs             — 4 palettes (dark/light × normal/high-contrast)
├── settings.rs          — Settings struct, serde JSON load/save
├── tabs/
│   ├── mod.rs
│   ├── main_tab.rs      — File selection, preview, run/cancel, progress
│   ├── settings_tab.rs  — Detection/blur config
│   ├── accessibility_tab.rs — Appearance and contrast
│   ├── privacy_tab.rs   — Static text
│   └── about_tab.rs     — Branding
├── widgets/
│   ├── mod.rs
│   ├── face_card.rs     — 76x76 thumbnail with selection
│   ├── face_group_card.rs — Stacked group card
│   └── faces_well.rs    — Grid container with group toggle
└── workers/
    ├── mod.rs
    ├── preview_worker.rs — Detect + group + save crops
    └── blur_worker.rs    — Apply blur
```

## Sub-Phases

### 7A: Skeleton App with Tab Navigation
- `main.rs`: `iced::application()` setup, window 520×360, title "Video Blur — Neutrino Graphics"
- `app.rs`: `App` struct, `Tab` enum (Main/Settings/Accessibility/Privacy/About), `Message::TabSelected`
- `tabs/privacy_tab.rs`: Static text (simplest tab, validates pattern)
- `tabs/about_tab.rs`: App name, version, company, website link
- `tabs/mod.rs`
- Footer: "neutrinographics.com" link → `open::that()`
- **Verify**: `cargo run -p video-blur-desktop` shows window with 5 tabs

### 7B: Theme System + Settings Persistence
- `theme.rs`: 4 color palettes matching Python `theme.py`. Dark/Light/System detection (macOS: `defaults read -g AppleInterfaceStyle`). Map to `iced::Theme::custom()`
- `settings.rs`: `Settings` struct with serde. Fields: detector, blur_shape, confidence (10-100), blur_strength (51-401), lookahead (0-30), appearance, high_contrast. Load/save to `dirs::config_dir()/Video Blur/settings.json`
- Wire into `App`: `App::theme()` uses settings
- **Verify**: Theme changes on settings update, persists across restarts

### 7C: Settings Tab + Accessibility Tab
- `tabs/settings_tab.rs`: detector pick_list, blur_shape pick_list, confidence slider (display "0.XX"), blur_strength slider (odd), lookahead slider, Restore Defaults button
- `tabs/accessibility_tab.rs`: appearance pick_list, high_contrast checkbox
- Messages: `DetectorChanged`, `BlurShapeChanged`, `ConfidenceChanged`, `BlurStrengthChanged`, `LookaheadChanged`, `RestoreDefaults`, `AppearanceChanged`, `HighContrastChanged`
- Auto-save on every change
- **Verify**: All controls work, persist, Restore Defaults resets

### 7D: Main Tab — File Selection
- `tabs/main_tab.rs`: Input/output file rows with browse buttons, auto-generated output path (`{stem}_blurred{ext}`)
- `rfd` for native file dialogs (async via `Task::perform`)
- Controls panel hidden until input selected
- Messages: `SelectInput`, `InputSelected`, `SelectOutput`, `OutputSelected`
- **Verify**: Browse opens native dialog, output auto-generates, controls reveal

### 7E: Background Workers + Progress
- `workers/preview_worker.rs`: Spawn thread, resolve model, build detector, run `PreviewFacesUseCase`, group faces (embedding → histogram fallback), send progress via crossbeam channel
- `workers/blur_worker.rs`: Spawn thread, build detector/blurrer/reader/writer, run `BlurFacesUseCase` or `BlurImageUseCase`, send progress
- Wiring follows CLI `main.rs` pattern
- `App::subscription()`: Poll crossbeam receiver, map `WorkerMessage` to `Message`
- Cancellation via `Arc<AtomicBool>`
- Messages: `RunPreview`, `RunBlur`, `Cancel`, `DownloadProgress`, `ScanProgress`, `BlurProgress`, `PreviewComplete`, `BlurComplete`, `WorkerError`, `WorkerCancelled`
- Progress bar in main tab, download progress display
- **Verify**: Preview runs, shows progress. Blur runs, shows progress with ETA

### 7F: Face Card Widgets + Faces Well
- `widgets/face_card.rs`: 76×76 thumbnail, selected/deselected states, click to toggle
- `widgets/face_group_card.rs`: Stacked card, count badge "x{N}", click toggles group
- `widgets/faces_well.rs`: Header ("Group similar faces" checkbox + count), separator, scrollable grid using manual row-based wrapping (76px cards + 8px spacing)
- Load crop JPEGs as `iced::widget::image::Handle`
- State: `face_selected: HashSet<u32>`, `group_faces: bool`
- Messages: `ToggleFace(u32)`, `ToggleGroup(usize)`, `GroupFacesToggled(bool)`
- **Verify**: Face cards render after preview, selection toggles, grouping works, count updates

### 7G: Completion Dialog
- Overlay after blur completes: "Processing complete!", OK + "Show in Folder" buttons
- `open::that()` on output directory for Show in Folder
- **Verify**: Dialog appears, Show in Folder opens Finder

### 7H: Visual Polish
- Custom button styles (browse, run, cancel, tab)
- Custom container styles (file rows, face grid, progress bar)
- Spacing/padding matching Python layout
- Typography sizes matching Python
- **Verify**: Visual comparison with Python app

### 7I: Edge Cases + Cleanup
- Temp directory cleanup on close/new input
- Detection invalidation when settings change after preview
- Cancel behavior (reset UI)
- Error display
- Image vs video detection (by extension)
- `get_selected_ids()`: None if all selected, Some(set) if partial
- **Verify**: Full workflow end-to-end, cancel, error handling, settings change after preview

## Execution Order

```
7A → 7B → 7C → 7D → 7E → 7F → 7G → 7H → 7I
```

Each sub-phase produces a buildable binary. 7A-7D are UI-only. 7E-7F connect to core. 7G-7I are polish.

## Reference Files
- CLI wiring pattern: `/Users/joel/git/da1nerd/blur/crates/cli/src/main.rs`
- Python main_window: `/Users/joel/git/da1nerd/video-blur/packages/desktop/src/video_blur_desktop/main_window.py`
- Python theme: `.../theme.py`
- Python settings_tab: `.../settings_tab.py`
- Python worker: `.../worker.py`
- Python face_card: `.../face_card.py`
- Python faces_well: `.../faces_well.py`

## Verification
```bash
cargo build -p video-blur-desktop
cargo run -p video-blur-desktop
cargo clippy -p video-blur-desktop -- -D warnings
cargo fmt -p video-blur-desktop --check
```
