# Whisper Speech Recognition Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the stub `WhisperRecognizer` with real whisper.cpp inference so keyword bleeping actually works.

**Architecture:** Add `whisper-rs` crate (Rust bindings for whisper.cpp), implement `SpeechRecognizer` trait using its API, download the `ggml-tiny.en.bin` model at app startup via the existing `ModelCache` pattern.

**Tech Stack:** `whisper-rs` 0.15, GGML model format, existing `model_resolver` for download

---

### Task 1: Add `whisper-rs` dependency

**Files:**
- Modify: `crates/core/Cargo.toml`

**Step 1: Add the dependency**

Add `whisper-rs = "0.15"` to `[dependencies]` in `crates/core/Cargo.toml`, after the existing `rustfft` entry.

**Step 2: Verify it compiles**

Run: `cargo check -p faceguard-core`
Expected: Compiles successfully (whisper.cpp will be built from source via whisper-rs-sys)

Note: This requires a C/C++ compiler. The project already requires one for ffmpeg.

**Step 3: Commit**

```bash
git add crates/core/Cargo.toml
git commit -m "feat: add whisper-rs dependency for speech recognition"
```

---

### Task 2: Update Whisper constants

**Files:**
- Modify: `crates/core/src/shared/constants.rs`
- Modify: `crates/core/src/audio/infrastructure/whisper_recognizer.rs`

**Step 1: Add model URL constant and rename model filename**

In `crates/core/src/shared/constants.rs`, update the Whisper constants:

```rust
pub const WHISPER_MODEL_NAME: &str = "ggml-tiny.en.bin";
pub const WHISPER_MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin";
pub const WHISPER_SAMPLE_RATE: u32 = 16000;
```

Note: `WHISPER_MODEL_FILENAME` should be renamed to `WHISPER_MODEL_NAME` to match the pattern used by `YOLO_MODEL_NAME`. If `WHISPER_MODEL_FILENAME` already exists, rename it.

**Step 2: Remove duplicate constant from whisper_recognizer.rs**

In `crates/core/src/audio/infrastructure/whisper_recognizer.rs`, remove the local `pub const WHISPER_SAMPLE_RATE: u32 = 16000;` and import it from `shared::constants` instead.

**Step 3: Verify compilation**

Run: `cargo check -p faceguard-core`
Expected: Compiles. Any existing references to the old constant name will be caught as compile errors — fix them.

**Step 4: Commit**

```bash
git add crates/core/src/shared/constants.rs crates/core/src/audio/infrastructure/whisper_recognizer.rs
git commit -m "feat: add Whisper model URL constant, deduplicate WHISPER_SAMPLE_RATE"
```

---

### Task 3: Implement WhisperRecognizer with real inference

**Files:**
- Modify: `crates/core/src/audio/infrastructure/whisper_recognizer.rs`

This is the core task. The existing file has a stub `transcribe()` that returns an empty vec. Replace it with real whisper-rs inference.

**Step 1: Write/update the failing test**

Add a test that verifies `transcribe()` returns words with timestamps when given real audio. This test requires the model file, so mark it `#[ignore]`.

```rust
#[test]
#[ignore] // Requires whisper model file
fn test_transcribe_returns_words_with_timestamps() {
    use crate::shared::constants::WHISPER_MODEL_NAME;

    // Resolve the model (downloads if needed)
    let model_path = crate::detection::infrastructure::model_resolver::resolve(
        WHISPER_MODEL_NAME,
        crate::shared::constants::WHISPER_MODEL_URL,
        None,
        None,
    )
    .expect("Failed to resolve whisper model");

    let recognizer = WhisperRecognizer::new(&model_path).expect("Failed to create recognizer");

    // Generate a 3-second sine wave (won't contain real speech, but should not crash)
    let sample_rate = 16000u32;
    let duration = 3.0;
    let len = (duration * sample_rate as f64) as usize;
    let samples: Vec<f32> = (0..len)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32
        })
        .collect();
    let audio = AudioSegment::new(samples, sample_rate, 1);

    let result = recognizer.transcribe(&audio);
    assert!(result.is_ok(), "Transcription should not error");
    // With a sine wave, transcript may be empty or contain hallucinated words — either is fine.
    // The important thing is that it doesn't crash and returns a valid Vec.
}
```

**Step 2: Implement WhisperRecognizer**

Replace the entire `whisper_recognizer.rs` with:

```rust
use std::path::{Path, PathBuf};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::speech_recognizer::SpeechRecognizer;
use crate::audio::domain::transcript::TranscriptWord;

/// Speech recognizer using whisper.cpp via whisper-rs.
///
/// Transcribes audio to word-level timestamped text using the Whisper tiny.en model.
pub struct WhisperRecognizer {
    model_path: PathBuf,
}

impl WhisperRecognizer {
    pub fn new(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        if !model_path.exists() {
            return Err(format!(
                "Whisper model not found at: {}",
                model_path.display()
            )
            .into());
        }
        Ok(Self {
            model_path: model_path.to_path_buf(),
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}

impl SpeechRecognizer for WhisperRecognizer {
    fn transcribe(
        &self,
        audio: &AudioSegment,
    ) -> Result<Vec<TranscriptWord>, Box<dyn std::error::Error>> {
        let ctx = WhisperContext::new_with_params(
            self.model_path.to_str().ok_or("Invalid model path")?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| format!("Failed to load Whisper model: {e}"))?;

        let mut state = ctx
            .create_state()
            .map_err(|e| format!("Failed to create Whisper state: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        params.set_language(Some("en"));
        params.set_translate(false);
        params.set_token_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_n_threads(num_cpus().min(4) as i32);

        let samples = audio.samples();
        state
            .full(params, samples)
            .map_err(|e| format!("Whisper inference failed: {e}"))?;

        let mut words = Vec::new();
        let num_segments = state
            .full_n_segments()
            .map_err(|e| format!("Failed to get segments: {e}"))?;

        for seg_idx in 0..num_segments {
            let n_tokens = state
                .full_n_tokens(seg_idx)
                .map_err(|e| format!("Failed to get token count: {e}"))?;

            for tok_idx in 0..n_tokens {
                let text = match state.full_get_token_text(seg_idx, tok_idx) {
                    Ok(t) => t,
                    Err(_) => continue,
                };

                // Skip special tokens (start with [, like [_BEG_], [_SOT_], etc.)
                let trimmed = text.trim();
                if trimmed.is_empty()
                    || trimmed.starts_with('[')
                    || trimmed.starts_with('<')
                {
                    continue;
                }

                let token_data = match state.full_get_token_data(seg_idx, tok_idx) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let prob = match state.full_get_token_p(seg_idx, tok_idx) {
                    Ok(p) => p,
                    Err(_) => 0.0,
                };

                // Token timestamps are in centiseconds (10ms units)
                let start_time = token_data.t0 as f64 / 100.0;
                let end_time = token_data.t1 as f64 / 100.0;

                // Skip tokens with invalid timestamps
                if end_time <= start_time {
                    continue;
                }

                words.push(TranscriptWord {
                    word: trimmed.to_string(),
                    start_time,
                    end_time,
                    confidence: prob,
                });
            }
        }

        Ok(words)
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_nonexistent_path_returns_error() {
        let result = WhisperRecognizer::new(std::path::Path::new("/nonexistent/model.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn test_new_nonexistent_path_error_message() {
        let result = WhisperRecognizer::new(std::path::Path::new("/nonexistent/model.bin"));
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found"),
            "Expected 'not found' in error, got: {err}"
        );
    }

    #[test]
    #[ignore] // Requires whisper model file
    fn test_transcribe_does_not_crash_on_sine_wave() {
        let model_path = crate::detection::infrastructure::model_resolver::resolve(
            crate::shared::constants::WHISPER_MODEL_NAME,
            crate::shared::constants::WHISPER_MODEL_URL,
            None,
            None,
        )
        .expect("Failed to resolve whisper model");

        let recognizer = WhisperRecognizer::new(&model_path).expect("Failed to create recognizer");

        let sample_rate = 16000u32;
        let len = (3.0 * sample_rate as f64) as usize;
        let samples: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32
            })
            .collect();
        let audio = AudioSegment::new(samples, sample_rate, 1);

        let result = recognizer.transcribe(&audio);
        assert!(result.is_ok(), "Transcription should not error: {result:?}");
    }
}
```

Key implementation notes:
- The `WhisperContext` is created per-call in `transcribe()`. This is intentional — it keeps the struct `Send`-safe and avoids lifetime issues. For a ~75MB model, context creation takes ~100ms which is negligible vs. inference time.
- Token timestamps (`t0`, `t1`) come from `full_get_token_data()`. These are in centiseconds (10ms units), so divide by 100.0 for seconds.
- Special tokens (like `[_BEG_]`, `[_SOT_]`) are filtered out.
- Thread count is capped at 4 to avoid oversubscribing on many-core machines.

**Step 3: Run tests**

Run: `cargo test -p faceguard-core -- whisper_recognizer`
Expected: Non-ignored tests pass. The `#[ignore]` test can be run manually with `cargo test -p faceguard-core -- whisper_recognizer --ignored` (requires model download).

**Step 4: Commit**

```bash
git add crates/core/src/audio/infrastructure/whisper_recognizer.rs
git commit -m "feat: implement WhisperRecognizer with whisper-rs inference"
```

---

### Task 4: Add Whisper model to ModelCache (desktop)

**Files:**
- Modify: `crates/desktop/src/workers/model_cache.rs`

**Step 1: Add whisper_path slot and resolve at startup**

Add a `whisper_path: Arc<ModelSlot>` field to `ModelCache`. In `ModelCache::new()`, resolve it in the background thread after the embedding model. Add a `wait_for_whisper()` method.

```rust
// In the imports, add:
use faceguard_core::shared::constants::{
    EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_URL, WHISPER_MODEL_NAME, WHISPER_MODEL_URL,
    YOLO_MODEL_NAME, YOLO_MODEL_URL,
};

// Add field to ModelCache struct:
pub struct ModelCache {
    yolo_path: Arc<ModelSlot>,
    embedding_path: Arc<ModelSlot>,
    whisper_path: Arc<ModelSlot>,
    yolo_session: Arc<SessionSlot>,
}

// In ModelCache::new(), add to constructor and background thread:
let cache = Arc::new(Self {
    yolo_path: Arc::new(ModelSlot::new()),
    embedding_path: Arc::new(ModelSlot::new()),
    whisper_path: Arc::new(ModelSlot::new()),
    yolo_session: Arc::new(SessionSlot::new()),
});

let whisper_path_slot = cache.whisper_path.clone();

// In the background thread, after embedding resolve:
whisper_path_slot.resolve(WHISPER_MODEL_NAME, WHISPER_MODEL_URL);

// Add wait method:
pub fn wait_for_whisper(
    &self,
    on_progress: &dyn Fn(u64, u64),
    cancelled: &AtomicBool,
) -> Result<PathBuf, String> {
    self.whisper_path.wait(on_progress, cancelled)
}
```

**Step 2: Verify compilation**

Run: `cargo check -p faceguard-desktop`
Expected: Compiles

**Step 3: Commit**

```bash
git add crates/desktop/src/workers/model_cache.rs
git commit -m "feat: add Whisper model download to ModelCache startup"
```

---

### Task 5: Wire WhisperRecognizer into desktop blur worker

**Files:**
- Modify: `crates/desktop/src/workers/blur_worker.rs`

**Step 1: Construct WhisperRecognizer when keywords are present**

In the `run_audio_processing()` function, replace the hardcoded `None` recognizer with a real `WhisperRecognizer` when keywords are non-empty.

```rust
// In the imports section of run_audio_processing(), add:
use faceguard_core::audio::infrastructure::whisper_recognizer::WhisperRecognizer;

// Replace the existing recognizer setup:
// OLD:
// let recognizer: Option<Box<dyn ...>> = None;

// NEW:
let recognizer: Option<
    Box<dyn faceguard_core::audio::domain::speech_recognizer::SpeechRecognizer>,
> = if !keywords.is_empty() {
    match params.model_cache.wait_for_whisper(&|_, _| {}, &AtomicBool::new(false)) {
        Ok(model_path) => match WhisperRecognizer::new(&model_path) {
            Ok(r) => Some(Box::new(r)),
            Err(e) => {
                log::warn!("Failed to create WhisperRecognizer: {e}");
                None
            }
        },
        Err(e) => {
            log::warn!("Whisper model not available: {e}");
            None
        }
    }
} else {
    None
};
```

Note: The `AtomicBool::new(false)` is used because audio processing happens after video blur completes — if the user cancelled, we'd already have returned. If you want the whisper wait to be cancellable, thread the existing `cancelled` flag through from `run_blur` to `run_audio_processing`.

**Step 2: Verify compilation**

Run: `cargo check -p faceguard-desktop`
Expected: Compiles

**Step 3: Commit**

```bash
git add crates/desktop/src/workers/blur_worker.rs
git commit -m "feat: wire WhisperRecognizer into desktop audio processing"
```

---

### Task 6: Wire WhisperRecognizer into CLI

**Files:**
- Modify: `crates/cli/src/main.rs`

**Step 1: Construct WhisperRecognizer when --audio-keywords is provided**

In `run_video_blur()`, replace the `None` recognizer with a real one:

```rust
// Replace:
// None, // recognizer (Whisper not yet wired in)

// With:
let recognizer: Option<
    Box<dyn faceguard_core::audio::domain::speech_recognizer::SpeechRecognizer>,
> = if !keywords.is_empty() {
    use faceguard_core::audio::infrastructure::whisper_recognizer::WhisperRecognizer;
    use faceguard_core::shared::constants::{WHISPER_MODEL_NAME, WHISPER_MODEL_URL};

    log::info!("Resolving Whisper model: {WHISPER_MODEL_NAME}");
    let whisper_path = faceguard_core::detection::infrastructure::model_resolver::resolve(
        WHISPER_MODEL_NAME,
        WHISPER_MODEL_URL,
        None,
        Some(Box::new(|downloaded, total| {
            if total > 0 {
                let pct = (downloaded as f64 / total as f64 * 100.0) as u32;
                eprint!("\rDownloading speech recognition model... {pct}%");
            }
        })),
    )?;
    eprintln!();
    Some(Box::new(WhisperRecognizer::new(&whisper_path)?))
} else {
    None
};
```

Then pass `recognizer` to `ProcessAudioUseCase::new()` instead of `None`.

**Step 2: Verify compilation**

Run: `cargo check -p faceguard-cli`
Expected: Compiles

**Step 3: Commit**

```bash
git add crates/cli/src/main.rs
git commit -m "feat: wire WhisperRecognizer into CLI audio processing"
```

---

### Task 7: Final verification

**Step 1: Run all tests**

Run: `cargo test`
Expected: All tests pass (325+ tests)

**Step 2: Run clippy**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: No warnings

**Step 3: Run fmt check**

Run: `cargo fmt --check`
Expected: No formatting issues

**Step 4: Run the ignored whisper test (optional, requires download)**

Run: `cargo test -p faceguard-core -- whisper_recognizer --ignored`
Expected: Model downloads, inference runs, test passes

**Step 5: Commit any fixes**

If any fixes were needed:
```bash
git add -A
git commit -m "fix: address clippy/fmt issues in whisper integration"
```
