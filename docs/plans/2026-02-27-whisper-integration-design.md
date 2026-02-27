# Whisper Speech Recognition Integration Design

## Goal

Add local speech-to-text using whisper.cpp so the keyword bleeping feature actually works. Currently the `WhisperRecognizer` is a stub that returns an empty transcript — this design replaces it with real inference.

## Decisions

- **Inference engine:** `whisper-rs` (Rust bindings for whisper.cpp). Provides word-level timestamps out of the box, uses GGML model format, battle-tested.
- **Model:** `ggml-tiny.en.bin` (~75 MB). English-only, fastest inference. Sufficient for keyword matching. Multilingual support can be added later by swapping to `ggml-tiny.bin` with a language setting.
- **Model hosting:** Hugging Face (`https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin`). No practical rate limits for single-file downloads.
- **Download strategy:** Background download at app startup alongside YOLO and embedding models, using the existing `ModelCache` / `model_resolver` pattern. Blocks at point of use if not yet ready.

## Architecture

The existing pipeline is fully wired — `SpeechRecognizer` trait, `ProcessAudioUseCase`, `WordCensor`, `AudioSegment`, `TranscriptWord` all work. Only the recognizer implementation and model download need to be filled in.

### Data Flow

1. App startup: `ModelCache` downloads `ggml-tiny.en.bin` from Hugging Face in background thread
2. User exports video with keywords: blur worker calls `wait_for_whisper()` to get model path
3. `WhisperRecognizer::new(path)` creates a `whisper_rs::WhisperContext`
4. `ProcessAudioUseCase` reads audio at 16 kHz mono, calls `recognizer.transcribe(&audio)`
5. `WhisperRecognizer` runs inference with word-level timestamps, returns `Vec<TranscriptWord>`
6. `WordCensor` matches transcript against keywords, applies bleep/silence
7. Rest of pipeline continues unchanged

## Components & Changes

### New dependency

`whisper-rs` added to `crates/core/Cargo.toml`.

### Files modified

1. **`shared/constants.rs`** — Add `WHISPER_MODEL_NAME` and `WHISPER_MODEL_URL`. Remove duplicate `WHISPER_SAMPLE_RATE` from `whisper_recognizer.rs`.

2. **`audio/infrastructure/whisper_recognizer.rs`** — Replace stub with real inference:
   - `new()` creates `whisper_rs::WhisperContext` from model path
   - `transcribe()` runs `full()` with `FullParams` configured for word-level timestamps (`set_token_timestamps(true)`), maps segments/tokens to `Vec<TranscriptWord>`

3. **`desktop/src/workers/model_cache.rs`** — Add `whisper_path: Arc<ModelSlot>` alongside existing slots. Resolve in startup thread. Add `wait_for_whisper()` method.

4. **`desktop/src/workers/blur_worker.rs`** — When keywords are non-empty and whisper model is available, construct `WhisperRecognizer` and pass as `Some(recognizer)` to `ProcessAudioUseCase`.

5. **`cli/src/main.rs`** — Resolve whisper model, construct `WhisperRecognizer` when `--audio-keywords` is provided.

### Files unchanged

`SpeechRecognizer` trait, `ProcessAudioUseCase`, `WordCensor`, `AudioSegment`, `TranscriptWord`, desktop settings UI — all already wired up correctly.

## Error Handling

- **No audio track:** Already handled — `ProcessAudioUseCase` returns `Ok(())`.
- **Model not ready at export time:** Blur worker blocks via `wait_for_whisper()` until download completes. If download failed, passes `None` for recognizer — audio passes through without bleeping (graceful degradation).
- **Inference failure:** `transcribe()` returns `Result` — errors propagate to blur worker, reported via `WorkerMessage::Error`.
- **Empty transcript / no matches:** Already handled — `find_censor_regions` returns empty vec, `apply_bleep` is a no-op.
- **Long audio:** whisper.cpp chunks internally in 30-second windows. No special handling needed.

## Testing

- Existing `WhisperRecognizer` tests (path validation, construction) updated for new constructor.
- Real inference test marked `#[ignore]` — requires model file, validates word timestamps are produced.
- All `ProcessAudioUseCase` tests use stub recognizers, remain unchanged.

## Future Extensibility

- **Language support:** Swap `ggml-tiny.en.bin` for `ggml-tiny.bin` (multilingual, same size). Add language dropdown in settings, pass language code to `FullParams::set_language()`. No architectural changes needed.
- **Larger models:** Same download/cache pattern, just change the constant URL and filename.
