# Audio Features Design

## Overview

Add audio anonymization capabilities to FaceGuard: keyword-based word bleeping (via local speech-to-text) and tiered voice disguising (via DSP transforms). Audio processing is optional, video-only, and runs as a second pass after the existing video pipeline.

## Requirements

- **Word bleeping**: User provides a keyword list. A local Whisper model transcribes the audio and matching words are replaced with a bleep tone or silence.
- **Voice disguising**: Three tiers of DSP-based voice transformation (Low/Medium/High), selectable by the user.
- **Offline**: Whisper ONNX model downloaded on first use, same pattern as the YOLO face detection model.
- **Optional**: Disabled by default. Configurable in desktop settings and via CLI flags.
- **Video only**: Audio processing applies to videos with audio tracks. No standalone audio file support.

## Approach: Two-Pass Pipeline

The existing video pipeline is unchanged. After video processing completes, a separate audio pass runs:

1. **Pass 1 (existing)**: Decode video -> detect faces -> blur -> encode video (no audio)
2. **Pass 2 (new)**: Decode audio from source -> transcribe with Whisper -> apply bleeps + voice transform -> encode audio -> mux into output

This avoids touching the battle-tested video pipeline and keeps audio processing independently testable.

If neither keywords nor voice disguise are enabled, the existing audio passthrough path is used (packet copy, no decode/encode).

## Architecture

### New Feature Slices

```
crates/core/src/
  audio/                              # Audio processing feature slice
    domain/
      audio_segment.rs                # AudioSegment entity (PCM samples + sample rate)
      transcript.rs                   # TranscriptWord entity (word, start_time, end_time)
      speech_recognizer.rs            # SpeechRecognizer trait
      audio_transformer.rs            # AudioTransformer trait (voice disguise)
      word_censor.rs                  # WordCensor service (keyword matching + bleep generation)
    infrastructure/
      whisper_recognizer.rs           # Whisper ONNX implementation of SpeechRecognizer
      pitch_shift_transformer.rs      # Low: pitch shift via phase vocoder
      formant_shift_transformer.rs    # Medium: pitch + formant shift via LPC
      voice_morph_transformer.rs      # High: pitch + formant + spectral randomization
  video/
    domain/
      audio_reader.rs                 # AudioReader trait (decode audio from video)
      audio_writer.rs                 # AudioWriter trait (encode audio into video)
    infrastructure/
      ffmpeg_audio_reader.rs          # FFmpeg audio decode to PCM
      ffmpeg_audio_writer.rs          # FFmpeg audio encode + mux
  pipeline/
    process_audio_use_case.rs         # Orchestrates the audio processing pass
```

### Dependency Rule

Same as existing: `infrastructure -> application -> domain`. The `audio/domain/` layer has zero external dependencies. FFmpeg usage is isolated in `video/infrastructure/`.

## Domain Entities

### AudioSegment

The audio equivalent of `Frame`. Holds decoded PCM data.

```rust
#[derive(Clone, Debug)]
pub struct AudioSegment {
    pub samples: Vec<f32>,      // Interleaved PCM, normalized [-1.0, 1.0]
    pub sample_rate: u32,       // e.g., 16000 for Whisper, 44100/48000 for output
    pub channels: u16,          // 1 (mono) or 2 (stereo)
}
```

### TranscriptWord

A recognized word with timing information.

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptWord {
    pub word: String,
    pub start_time: f64,        // seconds
    pub end_time: f64,          // seconds
    pub confidence: f32,        // 0.0-1.0
}
```

### CensorRegion

A time range to bleep, analogous to face `Region`.

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct CensorRegion {
    pub start_time: f64,        // seconds
    pub end_time: f64,          // seconds
    pub padding: f64,           // extra margin in seconds (e.g., 0.05s)
}
```

## Domain Traits

```rust
pub trait SpeechRecognizer: Send {
    fn transcribe(&self, audio: &AudioSegment) -> Result<Vec<TranscriptWord>, Box<dyn Error>>;
}

pub trait AudioTransformer: Send {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn Error>>;
}
```

## Domain Services

### WordCensor

Pure domain logic, no trait needed:

```rust
impl WordCensor {
    /// Match transcript words against keywords, return time ranges to censor.
    pub fn find_censor_regions(
        transcript: &[TranscriptWord],
        keywords: &[String],
        padding: f64,
    ) -> Vec<CensorRegion>;

    /// Replace audio in censor regions with a bleep tone or silence.
    pub fn apply_bleep(
        audio: &mut AudioSegment,
        regions: &[CensorRegion],
        bleep_frequency: f64,    // e.g., 1000.0 Hz
    );
}
```

## Pipeline: ProcessAudioUseCase

```
Source video
    |
    v
AudioReader (FFmpeg)
    | decode audio stream -> AudioSegment (mono, 16kHz for Whisper)
    v
SpeechRecognizer (Whisper)           <- only if keywords provided
    | -> Vec<TranscriptWord>
    v
WordCensor::find_censor_regions()    <- only if keywords provided
    | -> Vec<CensorRegion>
    v
WordCensor::apply_bleep()            <- only if censor regions found
    | mutates AudioSegment
    v
AudioTransformer                     <- only if voice disguise enabled
    | pitch/formant/morph transform
    v
AudioWriter (FFmpeg)
    | encode + mux into output video (replacing passthrough audio)
    v
Done
```

Key decisions:
- Whisper needs mono 16kHz input. Resample at decode time, process at 16kHz, resample back at encode time.
- Bleeping happens before voice transformation (otherwise transform distorts the bleep tone).
- Entire audio loaded into memory as one AudioSegment. 10 min video at 16kHz mono = ~19MB of f32 samples.
- If neither keywords nor voice disguise enabled, existing passthrough path is used.

## Voice Disguise Tiers

All tiers are pure DSP (no ML models). They build on each other incrementally.

### Low: Pitch Shift

- Shift pitch by configurable semitone offset (default: +4 semitones)
- Algorithm: phase vocoder with overlap-add (STFT -> shift bins -> ISTFT)
- Preserves speech intelligibility, changes perceived gender/age
- Constants: `window_size=2048`, `hop_size=512`, `semitones=4.0`

### Medium: Pitch + Formant Shift

- Pitch shift as above, plus formant frequency adjustment
- Formants encode vowel identity and speaker characteristics
- Algorithm: LPC (Linear Predictive Coding) to extract formants -> shift envelope -> resynthesize
- Constants: `lpc_order=16`, `formant_shift_ratio=1.2`

### High: Full Voice Morph

- Pitch + formant shift + spectral envelope randomization
- Adds jitter to break voiceprint patterns
- Makes voice unrecognizable even to forensic analysis
- Constants: `jitter_amount=0.15`, `spectral_smoothing=0.3`

## Whisper Model

- **Model**: Whisper tiny (~75MB) or base (~150MB), user choice in settings
- **Format**: ONNX, same runtime as YOLO face detection model
- **Download**: On first use, cached in app data directory (existing pattern)
- **Capability**: Word-level timestamps for precise bleep placement

## Settings (Desktop)

New settings in an Audio section:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| Audio processing | bool | off | Master toggle |
| Bleep keywords | String | "" | Comma-separated word list |
| Bleep sound | enum | Tone | Tone (1kHz sine) or Silence |
| Voice disguise | enum | Off | Off / Low / Medium / High |

## CLI Flags

```
--audio-keywords "name,address,phone"   # Comma-separated, enables bleeping
--voice-disguise low|medium|high        # Enables voice transform
```

If neither flag is provided, audio passthrough is used (current behavior).

## Integration with Existing Code

### FfmpegWriter Change

Add a flag to skip audio passthrough when audio processing will handle it:

```rust
pub struct FfmpegWriter {
    skip_audio_passthrough: bool,  // NEW
    // ... existing fields
}
```

### Blur Worker Change

After video processing completes, conditionally run the audio pass:

```rust
// Existing: video blur
blur_faces_use_case.run(...)?;

// New: audio processing (if enabled)
if audio_settings.is_enabled() {
    process_audio_use_case.run(source_path, output_path, &audio_settings)?;
}
```

### Model Download

Same infrastructure as face detection model download. The desktop app checks for the Whisper model on startup and downloads if missing.

## Constants

| Constant | Value | Location |
|----------|-------|----------|
| Bleep frequency | 1000.0 Hz | WordCensor |
| Bleep padding | 0.05s | WordCensor |
| STFT window size | 2048 | PitchShiftTransformer |
| STFT hop size | 512 | PitchShiftTransformer |
| Default pitch shift | 4.0 semitones | PitchShiftTransformer |
| LPC order | 16 | FormantShiftTransformer |
| Formant shift ratio | 1.2 | FormantShiftTransformer |
| Spectral jitter | 0.15 | VoiceMorphTransformer |
| Spectral smoothing | 0.3 | VoiceMorphTransformer |
| Whisper sample rate | 16000 Hz | WhisperRecognizer |
