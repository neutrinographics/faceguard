# Audio Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add keyword-based word bleeping and tiered voice disguising to FaceGuard videos.

**Architecture:** Two-pass pipeline — existing video blur runs first, then a separate audio pass decodes, transcribes (Whisper ONNX), censors keywords, applies DSP voice transforms, and muxes the processed audio into the output. Audio processing is optional and disabled by default.

**Tech Stack:** Rust, ONNX Runtime (Whisper), FFmpeg (audio decode/encode via ffmpeg-next), DSP (phase vocoder, LPC), iced (desktop GUI), clap (CLI).

**Design doc:** `docs/plans/2026-02-26-audio-features-design.md`

---

## Phase 1: Domain Entities & Word Censoring (Pure Logic)

### Task 1: AudioSegment Entity

**Files:**
- Create: `crates/core/src/audio/domain/audio_segment.rs`
- Create: `crates/core/src/audio/domain/mod.rs`
- Create: `crates/core/src/audio/mod.rs`
- Modify: `crates/core/src/lib.rs`

**Step 1: Write the failing test**

In `audio_segment.rs`, add a test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_segment_with_correct_fields() {
        let samples = vec![0.0f32; 16000];
        let seg = AudioSegment::new(samples.clone(), 16000, 1);
        assert_eq!(seg.samples(), &samples[..]);
        assert_eq!(seg.sample_rate(), 16000);
        assert_eq!(seg.channels(), 1);
    }

    #[test]
    fn test_duration_mono() {
        let seg = AudioSegment::new(vec![0.0; 48000], 16000, 1);
        assert_eq!(seg.duration(), 3.0);
    }

    #[test]
    fn test_duration_stereo() {
        let seg = AudioSegment::new(vec![0.0; 96000], 48000, 2);
        assert_eq!(seg.duration(), 1.0);
    }

    #[test]
    fn test_sample_at_time() {
        // 16kHz mono, 1 second
        let mut samples = vec![0.0f32; 16000];
        samples[8000] = 0.5;
        let seg = AudioSegment::new(samples, 16000, 1);
        assert_eq!(seg.sample_index_at_time(0.5), 8000);
    }

    #[test]
    fn test_samples_mut() {
        let mut seg = AudioSegment::new(vec![0.0; 100], 16000, 1);
        seg.samples_mut()[50] = 1.0;
        assert_eq!(seg.samples()[50], 1.0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p faceguard-core -- audio_segment`
Expected: FAIL — module `audio` not found

**Step 3: Write minimal implementation**

```rust
/// A segment of decoded audio: interleaved PCM samples normalized to [-1.0, 1.0].
#[derive(Clone, Debug)]
pub struct AudioSegment {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

impl AudioSegment {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self { samples, sample_rate, channels }
    }

    pub fn samples(&self) -> &[f32] { &self.samples }
    pub fn samples_mut(&mut self) -> &mut [f32] { &mut self.samples }
    pub fn sample_rate(&self) -> u32 { self.sample_rate }
    pub fn channels(&self) -> u16 { self.channels }

    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
    }

    pub fn sample_index_at_time(&self, time: f64) -> usize {
        (time * self.sample_rate as f64 * self.channels as f64) as usize
    }
}
```

Register modules:
- `crates/core/src/audio/mod.rs`: `pub mod domain;`
- `crates/core/src/audio/domain/mod.rs`: `pub mod audio_segment;`
- `crates/core/src/lib.rs`: add `pub mod audio;`

**Step 4: Run test to verify it passes**

Run: `cargo test -p faceguard-core -- audio_segment`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add crates/core/src/audio/ crates/core/src/lib.rs
git commit -m "feat(audio): add AudioSegment domain entity"
```

---

### Task 2: TranscriptWord and CensorRegion Entities

**Files:**
- Create: `crates/core/src/audio/domain/transcript.rs`
- Create: `crates/core/src/audio/domain/censor_region.rs`
- Modify: `crates/core/src/audio/domain/mod.rs`

**Step 1: Write the failing test**

In `transcript.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_word_fields() {
        let w = TranscriptWord {
            word: "hello".to_string(),
            start_time: 1.0,
            end_time: 1.5,
            confidence: 0.95,
        };
        assert_eq!(w.word, "hello");
        assert_eq!(w.start_time, 1.0);
        assert_eq!(w.end_time, 1.5);
        assert_eq!(w.confidence, 0.95);
    }

    #[test]
    fn test_transcript_word_duration() {
        let w = TranscriptWord {
            word: "test".to_string(),
            start_time: 2.0,
            end_time: 2.8,
            confidence: 0.9,
        };
        assert_relative_eq!(w.duration(), 0.8, epsilon = 0.001);
    }
}
```

In `censor_region.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_censor_region_effective_range() {
        let r = CensorRegion {
            start_time: 1.0,
            end_time: 2.0,
            padding: 0.05,
        };
        assert_relative_eq!(r.effective_start(), 0.95);
        assert_relative_eq!(r.effective_end(), 2.05);
    }

    #[test]
    fn test_censor_region_effective_start_clamps_to_zero() {
        let r = CensorRegion {
            start_time: 0.02,
            end_time: 0.5,
            padding: 0.05,
        };
        assert_relative_eq!(r.effective_start(), 0.0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p faceguard-core -- transcript censor_region`
Expected: FAIL — modules not found

**Step 3: Write minimal implementation**

`transcript.rs`:
```rust
use approx::assert_relative_eq;

#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptWord {
    pub word: String,
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: f32,
}

impl TranscriptWord {
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }
}
```

`censor_region.rs`:
```rust
#[derive(Clone, Debug, PartialEq)]
pub struct CensorRegion {
    pub start_time: f64,
    pub end_time: f64,
    pub padding: f64,
}

impl CensorRegion {
    pub fn effective_start(&self) -> f64 {
        (self.start_time - self.padding).max(0.0)
    }

    pub fn effective_end(&self) -> f64 {
        self.end_time + self.padding
    }
}
```

Update `crates/core/src/audio/domain/mod.rs`:
```rust
pub mod audio_segment;
pub mod censor_region;
pub mod transcript;
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p faceguard-core -- transcript censor_region`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add crates/core/src/audio/domain/
git commit -m "feat(audio): add TranscriptWord and CensorRegion entities"
```

---

### Task 3: WordCensor Service — find_censor_regions

**Files:**
- Create: `crates/core/src/audio/domain/word_censor.rs`
- Modify: `crates/core/src/audio/domain/mod.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::domain::transcript::TranscriptWord;

    fn word(w: &str, start: f64, end: f64) -> TranscriptWord {
        TranscriptWord {
            word: w.to_string(),
            start_time: start,
            end_time: end,
            confidence: 0.9,
        }
    }

    #[test]
    fn test_find_no_keywords_returns_empty() {
        let transcript = vec![word("hello", 0.0, 0.5), word("world", 0.5, 1.0)];
        let regions = WordCensor::find_censor_regions(&transcript, &[], DEFAULT_BLEEP_PADDING);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_find_matching_keyword() {
        let transcript = vec![
            word("my", 0.0, 0.3),
            word("name", 0.3, 0.6),
            word("is", 0.6, 0.8),
            word("john", 0.8, 1.2),
        ];
        let keywords = vec!["john".to_string()];
        let regions = WordCensor::find_censor_regions(&transcript, &keywords, 0.05);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start_time, 0.8);
        assert_eq!(regions[0].end_time, 1.2);
    }

    #[test]
    fn test_find_case_insensitive() {
        let transcript = vec![word("John", 1.0, 1.5)];
        let keywords = vec!["john".to_string()];
        let regions = WordCensor::find_censor_regions(&transcript, &keywords, 0.05);
        assert_eq!(regions.len(), 1);
    }

    #[test]
    fn test_find_multiple_matches() {
        let transcript = vec![
            word("call", 0.0, 0.3),
            word("john", 0.3, 0.6),
            word("or", 0.6, 0.8),
            word("jane", 0.8, 1.2),
        ];
        let keywords = vec!["john".to_string(), "jane".to_string()];
        let regions = WordCensor::find_censor_regions(&transcript, &keywords, 0.05);
        assert_eq!(regions.len(), 2);
    }

    #[test]
    fn test_find_no_matches() {
        let transcript = vec![word("hello", 0.0, 0.5)];
        let keywords = vec!["goodbye".to_string()];
        let regions = WordCensor::find_censor_regions(&transcript, &keywords, 0.05);
        assert!(regions.is_empty());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p faceguard-core -- word_censor`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```rust
use super::censor_region::CensorRegion;
use super::transcript::TranscriptWord;

pub const DEFAULT_BLEEP_PADDING: f64 = 0.05;
pub const DEFAULT_BLEEP_FREQUENCY: f64 = 1000.0;

pub struct WordCensor;

impl WordCensor {
    pub fn find_censor_regions(
        transcript: &[TranscriptWord],
        keywords: &[String],
        padding: f64,
    ) -> Vec<CensorRegion> {
        if keywords.is_empty() {
            return Vec::new();
        }

        let lower_keywords: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();

        transcript
            .iter()
            .filter(|w| lower_keywords.contains(&w.word.to_lowercase()))
            .map(|w| CensorRegion {
                start_time: w.start_time,
                end_time: w.end_time,
                padding,
            })
            .collect()
    }
}
```

Register in `mod.rs`: `pub mod word_censor;`

**Step 4: Run test to verify it passes**

Run: `cargo test -p faceguard-core -- word_censor`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add crates/core/src/audio/domain/
git commit -m "feat(audio): add WordCensor::find_censor_regions"
```

---

### Task 4: WordCensor Service — apply_bleep

**Files:**
- Modify: `crates/core/src/audio/domain/word_censor.rs`

**Step 1: Write the failing test**

Add to the existing test module in `word_censor.rs`:

```rust
    use crate::audio::domain::audio_segment::AudioSegment;
    use crate::audio::domain::censor_region::CensorRegion;
    use approx::assert_relative_eq;

    fn silent_segment(duration_secs: f64, sample_rate: u32) -> AudioSegment {
        let len = (duration_secs * sample_rate as f64) as usize;
        AudioSegment::new(vec![0.0; len], sample_rate, 1)
    }

    #[test]
    fn test_apply_bleep_tone_replaces_region() {
        let mut audio = silent_segment(2.0, 16000);
        let regions = vec![CensorRegion {
            start_time: 0.5,
            end_time: 1.0,
            padding: 0.0,
        }];
        WordCensor::apply_bleep(&mut audio, &regions, DEFAULT_BLEEP_FREQUENCY);

        // Samples in bleep region should be non-zero (sine wave)
        let start = audio.sample_index_at_time(0.5);
        let end = audio.sample_index_at_time(1.0);
        let bleep_energy: f64 = audio.samples()[start..end]
            .iter()
            .map(|s| (*s as f64) * (*s as f64))
            .sum();
        assert!(bleep_energy > 0.0, "Bleep region should have non-zero energy");
    }

    #[test]
    fn test_apply_bleep_leaves_non_region_untouched() {
        let mut audio = silent_segment(2.0, 16000);
        let regions = vec![CensorRegion {
            start_time: 0.5,
            end_time: 1.0,
            padding: 0.0,
        }];
        WordCensor::apply_bleep(&mut audio, &regions, DEFAULT_BLEEP_FREQUENCY);

        // Samples outside bleep region should still be zero
        let before_energy: f64 = audio.samples()[0..8000]
            .iter()
            .map(|s| (*s as f64) * (*s as f64))
            .sum();
        assert_relative_eq!(before_energy, 0.0);
    }

    #[test]
    fn test_apply_bleep_with_padding() {
        let mut audio = silent_segment(2.0, 16000);
        let regions = vec![CensorRegion {
            start_time: 1.0,
            end_time: 1.5,
            padding: 0.1,
        }];
        WordCensor::apply_bleep(&mut audio, &regions, DEFAULT_BLEEP_FREQUENCY);

        // Effective range is 0.9..1.6, so sample at 0.95s should be non-zero
        let idx = audio.sample_index_at_time(0.95);
        assert!(audio.samples()[idx].abs() > 0.0);
    }

    #[test]
    fn test_apply_bleep_empty_regions_no_change() {
        let mut audio = silent_segment(1.0, 16000);
        let original = audio.samples().to_vec();
        WordCensor::apply_bleep(&mut audio, &[], DEFAULT_BLEEP_FREQUENCY);
        assert_eq!(audio.samples(), &original[..]);
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p faceguard-core -- word_censor::tests::test_apply_bleep`
Expected: FAIL — method `apply_bleep` not found

**Step 3: Write minimal implementation**

Add to `WordCensor`:

```rust
    pub fn apply_bleep(
        audio: &mut AudioSegment,
        regions: &[CensorRegion],
        frequency: f64,
    ) {
        let sample_rate = audio.sample_rate() as f64;
        let channels = audio.channels() as usize;

        for region in regions {
            let start = audio.sample_index_at_time(region.effective_start());
            let end = audio.sample_index_at_time(region.effective_end()).min(audio.samples().len());

            let samples = audio.samples_mut();
            for i in start..end {
                let t = (i - start) as f64 / (sample_rate * channels as f64);
                samples[i] = (2.0 * std::f64::consts::PI * frequency * t).sin() as f32 * 0.3;
            }
        }
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p faceguard-core -- word_censor`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add crates/core/src/audio/domain/word_censor.rs
git commit -m "feat(audio): add WordCensor::apply_bleep"
```

---

## Phase 2: Domain Traits

### Task 5: SpeechRecognizer and AudioTransformer Traits

**Files:**
- Create: `crates/core/src/audio/domain/speech_recognizer.rs`
- Create: `crates/core/src/audio/domain/audio_transformer.rs`
- Modify: `crates/core/src/audio/domain/mod.rs`

**Step 1: Create the trait files**

`speech_recognizer.rs`:
```rust
use super::audio_segment::AudioSegment;
use super::transcript::TranscriptWord;

/// Domain interface for speech-to-text transcription.
///
/// Implementations run inference on audio to produce word-level timestamps.
pub trait SpeechRecognizer: Send {
    fn transcribe(
        &self,
        audio: &AudioSegment,
    ) -> Result<Vec<TranscriptWord>, Box<dyn std::error::Error>>;
}
```

`audio_transformer.rs`:
```rust
use super::audio_segment::AudioSegment;

/// Domain interface for audio transformation (voice disguise).
///
/// Implementations apply DSP effects to modify the speaker's voice.
pub trait AudioTransformer: Send {
    fn transform(
        &self,
        audio: &mut AudioSegment,
    ) -> Result<(), Box<dyn std::error::Error>>;
}
```

Update `mod.rs`:
```rust
pub mod audio_segment;
pub mod audio_transformer;
pub mod censor_region;
pub mod speech_recognizer;
pub mod transcript;
pub mod word_censor;
```

**Step 2: Verify it compiles**

Run: `cargo check -p faceguard-core`
Expected: OK

**Step 3: Commit**

```bash
git add crates/core/src/audio/domain/
git commit -m "feat(audio): add SpeechRecognizer and AudioTransformer domain traits"
```

---

### Task 6: AudioReader and AudioWriter Traits

**Files:**
- Create: `crates/core/src/video/domain/audio_reader.rs`
- Create: `crates/core/src/video/domain/audio_writer.rs`
- Modify: `crates/core/src/video/domain/mod.rs`

**Step 1: Create the trait files**

`audio_reader.rs`:
```rust
use std::path::Path;
use crate::audio::domain::audio_segment::AudioSegment;

/// Domain interface for decoding audio from a video file.
pub trait AudioReader: Send {
    /// Decode the audio track to a mono PCM AudioSegment at the given sample rate.
    /// Returns None if the video has no audio track.
    fn read_audio(
        &self,
        path: &Path,
        target_sample_rate: u32,
    ) -> Result<Option<AudioSegment>, Box<dyn std::error::Error>>;

    /// Return the original audio sample rate and channel count without decoding.
    fn audio_metadata(
        &self,
        path: &Path,
    ) -> Result<Option<(u32, u16)>, Box<dyn std::error::Error>>;
}
```

`audio_writer.rs`:
```rust
use std::path::Path;
use crate::audio::domain::audio_segment::AudioSegment;

/// Domain interface for encoding audio and muxing it into a video file.
pub trait AudioWriter: Send {
    /// Encode the AudioSegment and mux it into an existing video file,
    /// replacing any existing audio track.
    fn write_audio(
        &self,
        video_path: &Path,
        audio: &AudioSegment,
    ) -> Result<(), Box<dyn std::error::Error>>;
}
```

Update `crates/core/src/video/domain/mod.rs` to add the new modules.

**Step 2: Verify it compiles**

Run: `cargo check -p faceguard-core`
Expected: OK

**Step 3: Commit**

```bash
git add crates/core/src/video/domain/
git commit -m "feat(video): add AudioReader and AudioWriter domain traits"
```

---

## Phase 3: Voice Disguise DSP (Infrastructure)

### Task 7: Pitch Shift Transformer (Low)

**Files:**
- Create: `crates/core/src/audio/infrastructure/mod.rs`
- Create: `crates/core/src/audio/infrastructure/pitch_shift_transformer.rs`
- Modify: `crates/core/src/audio/mod.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::domain::audio_segment::AudioSegment;
    use crate::audio::domain::audio_transformer::AudioTransformer;
    use approx::assert_relative_eq;

    fn sine_segment(freq: f64, duration: f64, sample_rate: u32) -> AudioSegment {
        let len = (duration * sample_rate as f64) as usize;
        let samples: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32
            })
            .collect();
        AudioSegment::new(samples, sample_rate, 1)
    }

    #[test]
    fn test_pitch_shift_changes_audio() {
        let original = sine_segment(440.0, 1.0, 16000);
        let mut shifted = original.clone();
        let transformer = PitchShiftTransformer::new(DEFAULT_SEMITONES);
        transformer.transform(&mut shifted).unwrap();

        // Shifted audio should differ from original
        let diff: f64 = original.samples().iter()
            .zip(shifted.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum();
        assert!(diff > 0.0, "Pitch-shifted audio should differ from original");
    }

    #[test]
    fn test_pitch_shift_preserves_length() {
        let mut audio = sine_segment(440.0, 1.0, 16000);
        let original_len = audio.samples().len();
        let transformer = PitchShiftTransformer::new(DEFAULT_SEMITONES);
        transformer.transform(&mut audio).unwrap();
        assert_eq!(audio.samples().len(), original_len);
    }

    #[test]
    fn test_pitch_shift_preserves_amplitude_range() {
        let mut audio = sine_segment(440.0, 1.0, 16000);
        let transformer = PitchShiftTransformer::new(DEFAULT_SEMITONES);
        transformer.transform(&mut audio).unwrap();
        let max = audio.samples().iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(max <= 1.5, "Output should not clip excessively, got max={max}");
    }

    #[test]
    fn test_zero_semitones_near_identity() {
        let original = sine_segment(440.0, 1.0, 16000);
        let mut shifted = original.clone();
        let transformer = PitchShiftTransformer::new(0.0);
        transformer.transform(&mut shifted).unwrap();

        // With zero shift, output should be very close to input
        let diff: f64 = original.samples().iter()
            .zip(shifted.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            / original.samples().len() as f64;
        assert!(diff < 0.01, "Zero shift should be near-identity, MSE={diff}");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p faceguard-core -- pitch_shift`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

Implement a phase vocoder (STFT → frequency-domain pitch shift → ISTFT):

```rust
use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;

pub const DEFAULT_SEMITONES: f64 = 4.0;
const WINDOW_SIZE: usize = 2048;
const HOP_SIZE: usize = 512;

pub struct PitchShiftTransformer {
    semitones: f64,
}

impl PitchShiftTransformer {
    pub fn new(semitones: f64) -> Self {
        Self { semitones }
    }
}

impl AudioTransformer for PitchShiftTransformer {
    fn transform(
        &self,
        audio: &mut AudioSegment,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.semitones.abs() < 1e-6 {
            return Ok(());
        }

        let shift_ratio = 2.0f64.powf(self.semitones / 12.0);
        let samples = audio.samples_mut();
        let len = samples.len();

        // Phase vocoder implementation:
        // 1. STFT with Hann window
        // 2. Shift frequency bins by ratio
        // 3. ISTFT with overlap-add
        // (Full implementation — see code in step 3)

        phase_vocoder(samples, WINDOW_SIZE, HOP_SIZE, shift_ratio);
        Ok(())
    }
}

fn phase_vocoder(samples: &mut [f32], window_size: usize, hop_size: usize, shift_ratio: f64) {
    // Implementation uses:
    // - Hann window for analysis/synthesis
    // - FFT via rustfft crate
    // - Frequency bin shifting with phase interpolation
    // - Overlap-add reconstruction
    // Full implementation to be written — this is the core DSP routine
    // (~80 lines of FFT-based pitch shifting)
}
```

Note: This task requires adding `rustfft` to `Cargo.toml`:
```toml
rustfft = "6"
```

Register modules:
- `crates/core/src/audio/mod.rs`: add `pub mod infrastructure;`
- `crates/core/src/audio/infrastructure/mod.rs`: `pub mod pitch_shift_transformer;`

**Step 4: Run test to verify it passes**

Run: `cargo test -p faceguard-core -- pitch_shift`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add crates/core/src/audio/infrastructure/ crates/core/Cargo.toml
git commit -m "feat(audio): add PitchShiftTransformer (phase vocoder)"
```

---

### Task 8: Formant Shift Transformer (Medium)

**Files:**
- Create: `crates/core/src/audio/infrastructure/formant_shift_transformer.rs`
- Modify: `crates/core/src/audio/infrastructure/mod.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::domain::audio_segment::AudioSegment;
    use crate::audio::domain::audio_transformer::AudioTransformer;

    fn speech_like_segment(sample_rate: u32) -> AudioSegment {
        // Generate a signal with harmonic content resembling speech
        let duration = 1.0;
        let len = (duration * sample_rate as f64) as usize;
        let samples: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                let fundamental = (2.0 * std::f64::consts::PI * 150.0 * t).sin();
                let harmonic2 = 0.5 * (2.0 * std::f64::consts::PI * 300.0 * t).sin();
                let harmonic3 = 0.25 * (2.0 * std::f64::consts::PI * 450.0 * t).sin();
                (fundamental + harmonic2 + harmonic3) as f32 * 0.3
            })
            .collect();
        AudioSegment::new(samples, sample_rate, 1)
    }

    #[test]
    fn test_formant_shift_changes_audio() {
        let original = speech_like_segment(16000);
        let mut shifted = original.clone();
        let transformer = FormantShiftTransformer::new(
            DEFAULT_FORMANT_SEMITONES,
            DEFAULT_FORMANT_SHIFT_RATIO,
        );
        transformer.transform(&mut shifted).unwrap();
        let diff: f64 = original.samples().iter()
            .zip(shifted.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_formant_shift_preserves_length() {
        let mut audio = speech_like_segment(16000);
        let original_len = audio.samples().len();
        let transformer = FormantShiftTransformer::new(
            DEFAULT_FORMANT_SEMITONES,
            DEFAULT_FORMANT_SHIFT_RATIO,
        );
        transformer.transform(&mut audio).unwrap();
        assert_eq!(audio.samples().len(), original_len);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p faceguard-core -- formant_shift`
Expected: FAIL

**Step 3: Write minimal implementation**

Implements pitch shift (reusing phase vocoder from Task 7) plus LPC-based formant envelope modification.

```rust
use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;

pub const DEFAULT_FORMANT_SEMITONES: f64 = 4.0;
pub const DEFAULT_FORMANT_SHIFT_RATIO: f64 = 1.2;
const LPC_ORDER: usize = 16;

pub struct FormantShiftTransformer {
    semitones: f64,
    formant_ratio: f64,
}

impl FormantShiftTransformer {
    pub fn new(semitones: f64, formant_ratio: f64) -> Self {
        Self { semitones, formant_ratio }
    }
}

impl AudioTransformer for FormantShiftTransformer {
    fn transform(
        &self,
        audio: &mut AudioSegment,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Apply pitch shift (reuse phase_vocoder from pitch_shift_transformer)
        // 2. Extract formant envelope via LPC analysis
        // 3. Shift formant envelope by ratio
        // 4. Resynthesize with modified envelope
        // (~100 lines, LPC via Levinson-Durbin recursion)
        Ok(())
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p faceguard-core -- formant_shift`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add crates/core/src/audio/infrastructure/
git commit -m "feat(audio): add FormantShiftTransformer (pitch + formant shift)"
```

---

### Task 9: Voice Morph Transformer (High)

**Files:**
- Create: `crates/core/src/audio/infrastructure/voice_morph_transformer.rs`
- Modify: `crates/core/src/audio/infrastructure/mod.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::domain::audio_segment::AudioSegment;
    use crate::audio::domain::audio_transformer::AudioTransformer;

    // reuse speech_like_segment helper

    #[test]
    fn test_voice_morph_changes_audio() {
        let original = speech_like_segment(16000);
        let mut morphed = original.clone();
        let transformer = VoiceMorphTransformer::new(
            DEFAULT_MORPH_SEMITONES,
            DEFAULT_MORPH_FORMANT_RATIO,
            DEFAULT_JITTER_AMOUNT,
        );
        transformer.transform(&mut morphed).unwrap();
        let diff: f64 = original.samples().iter()
            .zip(morphed.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_voice_morph_preserves_length() {
        let mut audio = speech_like_segment(16000);
        let original_len = audio.samples().len();
        let transformer = VoiceMorphTransformer::new(
            DEFAULT_MORPH_SEMITONES,
            DEFAULT_MORPH_FORMANT_RATIO,
            DEFAULT_JITTER_AMOUNT,
        );
        transformer.transform(&mut audio).unwrap();
        assert_eq!(audio.samples().len(), original_len);
    }

    #[test]
    fn test_voice_morph_differs_from_formant_shift() {
        let mut morphed = speech_like_segment(16000);
        let mut formant_only = morphed.clone();

        VoiceMorphTransformer::new(4.0, 1.2, DEFAULT_JITTER_AMOUNT)
            .transform(&mut morphed).unwrap();
        crate::audio::infrastructure::formant_shift_transformer::FormantShiftTransformer::new(4.0, 1.2)
            .transform(&mut formant_only).unwrap();

        let diff: f64 = morphed.samples().iter()
            .zip(formant_only.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum();
        assert!(diff > 0.0, "Morph should differ from plain formant shift due to jitter");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p faceguard-core -- voice_morph`
Expected: FAIL

**Step 3: Write minimal implementation**

Extends formant shift with spectral envelope jitter.

```rust
use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;

pub const DEFAULT_MORPH_SEMITONES: f64 = 4.0;
pub const DEFAULT_MORPH_FORMANT_RATIO: f64 = 1.2;
pub const DEFAULT_JITTER_AMOUNT: f64 = 0.15;
const SPECTRAL_SMOOTHING: f64 = 0.3;

pub struct VoiceMorphTransformer {
    semitones: f64,
    formant_ratio: f64,
    jitter_amount: f64,
}

impl VoiceMorphTransformer {
    pub fn new(semitones: f64, formant_ratio: f64, jitter_amount: f64) -> Self {
        Self { semitones, formant_ratio, jitter_amount }
    }
}

impl AudioTransformer for VoiceMorphTransformer {
    fn transform(
        &self,
        audio: &mut AudioSegment,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Pitch + formant shift (reuse from formant_shift_transformer)
        // 2. Apply spectral envelope jitter (randomize formant frequencies per frame)
        // 3. Smooth jitter with exponential moving average
        Ok(())
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p faceguard-core -- voice_morph`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add crates/core/src/audio/infrastructure/
git commit -m "feat(audio): add VoiceMorphTransformer (full voice morph)"
```

---

## Phase 4: Audio I/O (FFmpeg Infrastructure)

### Task 10: FFmpeg Audio Reader

**Files:**
- Create: `crates/core/src/video/infrastructure/ffmpeg_audio_reader.rs`
- Modify: `crates/core/src/video/infrastructure/mod.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::domain::audio_reader::AudioReader;

    // These tests require a real video file — mark as #[ignore]
    #[test]
    #[ignore]
    fn test_read_audio_from_video_with_audio() {
        let reader = FfmpegAudioReader;
        let result = reader.read_audio(Path::new("tests/fixtures/sample.mp4"), 16000);
        assert!(result.is_ok());
        let segment = result.unwrap();
        assert!(segment.is_some());
        let seg = segment.unwrap();
        assert_eq!(seg.sample_rate(), 16000);
        assert_eq!(seg.channels(), 1);
        assert!(!seg.samples().is_empty());
    }

    #[test]
    #[ignore]
    fn test_read_audio_from_silent_video() {
        let reader = FfmpegAudioReader;
        let result = reader.read_audio(Path::new("tests/fixtures/no_audio.mp4"), 16000);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_read_audio_nonexistent_file() {
        let reader = FfmpegAudioReader;
        let result = reader.read_audio(Path::new("/nonexistent/file.mp4"), 16000);
        assert!(result.is_err());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p faceguard-core -- ffmpeg_audio_reader`
Expected: FAIL

**Step 3: Write minimal implementation**

```rust
use std::path::Path;
use crate::audio::domain::audio_segment::AudioSegment;
use crate::video::domain::audio_reader::AudioReader;

pub struct FfmpegAudioReader;

impl AudioReader for FfmpegAudioReader {
    fn read_audio(
        &self,
        path: &Path,
        target_sample_rate: u32,
    ) -> Result<Option<AudioSegment>, Box<dyn std::error::Error>> {
        let ictx = ffmpeg_next::format::input(path)?;

        let audio_stream = match ictx.streams().best(ffmpeg_next::media::Type::Audio) {
            Some(s) => s,
            None => return Ok(None),
        };

        let stream_index = audio_stream.index();
        let codec_params = audio_stream.parameters();
        let mut decoder = ffmpeg_next::codec::Context::from_parameters(codec_params)?
            .decoder()
            .audio()?;

        // Decode all audio packets to f32 samples
        // Resample to target_sample_rate mono using ffmpeg's swresample
        // Return AudioSegment

        // (~60 lines of FFmpeg audio decoding + resampling)
        todo!()
    }

    fn audio_metadata(
        &self,
        path: &Path,
    ) -> Result<Option<(u32, u16)>, Box<dyn std::error::Error>> {
        let ictx = ffmpeg_next::format::input(path)?;
        match ictx.streams().best(ffmpeg_next::media::Type::Audio) {
            Some(s) => {
                let params = s.parameters();
                let ctx = ffmpeg_next::codec::Context::from_parameters(params)?
                    .decoder()
                    .audio()?;
                Ok(Some((ctx.rate(), ctx.channels() as u16)))
            }
            None => Ok(None),
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p faceguard-core -- ffmpeg_audio_reader::tests::test_read_audio_nonexistent`
Expected: PASS (1 test; ignored tests skip)

**Step 5: Commit**

```bash
git add crates/core/src/video/infrastructure/
git commit -m "feat(video): add FfmpegAudioReader infrastructure"
```

---

### Task 11: FFmpeg Audio Writer

**Files:**
- Create: `crates/core/src/video/infrastructure/ffmpeg_audio_writer.rs`
- Modify: `crates/core/src/video/infrastructure/mod.rs`

Similar structure to Task 10. The writer:
1. Opens the output video file
2. Copies the video stream as-is
3. Encodes the AudioSegment as AAC audio
4. Muxes both into a new output file (replacing the temp output)

Tests marked `#[ignore]` for integration tests requiring real files; unit test for error path.

**Commit message:** `"feat(video): add FfmpegAudioWriter infrastructure"`

---

### Task 12: Skip Audio Passthrough Flag in FfmpegWriter

**Files:**
- Modify: `crates/core/src/video/infrastructure/ffmpeg_writer.rs`

**Step 1: Write the failing test**

Add a test that verifies when `skip_audio_passthrough` is true, the writer's `close()` does not copy audio.

**Step 2: Implement**

Add a `skip_audio_passthrough: bool` field to `FfmpegWriter`, defaulting to `false`. Check it in `close()` before calling `mux_audio_from_source()`.

Add a setter: `pub fn set_skip_audio_passthrough(&mut self, skip: bool)`.

**Step 3: Commit**

```bash
git commit -m "feat(video): add skip_audio_passthrough flag to FfmpegWriter"
```

---

## Phase 5: Whisper Integration

### Task 13: Whisper ONNX Recognizer

**Files:**
- Create: `crates/core/src/audio/infrastructure/whisper_recognizer.rs`
- Modify: `crates/core/src/audio/infrastructure/mod.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires Whisper ONNX model
    fn test_transcribe_english_speech() {
        let recognizer = WhisperRecognizer::new(Path::new("models/whisper-tiny.onnx")).unwrap();
        let audio = load_test_audio("tests/fixtures/hello_world.wav");
        let words = recognizer.transcribe(&audio).unwrap();
        assert!(!words.is_empty());
        let text: String = words.iter().map(|w| w.word.clone()).collect::<Vec<_>>().join(" ");
        assert!(text.to_lowercase().contains("hello"));
    }

    #[test]
    fn test_transcribe_silence_returns_empty() {
        // Can test with a stub or silent audio without model
    }
}
```

**Step 2: Implement**

```rust
use std::path::Path;
use std::sync::{Arc, Mutex};
use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::speech_recognizer::SpeechRecognizer;
use crate::audio::domain::transcript::TranscriptWord;

pub const WHISPER_SAMPLE_RATE: u32 = 16000;

pub struct WhisperRecognizer {
    session: Arc<Mutex<ort::session::Session>>,
}

impl WhisperRecognizer {
    pub fn new(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let session = ort::session::Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
        })
    }
}

impl SpeechRecognizer for WhisperRecognizer {
    fn transcribe(
        &self,
        audio: &AudioSegment,
    ) -> Result<Vec<TranscriptWord>, Box<dyn std::error::Error>> {
        // Run Whisper inference on 30-second chunks
        // Extract word-level timestamps from decoder output
        // (~150 lines of Whisper ONNX inference)
        todo!()
    }
}
```

**Step 3: Commit**

```bash
git commit -m "feat(audio): add WhisperRecognizer ONNX infrastructure"
```

---

### Task 14: Whisper Model Constants and Download Setup

**Files:**
- Modify: `crates/core/src/shared/constants.rs` (or wherever model URLs are defined)
- Modify: `crates/desktop/src/workers/model_cache.rs`

Add:
```rust
pub const WHISPER_MODEL_NAME: &str = "whisper-tiny-en.onnx";
pub const WHISPER_MODEL_URL: &str = "https://..."; // Hugging Face ONNX model URL
```

Add a `whisper_path` slot to `ModelCache` following the same pattern as `yolo_path`. Download is lazy — only triggered when audio processing is first enabled.

**Commit message:** `"feat(audio): add Whisper model download to ModelCache"`

---

## Phase 6: Pipeline Use Case

### Task 15: ProcessAudioUseCase

**Files:**
- Create: `crates/core/src/pipeline/process_audio_use_case.rs`
- Modify: `crates/core/src/pipeline/mod.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::domain::audio_segment::AudioSegment;
    use crate::audio::domain::audio_transformer::AudioTransformer;
    use crate::audio::domain::speech_recognizer::SpeechRecognizer;
    use crate::audio::domain::transcript::TranscriptWord;
    use crate::video::domain::audio_reader::AudioReader;
    use crate::video::domain::audio_writer::AudioWriter;
    use std::sync::{Arc, Mutex};

    // ─── Stubs ───

    struct StubAudioReader {
        segment: Option<AudioSegment>,
    }

    impl AudioReader for StubAudioReader {
        fn read_audio(&self, _: &Path, _: u32)
            -> Result<Option<AudioSegment>, Box<dyn std::error::Error>> {
            Ok(self.segment.clone())
        }
        fn audio_metadata(&self, _: &Path)
            -> Result<Option<(u32, u16)>, Box<dyn std::error::Error>> {
            Ok(self.segment.as_ref().map(|s| (s.sample_rate(), s.channels())))
        }
    }

    struct StubAudioWriter {
        written: Arc<Mutex<Option<AudioSegment>>>,
    }

    impl AudioWriter for StubAudioWriter {
        fn write_audio(&self, _: &Path, audio: &AudioSegment)
            -> Result<(), Box<dyn std::error::Error>> {
            *self.written.lock().unwrap() = Some(audio.clone());
            Ok(())
        }
    }

    struct StubRecognizer {
        words: Vec<TranscriptWord>,
    }

    impl SpeechRecognizer for StubRecognizer {
        fn transcribe(&self, _: &AudioSegment)
            -> Result<Vec<TranscriptWord>, Box<dyn std::error::Error>> {
            Ok(self.words.clone())
        }
    }

    struct StubTransformer {
        called: Arc<Mutex<bool>>,
    }

    impl AudioTransformer for StubTransformer {
        fn transform(&self, _: &mut AudioSegment)
            -> Result<(), Box<dyn std::error::Error>> {
            *self.called.lock().unwrap() = true;
            Ok(())
        }
    }

    fn silent_audio() -> AudioSegment {
        AudioSegment::new(vec![0.0; 16000], 16000, 1)
    }

    #[test]
    fn test_no_audio_track_skips_processing() {
        let writer = StubAudioWriter { written: Arc::new(Mutex::new(None)) };
        let written = writer.written.clone();
        let mut uc = ProcessAudioUseCase::new(
            Box::new(StubAudioReader { segment: None }),
            Box::new(writer),
            None, None, vec![],
        );
        uc.run(Path::new("in.mp4"), Path::new("out.mp4")).unwrap();
        assert!(written.lock().unwrap().is_none());
    }

    #[test]
    fn test_keywords_trigger_bleep() {
        let writer = StubAudioWriter { written: Arc::new(Mutex::new(None)) };
        let written = writer.written.clone();
        let recognizer = StubRecognizer {
            words: vec![TranscriptWord {
                word: "secret".to_string(),
                start_time: 0.5,
                end_time: 0.8,
                confidence: 0.95,
            }],
        };
        let mut uc = ProcessAudioUseCase::new(
            Box::new(StubAudioReader { segment: Some(silent_audio()) }),
            Box::new(writer),
            Some(Box::new(recognizer)),
            None,
            vec!["secret".to_string()],
        );
        uc.run(Path::new("in.mp4"), Path::new("out.mp4")).unwrap();

        let written = written.lock().unwrap();
        assert!(written.is_some());
        // Verify bleep was applied (samples in 0.5-0.8s range should be non-zero)
        let seg = written.as_ref().unwrap();
        let start = seg.sample_index_at_time(0.5);
        let end = seg.sample_index_at_time(0.8);
        let energy: f64 = seg.samples()[start..end].iter()
            .map(|s| (*s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_voice_transform_applied() {
        let writer = StubAudioWriter { written: Arc::new(Mutex::new(None)) };
        let transformer = StubTransformer { called: Arc::new(Mutex::new(false)) };
        let called = transformer.called.clone();
        let mut uc = ProcessAudioUseCase::new(
            Box::new(StubAudioReader { segment: Some(silent_audio()) }),
            Box::new(writer),
            None,
            Some(Box::new(transformer)),
            vec![],
        );
        uc.run(Path::new("in.mp4"), Path::new("out.mp4")).unwrap();
        assert!(*called.lock().unwrap());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p faceguard-core -- process_audio`
Expected: FAIL

**Step 3: Write minimal implementation**

```rust
use std::path::Path;
use crate::audio::domain::audio_transformer::AudioTransformer;
use crate::audio::domain::speech_recognizer::SpeechRecognizer;
use crate::audio::domain::word_censor::{WordCensor, DEFAULT_BLEEP_FREQUENCY, DEFAULT_BLEEP_PADDING};
use crate::video::domain::audio_reader::AudioReader;
use crate::video::domain::audio_writer::AudioWriter;

pub struct ProcessAudioUseCase {
    reader: Box<dyn AudioReader>,
    writer: Box<dyn AudioWriter>,
    recognizer: Option<Box<dyn SpeechRecognizer>>,
    transformer: Option<Box<dyn AudioTransformer>>,
    keywords: Vec<String>,
}

impl ProcessAudioUseCase {
    pub fn new(
        reader: Box<dyn AudioReader>,
        writer: Box<dyn AudioWriter>,
        recognizer: Option<Box<dyn SpeechRecognizer>>,
        transformer: Option<Box<dyn AudioTransformer>>,
        keywords: Vec<String>,
    ) -> Self {
        Self { reader, writer, recognizer, transformer, keywords }
    }

    pub fn run(
        &mut self,
        source_path: &Path,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Read audio
        let mut audio = match self.reader.read_audio(source_path, 16000)? {
            Some(a) => a,
            None => return Ok(()), // No audio track
        };

        // 2. Transcribe + bleep (if keywords provided)
        if !self.keywords.is_empty() {
            if let Some(ref recognizer) = self.recognizer {
                let transcript = recognizer.transcribe(&audio)?;
                let regions = WordCensor::find_censor_regions(
                    &transcript, &self.keywords, DEFAULT_BLEEP_PADDING,
                );
                WordCensor::apply_bleep(&mut audio, &regions, DEFAULT_BLEEP_FREQUENCY);
            }
        }

        // 3. Voice transform (if enabled)
        if let Some(ref transformer) = self.transformer {
            transformer.transform(&mut audio)?;
        }

        // 4. Write processed audio
        self.writer.write_audio(output_path, &audio)?;

        Ok(())
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p faceguard-core -- process_audio`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add crates/core/src/pipeline/
git commit -m "feat(pipeline): add ProcessAudioUseCase"
```

---

## Phase 7: Desktop Integration

### Task 16: Audio Settings

**Files:**
- Modify: `crates/desktop/src/settings.rs`

**Step 1: Add new types and fields**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceDisguise {
    Off,
    Low,
    Medium,
    High,
}

impl VoiceDisguise {
    pub const ALL: &[VoiceDisguise] = &[
        VoiceDisguise::Off,
        VoiceDisguise::Low,
        VoiceDisguise::Medium,
        VoiceDisguise::High,
    ];
}

impl std::fmt::Display for VoiceDisguise { ... }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BleepSound {
    Tone,
    Silence,
}

impl BleepSound {
    pub const ALL: &[BleepSound] = &[BleepSound::Tone, BleepSound::Silence];
}

impl std::fmt::Display for BleepSound { ... }
```

Add to `Settings` struct:
```rust
    #[serde(default)]
    pub audio_processing: bool,
    #[serde(default)]
    pub bleep_keywords: String,
    #[serde(default = "default_bleep_sound")]
    pub bleep_sound: BleepSound,
    #[serde(default = "default_voice_disguise")]
    pub voice_disguise: VoiceDisguise,
```

Add defaults:
```rust
fn default_bleep_sound() -> BleepSound { BleepSound::Tone }
fn default_voice_disguise() -> VoiceDisguise { VoiceDisguise::Off }
```

Update `Default::default()` to include the new fields.

**Step 2: Verify it compiles**

Run: `cargo check -p faceguard-desktop`

**Step 3: Commit**

```bash
git commit -m "feat(desktop): add audio settings (keywords, voice disguise, bleep sound)"
```

---

### Task 17: Audio Settings UI Tab

**Files:**
- Modify: `crates/desktop/src/tabs/settings_tab.rs`

Add an "Audio" section to the settings tab with:
- Toggle: Audio processing on/off
- Text input: Bleep keywords (comma-separated)
- Picker: Bleep sound (Tone / Silence)
- Picker: Voice disguise (Off / Low / Medium / High)

Follow the existing card/slider pattern from the Blur section.

**Commit message:** `"feat(desktop): add Audio section to settings UI"`

---

### Task 18: Wire Audio Processing into Blur Worker

**Files:**
- Modify: `crates/desktop/src/workers/blur_worker.rs`

Add audio fields to `BlurParams`:
```rust
pub audio_processing: bool,
pub bleep_keywords: String,
pub bleep_sound: crate::settings::BleepSound,
pub voice_disguise: crate::settings::VoiceDisguise,
```

After the video blur completes in `run_blur()`, if `audio_processing` is true:
1. Create the appropriate `AudioTransformer` based on `voice_disguise` level
2. Create `WhisperRecognizer` if keywords are provided (download model if needed)
3. Create `FfmpegAudioReader` and `FfmpegAudioWriter`
4. Run `ProcessAudioUseCase`

Also set `skip_audio_passthrough = true` on the `FfmpegWriter` when audio processing is enabled.

**Commit message:** `"feat(desktop): wire audio processing into blur worker"`

---

### Task 19: Wire Audio Processing in App Messages

**Files:**
- Modify: `crates/desktop/src/app.rs`

Add messages:
```rust
AudioProcessingChanged(bool),
BleepKeywordsChanged(String),
BleepSoundChanged(BleepSound),
VoiceDisguiseChanged(VoiceDisguise),
```

Handle each: update settings, save, no cache invalidation needed (audio doesn't affect detection).

Pass audio settings to `BlurParams` when spawning the blur worker.

**Commit message:** `"feat(desktop): add audio settings messages and wiring"`

---

## Phase 8: CLI Integration

### Task 20: CLI Audio Flags

**Files:**
- Modify: `crates/cli/src/main.rs`

Add to `Cli` struct:
```rust
    /// Comma-separated keywords to bleep out (enables audio processing).
    #[arg(long, value_delimiter = ',')]
    audio_keywords: Option<Vec<String>>,

    /// Voice disguise level: off, low, medium, high.
    #[arg(long, default_value = "off")]
    voice_disguise: String,
```

In `run_video_blur()`, after the video blur use case completes:
1. Parse `voice_disguise` string to enum
2. If keywords or voice_disguise != off, create and run `ProcessAudioUseCase`
3. Set `skip_audio_passthrough` on the writer when audio processing is active

**Commit message:** `"feat(cli): add --audio-keywords and --voice-disguise flags"`

---

## Phase 9: Final Verification

### Task 21: Clippy + Format + Full Test Suite

**Step 1:** Run `cargo fmt`
**Step 2:** Run `cargo clippy --all-targets -- -D warnings`
**Step 3:** Run `cargo test`
**Step 4:** Fix any issues
**Step 5:** Final commit

```bash
git commit -m "chore: clippy and format fixes for audio features"
```

---

## Dependency Summary

New crate dependencies to add to `crates/core/Cargo.toml`:
```toml
rustfft = "6"        # FFT for phase vocoder
```

No new dependencies needed for LPC (implemented from scratch) or the bleep tone (sine generation).

## Task Dependency Graph

```
Phase 1: Entities & Services
  Task 1 (AudioSegment) ──→ Task 2 (TranscriptWord, CensorRegion) ──→ Task 3 (find_censor_regions) ──→ Task 4 (apply_bleep)

Phase 2: Traits
  Task 1 ──→ Task 5 (SpeechRecognizer, AudioTransformer traits)
  Task 1 ──→ Task 6 (AudioReader, AudioWriter traits)

Phase 3: DSP (depends on Task 5)
  Task 5 ──→ Task 7 (PitchShift) ──→ Task 8 (FormantShift) ──→ Task 9 (VoiceMorph)

Phase 4: FFmpeg I/O (depends on Task 6)
  Task 6 ──→ Task 10 (FfmpegAudioReader) ──→ Task 11 (FfmpegAudioWriter) ──→ Task 12 (skip passthrough flag)

Phase 5: Whisper (depends on Task 5)
  Task 5 ──→ Task 13 (WhisperRecognizer) ──→ Task 14 (model download)

Phase 6: Pipeline (depends on Tasks 4, 5, 6)
  Tasks 4+5+6 ──→ Task 15 (ProcessAudioUseCase)

Phase 7: Desktop (depends on Task 15)
  Task 15 ──→ Task 16 (settings) ──→ Task 17 (UI) ──→ Task 18 (worker) ──→ Task 19 (app messages)

Phase 8: CLI (depends on Task 15)
  Task 15 ──→ Task 20 (CLI flags)

Phase 9: Verification
  All ──→ Task 21 (clippy + tests)
```

Phases 3, 4, and 5 can be worked on in parallel since they depend only on Phase 2.
