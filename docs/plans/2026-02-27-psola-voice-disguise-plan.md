# PSOLA Voice Disguise Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the phase vocoder voice disguising with PSOLA for natural-sounding voice anonymization.

**Architecture:** Time-domain PSOLA pitch shifting replaces the FFT-based phase vocoder. FormantShiftTransformer becomes a standalone single-pass LPC warp (no longer wrapping PitchShiftTransformer). VoiceMorphTransformer composes PSOLA with pitch contour warping + formant warp, replacing spectral jitter.

**Tech Stack:** Rust, rustfft (retained for LPC spectral envelope computation only), no new dependencies.

---

### Task 1: Pitch Detection — Autocorrelation-Based F0 Estimator

Build the core pitch detection function that analyzes a frame of audio and returns the fundamental period and voiced/unvoiced classification.

**Files:**
- Modify: `crates/core/src/audio/infrastructure/pitch_shift_transformer.rs`

**Step 1: Write failing tests for pitch detection**

Replace the entire contents of `crates/core/src/audio/infrastructure/pitch_shift_transformer.rs` with the new module structure. Start with just the pitch detection tests and types:

```rust
use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;
use std::f64::consts::PI;

/// Default pitch shift in semitones (upward).
pub const DEFAULT_SEMITONES: f64 = 2.5;

/// Minimum detectable F0 in Hz (deep male voice).
const MIN_F0_HZ: f64 = 60.0;

/// Maximum detectable F0 in Hz (high female/child voice).
const MAX_F0_HZ: f64 = 500.0;

/// Frame size for pitch analysis (~30ms at 16kHz).
const ANALYSIS_FRAME_SIZE: usize = 512;

/// Hop between analysis frames.
const ANALYSIS_HOP: usize = 256;

/// Voicing threshold: normalized autocorrelation peak must exceed this
/// to classify a frame as voiced.
const VOICING_THRESHOLD: f64 = 0.3;

/// Fixed pitch mark spacing for unvoiced segments (in samples at 16kHz = ~5ms).
const UNVOICED_MARK_SPACING: usize = 80;

/// Result of analyzing one frame for pitch.
#[derive(Clone, Debug)]
struct PitchFrame {
    /// True if the frame is voiced (periodic).
    voiced: bool,
    /// Detected pitch period in samples (only meaningful if voiced).
    period_samples: usize,
}

/// Detect pitch in a single frame using autocorrelation.
///
/// Returns a PitchFrame with voiced/unvoiced classification and detected period.
/// `sample_rate` is needed to convert min/max F0 to lag bounds.
fn detect_pitch(frame: &[f64], sample_rate: u32) -> PitchFrame {
    todo!()
}

pub struct PitchShiftTransformer {
    semitones: f64,
}

impl PitchShiftTransformer {
    pub fn new(semitones: f64) -> Self {
        Self { semitones }
    }
}

impl AudioTransformer for PitchShiftTransformer {
    fn transform(&self, _audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pure sine wave as f64 samples.
    fn sine_f64(freq: f64, duration: f64, sample_rate: u32) -> Vec<f64> {
        let len = (duration * sample_rate as f64) as usize;
        (0..len)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * PI * freq * t).sin()
            })
            .collect()
    }

    fn sine_segment(freq: f64, duration: f64, sample_rate: u32) -> AudioSegment {
        let samples: Vec<f32> = sine_f64(freq, duration, sample_rate)
            .iter()
            .map(|&s| s as f32)
            .collect();
        AudioSegment::new(samples, sample_rate, 1)
    }

    #[test]
    fn test_detect_pitch_finds_100hz_fundamental() {
        // 100 Hz at 16kHz sample rate -> period of 160 samples
        let wave = sine_f64(100.0, 0.05, 16000);
        let frame = &wave[..ANALYSIS_FRAME_SIZE.min(wave.len())];
        let result = detect_pitch(frame, 16000);
        assert!(result.voiced, "100 Hz sine should be detected as voiced");
        // Period should be ~160 samples (16000/100)
        let expected_period = 160;
        assert!(
            (result.period_samples as i32 - expected_period as i32).unsigned_abs() <= 3,
            "Expected period ~{expected_period}, got {}",
            result.period_samples
        );
    }

    #[test]
    fn test_detect_pitch_finds_200hz_fundamental() {
        let wave = sine_f64(200.0, 0.05, 16000);
        let frame = &wave[..ANALYSIS_FRAME_SIZE.min(wave.len())];
        let result = detect_pitch(frame, 16000);
        assert!(result.voiced);
        let expected_period = 80; // 16000/200
        assert!(
            (result.period_samples as i32 - expected_period as i32).unsigned_abs() <= 2,
            "Expected period ~{expected_period}, got {}",
            result.period_samples
        );
    }

    #[test]
    fn test_detect_pitch_classifies_noise_as_unvoiced() {
        // White noise should not have a clear pitch
        let mut rng_state: u64 = 12345;
        let noise: Vec<f64> = (0..ANALYSIS_FRAME_SIZE)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let bits = (rng_state >> 33) as f64;
                let max = (1u64 << 31) as f64;
                (bits / max) * 2.0 - 1.0
            })
            .collect();
        let result = detect_pitch(&noise, 16000);
        assert!(
            !result.voiced,
            "White noise should be classified as unvoiced"
        );
    }

    #[test]
    fn test_detect_pitch_silence_is_unvoiced() {
        let silence = vec![0.0f64; ANALYSIS_FRAME_SIZE];
        let result = detect_pitch(&silence, 16000);
        assert!(!result.voiced, "Silence should be classified as unvoiced");
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p faceguard-core -- pitch_shift`
Expected: FAIL — `todo!()` panics

**Step 3: Implement detect_pitch**

Replace the `todo!()` in `detect_pitch` with:

```rust
fn detect_pitch(frame: &[f64], sample_rate: u32) -> PitchFrame {
    let n = frame.len();

    // Compute normalized autocorrelation
    // R(0) is the energy; R(lag)/R(0) gives normalized correlation
    let energy: f64 = frame.iter().map(|&s| s * s).sum();
    if energy < 1e-10 {
        return PitchFrame {
            voiced: false,
            period_samples: UNVOICED_MARK_SPACING,
        };
    }

    // Lag bounds from F0 range
    let min_lag = (sample_rate as f64 / MAX_F0_HZ).floor() as usize;
    let max_lag = (sample_rate as f64 / MIN_F0_HZ).ceil() as usize;
    let max_lag = max_lag.min(n - 1);

    if min_lag >= max_lag || max_lag >= n {
        return PitchFrame {
            voiced: false,
            period_samples: UNVOICED_MARK_SPACING,
        };
    }

    let mut best_lag = min_lag;
    let mut best_corr = f64::NEG_INFINITY;

    for lag in min_lag..=max_lag {
        let mut sum = 0.0;
        for i in 0..n - lag {
            sum += frame[i] * frame[i + lag];
        }
        // Normalize by geometric mean of energies of the two segments
        let energy_a: f64 = frame[..n - lag].iter().map(|&s| s * s).sum();
        let energy_b: f64 = frame[lag..].iter().map(|&s| s * s).sum();
        let denom = (energy_a * energy_b).sqrt();
        let normalized = if denom > 1e-10 { sum / denom } else { 0.0 };

        if normalized > best_corr {
            best_corr = normalized;
            best_lag = lag;
        }
    }

    PitchFrame {
        voiced: best_corr > VOICING_THRESHOLD,
        period_samples: best_lag,
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p faceguard-core -- pitch_shift`
Expected: All 4 pitch detection tests PASS

**Step 5: Commit**

```bash
git add crates/core/src/audio/infrastructure/pitch_shift_transformer.rs
git commit -m "feat(audio): add PSOLA pitch detection with autocorrelation F0 estimator"
```

---

### Task 2: PSOLA Pitch Marking and Synthesis

Add pitch mark placement and the overlap-add synthesis that shifts pitch by repositioning marks.

**Files:**
- Modify: `crates/core/src/audio/infrastructure/pitch_shift_transformer.rs`

**Step 1: Write failing tests for PSOLA synthesis**

Add these tests to the existing `mod tests` block:

```rust
    #[test]
    fn test_psola_shift_changes_audio() {
        let original = sine_segment(150.0, 1.0, 16000);
        let mut shifted = original.clone();
        let transformer = PitchShiftTransformer::new(DEFAULT_SEMITONES);
        transformer.transform(&mut shifted).unwrap();
        let diff: f64 = original
            .samples()
            .iter()
            .zip(shifted.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum();
        assert!(
            diff > 0.0,
            "Pitch-shifted audio should differ from original"
        );
    }

    #[test]
    fn test_psola_shift_preserves_length() {
        let mut audio = sine_segment(150.0, 1.0, 16000);
        let original_len = audio.samples().len();
        let transformer = PitchShiftTransformer::new(DEFAULT_SEMITONES);
        transformer.transform(&mut audio).unwrap();
        assert_eq!(audio.samples().len(), original_len);
    }

    #[test]
    fn test_psola_shift_preserves_amplitude_range() {
        let mut audio = sine_segment(150.0, 1.0, 16000);
        let transformer = PitchShiftTransformer::new(DEFAULT_SEMITONES);
        transformer.transform(&mut audio).unwrap();
        let max = audio
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(
            max <= 1.5,
            "Output should not clip excessively, got max={max}"
        );
    }

    #[test]
    fn test_zero_semitones_near_identity() {
        let original = sine_segment(150.0, 1.0, 16000);
        let mut shifted = original.clone();
        let transformer = PitchShiftTransformer::new(0.0);
        transformer.transform(&mut shifted).unwrap();
        let diff: f64 = original
            .samples()
            .iter()
            .zip(shifted.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            / original.samples().len() as f64;
        assert!(
            diff < 0.01,
            "Zero shift should be near-identity, MSE={diff}"
        );
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p faceguard-core -- psola_shift`
Expected: FAIL — `todo!()` in transform

**Step 3: Implement PSOLA marking and synthesis**

Add these helper functions before the `impl AudioTransformer` block:

```rust
/// Analyze the entire signal and return per-frame pitch information.
fn analyze_pitch(samples: &[f64], sample_rate: u32) -> Vec<PitchFrame> {
    let n = samples.len();
    let mut frames = Vec::new();
    let mut pos = 0;
    while pos + ANALYSIS_FRAME_SIZE <= n {
        frames.push(detect_pitch(&samples[pos..pos + ANALYSIS_FRAME_SIZE], sample_rate));
        pos += ANALYSIS_HOP;
    }
    frames
}

/// Place pitch marks throughout the signal based on detected pitch.
/// Returns sample indices where each pitch mark is placed.
fn place_pitch_marks(samples: &[f64], pitch_frames: &[PitchFrame]) -> Vec<usize> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    let mut marks = Vec::new();
    let mut pos: usize = 0;

    while pos < n {
        marks.push(pos);
        // Find which analysis frame this position falls in
        let frame_idx = (pos / ANALYSIS_HOP).min(pitch_frames.len().saturating_sub(1));
        let period = if frame_idx < pitch_frames.len() && pitch_frames[frame_idx].voiced {
            pitch_frames[frame_idx].period_samples
        } else {
            UNVOICED_MARK_SPACING
        };
        pos += period.max(1);
    }
    marks
}

/// PSOLA synthesis: given analysis marks and a shift ratio, produce output.
/// shift_ratio > 1 means higher pitch (shorter periods).
fn psola_synthesize(
    samples: &[f64],
    analysis_marks: &[usize],
    pitch_frames: &[PitchFrame],
    shift_ratio: f64,
) -> Vec<f64> {
    let n = samples.len();
    let mut output = vec![0.0f64; n];
    let mut window_sum = vec![0.0f64; n];

    if analysis_marks.is_empty() {
        return output;
    }

    // Compute synthesis marks by adjusting spacing
    let mut synthesis_marks = Vec::with_capacity(analysis_marks.len());
    synthesis_marks.push(0.0f64); // first mark at position 0

    for i in 1..analysis_marks.len() {
        let analysis_spacing = analysis_marks[i] as f64 - analysis_marks[i - 1] as f64;
        let synthesis_spacing = analysis_spacing / shift_ratio;
        let prev_synth = synthesis_marks[i - 1];
        synthesis_marks.push(prev_synth + synthesis_spacing);
    }

    // For each analysis mark, extract a windowed grain and place it at the synthesis mark
    for (i, &analysis_pos) in analysis_marks.iter().enumerate() {
        // Determine local pitch period for window sizing
        let frame_idx =
            (analysis_pos / ANALYSIS_HOP).min(pitch_frames.len().saturating_sub(1));
        let local_period = if frame_idx < pitch_frames.len() && pitch_frames[frame_idx].voiced {
            pitch_frames[frame_idx].period_samples
        } else {
            UNVOICED_MARK_SPACING
        };

        // Window size = 2 * local period (covers two full cycles)
        let half_win = local_period;
        let win_size = 2 * half_win;
        if win_size == 0 {
            continue;
        }

        // Hann window for this grain
        let hann: Vec<f64> = (0..win_size)
            .map(|j| 0.5 * (1.0 - (2.0 * PI * j as f64 / win_size as f64).cos()))
            .collect();

        // Extract grain centered on analysis mark
        let grain_start = if analysis_pos >= half_win {
            analysis_pos - half_win
        } else {
            0
        };
        let grain_end = (analysis_pos + half_win).min(n);

        // Synthesis position (center of output grain)
        let synth_center = synthesis_marks[i];
        let synth_start = synth_center - half_win as f64;

        for j in 0..(grain_end - grain_start) {
            let src_idx = grain_start + j;
            let dst_f = synth_start + j as f64;
            let dst_idx = dst_f.round() as i64;
            if dst_idx >= 0 && (dst_idx as usize) < n {
                let d = dst_idx as usize;
                let win_idx = if analysis_pos >= half_win {
                    j
                } else {
                    j + (half_win - analysis_pos)
                };
                if win_idx < win_size {
                    output[d] += samples[src_idx] * hann[win_idx];
                    window_sum[d] += hann[win_idx];
                }
            }
        }
    }

    // Normalize by window sum
    for i in 0..n {
        if window_sum[i] > 1e-10 {
            output[i] /= window_sum[i];
        }
    }

    output
}
```

Now replace the `todo!()` in `impl AudioTransformer for PitchShiftTransformer`:

```rust
impl AudioTransformer for PitchShiftTransformer {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>> {
        if self.semitones.abs() < 1e-10 {
            return Ok(());
        }

        let samples: Vec<f64> = audio.samples().iter().map(|&s| s as f64).collect();
        let n = samples.len();
        if n < ANALYSIS_FRAME_SIZE {
            return Ok(());
        }

        let shift_ratio = 2.0_f64.powf(self.semitones / 12.0);
        let sample_rate = audio.sample_rate();

        // Step 1: Analyze pitch across the signal
        let pitch_frames = analyze_pitch(&samples, sample_rate);

        // Step 2: Place analysis pitch marks
        let analysis_marks = place_pitch_marks(&samples, &pitch_frames);

        // Step 3: PSOLA synthesis with shifted marks
        let output = psola_synthesize(&samples, &analysis_marks, &pitch_frames, shift_ratio);

        // Step 4: Peak-normalize to avoid clipping
        let input_peak = samples.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        let output_peak = output.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        let gain = if output_peak > 1e-10 && output_peak > input_peak {
            input_peak / output_peak
        } else {
            1.0
        };

        let out_samples = audio.samples_mut();
        for i in 0..n {
            out_samples[i] = (output[i] * gain) as f32;
        }

        Ok(())
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p faceguard-core -- pitch_shift`
Expected: All 8 tests PASS (4 pitch detection + 4 synthesis)

**Step 5: Commit**

```bash
git add crates/core/src/audio/infrastructure/pitch_shift_transformer.rs
git commit -m "feat(audio): implement PSOLA pitch marking and synthesis"
```

---

### Task 3: Simplify FormantShiftTransformer to Standalone LPC Warp

Decouple formant shifting from pitch shifting. The formant transformer now only does spectral envelope modification — it no longer wraps PitchShiftTransformer internally.

**Files:**
- Modify: `crates/core/src/audio/infrastructure/formant_shift_transformer.rs`

**Step 1: Write failing tests**

Replace the test module with updated tests that verify standalone behavior:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::domain::audio_segment::AudioSegment;
    use crate::audio::domain::audio_transformer::AudioTransformer;

    fn speech_like_segment(sample_rate: u32) -> AudioSegment {
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
    fn test_formant_warp_changes_audio() {
        let original = speech_like_segment(16000);
        let mut warped = original.clone();
        let transformer = FormantShiftTransformer::new(DEFAULT_FORMANT_SHIFT_RATIO);
        transformer.transform(&mut warped).unwrap();
        let diff: f64 = original
            .samples()
            .iter()
            .zip(warped.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_formant_warp_preserves_length() {
        let mut audio = speech_like_segment(16000);
        let original_len = audio.samples().len();
        let transformer = FormantShiftTransformer::new(DEFAULT_FORMANT_SHIFT_RATIO);
        transformer.transform(&mut audio).unwrap();
        assert_eq!(audio.samples().len(), original_len);
    }

    #[test]
    fn test_unity_ratio_near_identity() {
        let original = speech_like_segment(16000);
        let mut warped = original.clone();
        let transformer = FormantShiftTransformer::new(1.0);
        transformer.transform(&mut warped).unwrap();
        let mse: f64 = original
            .samples()
            .iter()
            .zip(warped.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            / original.samples().len() as f64;
        assert!(
            mse < 0.001,
            "Unity formant ratio should be near-identity, MSE={mse}"
        );
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p faceguard-core -- formant`
Expected: FAIL — constructor signature changed (now takes 1 arg instead of 2)

**Step 3: Update FormantShiftTransformer to standalone**

Rewrite the struct and constructor (keep the existing LPC helper functions unchanged):

```rust
use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Default ratio by which formant frequencies are shifted.
pub const DEFAULT_FORMANT_SHIFT_RATIO: f64 = 1.15;

/// Order of the LPC (Linear Predictive Coding) analysis.
const LPC_ORDER: usize = 16;

/// STFT frame size for formant analysis.
const WINDOW_SIZE: usize = 2048;

/// Hop size between successive analysis frames.
const HOP_SIZE: usize = 512;

/// Standalone spectral envelope modifier via LPC formant analysis.
///
/// Applies frequency-domain envelope reshaping without any pitch shifting.
/// Used as the second stage in Medium/High tier voice disguise,
/// after PSOLA pitch shifting has already been applied.
pub struct FormantShiftTransformer {
    formant_ratio: f64,
}

impl FormantShiftTransformer {
    pub fn new(formant_ratio: f64) -> Self {
        Self { formant_ratio }
    }
}
```

The `impl AudioTransformer` block stays the same as the current code, minus the `self.pitch_shifter.transform(audio)?;` call at the top. Everything from the `if (self.formant_ratio - 1.0).abs() < 1e-10` check onward remains identical.

Remove the `DEFAULT_FORMANT_SEMITONES` constant (no longer needed).

Remove the `use crate::audio::infrastructure::pitch_shift_transformer::PitchShiftTransformer;` import.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p faceguard-core -- formant`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add crates/core/src/audio/infrastructure/formant_shift_transformer.rs
git commit -m "refactor(audio): decouple FormantShiftTransformer from pitch shifting"
```

---

### Task 4: Rewrite VoiceMorphTransformer with Pitch Contour Warping

Replace spectral jitter with pitch contour warping. The transformer now composes PSOLA (with per-frame varying shift) + formant warp.

**Files:**
- Modify: `crates/core/src/audio/infrastructure/voice_morph_transformer.rs`

**Step 1: Write failing tests**

Replace the entire file with:

```rust
use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;
use crate::audio::infrastructure::formant_shift_transformer::{
    FormantShiftTransformer, DEFAULT_FORMANT_SHIFT_RATIO,
};
use crate::audio::infrastructure::pitch_shift_transformer::DEFAULT_SEMITONES;
use std::f64::consts::PI;

/// Default pitch contour warp range in semitones.
/// The shift varies by +/- this amount around the base shift via random walk.
pub const DEFAULT_CONTOUR_WARP_RANGE: f64 = 0.5;

/// Step size for the random walk (fraction of warp range per analysis hop).
const CONTOUR_STEP_SIZE: f64 = 0.1;

/// Highest tier voice disguise: PSOLA with pitch contour warping + formant shift.
///
/// Instead of a constant pitch shift, varies the shift smoothly over time
/// via a deterministic random walk. This breaks prosody patterns used in
/// forensic voice analysis while maintaining natural-sounding speech.
pub struct VoiceMorphTransformer {
    base_semitones: f64,
    contour_warp_range: f64,
    formant_shifter: FormantShiftTransformer,
}

impl VoiceMorphTransformer {
    pub fn new(base_semitones: f64, formant_ratio: f64, contour_warp_range: f64) -> Self {
        Self {
            base_semitones,
            contour_warp_range,
            formant_shifter: FormantShiftTransformer::new(formant_ratio),
        }
    }
}

impl AudioTransformer for VoiceMorphTransformer {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech_like_segment(sample_rate: u32) -> AudioSegment {
        let duration = 1.0;
        let len = (duration * sample_rate as f64) as usize;
        let samples: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                let fundamental = (2.0 * PI * 150.0 * t).sin();
                let harmonic2 = 0.5 * (2.0 * PI * 300.0 * t).sin();
                let harmonic3 = 0.25 * (2.0 * PI * 450.0 * t).sin();
                (fundamental + harmonic2 + harmonic3) as f32 * 0.3
            })
            .collect();
        AudioSegment::new(samples, sample_rate, 1)
    }

    #[test]
    fn test_voice_morph_changes_audio() {
        let original = speech_like_segment(16000);
        let mut morphed = original.clone();
        let transformer = VoiceMorphTransformer::new(
            DEFAULT_SEMITONES,
            DEFAULT_FORMANT_SHIFT_RATIO,
            DEFAULT_CONTOUR_WARP_RANGE,
        );
        transformer.transform(&mut morphed).unwrap();
        let diff: f64 = original
            .samples()
            .iter()
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
            DEFAULT_SEMITONES,
            DEFAULT_FORMANT_SHIFT_RATIO,
            DEFAULT_CONTOUR_WARP_RANGE,
        );
        transformer.transform(&mut audio).unwrap();
        assert_eq!(audio.samples().len(), original_len);
    }

    #[test]
    fn test_voice_morph_differs_from_constant_shift() {
        let mut morphed = speech_like_segment(16000);
        let mut constant = morphed.clone();

        VoiceMorphTransformer::new(
            DEFAULT_SEMITONES,
            DEFAULT_FORMANT_SHIFT_RATIO,
            DEFAULT_CONTOUR_WARP_RANGE,
        )
        .transform(&mut morphed)
        .unwrap();

        // Apply just constant pitch shift + formant for comparison
        use crate::audio::infrastructure::pitch_shift_transformer::PitchShiftTransformer;
        PitchShiftTransformer::new(DEFAULT_SEMITONES)
            .transform(&mut constant)
            .unwrap();
        FormantShiftTransformer::new(DEFAULT_FORMANT_SHIFT_RATIO)
            .transform(&mut constant)
            .unwrap();

        let diff: f64 = morphed
            .samples()
            .iter()
            .zip(constant.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum();
        assert!(
            diff > 0.0,
            "Morph with contour warping should differ from constant shift"
        );
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p faceguard-core -- voice_morph`
Expected: FAIL — `todo!()` panics

**Step 3: Implement VoiceMorphTransformer**

The key insight: instead of using `PitchShiftTransformer` as a black box, this transformer calls the PSOLA helper functions directly from `pitch_shift_transformer` with a per-frame varying shift ratio.

First, make the PSOLA helpers public within the crate. In `pitch_shift_transformer.rs`, change these functions from private to `pub(crate)`:

```rust
pub(crate) fn detect_pitch(frame: &[f64], sample_rate: u32) -> PitchFrame { ... }
pub(crate) fn analyze_pitch(samples: &[f64], sample_rate: u32) -> Vec<PitchFrame> { ... }
pub(crate) fn place_pitch_marks(samples: &[f64], pitch_frames: &[PitchFrame]) -> Vec<usize> { ... }
```

Also make `PitchFrame`, `ANALYSIS_HOP`, and `UNVOICED_MARK_SPACING` `pub(crate)`.

Then implement the `transform` method in `voice_morph_transformer.rs`:

```rust
/// Simple LCG for deterministic random walk.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.state >> 33) as f64;
        let max = (1u64 << 31) as f64;
        (bits / max) * 2.0 - 1.0
    }
}

impl AudioTransformer for VoiceMorphTransformer {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>> {
        use crate::audio::infrastructure::pitch_shift_transformer::{
            analyze_pitch, place_pitch_marks, ANALYSIS_HOP, UNVOICED_MARK_SPACING,
        };

        let samples: Vec<f64> = audio.samples().iter().map(|&s| s as f64).collect();
        let n = samples.len();
        if n < 512 {
            return Ok(());
        }

        let sample_rate = audio.sample_rate();

        // Step 1: Analyze pitch
        let pitch_frames = analyze_pitch(&samples, sample_rate);
        let analysis_marks = place_pitch_marks(&samples, &pitch_frames);

        if analysis_marks.is_empty() {
            return Ok(());
        }

        // Step 2: Generate per-mark shift ratios via random walk
        let mut rng = Lcg::new(42);
        let mut warp_offset = 0.0f64; // current random walk offset in semitones

        let mut output = vec![0.0f64; n];
        let mut window_sum = vec![0.0f64; n];

        // Build synthesis marks with per-mark varying shift
        let mut synthesis_marks = Vec::with_capacity(analysis_marks.len());
        synthesis_marks.push(0.0f64);

        for i in 1..analysis_marks.len() {
            // Random walk step
            warp_offset += rng.next_f64() * CONTOUR_STEP_SIZE * self.contour_warp_range;
            warp_offset = warp_offset.clamp(-self.contour_warp_range, self.contour_warp_range);

            let local_semitones = self.base_semitones + warp_offset;
            let local_ratio = 2.0_f64.powf(local_semitones / 12.0);

            let analysis_spacing =
                analysis_marks[i] as f64 - analysis_marks[i - 1] as f64;
            let synthesis_spacing = analysis_spacing / local_ratio;
            synthesis_marks.push(synthesis_marks[i - 1] + synthesis_spacing);
        }

        // Step 3: PSOLA overlap-add with the warped synthesis marks
        for (i, &analysis_pos) in analysis_marks.iter().enumerate() {
            let frame_idx = (analysis_pos / ANALYSIS_HOP)
                .min(pitch_frames.len().saturating_sub(1));
            let local_period = if frame_idx < pitch_frames.len()
                && pitch_frames[frame_idx].voiced
            {
                pitch_frames[frame_idx].period_samples
            } else {
                UNVOICED_MARK_SPACING
            };

            let half_win = local_period;
            let win_size = 2 * half_win;
            if win_size == 0 {
                continue;
            }

            let hann: Vec<f64> = (0..win_size)
                .map(|j| {
                    0.5 * (1.0 - (2.0 * PI * j as f64 / win_size as f64).cos())
                })
                .collect();

            let grain_start = analysis_pos.saturating_sub(half_win);
            let grain_end = (analysis_pos + half_win).min(n);

            let synth_center = synthesis_marks[i];
            let synth_start = synth_center - half_win as f64;

            for j in 0..(grain_end - grain_start) {
                let src_idx = grain_start + j;
                let dst_f = synth_start + j as f64;
                let dst_idx = dst_f.round() as i64;
                if dst_idx >= 0 && (dst_idx as usize) < n {
                    let d = dst_idx as usize;
                    let win_idx = if analysis_pos >= half_win {
                        j
                    } else {
                        j + (half_win - analysis_pos)
                    };
                    if win_idx < win_size {
                        output[d] += samples[src_idx] * hann[win_idx];
                        window_sum[d] += hann[win_idx];
                    }
                }
            }
        }

        // Normalize
        for i in 0..n {
            if window_sum[i] > 1e-10 {
                output[i] /= window_sum[i];
            }
        }

        // Peak-normalize
        let input_peak = samples.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        let output_peak = output.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        let gain = if output_peak > 1e-10 && output_peak > input_peak {
            input_peak / output_peak
        } else {
            1.0
        };

        // Write PSOLA output back, then apply formant warp
        let out_samples = audio.samples_mut();
        for i in 0..n {
            out_samples[i] = (output[i] * gain) as f32;
        }

        // Step 4: Apply formant warp
        self.formant_shifter.transform(audio)?;

        Ok(())
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p faceguard-core -- voice_morph`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add crates/core/src/audio/infrastructure/pitch_shift_transformer.rs
git add crates/core/src/audio/infrastructure/voice_morph_transformer.rs
git commit -m "feat(audio): rewrite VoiceMorphTransformer with pitch contour warping"
```

---

### Task 5: Update CLI and Desktop Wiring

Update the constructor calls in CLI and desktop to match the new signatures.

**Files:**
- Modify: `crates/cli/src/main.rs:254-277`
- Modify: `crates/desktop/src/workers/blur_worker.rs:259-274`

**Step 1: Update CLI wiring**

In `crates/cli/src/main.rs`, change the transformer selection block (lines ~254-277) to:

```rust
let transformer: Option<
    Box<dyn faceguard_core::audio::domain::audio_transformer::AudioTransformer>,
> = match voice_disguise {
    "low" => {
        use faceguard_core::audio::infrastructure::pitch_shift_transformer::*;
        Some(Box::new(PitchShiftTransformer::new(DEFAULT_SEMITONES)))
    }
    "medium" => {
        use faceguard_core::audio::infrastructure::formant_shift_transformer::*;
        use faceguard_core::audio::infrastructure::pitch_shift_transformer::PitchShiftTransformer;
        // Medium: PSOLA pitch shift + standalone formant warp, composed sequentially
        Some(Box::new(ComposedTransformer::new(vec![
            Box::new(PitchShiftTransformer::new(
                faceguard_core::audio::infrastructure::pitch_shift_transformer::DEFAULT_SEMITONES,
            )),
            Box::new(FormantShiftTransformer::new(DEFAULT_FORMANT_SHIFT_RATIO)),
        ])))
    }
    "high" => {
        use faceguard_core::audio::infrastructure::voice_morph_transformer::*;
        Some(Box::new(VoiceMorphTransformer::new(
            faceguard_core::audio::infrastructure::pitch_shift_transformer::DEFAULT_SEMITONES,
            faceguard_core::audio::infrastructure::formant_shift_transformer::DEFAULT_FORMANT_SHIFT_RATIO,
            DEFAULT_CONTOUR_WARP_RANGE,
        )))
    }
    _ => None,
};
```

Wait — this introduces a `ComposedTransformer` that doesn't exist yet. A simpler approach: keep Medium using `FormantShiftTransformer` but have it also do its own pitch shift internally. But we just decoupled them.

**Simpler approach**: For the Medium tier, create a small `ComposedTransformer` in the domain layer since it's just glue:

Add to `crates/core/src/audio/domain/audio_transformer.rs`:

```rust
/// Applies multiple transformers in sequence.
pub struct ComposedTransformer {
    transformers: Vec<Box<dyn AudioTransformer>>,
}

impl ComposedTransformer {
    pub fn new(transformers: Vec<Box<dyn AudioTransformer>>) -> Self {
        Self { transformers }
    }
}

impl AudioTransformer for ComposedTransformer {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>> {
        for t in &self.transformers {
            t.transform(audio)?;
        }
        Ok(())
    }
}
```

Then update CLI (`crates/cli/src/main.rs` lines ~254-277):

```rust
let transformer: Option<
    Box<dyn faceguard_core::audio::domain::audio_transformer::AudioTransformer>,
> = match voice_disguise {
    "low" => {
        use faceguard_core::audio::infrastructure::pitch_shift_transformer::*;
        Some(Box::new(PitchShiftTransformer::new(DEFAULT_SEMITONES)))
    }
    "medium" => {
        use faceguard_core::audio::domain::audio_transformer::ComposedTransformer;
        use faceguard_core::audio::infrastructure::formant_shift_transformer::*;
        use faceguard_core::audio::infrastructure::pitch_shift_transformer::{
            PitchShiftTransformer, DEFAULT_SEMITONES,
        };
        Some(Box::new(ComposedTransformer::new(vec![
            Box::new(PitchShiftTransformer::new(DEFAULT_SEMITONES)),
            Box::new(FormantShiftTransformer::new(DEFAULT_FORMANT_SHIFT_RATIO)),
        ])))
    }
    "high" => {
        use faceguard_core::audio::infrastructure::formant_shift_transformer::DEFAULT_FORMANT_SHIFT_RATIO;
        use faceguard_core::audio::infrastructure::pitch_shift_transformer::DEFAULT_SEMITONES;
        use faceguard_core::audio::infrastructure::voice_morph_transformer::*;
        Some(Box::new(VoiceMorphTransformer::new(
            DEFAULT_SEMITONES,
            DEFAULT_FORMANT_SHIFT_RATIO,
            DEFAULT_CONTOUR_WARP_RANGE,
        )))
    }
    _ => None,
};
```

And update Desktop (`crates/desktop/src/workers/blur_worker.rs` lines ~259-274):

```rust
let transformer: Option<Box<dyn AudioTransformer>> = match params.voice_disguise {
    crate::settings::VoiceDisguise::Off => None,
    crate::settings::VoiceDisguise::Low => Some(Box::new(PitchShiftTransformer::new(
        faceguard_core::audio::infrastructure::pitch_shift_transformer::DEFAULT_SEMITONES,
    ))),
    crate::settings::VoiceDisguise::Medium => {
        use faceguard_core::audio::domain::audio_transformer::ComposedTransformer;
        Some(Box::new(ComposedTransformer::new(vec![
            Box::new(PitchShiftTransformer::new(
                faceguard_core::audio::infrastructure::pitch_shift_transformer::DEFAULT_SEMITONES,
            )),
            Box::new(FormantShiftTransformer::new(
                faceguard_core::audio::infrastructure::formant_shift_transformer::DEFAULT_FORMANT_SHIFT_RATIO,
            )),
        ])))
    }
    crate::settings::VoiceDisguise::High => Some(Box::new(VoiceMorphTransformer::new(
        faceguard_core::audio::infrastructure::pitch_shift_transformer::DEFAULT_SEMITONES,
        faceguard_core::audio::infrastructure::formant_shift_transformer::DEFAULT_FORMANT_SHIFT_RATIO,
        faceguard_core::audio::infrastructure::voice_morph_transformer::DEFAULT_CONTOUR_WARP_RANGE,
    ))),
};
```

**Step 2: Verify it compiles**

Run: `cargo build`
Expected: Compiles without errors

**Step 3: Run all tests**

Run: `cargo test`
Expected: All tests pass

**Step 4: Run clippy**

Run: `cargo clippy --all-targets`
Expected: No warnings

**Step 5: Commit**

```bash
git add crates/core/src/audio/domain/audio_transformer.rs
git add crates/cli/src/main.rs
git add crates/desktop/src/workers/blur_worker.rs
git commit -m "feat(audio): wire up PSOLA transformers in CLI and desktop"
```

---

### Task 6: Update CLAUDE.md Constants Table

Update the domain constants table in CLAUDE.md to reflect the new values.

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update constants table**

Remove the old audio constants and add the new ones. In the "Key Domain Constants" table, the audio-related entries should now be:

| Constant | Value | Location |
|----------|-------|----------|
| Default semitones | 2.5 | PitchShiftTransformer |
| Default formant ratio | 1.15 | FormantShiftTransformer |
| Contour warp range | 0.5 | VoiceMorphTransformer |
| Voicing threshold | 0.3 | pitch_shift_transformer::detect_pitch |
| Analysis frame size | 512 | pitch_shift_transformer |
| Unvoiced mark spacing | 80 | pitch_shift_transformer |

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update domain constants for PSOLA voice disguise"
```

---

### Task 7: Clean Up Unused Dependencies

Remove `rustfft` usage from `pitch_shift_transformer.rs` and `voice_morph_transformer.rs` (they no longer use FFT). Only `formant_shift_transformer.rs` still needs it.

**Files:**
- Modify: `crates/core/src/audio/infrastructure/pitch_shift_transformer.rs` — remove `use rustfft` if present
- Modify: `crates/core/src/audio/infrastructure/voice_morph_transformer.rs` — remove `use rustfft` if present

**Step 1: Remove unused imports**

In `pitch_shift_transformer.rs`: remove any `use rustfft::*` lines.
In `voice_morph_transformer.rs`: remove any `use rustfft::*` lines.

**Step 2: Verify**

Run: `cargo clippy --all-targets`
Expected: No warnings about unused imports

**Step 3: Commit**

```bash
git add crates/core/src/audio/infrastructure/pitch_shift_transformer.rs
git add crates/core/src/audio/infrastructure/voice_morph_transformer.rs
git commit -m "chore: remove unused rustfft imports from PSOLA transformers"
```
