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
pub(crate) const ANALYSIS_FRAME_SIZE: usize = 512;

/// Hop between analysis frames.
pub(crate) const ANALYSIS_HOP: usize = 256;

/// Voicing threshold: normalized autocorrelation peak must exceed this
/// to classify a frame as voiced.
const VOICING_THRESHOLD: f64 = 0.3;

/// Fixed pitch mark spacing for unvoiced segments (in samples at 16kHz = ~5ms).
pub(crate) const UNVOICED_MARK_SPACING: usize = 80;

/// Result of analyzing one frame for pitch.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PitchFrame {
    /// True if the frame is voiced (periodic).
    pub(crate) voiced: bool,
    /// Detected pitch period in samples (only meaningful if voiced).
    pub(crate) period_samples: usize,
}

/// Detect pitch in a single frame using autocorrelation.
///
/// Returns a PitchFrame with voiced/unvoiced classification and detected period.
/// `sample_rate` is needed to convert min/max F0 to lag bounds.
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

    if min_lag >= max_lag {
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

pub struct PitchShiftTransformer {
    semitones: f64,
}

impl PitchShiftTransformer {
    pub fn new(semitones: f64) -> Self {
        Self { semitones }
    }
}

/// Analyze the entire signal and return per-frame pitch information.
pub(crate) fn analyze_pitch(samples: &[f64], sample_rate: u32) -> Vec<PitchFrame> {
    let n = samples.len();
    let mut frames = Vec::new();
    let mut pos = 0;
    while pos + ANALYSIS_FRAME_SIZE <= n {
        frames.push(detect_pitch(
            &samples[pos..pos + ANALYSIS_FRAME_SIZE],
            sample_rate,
        ));
        pos += ANALYSIS_HOP;
    }
    frames
}

/// Place pitch marks throughout the signal based on detected pitch.
/// Returns sample indices where each pitch mark is placed.
pub(crate) fn place_pitch_marks(samples: &[f64], pitch_frames: &[PitchFrame]) -> Vec<usize> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    let mut marks = Vec::new();
    let mut pos: usize = 0;

    while pos < n {
        marks.push(pos);
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

/// Find the index of the analysis mark nearest to `target_pos`.
pub(crate) fn find_nearest_mark(analysis_marks: &[usize], target_pos: f64) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (i, &mark) in analysis_marks.iter().enumerate() {
        let dist = (mark as f64 - target_pos).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        } else {
            // Marks are sorted, so once distance increases we can stop
            break;
        }
    }
    best_idx
}

/// Get the local pitch period at a given analysis mark position.
pub(crate) fn local_period_at(analysis_pos: usize, pitch_frames: &[PitchFrame]) -> usize {
    let frame_idx = (analysis_pos / ANALYSIS_HOP).min(pitch_frames.len().saturating_sub(1));
    if frame_idx < pitch_frames.len() && pitch_frames[frame_idx].voiced {
        pitch_frames[frame_idx].period_samples
    } else {
        UNVOICED_MARK_SPACING
    }
}

/// Core PSOLA overlap-add: place grains from analysis positions at synthesis positions.
///
/// `grain_sources` is a list of (synthesis_center, analysis_mark_index) pairs.
/// Each pair says "place the grain from analysis_marks[idx] at synthesis_center".
pub(crate) fn psola_overlap_add(
    samples: &[f64],
    analysis_marks: &[usize],
    pitch_frames: &[PitchFrame],
    grain_sources: &[(f64, usize)],
) -> Vec<f64> {
    let n = samples.len();
    let mut output = vec![0.0f64; n];
    let mut window_sum = vec![0.0f64; n];

    for &(synth_center, analysis_idx) in grain_sources {
        let analysis_pos = analysis_marks[analysis_idx];
        let local_period = local_period_at(analysis_pos, pitch_frames);

        let half_win = local_period;
        let win_size = 2 * half_win;
        if win_size == 0 {
            continue;
        }

        let hann: Vec<f64> = (0..win_size)
            .map(|j| 0.5 * (1.0 - (2.0 * PI * j as f64 / win_size as f64).cos()))
            .collect();

        let grain_start = analysis_pos.saturating_sub(half_win);
        let grain_end = (analysis_pos + half_win).min(n);

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

/// PSOLA synthesis: given analysis marks and a shift ratio, produce output.
/// shift_ratio > 1 means higher pitch (shorter periods).
///
/// Generates synthesis marks covering the full output duration. Each synthesis
/// mark maps back to the nearest analysis grain via `synth_pos * shift_ratio`.
fn psola_synthesize(
    samples: &[f64],
    analysis_marks: &[usize],
    pitch_frames: &[PitchFrame],
    shift_ratio: f64,
) -> Vec<f64> {
    let n = samples.len();

    if analysis_marks.is_empty() {
        return vec![0.0f64; n];
    }

    // Generate synthesis marks covering the full output duration.
    // For each synthesis position, map back to analysis space to find the
    // nearest grain to borrow. This ensures the output covers the entire
    // signal without trailing silence.
    let mut grain_sources: Vec<(f64, usize)> = Vec::new();
    let mut synth_pos = 0.0f64;

    while synth_pos < n as f64 {
        // Use grain from the same time position (time-preserving pitch shift)
        let nearest_idx = find_nearest_mark(analysis_marks, synth_pos);
        grain_sources.push((synth_pos, nearest_idx));

        // Advance by the local period at the desired output pitch
        let analysis_pos = analysis_marks[nearest_idx];
        let period = local_period_at(analysis_pos, pitch_frames);
        synth_pos += (period as f64 / shift_ratio).max(1.0);
    }

    psola_overlap_add(samples, analysis_marks, pitch_frames, &grain_sources)
}

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

        let pitch_frames = analyze_pitch(&samples, sample_rate);
        let analysis_marks = place_pitch_marks(&samples, &pitch_frames);
        let output = psola_synthesize(&samples, &analysis_marks, &pitch_frames, shift_ratio);

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
            (result.period_samples as i32 - expected_period).unsigned_abs() <= 3,
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
            (result.period_samples as i32 - expected_period).unsigned_abs() <= 2,
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
                rng_state = rng_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
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
}
