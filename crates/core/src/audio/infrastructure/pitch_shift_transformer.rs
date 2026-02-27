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
