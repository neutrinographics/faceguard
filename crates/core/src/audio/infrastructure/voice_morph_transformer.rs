use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;
use crate::audio::infrastructure::formant_shift_transformer::FormantShiftTransformer;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Default pitch shift in semitones for the voice morph transformer.
pub const DEFAULT_MORPH_SEMITONES: f64 = 4.0;

/// Default ratio by which formant frequencies are shifted in the morph transformer.
pub const DEFAULT_MORPH_FORMANT_RATIO: f64 = 1.2;

/// Default amount of spectral jitter applied to break voiceprint patterns.
pub const DEFAULT_JITTER_AMOUNT: f64 = 0.15;

/// Exponential smoothing factor for jitter across frames (private).
const SPECTRAL_SMOOTHING: f64 = 0.3;

/// STFT frame size (matches other transformers).
const WINDOW_SIZE: usize = 2048;

/// Hop size between successive analysis frames.
const HOP_SIZE: usize = 512;

/// Highest tier voice disguise that combines pitch shift, formant shift, and
/// spectral envelope jitter to make the voice unrecognizable.
///
/// It first delegates to `FormantShiftTransformer` for pitch and formant modification,
/// then applies per-bin random gain perturbation with exponential smoothing across
/// frames to break voiceprint patterns while keeping speech intelligible.
pub struct VoiceMorphTransformer {
    formant_shifter: FormantShiftTransformer,
    jitter_amount: f64,
}

/// Simple linear congruential generator for deterministic pseudo-random values.
/// No external dependency needed.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a pseudo-random f64 in the range [-1.0, 1.0].
    fn next_f64(&mut self) -> f64 {
        // LCG parameters from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // Use upper bits for better quality
        let bits = (self.state >> 33) as f64;
        let max = (1u64 << 31) as f64;
        (bits / max) * 2.0 - 1.0
    }
}

impl VoiceMorphTransformer {
    pub fn new(semitones: f64, formant_ratio: f64, jitter_amount: f64) -> Self {
        Self {
            formant_shifter: FormantShiftTransformer::new(formant_ratio),
            jitter_amount,
        }
    }
}

impl AudioTransformer for VoiceMorphTransformer {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>> {
        // Step 1: Apply formant shift (which internally applies pitch shift)
        self.formant_shifter.transform(audio)?;

        // Step 2: Apply spectral jitter
        if self.jitter_amount.abs() < 1e-10 {
            return Ok(());
        }

        let samples: Vec<f64> = audio.samples().iter().map(|&s| s as f64).collect();
        let n = samples.len();
        if n < WINDOW_SIZE {
            return Ok(());
        }

        let half_window = WINDOW_SIZE / 2 + 1;

        // Precompute Hann window
        let hann: Vec<f64> = (0..WINDOW_SIZE)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / WINDOW_SIZE as f64).cos()))
            .collect();

        let mut output = vec![0.0f64; n];
        let mut window_sum = vec![0.0f64; n];

        let mut planner = FftPlanner::<f64>::new();
        let fft_forward = planner.plan_fft_forward(WINDOW_SIZE);
        let fft_inverse = planner.plan_fft_inverse(WINDOW_SIZE);

        let num_frames = (n - WINDOW_SIZE) / HOP_SIZE + 1;

        // Smoothed jitter gains across frames (initialized to 1.0 = no change)
        let mut smoothed_gains = vec![1.0f64; half_window];

        // Deterministic PRNG with fixed seed
        let mut rng = Lcg::new(42);

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_SIZE;

            // Window the frame
            let mut fft_buf: Vec<Complex<f64>> = (0..WINDOW_SIZE)
                .map(|i| Complex::new(samples[start + i] * hann[i], 0.0))
                .collect();

            // Forward FFT
            fft_forward.process(&mut fft_buf);

            // Apply jittered gains to each frequency bin
            for k in 0..half_window {
                // Generate deterministic random perturbation
                let random_val = rng.next_f64();
                let target_gain = 1.0 + self.jitter_amount * random_val;

                // Exponential smoothing across frames to avoid crackling
                smoothed_gains[k] = SPECTRAL_SMOOTHING * target_gain
                    + (1.0 - SPECTRAL_SMOOTHING) * smoothed_gains[k];

                fft_buf[k] *= smoothed_gains[k];
            }

            // Mirror for negative frequencies (conjugate symmetry)
            for k in 1..half_window - 1 {
                fft_buf[WINDOW_SIZE - k] = fft_buf[k].conj();
            }

            // Inverse FFT
            fft_inverse.process(&mut fft_buf);
            let norm = 1.0 / WINDOW_SIZE as f64;

            // Overlap-add with synthesis window
            for i in 0..WINDOW_SIZE {
                if start + i < n {
                    let val = fft_buf[i].re * norm * hann[i];
                    if val.is_finite() {
                        output[start + i] += val;
                    }
                    window_sum[start + i] += hann[i] * hann[i];
                }
            }
        }

        // Normalize by window sum
        let max_ws = window_sum.iter().cloned().fold(0.0f64, f64::max);
        let ws_threshold = max_ws * 0.1;

        for i in 0..n {
            if window_sum[i] >= ws_threshold {
                output[i] /= window_sum[i];
            } else {
                output[i] = 0.0;
            }
        }

        // Peak-normalize to avoid clipping
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
    fn test_voice_morph_changes_audio() {
        let original = speech_like_segment(16000);
        let mut morphed = original.clone();
        let transformer = VoiceMorphTransformer::new(
            DEFAULT_MORPH_SEMITONES,
            DEFAULT_MORPH_FORMANT_RATIO,
            DEFAULT_JITTER_AMOUNT,
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
            .transform(&mut morphed)
            .unwrap();
        crate::audio::infrastructure::formant_shift_transformer::FormantShiftTransformer::new(1.2)
        .transform(&mut formant_only)
        .unwrap();

        let diff: f64 = morphed
            .samples()
            .iter()
            .zip(formant_only.samples().iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum();
        assert!(
            diff > 0.0,
            "Morph should differ from plain formant shift due to jitter"
        );
    }
}
