use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Default ratio by which formant frequencies are shifted.
pub const DEFAULT_FORMANT_SHIFT_RATIO: f64 = 1.15;

/// Order of the LPC (Linear Predictive Coding) analysis.
const LPC_ORDER: usize = 16;

/// STFT frame size for formant analysis (matches pitch shift transformer).
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

/// Compute autocorrelation of `x` up to lag `order`.
fn autocorrelation(x: &[f64], order: usize) -> Vec<f64> {
    let n = x.len();
    (0..=order)
        .map(|lag| {
            let mut sum = 0.0;
            for i in 0..n - lag {
                sum += x[i] * x[i + lag];
            }
            sum
        })
        .collect()
}

/// Levinson-Durbin recursion: given autocorrelation values, compute LPC coefficients.
/// Returns (coefficients of length order+1 with a[0]=1, prediction_error).
fn levinson_durbin(r: &[f64], order: usize) -> (Vec<f64>, f64) {
    let mut a = vec![0.0; order + 1];
    let mut a_prev = vec![0.0; order + 1];
    a[0] = 1.0;
    a_prev[0] = 1.0;

    let mut error = r[0];
    if error.abs() < 1e-30 {
        return (a, error);
    }

    for i in 1..=order {
        // Compute reflection coefficient
        let mut lambda = 0.0;
        for j in 0..i {
            lambda -= a_prev[j] * r[i - j];
        }
        lambda /= error;

        // Clamp reflection coefficient for stability
        lambda = lambda.clamp(-0.999, 0.999);

        // Update coefficients
        for j in 0..=i {
            a[j] = a_prev[j] + lambda * a_prev[i.saturating_sub(j)];
        }

        error *= 1.0 - lambda * lambda;
        if error.abs() < 1e-30 {
            break;
        }

        a_prev[..=i].copy_from_slice(&a[..=i]);
    }

    (a, error)
}

/// Compute the LPC spectral envelope magnitude at `num_bins` frequency points.
/// The envelope is |1 / A(e^jw)| where A(z) = sum_k a[k] z^{-k}.
fn lpc_spectral_envelope(a: &[f64], gain: f64, num_bins: usize) -> Vec<f64> {
    let order = a.len() - 1;
    (0..num_bins)
        .map(|k| {
            let omega = PI * k as f64 / num_bins as f64;
            let mut re = 0.0;
            let mut im = 0.0;
            for (i, &coeff) in a.iter().enumerate().take(order + 1) {
                re += coeff * (omega * i as f64).cos();
                im -= coeff * (omega * i as f64).sin();
            }
            let mag_sq = re * re + im * im;
            if mag_sq > 1e-30 {
                gain / mag_sq.sqrt()
            } else {
                gain * 1e15
            }
        })
        .collect()
}

/// Shift the spectral envelope by `ratio`, resampling via linear interpolation.
fn shift_envelope(envelope: &[f64], ratio: f64) -> Vec<f64> {
    let n = envelope.len();
    (0..n)
        .map(|k| {
            // For ratio > 1, formants shift up: source frequency is lower
            let src = k as f64 / ratio;
            let idx = src.floor() as usize;
            let frac = src - idx as f64;
            if idx + 1 < n {
                envelope[idx] * (1.0 - frac) + envelope[idx + 1] * frac
            } else if idx < n {
                envelope[idx] * (1.0 - frac)
            } else {
                envelope[n - 1]
            }
        })
        .collect()
}

impl AudioTransformer for FormantShiftTransformer {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>> {
        // Apply formant modification via spectral envelope reshaping
        if (self.formant_ratio - 1.0).abs() < 1e-10 {
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

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_SIZE;

            // Window the frame
            let windowed: Vec<f64> = (0..WINDOW_SIZE)
                .map(|i| samples[start + i] * hann[i])
                .collect();

            // LPC analysis on the windowed frame
            let r = autocorrelation(&windowed, LPC_ORDER);
            if r[0].abs() < 1e-30 {
                // Silent frame
                for i in 0..WINDOW_SIZE {
                    if start + i < n {
                        window_sum[start + i] += hann[i] * hann[i];
                    }
                }
                continue;
            }

            let (lpc_coeffs, lpc_error) = levinson_durbin(&r, LPC_ORDER);
            let lpc_gain = lpc_error.abs().sqrt().max(1e-15);

            // Compute original and shifted spectral envelopes
            let original_env = lpc_spectral_envelope(&lpc_coeffs, lpc_gain, half_window);
            let shifted_env = shift_envelope(&original_env, self.formant_ratio);

            // FFT the windowed frame
            let mut fft_buf: Vec<Complex<f64>> =
                windowed.iter().map(|&s| Complex::new(s, 0.0)).collect();
            fft_forward.process(&mut fft_buf);

            // Modify the spectrum: divide by original envelope, multiply by shifted
            for k in 0..half_window {
                let orig_mag = original_env[k].max(1e-15);
                let new_mag = shifted_env[k];
                let ratio = new_mag / orig_mag;
                // Clamp the ratio to avoid extreme amplification
                let clamped_ratio = ratio.clamp(0.01, 100.0);
                fft_buf[k] *= clamped_ratio;
            }
            // Mirror for negative frequencies (conjugate symmetry)
            for k in 1..half_window - 1 {
                fft_buf[WINDOW_SIZE - k] = fft_buf[k].conj();
            }

            // IFFT
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
