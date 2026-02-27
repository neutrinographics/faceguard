use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Default pitch shift in semitones (upward).
pub const DEFAULT_SEMITONES: f64 = 4.0;

/// STFT analysis/synthesis window size.
const WINDOW_SIZE: usize = 2048;

/// Hop size between successive STFT frames.
const HOP_SIZE: usize = 512;

/// Phase vocoder-based pitch shifter.
///
/// Shifts pitch by a configurable number of semitones using
/// STFT -> frequency bin shifting -> ISTFT with overlap-add.
pub struct PitchShiftTransformer {
    semitones: f64,
}

impl PitchShiftTransformer {
    pub fn new(semitones: f64) -> Self {
        Self { semitones }
    }
}

impl AudioTransformer for PitchShiftTransformer {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>> {
        // Zero semitone shift is identity -- return early.
        if self.semitones.abs() < 1e-10 {
            return Ok(());
        }

        let samples = audio.samples().to_vec();
        let n = samples.len();
        if n < WINDOW_SIZE {
            return Ok(());
        }

        let shift_ratio = 2.0_f64.powf(self.semitones / 12.0);
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

        // Phase tracking for synthesis
        let mut prev_phase = vec![0.0f64; half_window];
        let mut synth_phase = vec![0.0f64; half_window];

        let expected_phase_advance: Vec<f64> = (0..half_window)
            .map(|k| 2.0 * PI * k as f64 * HOP_SIZE as f64 / WINDOW_SIZE as f64)
            .collect();

        let num_frames = if n >= WINDOW_SIZE {
            (n - WINDOW_SIZE) / HOP_SIZE + 1
        } else {
            0
        };

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_SIZE;

            // Analysis: window the input frame and FFT
            let mut fft_buf: Vec<Complex<f64>> = (0..WINDOW_SIZE)
                .map(|i| Complex::new(samples[start + i] as f64 * hann[i], 0.0))
                .collect();

            fft_forward.process(&mut fft_buf);

            // Extract magnitude and phase for positive frequencies
            let magnitudes: Vec<f64> = fft_buf[..half_window].iter().map(|c| c.norm()).collect();
            let phases: Vec<f64> = fft_buf[..half_window]
                .iter()
                .map(|c| c.im.atan2(c.re))
                .collect();

            // Compute instantaneous frequency for each bin
            let inst_freq: Vec<f64> = (0..half_window)
                .map(|k| {
                    let phase_diff = phases[k] - prev_phase[k] - expected_phase_advance[k];
                    // Wrap to [-pi, pi]
                    let wrapped = phase_diff - (2.0 * PI) * (phase_diff / (2.0 * PI)).round();
                    expected_phase_advance[k] + wrapped
                })
                .collect();

            // Shift bins: map each analysis bin to a new synthesis bin.
            // Use max magnitude when multiple source bins map to the same target
            // to avoid amplitude blowup.
            let mut new_magnitudes = vec![0.0f64; half_window];
            let mut new_inst_freq = vec![0.0f64; half_window];

            for k in 0..half_window {
                let new_bin = (k as f64 * shift_ratio).round() as usize;
                if new_bin < half_window && magnitudes[k] > new_magnitudes[new_bin] {
                    new_magnitudes[new_bin] = magnitudes[k];
                    new_inst_freq[new_bin] = inst_freq[k] * shift_ratio;
                }
            }

            // Accumulate synthesis phase
            for k in 0..half_window {
                synth_phase[k] += new_inst_freq[k];
            }

            // Reconstruct complex spectrum
            let mut synth_buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); WINDOW_SIZE];
            for k in 0..half_window {
                synth_buf[k] = Complex::new(
                    new_magnitudes[k] * synth_phase[k].cos(),
                    new_magnitudes[k] * synth_phase[k].sin(),
                );
            }
            // Mirror for negative frequencies (conjugate symmetry for real output)
            for k in 1..half_window - 1 {
                synth_buf[WINDOW_SIZE - k] = synth_buf[k].conj();
            }

            fft_inverse.process(&mut synth_buf);

            // Normalize IFFT (rustfft does not normalize)
            let norm = 1.0 / WINDOW_SIZE as f64;

            // Overlap-add with synthesis window
            for i in 0..WINDOW_SIZE {
                if start + i < n {
                    let val = synth_buf[i].re * norm * hann[i];
                    output[start + i] += val;
                    window_sum[start + i] += hann[i] * hann[i];
                }
            }

            // Save current phase for next frame
            prev_phase.copy_from_slice(&phases);
        }

        // Determine a stable window_sum threshold.
        // Use the median of non-zero window_sum values to find the "steady state" value,
        // then only trust samples where window_sum is at least half of that.
        let max_window_sum = window_sum.iter().cloned().fold(0.0f64, f64::max);
        let ws_threshold = max_window_sum * 0.1;

        // Normalize by window sum to compensate overlap-add
        for i in 0..n {
            if window_sum[i] >= ws_threshold {
                output[i] /= window_sum[i];
            } else {
                // Edge region: fade to zero to avoid artifacts
                output[i] = 0.0;
            }
        }

        // Peak-normalize: ensure output peak does not exceed input peak
        let input_peak = samples
            .iter()
            .map(|s| s.abs() as f64)
            .fold(0.0f64, f64::max);
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
        let original = sine_segment(440.0, 1.0, 16000);
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
