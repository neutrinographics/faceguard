use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;
use crate::audio::infrastructure::formant_shift_transformer::FormantShiftTransformer;
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
            analyze_pitch, place_pitch_marks, ANALYSIS_FRAME_SIZE, ANALYSIS_HOP,
            UNVOICED_MARK_SPACING,
        };

        let samples: Vec<f64> = audio.samples().iter().map(|&s| s as f64).collect();
        let n = samples.len();
        if n < ANALYSIS_FRAME_SIZE {
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
        let mut warp_offset = 0.0f64;

        let mut output = vec![0.0f64; n];
        let mut window_sum = vec![0.0f64; n];

        // Build synthesis marks with per-mark varying shift
        let mut synthesis_marks = Vec::with_capacity(analysis_marks.len());
        synthesis_marks.push(0.0f64);

        for i in 1..analysis_marks.len() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::infrastructure::formant_shift_transformer::DEFAULT_FORMANT_SHIFT_RATIO;
    use crate::audio::infrastructure::pitch_shift_transformer::DEFAULT_SEMITONES;

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
