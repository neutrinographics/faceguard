use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::audio_transformer::AudioTransformer;
use crate::audio::infrastructure::formant_shift_transformer::FormantShiftTransformer;

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
            analyze_pitch, find_nearest_mark, local_period_at, place_pitch_marks,
            psola_overlap_add, ANALYSIS_FRAME_SIZE,
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

        // Step 2: Generate synthesis marks covering the full output duration
        // with per-mark varying shift via random walk
        let mut rng = Lcg::new(42);
        let mut warp_offset = 0.0f64;
        let mut grain_sources: Vec<(f64, usize)> = Vec::new();
        let mut synth_pos = 0.0f64;

        while synth_pos < n as f64 {
            // Random walk step for pitch contour warping
            warp_offset += rng.next_f64() * CONTOUR_STEP_SIZE * self.contour_warp_range;
            warp_offset = warp_offset.clamp(-self.contour_warp_range, self.contour_warp_range);

            let local_semitones = self.base_semitones + warp_offset;
            let local_ratio = 2.0_f64.powf(local_semitones / 12.0);

            // Map synthesis position back to analysis time
            let analysis_time = synth_pos * local_ratio;
            let nearest_idx =
                find_nearest_mark(&analysis_marks, analysis_time);
            grain_sources.push((synth_pos, nearest_idx));

            // Advance by local period at output pitch
            let analysis_pos = analysis_marks[nearest_idx];
            let period = local_period_at(analysis_pos, &pitch_frames);
            synth_pos += (period as f64 / local_ratio).max(1.0);
        }

        // Step 3: PSOLA overlap-add using shared core
        let output =
            psola_overlap_add(&samples, &analysis_marks, &pitch_frames, &grain_sources);

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
    use std::f64::consts::PI;

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
