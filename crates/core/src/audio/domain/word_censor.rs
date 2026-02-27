use super::audio_segment::AudioSegment;
use super::censor_region::CensorRegion;
use super::transcript::TranscriptWord;

pub const DEFAULT_BLEEP_PADDING: f64 = 0.05;
pub const DEFAULT_BLEEP_FREQUENCY: f64 = 1000.0;

/// How censored regions are replaced in the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BleepMode {
    /// Replace with a sine-wave tone at a given frequency.
    Tone,
    /// Replace with silence (zero samples).
    Silence,
}

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

    pub fn apply_bleep(
        audio: &mut AudioSegment,
        regions: &[CensorRegion],
        frequency: f64,
        mode: BleepMode,
    ) {
        let sample_rate = audio.sample_rate() as f64;
        let channels = audio.channels() as usize;

        for region in regions {
            let start = audio.sample_index_at_time(region.effective_start());
            let end = audio
                .sample_index_at_time(region.effective_end())
                .min(audio.samples().len());

            let samples = audio.samples_mut();
            match mode {
                BleepMode::Tone => {
                    for (offset, sample) in samples[start..end].iter_mut().enumerate() {
                        let t = offset as f64 / (sample_rate * channels as f64);
                        *sample = (2.0 * std::f64::consts::PI * frequency * t).sin() as f32 * 0.3;
                    }
                }
                BleepMode::Silence => {
                    for sample in samples[start..end].iter_mut() {
                        *sample = 0.0;
                    }
                }
            }
        }
    }
}

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
        WordCensor::apply_bleep(
            &mut audio,
            &regions,
            DEFAULT_BLEEP_FREQUENCY,
            BleepMode::Tone,
        );

        let start = audio.sample_index_at_time(0.5);
        let end = audio.sample_index_at_time(1.0);
        let bleep_energy: f64 = audio.samples()[start..end]
            .iter()
            .map(|s| (*s as f64) * (*s as f64))
            .sum();
        assert!(
            bleep_energy > 0.0,
            "Bleep region should have non-zero energy"
        );
    }

    #[test]
    fn test_apply_bleep_leaves_non_region_untouched() {
        let mut audio = silent_segment(2.0, 16000);
        let regions = vec![CensorRegion {
            start_time: 0.5,
            end_time: 1.0,
            padding: 0.0,
        }];
        WordCensor::apply_bleep(
            &mut audio,
            &regions,
            DEFAULT_BLEEP_FREQUENCY,
            BleepMode::Tone,
        );

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
        WordCensor::apply_bleep(
            &mut audio,
            &regions,
            DEFAULT_BLEEP_FREQUENCY,
            BleepMode::Tone,
        );

        let idx = audio.sample_index_at_time(0.95);
        assert!(audio.samples()[idx].abs() > 0.0);
    }

    #[test]
    fn test_apply_bleep_empty_regions_no_change() {
        let mut audio = silent_segment(1.0, 16000);
        let original = audio.samples().to_vec();
        WordCensor::apply_bleep(&mut audio, &[], DEFAULT_BLEEP_FREQUENCY, BleepMode::Tone);
        assert_eq!(audio.samples(), &original[..]);
    }

    #[test]
    fn test_apply_bleep_silence_zeroes_region() {
        let mut audio = AudioSegment::new(vec![0.5f32; 32000], 16000, 1);
        let regions = vec![CensorRegion {
            start_time: 0.5,
            end_time: 1.0,
            padding: 0.0,
        }];
        WordCensor::apply_bleep(
            &mut audio,
            &regions,
            DEFAULT_BLEEP_FREQUENCY,
            BleepMode::Silence,
        );

        let start = audio.sample_index_at_time(0.5);
        let end = audio.sample_index_at_time(1.0);

        // Censored region should be all zeros
        let region_energy: f64 = audio.samples()[start..end]
            .iter()
            .map(|s| (*s as f64).powi(2))
            .sum();
        assert_relative_eq!(region_energy, 0.0);

        // Before and after should still be non-zero
        assert!(audio.samples()[0].abs() > 0.0);
        assert!(audio.samples()[end + 1].abs() > 0.0);
    }
}
