use std::path::{Path, PathBuf};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::speech_recognizer::SpeechRecognizer;
use crate::audio::domain::transcript::TranscriptWord;

/// Speech recognizer using whisper.cpp via whisper-rs.
///
/// Transcribes audio to word-level timestamped text using the Whisper tiny.en model.
#[derive(Debug)]
pub struct WhisperRecognizer {
    model_path: PathBuf,
}

impl WhisperRecognizer {
    pub fn new(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        if !model_path.exists() {
            return Err(format!("Whisper model not found at: {}", model_path.display()).into());
        }
        Ok(Self {
            model_path: model_path.to_path_buf(),
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}

impl SpeechRecognizer for WhisperRecognizer {
    fn transcribe(
        &self,
        audio: &AudioSegment,
    ) -> Result<Vec<TranscriptWord>, Box<dyn std::error::Error>> {
        let ctx = WhisperContext::new_with_params(
            self.model_path.to_str().ok_or("Invalid model path")?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| format!("Failed to load Whisper model: {e}"))?;

        let mut state = ctx
            .create_state()
            .map_err(|e| format!("Failed to create Whisper state: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        params.set_language(Some("en"));
        params.set_translate(false);
        params.set_token_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_n_threads(num_cpus().min(4) as i32);

        let samples = audio.samples();
        state
            .full(params, samples)
            .map_err(|e| format!("Whisper inference failed: {e}"))?;

        let mut words = Vec::new();
        let num_segments = state.full_n_segments();

        for seg_idx in 0..num_segments {
            let segment = match state.get_segment(seg_idx) {
                Some(s) => s,
                None => continue,
            };

            let n_tokens = segment.n_tokens();
            for tok_idx in 0..n_tokens {
                let token = match segment.get_token(tok_idx) {
                    Some(t) => t,
                    None => continue,
                };

                let text = match token.to_str() {
                    Ok(t) => t,
                    Err(_) => continue,
                };

                // Skip special tokens (start with [, like [_BEG_], [_SOT_], etc.)
                let trimmed = text.trim();
                if trimmed.is_empty() || trimmed.starts_with('[') || trimmed.starts_with('<') {
                    continue;
                }

                let token_data = token.token_data();
                let prob = token.token_probability();

                // Token timestamps are in centiseconds (10ms units)
                let start_time = token_data.t0 as f64 / 100.0;
                let end_time = token_data.t1 as f64 / 100.0;

                // Skip tokens with invalid timestamps
                if end_time <= start_time {
                    continue;
                }

                words.push(TranscriptWord {
                    word: trimmed.to_string(),
                    start_time,
                    end_time,
                    confidence: prob,
                });
            }
        }

        Ok(words)
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_nonexistent_path_returns_error() {
        let result = WhisperRecognizer::new(std::path::Path::new("/nonexistent/model.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn test_new_nonexistent_path_error_message() {
        let result = WhisperRecognizer::new(std::path::Path::new("/nonexistent/model.bin"));
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found"),
            "Expected 'not found' in error, got: {err}"
        );
    }

    #[test]
    #[ignore] // Requires whisper model file
    fn test_transcribe_does_not_crash_on_sine_wave() {
        let model_path = crate::detection::infrastructure::model_resolver::resolve(
            crate::shared::constants::WHISPER_MODEL_NAME,
            crate::shared::constants::WHISPER_MODEL_URL,
            None,
            None,
        )
        .expect("Failed to resolve whisper model");

        let recognizer = WhisperRecognizer::new(&model_path).expect("Failed to create recognizer");

        let sample_rate = 16000u32;
        let len = (3.0 * sample_rate as f64) as usize;
        let samples: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32
            })
            .collect();
        let audio = AudioSegment::new(samples, sample_rate, 1);

        let result = recognizer.transcribe(&audio);
        assert!(result.is_ok(), "Transcription should not error: {result:?}");
    }
}
