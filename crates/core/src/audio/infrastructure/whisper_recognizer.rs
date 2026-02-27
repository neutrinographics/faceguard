use std::path::{Path, PathBuf};

use crate::audio::domain::audio_segment::AudioSegment;
use crate::audio::domain::speech_recognizer::SpeechRecognizer;
use crate::audio::domain::transcript::TranscriptWord;

/// Speech recognizer using whisper.cpp via whisper-rs.
///
/// Requires a Whisper ONNX model directory containing the encoder and decoder models.
/// Full inference implementation is planned — currently returns an empty transcript
/// (safe default: no keywords match, so no bleeping occurs).
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
        _audio: &AudioSegment,
    ) -> Result<Vec<TranscriptWord>, Box<dyn std::error::Error>> {
        // TODO: Implement full Whisper ONNX inference
        // This requires:
        // 1. Computing log-mel spectrogram from audio
        // 2. Running encoder to get audio features
        // 3. Running decoder with beam search for token generation
        // 4. Extracting word-level timestamps from decoder output
        //
        // For now, return empty transcript (no words detected).
        // This allows the pipeline to function — no keywords will match,
        // so no bleeping occurs, which is safe default behavior.
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_nonexistent_path_returns_error() {
        let result = WhisperRecognizer::new(Path::new("/nonexistent/whisper/model"));
        assert!(result.is_err());
    }

    #[test]
    fn test_new_with_existing_path_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let recognizer = WhisperRecognizer::new(dir.path());
        assert!(recognizer.is_ok());
    }

    #[test]
    fn test_model_path_returns_stored_path() {
        let dir = tempfile::tempdir().unwrap();
        let recognizer = WhisperRecognizer::new(dir.path()).unwrap();
        assert_eq!(recognizer.model_path(), dir.path());
    }

    #[test]
    fn test_transcribe_returns_empty_for_now() {
        let dir = tempfile::tempdir().unwrap();
        let recognizer = WhisperRecognizer::new(dir.path()).unwrap();
        let audio = AudioSegment::new(vec![0.0; 16000], 16000, 1);
        let result = recognizer.transcribe(&audio).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_error_message_contains_path() {
        let bad_path = Path::new("/nonexistent/whisper/model");
        let err = WhisperRecognizer::new(bad_path).unwrap_err();
        assert!(err.to_string().contains("/nonexistent/whisper/model"));
    }
}
