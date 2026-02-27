use super::audio_segment::AudioSegment;
use super::transcript::TranscriptWord;

/// Domain interface for speech-to-text transcription.
///
/// Implementations run inference on audio to produce word-level timestamps.
pub trait SpeechRecognizer: Send {
    fn transcribe(
        &self,
        audio: &AudioSegment,
    ) -> Result<Vec<TranscriptWord>, Box<dyn std::error::Error>>;
}
