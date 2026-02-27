use super::audio_segment::AudioSegment;

/// Domain interface for audio transformation (voice disguise).
///
/// Implementations apply DSP effects to modify the speaker's voice.
pub trait AudioTransformer: Send {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>>;
}
