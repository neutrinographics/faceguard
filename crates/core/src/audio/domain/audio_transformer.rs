use super::audio_segment::AudioSegment;

/// Domain interface for audio transformation (voice disguise).
///
/// Implementations apply DSP effects to modify the speaker's voice.
pub trait AudioTransformer: Send {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>>;
}

/// Applies multiple transformers in sequence.
pub struct ComposedTransformer {
    transformers: Vec<Box<dyn AudioTransformer>>,
}

impl ComposedTransformer {
    pub fn new(transformers: Vec<Box<dyn AudioTransformer>>) -> Self {
        Self { transformers }
    }
}

impl AudioTransformer for ComposedTransformer {
    fn transform(&self, audio: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>> {
        for t in &self.transformers {
            t.transform(audio)?;
        }
        Ok(())
    }
}
