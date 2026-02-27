use crate::audio::domain::audio_segment::AudioSegment;
use std::path::Path;

/// Domain interface for decoding audio from a video file.
pub trait AudioReader: Send {
    /// Decode the audio track to a mono PCM AudioSegment at the given sample rate.
    /// Returns None if the video has no audio track.
    fn read_audio(
        &self,
        path: &Path,
        target_sample_rate: u32,
    ) -> Result<Option<AudioSegment>, Box<dyn std::error::Error>>;

    /// Return the original audio sample rate and channel count without decoding.
    fn audio_metadata(&self, path: &Path)
        -> Result<Option<(u32, u16)>, Box<dyn std::error::Error>>;
}
