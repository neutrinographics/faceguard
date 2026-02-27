use crate::audio::domain::audio_segment::AudioSegment;
use std::path::Path;

/// Domain interface for encoding audio and muxing it into a video file.
pub trait AudioWriter: Send {
    /// Encode the AudioSegment and mux it into an existing video file,
    /// replacing any existing audio track.
    fn write_audio(
        &self,
        video_path: &Path,
        audio: &AudioSegment,
    ) -> Result<(), Box<dyn std::error::Error>>;
}
