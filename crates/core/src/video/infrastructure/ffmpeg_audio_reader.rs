use std::path::Path;

use crate::audio::domain::audio_segment::AudioSegment;
use crate::video::domain::audio_reader::AudioReader;

/// Decodes audio from a video file using ffmpeg-next.
pub struct FfmpegAudioReader;

impl AudioReader for FfmpegAudioReader {
    fn read_audio(
        &self,
        path: &Path,
        target_sample_rate: u32,
    ) -> Result<Option<AudioSegment>, Box<dyn std::error::Error>> {
        ffmpeg_next::init()?;

        let mut ictx = ffmpeg_next::format::input(path)?;

        let audio_stream = match ictx.streams().best(ffmpeg_next::media::Type::Audio) {
            Some(stream) => stream,
            None => return Ok(None),
        };

        let audio_stream_index = audio_stream.index();
        let codec_params = audio_stream.parameters();

        let codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(codec_params)?;
        let mut decoder = codec_ctx.decoder().audio()?;

        let mut resampler = ffmpeg_next::software::resampling::Context::get(
            decoder.format(),
            decoder.channel_layout(),
            decoder.rate(),
            ffmpeg_next::format::Sample::F32(ffmpeg_next::format::sample::Type::Planar),
            ffmpeg_next::ChannelLayout::MONO,
            target_sample_rate,
        )?;

        let mut all_samples: Vec<f32> = Vec::new();
        let mut decoded_frame = ffmpeg_next::util::frame::audio::Audio::empty();
        let mut resampled_frame = ffmpeg_next::util::frame::audio::Audio::empty();

        for (stream, packet) in ictx.packets() {
            if stream.index() != audio_stream_index {
                continue;
            }

            decoder.send_packet(&packet)?;

            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                resampler.run(&decoded_frame, &mut resampled_frame)?;
                extract_f32_samples(&resampled_frame, &mut all_samples);
            }
        }

        // Flush the decoder
        decoder.send_eof()?;
        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            resampler.run(&decoded_frame, &mut resampled_frame)?;
            extract_f32_samples(&resampled_frame, &mut all_samples);
        }

        // Flush the resampler (may have buffered samples)
        if let Ok(Some(delay)) = resampler.flush(&mut resampled_frame) {
            if delay.output > 0 {
                extract_f32_samples(&resampled_frame, &mut all_samples);
            }
        }

        Ok(Some(AudioSegment::new(all_samples, target_sample_rate, 1)))
    }

    fn audio_metadata(
        &self,
        path: &Path,
    ) -> Result<Option<(u32, u16)>, Box<dyn std::error::Error>> {
        ffmpeg_next::init()?;

        let ictx = ffmpeg_next::format::input(path)?;

        let audio_stream = match ictx.streams().best(ffmpeg_next::media::Type::Audio) {
            Some(stream) => stream,
            None => return Ok(None),
        };

        let codec_ctx =
            ffmpeg_next::codec::context::Context::from_parameters(audio_stream.parameters())?;
        let decoder = codec_ctx.decoder().audio()?;

        let sample_rate = decoder.rate();
        let channels = decoder.channels() as u16;

        Ok(Some((sample_rate, channels)))
    }
}

/// Extract f32 samples from a planar mono resampled frame.
fn extract_f32_samples(frame: &ffmpeg_next::util::frame::audio::Audio, out: &mut Vec<f32>) {
    let num_samples = frame.samples();
    if num_samples == 0 {
        return;
    }
    let data = frame.data(0);
    let floats = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, num_samples) };
    out.extend_from_slice(floats);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_read_audio_nonexistent_file() {
        let reader = FfmpegAudioReader;
        let path = if cfg!(windows) {
            Path::new("Z:\\nonexistent\\file.mp4")
        } else {
            Path::new("/nonexistent/file.mp4")
        };
        let result = reader.read_audio(path, 16000);
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_metadata_nonexistent_file() {
        let reader = FfmpegAudioReader;
        let path = if cfg!(windows) {
            Path::new("Z:\\nonexistent\\file.mp4")
        } else {
            Path::new("/nonexistent/file.mp4")
        };
        let result = reader.audio_metadata(path);
        assert!(result.is_err());
    }
}
