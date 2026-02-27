use std::path::Path;

use crate::audio::domain::audio_segment::AudioSegment;
use crate::video::domain::audio_writer::AudioWriter;

/// Muxes processed audio into an existing video file using ffmpeg-next.
///
/// The writer opens the existing video-only file, creates a temp output with
/// both the original video stream and newly encoded AAC audio, then replaces
/// the original file.
pub struct FfmpegAudioWriter;

impl AudioWriter for FfmpegAudioWriter {
    fn write_audio(
        &self,
        video_path: &Path,
        audio: &AudioSegment,
    ) -> Result<(), Box<dyn std::error::Error>> {
        ffmpeg_next::init()?;

        let temp_path = video_path.with_extension("tmp.mp4");

        // Open the existing video (no audio)
        let mut ictx = ffmpeg_next::format::input(video_path)?;

        // Create output
        let mut octx = ffmpeg_next::format::output(&temp_path)?;

        // Copy video stream parameters
        let video_stream = ictx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or("No video stream in source file")?;
        let video_src_idx = video_stream.index();
        let video_in_tb = video_stream.time_base();

        let mut ost_video =
            octx.add_stream(ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::None))?;
        ost_video.set_parameters(video_stream.parameters());
        unsafe {
            (*ost_video.parameters().as_mut_ptr()).codec_tag = 0;
        }
        let video_ost_idx = ost_video.index();

        // Set up AAC audio encoder
        let aac_codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::AAC)
            .ok_or("AAC encoder not found")?;
        let mut ost_audio = octx.add_stream(Some(aac_codec))?;
        let audio_ost_idx = ost_audio.index();

        let mut audio_encoder = ffmpeg_next::codec::context::Context::new_with_codec(aac_codec)
            .encoder()
            .audio()?;

        audio_encoder.set_rate(audio.sample_rate() as i32);
        audio_encoder.set_channel_layout(ffmpeg_next::ChannelLayout::MONO);

        // Pick a format the AAC encoder supports — try FLTP (most common for AAC)
        audio_encoder.set_format(ffmpeg_next::format::Sample::F32(
            ffmpeg_next::format::sample::Type::Planar,
        ));

        let mut audio_encoder = audio_encoder.open_as(aac_codec)?;
        ost_audio.set_parameters(&audio_encoder);

        let audio_time_base = audio_encoder.time_base();
        let frame_size = audio_encoder.frame_size() as usize;

        // Write header
        octx.write_header()?;

        let ost_video_tb = octx.stream(video_ost_idx).unwrap().time_base();
        let ost_audio_tb = octx.stream(audio_ost_idx).unwrap().time_base();

        // Copy video packets
        for (stream, mut packet) in ictx.packets() {
            if stream.index() != video_src_idx {
                continue;
            }
            packet.rescale_ts(video_in_tb, ost_video_tb);
            packet.set_position(-1);
            packet.set_stream(video_ost_idx);
            packet.write_interleaved(&mut octx)?;
        }

        // Encode audio
        encode_audio_segment(
            &mut audio_encoder,
            audio,
            &mut octx,
            audio_ost_idx,
            audio_time_base,
            ost_audio_tb,
            frame_size,
        )?;

        octx.write_trailer()?;

        // Drop contexts before file operations
        drop(octx);
        drop(ictx);

        std::fs::rename(&temp_path, video_path)?;
        Ok(())
    }
}

/// Encode an AudioSegment into AAC packets and write them to the output.
fn encode_audio_segment(
    encoder: &mut ffmpeg_next::codec::encoder::audio::Encoder,
    audio: &AudioSegment,
    octx: &mut ffmpeg_next::format::context::Output,
    stream_idx: usize,
    enc_time_base: ffmpeg_next::Rational,
    ost_time_base: ffmpeg_next::Rational,
    frame_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let samples = audio.samples();
    let sample_rate = audio.sample_rate();
    let effective_frame_size = if frame_size == 0 { 1024 } else { frame_size };

    let mut pts: i64 = 0;

    for chunk in samples.chunks(effective_frame_size) {
        let mut frame = ffmpeg_next::util::frame::audio::Audio::new(
            ffmpeg_next::format::Sample::F32(ffmpeg_next::format::sample::Type::Planar),
            chunk.len(),
            ffmpeg_next::ChannelLayout::MONO,
        );
        frame.set_rate(sample_rate);
        frame.set_pts(Some(pts));

        // Copy f32 samples into the frame's data plane
        let dst = frame.data_mut(0);
        let src_bytes =
            unsafe { std::slice::from_raw_parts(chunk.as_ptr() as *const u8, chunk.len() * 4) };
        dst[..src_bytes.len()].copy_from_slice(src_bytes);

        encoder.send_frame(&frame)?;
        flush_audio_packets(encoder, octx, stream_idx, enc_time_base, ost_time_base)?;

        pts += chunk.len() as i64;
    }

    // Flush encoder
    encoder.send_eof()?;
    flush_audio_packets(encoder, octx, stream_idx, enc_time_base, ost_time_base)?;

    Ok(())
}

fn flush_audio_packets(
    encoder: &mut ffmpeg_next::codec::encoder::audio::Encoder,
    octx: &mut ffmpeg_next::format::context::Output,
    stream_idx: usize,
    enc_time_base: ffmpeg_next::Rational,
    ost_time_base: ffmpeg_next::Rational,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut encoded = ffmpeg_next::Packet::empty();
    while encoder.receive_packet(&mut encoded).is_ok() {
        encoded.set_stream(stream_idx);
        encoded.rescale_ts(enc_time_base, ost_time_base);
        encoded.write_interleaved(octx)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_write_audio_nonexistent_file() {
        let writer = FfmpegAudioWriter;
        let audio = AudioSegment::new(vec![0.0; 16000], 16000, 1);
        let path = if cfg!(windows) {
            Path::new("Z:\\nonexistent\\file.mp4")
        } else {
            Path::new("/nonexistent/file.mp4")
        };
        let result = writer.write_audio(path, &audio);
        assert!(result.is_err());
    }
}
