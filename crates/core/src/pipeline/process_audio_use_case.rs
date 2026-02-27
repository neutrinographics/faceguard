use std::path::Path;

use crate::audio::domain::audio_transformer::AudioTransformer;
use crate::audio::domain::speech_recognizer::SpeechRecognizer;
use crate::audio::domain::word_censor::{
    BleepMode, WordCensor, DEFAULT_BLEEP_FREQUENCY, DEFAULT_BLEEP_PADDING,
};
use crate::video::domain::audio_reader::AudioReader;
use crate::video::domain::audio_writer::AudioWriter;

pub struct ProcessAudioUseCase {
    reader: Box<dyn AudioReader>,
    writer: Box<dyn AudioWriter>,
    recognizer: Option<Box<dyn SpeechRecognizer>>,
    transformer: Option<Box<dyn AudioTransformer>>,
    keywords: Vec<String>,
    bleep_mode: BleepMode,
}

impl ProcessAudioUseCase {
    pub fn new(
        reader: Box<dyn AudioReader>,
        writer: Box<dyn AudioWriter>,
        recognizer: Option<Box<dyn SpeechRecognizer>>,
        transformer: Option<Box<dyn AudioTransformer>>,
        keywords: Vec<String>,
        bleep_mode: BleepMode,
    ) -> Self {
        Self {
            reader,
            writer,
            recognizer,
            transformer,
            keywords,
            bleep_mode,
        }
    }

    pub fn run(
        &self,
        source_path: &Path,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Read audio from source
        let mut audio = match self.reader.read_audio(source_path, 16000)? {
            Some(a) => a,
            None => return Ok(()), // No audio track — skip
        };

        // 2. Transcribe keywords on the original audio (before voice transform)
        let censor_regions = if !self.keywords.is_empty() {
            if let Some(ref recognizer) = self.recognizer {
                let transcript = recognizer.transcribe(&audio)?;
                WordCensor::find_censor_regions(&transcript, &self.keywords, DEFAULT_BLEEP_PADDING)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // 3. Voice transform (if enabled) — must happen before bleeping,
        //    otherwise PSOLA overlap-add corrupts the bleep tones
        if let Some(ref transformer) = self.transformer {
            transformer.transform(&mut audio)?;
        }

        // 4. Apply bleeps after voice transform so they cleanly overwrite
        if !censor_regions.is_empty() {
            WordCensor::apply_bleep(
                &mut audio,
                &censor_regions,
                DEFAULT_BLEEP_FREQUENCY,
                self.bleep_mode,
            );
        }

        // 5. Write processed audio to output
        self.writer.write_audio(output_path, &audio)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::domain::audio_segment::AudioSegment;
    use crate::audio::domain::audio_transformer::AudioTransformer;
    use crate::audio::domain::speech_recognizer::SpeechRecognizer;
    use crate::audio::domain::transcript::TranscriptWord;
    use crate::video::domain::audio_reader::AudioReader;
    use crate::video::domain::audio_writer::AudioWriter;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    // ─── Stubs ───

    struct StubAudioReader {
        segment: Option<AudioSegment>,
    }

    impl AudioReader for StubAudioReader {
        fn read_audio(
            &self,
            _: &Path,
            _: u32,
        ) -> Result<Option<AudioSegment>, Box<dyn std::error::Error>> {
            Ok(self.segment.clone())
        }

        fn audio_metadata(
            &self,
            _: &Path,
        ) -> Result<Option<(u32, u16)>, Box<dyn std::error::Error>> {
            Ok(self
                .segment
                .as_ref()
                .map(|s| (s.sample_rate(), s.channels())))
        }
    }

    struct StubAudioWriter {
        written: Arc<Mutex<Option<AudioSegment>>>,
    }

    impl AudioWriter for StubAudioWriter {
        fn write_audio(
            &self,
            _: &Path,
            audio: &AudioSegment,
        ) -> Result<(), Box<dyn std::error::Error>> {
            *self.written.lock().unwrap() = Some(audio.clone());
            Ok(())
        }
    }

    struct StubRecognizer {
        words: Vec<TranscriptWord>,
    }

    impl SpeechRecognizer for StubRecognizer {
        fn transcribe(
            &self,
            _: &AudioSegment,
        ) -> Result<Vec<TranscriptWord>, Box<dyn std::error::Error>> {
            Ok(self.words.clone())
        }
    }

    struct StubTransformer {
        called: Arc<Mutex<bool>>,
    }

    impl AudioTransformer for StubTransformer {
        fn transform(&self, _: &mut AudioSegment) -> Result<(), Box<dyn std::error::Error>> {
            *self.called.lock().unwrap() = true;
            Ok(())
        }
    }

    fn silent_audio() -> AudioSegment {
        AudioSegment::new(vec![0.0; 16000], 16000, 1)
    }

    #[test]
    fn test_no_audio_track_skips_processing() {
        let writer = StubAudioWriter {
            written: Arc::new(Mutex::new(None)),
        };
        let written = writer.written.clone();
        let uc = ProcessAudioUseCase::new(
            Box::new(StubAudioReader { segment: None }),
            Box::new(writer),
            None,
            None,
            vec![],
            BleepMode::Tone,
        );
        uc.run(Path::new("in.mp4"), Path::new("out.mp4")).unwrap();
        assert!(written.lock().unwrap().is_none());
    }

    #[test]
    fn test_keywords_trigger_bleep() {
        let writer = StubAudioWriter {
            written: Arc::new(Mutex::new(None)),
        };
        let written = writer.written.clone();
        let recognizer = StubRecognizer {
            words: vec![TranscriptWord {
                word: "secret".to_string(),
                start_time: 0.5,
                end_time: 0.8,
                confidence: 0.95,
            }],
        };
        let uc = ProcessAudioUseCase::new(
            Box::new(StubAudioReader {
                segment: Some(silent_audio()),
            }),
            Box::new(writer),
            Some(Box::new(recognizer)),
            None,
            vec!["secret".to_string()],
            BleepMode::Tone,
        );
        uc.run(Path::new("in.mp4"), Path::new("out.mp4")).unwrap();

        let written = written.lock().unwrap();
        assert!(written.is_some());
        let seg = written.as_ref().unwrap();
        let start = seg.sample_index_at_time(0.5);
        let end = seg.sample_index_at_time(0.8);
        let energy: f64 = seg.samples()[start..end]
            .iter()
            .map(|s| (*s as f64).powi(2))
            .sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_voice_transform_applied() {
        let writer = StubAudioWriter {
            written: Arc::new(Mutex::new(None)),
        };
        let transformer = StubTransformer {
            called: Arc::new(Mutex::new(false)),
        };
        let called = transformer.called.clone();
        let uc = ProcessAudioUseCase::new(
            Box::new(StubAudioReader {
                segment: Some(silent_audio()),
            }),
            Box::new(writer),
            None,
            Some(Box::new(transformer)),
            vec![],
            BleepMode::Tone,
        );
        uc.run(Path::new("in.mp4"), Path::new("out.mp4")).unwrap();
        assert!(*called.lock().unwrap());
    }
}
