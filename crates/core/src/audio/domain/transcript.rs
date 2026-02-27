#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptWord {
    pub word: String,
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: f32,
}

impl TranscriptWord {
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_transcript_word_fields() {
        let w = TranscriptWord {
            word: "hello".to_string(),
            start_time: 1.0,
            end_time: 1.5,
            confidence: 0.95,
        };
        assert_eq!(w.word, "hello");
        assert_eq!(w.start_time, 1.0);
        assert_eq!(w.end_time, 1.5);
        assert_eq!(w.confidence, 0.95);
    }

    #[test]
    fn test_transcript_word_duration() {
        let w = TranscriptWord {
            word: "test".to_string(),
            start_time: 2.0,
            end_time: 2.8,
            confidence: 0.9,
        };
        assert_relative_eq!(w.duration(), 0.8, epsilon = 0.001);
    }
}
