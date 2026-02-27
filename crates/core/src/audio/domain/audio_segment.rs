/// A segment of decoded audio: interleaved PCM samples normalized to [-1.0, 1.0].
#[derive(Clone, Debug)]
pub struct AudioSegment {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

impl AudioSegment {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.samples
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
    }

    pub fn sample_index_at_time(&self, time: f64) -> usize {
        (time * self.sample_rate as f64 * self.channels as f64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_segment_with_correct_fields() {
        let samples = vec![0.0f32; 16000];
        let seg = AudioSegment::new(samples.clone(), 16000, 1);
        assert_eq!(seg.samples(), &samples[..]);
        assert_eq!(seg.sample_rate(), 16000);
        assert_eq!(seg.channels(), 1);
    }

    #[test]
    fn test_duration_mono() {
        let seg = AudioSegment::new(vec![0.0; 48000], 16000, 1);
        assert_eq!(seg.duration(), 3.0);
    }

    #[test]
    fn test_duration_stereo() {
        let seg = AudioSegment::new(vec![0.0; 96000], 48000, 2);
        assert_eq!(seg.duration(), 1.0);
    }

    #[test]
    fn test_sample_index_at_time() {
        let mut samples = vec![0.0f32; 16000];
        samples[8000] = 0.5;
        let seg = AudioSegment::new(samples, 16000, 1);
        assert_eq!(seg.sample_index_at_time(0.5), 8000);
    }

    #[test]
    fn test_samples_mut() {
        let mut seg = AudioSegment::new(vec![0.0; 100], 16000, 1);
        seg.samples_mut()[50] = 1.0;
        assert_eq!(seg.samples()[50], 1.0);
    }
}
