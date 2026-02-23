/// A single video/image frame with pixel data and sequence index.
///
/// Data is stored as contiguous RGB bytes in row-major order.
/// The domain layer treats pixel data as opaque — format conversion
/// happens at I/O boundaries only.
#[derive(Clone, Debug)]
pub struct Frame {
    data: Vec<u8>,
    width: u32,
    height: u32,
    channels: u8,
    index: usize,
}

impl Frame {
    pub fn new(data: Vec<u8>, width: u32, height: u32, channels: u8, index: usize) -> Self {
        debug_assert_eq!(
            data.len(),
            (width as usize) * (height as usize) * (channels as usize),
            "data length must equal width * height * channels"
        );
        Self {
            data,
            width,
            height,
            channels,
            index,
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_and_accessors() {
        let data = vec![0u8; 12]; // 2x2x3
        let frame = Frame::new(data.clone(), 2, 2, 3, 5);
        assert_eq!(frame.width(), 2);
        assert_eq!(frame.height(), 2);
        assert_eq!(frame.channels(), 3);
        assert_eq!(frame.index(), 5);
        assert_eq!(frame.data(), &data[..]);
    }

    #[test]
    fn test_data_mut_allows_modification() {
        let data = vec![0u8; 6]; // 2x1x3
        let mut frame = Frame::new(data, 2, 1, 3, 0);
        frame.data_mut()[0] = 255;
        assert_eq!(frame.data()[0], 255);
    }

    #[test]
    fn test_clone_is_independent() {
        let data = vec![100u8; 12];
        let frame = Frame::new(data, 2, 2, 3, 0);
        let mut cloned = frame.clone();
        cloned.data_mut()[0] = 0;
        assert_eq!(frame.data()[0], 100);
        assert_eq!(cloned.data()[0], 0);
    }

    #[test]
    #[should_panic(expected = "data length must equal width * height * channels")]
    fn test_mismatched_data_length_panics_in_debug() {
        let data = vec![0u8; 10]; // wrong size for 2x2x3
        Frame::new(data, 2, 2, 3, 0);
    }
}
