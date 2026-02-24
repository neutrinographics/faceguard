use ndarray::{ArrayView3, ArrayViewMut3};

/// A single video/image frame: contiguous RGB bytes in row-major order.
///
/// Format conversion happens at I/O boundaries only; the domain layer
/// treats pixel data as opaque.
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

    pub fn as_ndarray(&self) -> ArrayView3<'_, u8> {
        ArrayView3::from_shape(self.shape(), &self.data)
            .expect("Frame data length must match dimensions")
    }

    pub fn as_ndarray_mut(&mut self) -> ArrayViewMut3<'_, u8> {
        ArrayViewMut3::from_shape(self.shape(), &mut self.data)
            .expect("Frame data length must match dimensions")
    }

    fn shape(&self) -> (usize, usize, usize) {
        (
            self.height as usize,
            self.width as usize,
            self.channels as usize,
        )
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

    #[test]
    fn test_as_ndarray_shape() {
        let data = vec![0u8; 24]; // 2x4x3
        let frame = Frame::new(data, 4, 2, 3, 0);
        let arr = frame.as_ndarray();
        assert_eq!(arr.shape(), &[2, 4, 3]); // (height, width, channels)
    }

    #[test]
    fn test_as_ndarray_pixel_access() {
        // 2x2 RGB: set pixel (row=1, col=0) to red
        let mut data = vec![0u8; 12];
        data[6] = 255; // row=1, col=0, R
        let frame = Frame::new(data, 2, 2, 3, 0);
        let arr = frame.as_ndarray();
        assert_eq!(arr[[1, 0, 0]], 255); // R
        assert_eq!(arr[[1, 0, 1]], 0); // G
        assert_eq!(arr[[1, 0, 2]], 0); // B
    }

    #[test]
    fn test_as_ndarray_mut_modification() {
        let data = vec![0u8; 12]; // 2x2x3
        let mut frame = Frame::new(data, 2, 2, 3, 0);
        {
            let mut arr = frame.as_ndarray_mut();
            arr[[0, 1, 2]] = 128; // row=0, col=1, B channel
        }
        assert_eq!(frame.as_ndarray()[[0, 1, 2]], 128);
    }
}
