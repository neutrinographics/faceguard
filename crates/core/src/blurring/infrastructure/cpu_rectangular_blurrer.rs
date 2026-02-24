use std::cell::RefCell;

use crate::blurring::domain::frame_blurrer::FrameBlurrer;
use crate::shared::frame::Frame;
use crate::shared::region::Region;

use super::gaussian;

/// Default kernel size for Gaussian blur.
const DEFAULT_KERNEL_SIZE: usize = 201;

/// CPU rectangular blurrer using separable Gaussian blur.
///
/// Blurs the entire rectangular ROI for each region. Uses the same
/// downscale-blur-upscale optimization as the Python implementation
/// for large kernel sizes.
pub struct CpuRectangularBlurrer {
    kernel: Vec<f32>,
    scale: usize,
    small_kernel: Vec<f32>,
    roi_buf: RefCell<Vec<u8>>,
    blur_temp: RefCell<Vec<f32>>,
}

impl CpuRectangularBlurrer {
    pub fn new(kernel_size: usize) -> Self {
        let scale = (kernel_size / 50).max(1);
        let small_k = (kernel_size / scale) | 1; // ensure odd
        Self {
            kernel: gaussian::gaussian_kernel_1d(kernel_size),
            scale,
            small_kernel: gaussian::gaussian_kernel_1d(small_k),
            roi_buf: RefCell::new(Vec::new()),
            blur_temp: RefCell::new(Vec::new()),
        }
    }
}

impl Default for CpuRectangularBlurrer {
    fn default() -> Self {
        Self::new(DEFAULT_KERNEL_SIZE)
    }
}

impl FrameBlurrer for CpuRectangularBlurrer {
    fn blur(
        &self,
        frame: &mut Frame,
        regions: &[Region],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fw = frame.width() as usize;
        let channels = frame.channels() as usize;
        let data = frame.data_mut();

        for r in regions {
            let rx = r.x.max(0) as usize;
            let ry = r.y.max(0) as usize;
            let rw = r.width.max(0) as usize;
            let rh = r.height.max(0) as usize;

            if rw == 0 || rh == 0 {
                continue;
            }

            // Extract ROI (reuse buffer across regions)
            let mut roi = self.roi_buf.borrow_mut();
            let roi_size = rw * rh * channels;
            roi.resize(roi_size, 0);
            for row in 0..rh {
                let src_offset = ((ry + row) * fw + rx) * channels;
                let dst_offset = row * rw * channels;
                roi[dst_offset..dst_offset + rw * channels]
                    .copy_from_slice(&data[src_offset..src_offset + rw * channels]);
            }

            // Blur ROI (with downscale optimization for large kernels)
            let mut temp = self.blur_temp.borrow_mut();
            if self.scale <= 1 || rh < self.scale * 2 || rw < self.scale * 2 {
                gaussian::separable_gaussian_blur_with_kernel(
                    &mut roi,
                    rw,
                    rh,
                    channels,
                    &self.kernel,
                    &mut temp,
                );
            } else {
                let (mut small, sw, sh) = gaussian::downscale(&roi, rw, rh, channels, self.scale);
                gaussian::separable_gaussian_blur_with_kernel(
                    &mut small,
                    sw,
                    sh,
                    channels,
                    &self.small_kernel,
                    &mut temp,
                );
                let upscaled = gaussian::upscale(&small, sw, sh, channels, rw, rh);
                roi[..roi_size].copy_from_slice(&upscaled);
            }

            // Write blurred ROI back
            for row in 0..rh {
                let dst_offset = ((ry + row) * fw + rx) * channels;
                let src_offset = row * rw * channels;
                data[dst_offset..dst_offset + rw * channels]
                    .copy_from_slice(&roi[src_offset..src_offset + rw * channels]);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: u32, height: u32, value: u8) -> Frame {
        let data = vec![value; (width * height * 3) as usize];
        Frame::new(data, width, height, 3, 0)
    }

    fn region(x: i32, y: i32, w: i32, h: i32) -> Region {
        Region {
            x,
            y,
            width: w,
            height: h,
            track_id: None,
            full_width: None,
            full_height: None,
            unclamped_x: None,
            unclamped_y: None,
        }
    }

    #[test]
    fn test_no_regions_frame_unchanged() {
        let mut frame = make_frame(100, 100, 128);
        let original = frame.data().to_vec();
        let blurrer = CpuRectangularBlurrer::new(5);
        blurrer.blur(&mut frame, &[]).unwrap();
        assert_eq!(frame.data(), &original[..]);
    }

    #[test]
    fn test_preserves_frame_index() {
        let data = vec![128u8; 100 * 100 * 3];
        let mut frame = Frame::new(data, 100, 100, 3, 42);
        let blurrer = CpuRectangularBlurrer::new(5);
        blurrer.blur(&mut frame, &[]).unwrap();
        assert_eq!(frame.index(), 42);
    }

    #[test]
    fn test_blur_modifies_region_pixels() {
        let mut frame = make_frame(100, 100, 0);
        // Set a small bright patch in the region
        let data = frame.data_mut();
        for y in 10..15 {
            for x in 10..15 {
                let idx = (y * 100 + x) * 3;
                data[idx] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
            }
        }

        let blurrer = CpuRectangularBlurrer::new(5);
        blurrer.blur(&mut frame, &[region(5, 5, 30, 30)]).unwrap();

        // The bright patch borders dark pixels within the ROI — the Gaussian blur
        // should spread brightness into adjacent dark pixels.
        // Check a pixel just outside the bright patch but inside the region.
        let neighbor = (9 * 100 + 12) * 3; // one row above the bright patch
        assert!(
            frame.data()[neighbor] > 0,
            "blur should spread to adjacent pixels"
        );
    }

    #[test]
    fn test_pixels_outside_region_unchanged() {
        let mut frame = make_frame(100, 100, 0);
        // Set bright pixels everywhere
        frame.data_mut().fill(200);

        let original = frame.data().to_vec();
        let blurrer = CpuRectangularBlurrer::new(5);
        blurrer.blur(&mut frame, &[region(10, 10, 20, 20)]).unwrap();

        // Pixel at (0,0) should be unchanged
        assert_eq!(frame.data()[0], original[0]);
        // Pixel at (50,50) should be unchanged
        let idx = (50 * 100 + 50) * 3;
        assert_eq!(frame.data()[idx], original[idx]);
    }

    #[test]
    fn test_multiple_regions() {
        let mut frame = make_frame(100, 100, 0);
        let data = frame.data_mut();
        // Bright spot in first region
        let idx1 = (15 * 100 + 15) * 3;
        data[idx1] = 255;
        // Bright spot in second region
        let idx2 = (75 * 100 + 75) * 3;
        data[idx2] = 255;

        let blurrer = CpuRectangularBlurrer::new(5);
        blurrer
            .blur(
                &mut frame,
                &[region(10, 10, 20, 20), region(70, 70, 20, 20)],
            )
            .unwrap();

        // Both bright spots should have been blurred (spread)
        assert!(frame.data()[idx1] < 255);
        assert!(frame.data()[idx2] < 255);
    }

    #[test]
    fn test_zero_size_region_skipped() {
        let mut frame = make_frame(100, 100, 128);
        let original = frame.data().to_vec();
        let blurrer = CpuRectangularBlurrer::new(5);
        blurrer.blur(&mut frame, &[region(10, 10, 0, 20)]).unwrap();
        assert_eq!(frame.data(), &original[..]);
    }

    #[test]
    fn test_full_frame_region() {
        let mut frame = make_frame(50, 50, 0);
        let data = frame.data_mut();
        let center = (25 * 50 + 25) * 3;
        data[center] = 255;

        let blurrer = CpuRectangularBlurrer::new(5);
        blurrer.blur(&mut frame, &[region(0, 0, 50, 50)]).unwrap();

        // Center should be blurred down
        assert!(frame.data()[center] < 255);
    }

    #[test]
    fn test_default_kernel_size() {
        let blurrer = CpuRectangularBlurrer::default();
        assert_eq!(blurrer.kernel.len(), DEFAULT_KERNEL_SIZE);
    }

    #[test]
    fn test_downscale_optimization_used_for_large_kernel() {
        let blurrer = CpuRectangularBlurrer::new(201);
        assert!(blurrer.scale > 1);
        assert!(blurrer.small_kernel.len() < blurrer.kernel.len());
        assert_eq!(blurrer.small_kernel.len() % 2, 1); // must be odd
    }
}
