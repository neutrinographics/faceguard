use crate::blurring::domain::frame_blurrer::FrameBlurrer;
use crate::shared::frame::Frame;
use crate::shared::region::Region;

use super::gaussian;

/// Default kernel size for Gaussian blur.
const DEFAULT_KERNEL_SIZE: usize = 201;

/// CPU elliptical blurrer using separable Gaussian blur with ellipse masking.
///
/// Blurs the rectangular ROI and then composites only the elliptical area
/// back into the frame. Uses the Region's edge-aware ellipse geometry so
/// the blur extends smoothly off frame edges.
pub struct CpuEllipticalBlurrer {
    kernel_size: usize,
    scale: usize,
    small_k: usize,
}

impl CpuEllipticalBlurrer {
    pub fn new(kernel_size: usize) -> Self {
        let scale = (kernel_size / 50).max(1);
        let small_k = (kernel_size / scale) | 1; // ensure odd
        Self {
            kernel_size,
            scale,
            small_k,
        }
    }
}

impl Default for CpuEllipticalBlurrer {
    fn default() -> Self {
        Self::new(DEFAULT_KERNEL_SIZE)
    }
}

impl FrameBlurrer for CpuEllipticalBlurrer {
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

            // Extract ROI
            let mut roi = vec![0u8; rw * rh * channels];
            for row in 0..rh {
                let src_offset = ((ry + row) * fw + rx) * channels;
                let dst_offset = row * rw * channels;
                roi[dst_offset..dst_offset + rw * channels]
                    .copy_from_slice(&data[src_offset..src_offset + rw * channels]);
            }

            // Blur ROI (with downscale optimization for large kernels)
            let blurred_roi = if self.scale <= 1 || rh < self.scale * 2 || rw < self.scale * 2 {
                gaussian::separable_gaussian_blur(&mut roi, rw, rh, channels, self.kernel_size);
                roi
            } else {
                let (mut small, sw, sh) = gaussian::downscale(&roi, rw, rh, channels, self.scale);
                gaussian::separable_gaussian_blur(&mut small, sw, sh, channels, self.small_k);
                gaussian::upscale(&small, sw, sh, channels, rw, rh)
            };

            // Get ellipse geometry from region
            let (ecx, ecy) = r.ellipse_center_in_roi();
            let (semi_a, semi_b) = r.ellipse_axes();
            let inv_a_sq = if semi_a > 0.0 { 1.0 / (semi_a * semi_a) } else { 0.0 };
            let inv_b_sq = if semi_b > 0.0 { 1.0 / (semi_b * semi_b) } else { 0.0 };
            let ellipse_valid = semi_a > 0.0 && semi_b > 0.0;

            // Composite blurred pixels within ellipse mask back into frame
            for row in 0..rh {
                for col in 0..rw {
                    // Ellipse SDF: (dx^2 * inv_a_sq) + (dy^2 * inv_b_sq) <= 1
                    let dx = col as f64 - ecx;
                    let dy = row as f64 - ecy;
                    let ellipse_dist = if ellipse_valid {
                        dx * dx * inv_a_sq + dy * dy * inv_b_sq
                    } else {
                        f64::MAX
                    };

                    if ellipse_dist <= 1.0 {
                        let frame_offset = ((ry + row) * fw + (rx + col)) * channels;
                        let roi_offset = (row * rw + col) * channels;
                        data[frame_offset..frame_offset + channels]
                            .copy_from_slice(&blurred_roi[roi_offset..roi_offset + channels]);
                    }
                }
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
        let blurrer = CpuEllipticalBlurrer::new(5);
        blurrer.blur(&mut frame, &[]).unwrap();
        assert_eq!(frame.data(), &original[..]);
    }

    #[test]
    fn test_preserves_frame_index() {
        let data = vec![128u8; 100 * 100 * 3];
        let mut frame = Frame::new(data, 100, 100, 3, 42);
        let blurrer = CpuEllipticalBlurrer::new(5);
        blurrer.blur(&mut frame, &[]).unwrap();
        assert_eq!(frame.index(), 42);
    }

    #[test]
    fn test_blur_modifies_region_pixels() {
        let mut frame = make_frame(100, 100, 0);
        // Set a bright patch inside where the ellipse will be
        let data = frame.data_mut();
        for y in 18..22 {
            for x in 18..22 {
                let idx = (y * 100 + x) * 3;
                data[idx] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
            }
        }

        let blurrer = CpuEllipticalBlurrer::new(5);
        // Region centered at (20, 20) with size 30x30 — ellipse covers the center
        blurrer.blur(&mut frame, &[region(5, 5, 30, 30)]).unwrap();

        // Bright pixels inside the ellipse should have been blurred
        let center = (20 * 100 + 20) * 3;
        // At minimum, some spreading should have occurred
        assert!(frame.data()[center] < 255 || frame.data()[(18 * 100 + 18) * 3] < 255);
    }

    #[test]
    fn test_pixels_outside_region_unchanged() {
        let mut frame = make_frame(100, 100, 200);
        let original = frame.data().to_vec();
        let blurrer = CpuEllipticalBlurrer::new(5);
        blurrer.blur(&mut frame, &[region(10, 10, 20, 20)]).unwrap();

        // Pixel at (0,0) should be unchanged
        assert_eq!(frame.data()[0], original[0]);
        // Pixel at (50,50) should be unchanged
        let idx = (50 * 100 + 50) * 3;
        assert_eq!(frame.data()[idx], original[idx]);
    }

    #[test]
    fn test_ellipse_does_not_blur_corners() {
        // Create a region where corners are outside the ellipse
        let mut frame = make_frame(100, 100, 0);
        let data = frame.data_mut();
        // Set the entire region to white
        for y in 0..40 {
            for x in 0..40 {
                let idx = (y * 100 + x) * 3;
                data[idx] = 200;
                data[idx + 1] = 200;
                data[idx + 2] = 200;
            }
        }

        let original = frame.data().to_vec();
        let blurrer = CpuEllipticalBlurrer::new(5);
        blurrer.blur(&mut frame, &[region(0, 0, 40, 40)]).unwrap();

        // Corner pixel (0,0) is outside the ellipse — should be unchanged
        // Ellipse center is at (20, 20) with semi-axes (20, 20)
        // Corner (0,0) has distance ((0-20)/20)^2 + ((0-20)/20)^2 = 1+1 = 2 > 1
        assert_eq!(frame.data()[0], original[0]);
    }

    #[test]
    fn test_multiple_regions() {
        let mut frame = make_frame(100, 100, 0);
        let data = frame.data_mut();
        // Bright spot in center of first region
        let idx1 = (20 * 100 + 20) * 3;
        data[idx1] = 255;
        data[idx1 + 1] = 255;
        data[idx1 + 2] = 255;
        // Bright spot in center of second region
        let idx2 = (75 * 100 + 75) * 3;
        data[idx2] = 255;
        data[idx2 + 1] = 255;
        data[idx2 + 2] = 255;

        let blurrer = CpuEllipticalBlurrer::new(5);
        blurrer
            .blur(
                &mut frame,
                &[region(10, 10, 20, 20), region(65, 65, 20, 20)],
            )
            .unwrap();

        // Both spots should have been blurred (are within their ellipses)
        assert!(frame.data()[idx1] < 255);
        assert!(frame.data()[idx2] < 255);
    }

    #[test]
    fn test_zero_size_region_skipped() {
        let mut frame = make_frame(100, 100, 128);
        let original = frame.data().to_vec();
        let blurrer = CpuEllipticalBlurrer::new(5);
        blurrer.blur(&mut frame, &[region(10, 10, 0, 20)]).unwrap();
        assert_eq!(frame.data(), &original[..]);
    }

    #[test]
    fn test_ellipse_uses_full_dimensions() {
        // Region clipped at left edge: full ellipse extends off-screen
        let r = Region {
            x: 0,
            y: 10,
            width: 30,
            height: 40,
            track_id: None,
            full_width: Some(60),
            full_height: Some(40),
            unclamped_x: Some(-30),
            unclamped_y: Some(10),
        };

        let (ecx, _ecy) = r.ellipse_center_in_roi();
        let (sa, sb) = r.ellipse_axes();

        // The ellipse center should be offset (to the left of the visible region)
        assert!(ecx < r.width as f64 / 2.0);
        // Semi-axes use full dimensions
        assert_eq!(sa, 30.0); // full_width / 2
        assert_eq!(sb, 20.0); // full_height / 2

        // Verify blurring doesn't crash with this region
        let mut frame = make_frame(100, 100, 128);
        let blurrer = CpuEllipticalBlurrer::new(5);
        blurrer.blur(&mut frame, &[r]).unwrap();
    }

    #[test]
    fn test_default_kernel_size() {
        let blurrer = CpuEllipticalBlurrer::default();
        assert_eq!(blurrer.kernel_size, DEFAULT_KERNEL_SIZE);
    }
}
