use std::cell::RefCell;

use crate::blurring::domain::frame_blurrer::FrameBlurrer;
use crate::shared::frame::Frame;
use crate::shared::region::Region;

use super::gaussian::{self, RoiRect};

const DEFAULT_KERNEL_SIZE: usize = 201;

/// CPU elliptical blurrer using separable Gaussian blur with ellipse masking.
///
/// Blurs the rectangular ROI and then composites only the elliptical area
/// back into the frame. Uses the Region's edge-aware ellipse geometry so
/// the blur extends smoothly off frame edges.
pub struct CpuEllipticalBlurrer {
    kernel: Vec<f32>,
    scale: usize,
    small_kernel: Vec<f32>,
    roi_buf: RefCell<Vec<u8>>,
    blur_temp: RefCell<Vec<f32>>,
}

impl CpuEllipticalBlurrer {
    pub fn new(kernel_size: usize) -> Self {
        let scale = (kernel_size / 50).max(1);
        let small_k = (kernel_size / scale) | 1;
        Self {
            kernel: gaussian::gaussian_kernel_1d(kernel_size),
            scale,
            small_kernel: gaussian::gaussian_kernel_1d(small_k),
            roi_buf: RefCell::new(Vec::new()),
            blur_temp: RefCell::new(Vec::new()),
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

            let rect = RoiRect {
                x: rx,
                y: ry,
                w: rw,
                h: rh,
            };
            let mut roi = self.roi_buf.borrow_mut();
            let mut temp = self.blur_temp.borrow_mut();

            gaussian::extract_roi(data, fw, channels, rect, &mut roi);
            gaussian::blur_roi_in_place(
                &mut roi,
                rw,
                rh,
                channels,
                &self.kernel,
                &self.small_kernel,
                self.scale,
                &mut temp,
            );

            composite_ellipse(data, &roi, fw, channels, rect, r);
        }

        Ok(())
    }
}

/// Write blurred pixels back to the frame only within the ellipse mask.
fn composite_ellipse(
    data: &mut [u8],
    roi: &[u8],
    frame_width: usize,
    channels: usize,
    rect: RoiRect,
    region: &Region,
) {
    let (ecx, ecy) = region.ellipse_center_in_roi();
    let (semi_a, semi_b) = region.ellipse_axes();

    if semi_a <= 0.0 || semi_b <= 0.0 {
        return;
    }

    let inv_a_sq = 1.0 / (semi_a * semi_a);
    let inv_b_sq = 1.0 / (semi_b * semi_b);

    for row in 0..rect.h {
        for col in 0..rect.w {
            let dx = col as f64 - ecx;
            let dy = row as f64 - ecy;

            if dx * dx * inv_a_sq + dy * dy * inv_b_sq <= 1.0 {
                let frame_offset = ((rect.y + row) * frame_width + (rect.x + col)) * channels;
                let roi_offset = (row * rect.w + col) * channels;
                data[frame_offset..frame_offset + channels]
                    .copy_from_slice(&roi[roi_offset..roi_offset + channels]);
            }
        }
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
        blurrer.blur(&mut frame, &[region(5, 5, 30, 30)]).unwrap();

        let center = (20 * 100 + 20) * 3;
        assert!(frame.data()[center] < 255 || frame.data()[(18 * 100 + 18) * 3] < 255);
    }

    #[test]
    fn test_pixels_outside_region_unchanged() {
        let mut frame = make_frame(100, 100, 200);
        let original = frame.data().to_vec();
        let blurrer = CpuEllipticalBlurrer::new(5);
        blurrer.blur(&mut frame, &[region(10, 10, 20, 20)]).unwrap();

        assert_eq!(frame.data()[0], original[0]);
        let idx = (50 * 100 + 50) * 3;
        assert_eq!(frame.data()[idx], original[idx]);
    }

    #[test]
    fn test_ellipse_does_not_blur_corners() {
        let mut frame = make_frame(100, 100, 0);
        let data = frame.data_mut();
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

        // Corner (0,0) is outside the ellipse — ellipse distance is 2.0 > 1.0
        assert_eq!(frame.data()[0], original[0]);
    }

    #[test]
    fn test_multiple_regions() {
        let mut frame = make_frame(100, 100, 0);
        let data = frame.data_mut();
        let idx1 = (20 * 100 + 20) * 3;
        data[idx1] = 255;
        data[idx1 + 1] = 255;
        data[idx1 + 2] = 255;
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

        assert!(ecx < r.width as f64 / 2.0);
        assert_eq!(sa, 30.0);
        assert_eq!(sb, 20.0);

        let mut frame = make_frame(100, 100, 128);
        let blurrer = CpuEllipticalBlurrer::new(5);
        blurrer.blur(&mut frame, &[r]).unwrap();
    }

    #[test]
    fn test_default_kernel_size() {
        let blurrer = CpuEllipticalBlurrer::default();
        assert_eq!(blurrer.kernel.len(), DEFAULT_KERNEL_SIZE);
    }
}
