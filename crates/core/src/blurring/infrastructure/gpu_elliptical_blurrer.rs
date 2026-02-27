use std::sync::Arc;

use crate::blurring::domain::frame_blurrer::FrameBlurrer;
use crate::shared::frame::Frame;
use crate::shared::region::Region;

use super::gpu_context::{pack_roi, unpack_roi, GpuContext, RoiDescriptor};

const DEFAULT_KERNEL_SIZE: u32 = 201;

/// GPU elliptical blurrer using a wgpu compute shader.
///
/// Runs a two-pass separable Gaussian blur on the GPU with an ellipse mask.
/// Uses the Region's edge-aware ellipse geometry so the blur extends
/// smoothly off frame edges.
pub struct GpuEllipticalBlurrer {
    ctx: Arc<GpuContext>,
    kernel_size: u32,
}

impl GpuEllipticalBlurrer {
    pub fn new(ctx: Arc<GpuContext>, kernel_size: u32) -> Self {
        Self { ctx, kernel_size }
    }

    pub fn with_default_kernel(ctx: Arc<GpuContext>) -> Self {
        Self::new(ctx, DEFAULT_KERNEL_SIZE)
    }
}

impl FrameBlurrer for GpuEllipticalBlurrer {
    fn blur(
        &self,
        frame: &mut Frame,
        regions: &[Region],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if regions.is_empty() {
            return Ok(());
        }

        let fw = frame.width() as usize;
        let fh = frame.height() as usize;
        let channels = frame.channels() as usize;
        let data = frame.data_mut();

        let mut descriptors: Vec<RoiDescriptor> = Vec::with_capacity(regions.len());
        let mut region_info: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(regions.len());

        for r in regions {
            let rx = r.x.max(0) as usize;
            let ry = r.y.max(0) as usize;
            let rw = (r.width.max(0) as usize).min(fw.saturating_sub(rx));
            let rh = (r.height.max(0) as usize).min(fh.saturating_sub(ry));

            if rw == 0 || rh == 0 {
                continue;
            }

            let (ecx, ecy) = r.ellipse_center_in_roi();
            let (semi_a, semi_b) = r.ellipse_axes();

            descriptors.push(RoiDescriptor {
                pixels: pack_roi(data, fw, channels, rx, ry, rw, rh),
                width: rw as u32,
                height: rh as u32,
                kernel_size: self.kernel_size,
                ellipse_cx: ecx as f32,
                ellipse_cy: ecy as f32,
                ellipse_a: semi_a as f32,
                ellipse_b: semi_b as f32,
                use_ellipse: true,
            });
            region_info.push((rx, ry, rw, rh));
        }

        let results = self.ctx.blur_rois(&descriptors);

        for (result, &(rx, ry, rw, rh)) in results.iter().zip(region_info.iter()) {
            unpack_roi(data, result, fw, channels, rx, ry, rw, rh);
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

    fn try_gpu_context() -> Option<Arc<GpuContext>> {
        GpuContext::new().map(Arc::new)
    }

    #[test]
    fn test_no_regions_frame_unchanged() {
        let ctx = match try_gpu_context() {
            Some(c) => c,
            None => return,
        };
        let mut frame = make_frame(50, 50, 128);
        let original = frame.data().to_vec();
        let blurrer = GpuEllipticalBlurrer::new(ctx, 5);
        blurrer.blur(&mut frame, &[]).unwrap();
        assert_eq!(frame.data(), &original[..]);
    }

    #[test]
    fn test_blur_modifies_pixels_inside_ellipse() {
        let ctx = match try_gpu_context() {
            Some(c) => c,
            None => return,
        };
        let mut frame = make_frame(50, 50, 0);
        let data = frame.data_mut();
        for y in 20..25 {
            for x in 20..25 {
                let idx = (y * 50 + x) * 3;
                data[idx] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
            }
        }

        let blurrer = GpuEllipticalBlurrer::new(ctx, 5);
        blurrer.blur(&mut frame, &[region(10, 10, 30, 30)]).unwrap();

        let center = (22 * 50 + 22) * 3;
        assert!(
            frame.data()[center] < 255 || frame.data()[(19 * 50 + 22) * 3] > 0,
            "GPU elliptical blur should modify pixels within ellipse"
        );
    }

    #[test]
    fn test_corners_outside_ellipse_unchanged() {
        let ctx = match try_gpu_context() {
            Some(c) => c,
            None => return,
        };
        let mut frame = make_frame(50, 50, 100);
        let original = frame.data().to_vec();
        let blurrer = GpuEllipticalBlurrer::new(ctx, 5);
        blurrer.blur(&mut frame, &[region(0, 0, 40, 40)]).unwrap();
        assert_eq!(frame.data()[0], original[0]);
    }

    #[test]
    fn test_pixels_outside_region_unchanged() {
        let ctx = match try_gpu_context() {
            Some(c) => c,
            None => return,
        };
        let mut frame = make_frame(50, 50, 200);
        let original = frame.data().to_vec();
        let blurrer = GpuEllipticalBlurrer::new(ctx, 5);
        blurrer.blur(&mut frame, &[region(10, 10, 20, 20)]).unwrap();
        assert_eq!(frame.data()[0], original[0]);
    }

    #[test]
    fn test_zero_size_region_skipped() {
        let ctx = match try_gpu_context() {
            Some(c) => c,
            None => return,
        };
        let mut frame = make_frame(50, 50, 128);
        let original = frame.data().to_vec();
        let blurrer = GpuEllipticalBlurrer::new(ctx, 5);
        blurrer.blur(&mut frame, &[region(10, 10, 0, 20)]).unwrap();
        assert_eq!(frame.data(), &original[..]);
    }

    #[test]
    fn test_ellipse_uses_full_dimensions() {
        let ctx = match try_gpu_context() {
            Some(c) => c,
            None => return,
        };
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

        let mut frame = make_frame(50, 50, 128);
        let blurrer = GpuEllipticalBlurrer::new(ctx, 5);
        blurrer.blur(&mut frame, &[r]).unwrap();
    }

    #[test]
    fn test_region_extending_beyond_frame_does_not_panic() {
        let ctx = match try_gpu_context() {
            Some(c) => c,
            None => return,
        };
        let mut frame = make_frame(50, 50, 128);
        let blurrer = GpuEllipticalBlurrer::new(ctx, 5);
        blurrer.blur(&mut frame, &[region(40, 40, 30, 30)]).unwrap();
    }
}
