use crate::blurring::domain::frame_blurrer::FrameBlurrer;
use crate::shared::frame::Frame;
use crate::shared::region::Region;

use super::gpu_context::GpuContext;

/// Default kernel size for GPU Gaussian blur.
const DEFAULT_KERNEL_SIZE: u32 = 201;

/// GPU elliptical blurrer using a wgpu compute shader.
///
/// Runs a two-pass separable Gaussian blur on the GPU with an ellipse mask.
/// Uses the Region's edge-aware ellipse geometry so the blur extends
/// smoothly off frame edges.
pub struct GpuEllipticalBlurrer {
    ctx: GpuContext,
    kernel_size: u32,
}

impl GpuEllipticalBlurrer {
    pub fn new(ctx: GpuContext, kernel_size: u32) -> Self {
        Self { ctx, kernel_size }
    }

    pub fn with_default_kernel(ctx: GpuContext) -> Self {
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

            // Get ellipse geometry from region
            let (ecx, ecy) = r.ellipse_center_in_roi();
            let (semi_a, semi_b) = r.ellipse_axes();

            // Extract ROI and pack into RGBA u32
            let mut packed: Vec<u32> = Vec::with_capacity(rw * rh);
            for row in 0..rh {
                for col in 0..rw {
                    let offset = ((ry + row) * fw + (rx + col)) * channels;
                    let r_val = data[offset] as u32;
                    let g_val = if channels > 1 {
                        data[offset + 1] as u32
                    } else {
                        0
                    };
                    let b_val = if channels > 2 {
                        data[offset + 2] as u32
                    } else {
                        0
                    };
                    packed.push(r_val | (g_val << 8) | (b_val << 16) | (255 << 24));
                }
            }

            // Run GPU blur with ellipse mask
            let result = self.ctx.blur_roi(
                &packed,
                rw as u32,
                rh as u32,
                self.kernel_size,
                ecx as f32,
                ecy as f32,
                semi_a as f32,
                semi_b as f32,
                true,
            );

            // Unpack and write back
            for row in 0..rh {
                for col in 0..rw {
                    let pixel = result[row * rw + col];
                    let offset = ((ry + row) * fw + (rx + col)) * channels;
                    data[offset] = (pixel & 0xFF) as u8;
                    if channels > 1 {
                        data[offset + 1] = ((pixel >> 8) & 0xFF) as u8;
                    }
                    if channels > 2 {
                        data[offset + 2] = ((pixel >> 16) & 0xFF) as u8;
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

    fn try_gpu_context() -> Option<GpuContext> {
        GpuContext::new()
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
        // Bright spot in center of region (inside ellipse)
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

        // Center pixel (within ellipse) should be blurred
        let center = (22 * 50 + 22) * 3;
        // The bright patch should have been blurred
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
        // Region from (0,0) to (40,40), ellipse center at (20,20)
        blurrer.blur(&mut frame, &[region(0, 0, 40, 40)]).unwrap();

        // Corner pixel (0,0) is outside ellipse — should be original value
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

        // Pixel at (0,0) outside region
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
        // Region clipped at left edge with full dimensions
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
        // Should not crash with edge-clipped regions
        blurrer.blur(&mut frame, &[r]).unwrap();
    }
}
