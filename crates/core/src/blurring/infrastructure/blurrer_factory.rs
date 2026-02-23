use crate::blurring::domain::frame_blurrer::FrameBlurrer;

use super::cpu_elliptical_blurrer::CpuEllipticalBlurrer;
use super::cpu_rectangular_blurrer::CpuRectangularBlurrer;
use super::gpu_context::GpuContext;
use super::gpu_elliptical_blurrer::GpuEllipticalBlurrer;
use super::gpu_rectangular_blurrer::GpuRectangularBlurrer;

/// Blur shape preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlurShape {
    Elliptical,
    Rectangular,
}

/// Creates the best available blurrer, preferring GPU when available.
///
/// Probes for a wgpu adapter at startup. If one is found, returns a GPU
/// blurrer; otherwise falls back to the CPU implementation. Logs which
/// backend is selected.
pub fn create_blurrer(shape: BlurShape, kernel_size: usize) -> Box<dyn FrameBlurrer> {
    if let Some(ctx) = GpuContext::new() {
        log::info!("Using GPU backend for {:?} blur (kernel_size={})", shape, kernel_size);
        match shape {
            BlurShape::Elliptical => Box::new(GpuEllipticalBlurrer::new(ctx, kernel_size as u32)),
            BlurShape::Rectangular => Box::new(GpuRectangularBlurrer::new(ctx, kernel_size as u32)),
        }
    } else {
        log::info!("No GPU available, using CPU backend for {:?} blur (kernel_size={})", shape, kernel_size);
        match shape {
            BlurShape::Elliptical => Box::new(CpuEllipticalBlurrer::new(kernel_size)),
            BlurShape::Rectangular => Box::new(CpuRectangularBlurrer::new(kernel_size)),
        }
    }
}

/// Returns true if a GPU adapter is available for compute shaders.
pub fn gpu_available() -> bool {
    GpuContext::new().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::frame::Frame;
    use crate::shared::region::Region;

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
    fn test_create_elliptical_blurrer_works() {
        let blurrer = create_blurrer(BlurShape::Elliptical, 5);
        let mut frame = make_frame(50, 50, 128);
        blurrer.blur(&mut frame, &[]).unwrap();
    }

    #[test]
    fn test_create_rectangular_blurrer_works() {
        let blurrer = create_blurrer(BlurShape::Rectangular, 5);
        let mut frame = make_frame(50, 50, 128);
        blurrer.blur(&mut frame, &[]).unwrap();
    }

    #[test]
    fn test_factory_blurrer_actually_blurs() {
        let blurrer = create_blurrer(BlurShape::Rectangular, 5);
        let mut frame = make_frame(50, 50, 0);
        let data = frame.data_mut();
        for y in 20..25 {
            for x in 20..25 {
                let idx = (y * 50 + x) * 3;
                data[idx] = 255;
            }
        }

        blurrer.blur(&mut frame, &[region(10, 10, 30, 30)]).unwrap();

        // Neighboring pixel should have blur spread
        let neighbor = (19 * 50 + 22) * 3;
        assert!(frame.data()[neighbor] > 0);
    }

    #[test]
    fn test_gpu_available_returns_bool() {
        // Just verify it doesn't panic — result depends on hardware
        let _ = gpu_available();
    }
}
