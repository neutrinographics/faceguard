/// Precompute a 1D Gaussian kernel of the given size.
///
/// `kernel_size` must be odd and >= 1. Sigma is derived as `kernel_size / 6.0`
/// (matching OpenCV's sigma=0 convention).
pub fn gaussian_kernel_1d(kernel_size: usize) -> Vec<f64> {
    debug_assert!(kernel_size >= 1 && kernel_size % 2 == 1);
    let sigma = kernel_size as f64 / 6.0;
    let half = (kernel_size / 2) as f64;
    let mut kernel: Vec<f64> = (0..kernel_size)
        .map(|i| {
            let x = i as f64 - half;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let sum: f64 = kernel.iter().sum();
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Apply a separable Gaussian blur to an RGB image stored as `[height][width][3]`.
///
/// The image is modified in-place. `kernel_size` must be odd.
pub fn separable_gaussian_blur(
    data: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    kernel_size: usize,
) {
    if kernel_size <= 1 || width == 0 || height == 0 {
        return;
    }
    let kernel = gaussian_kernel_1d(kernel_size);
    let half = kernel_size / 2;

    // Temporary buffer for intermediate results
    let mut temp = vec![0.0f64; width * height * channels];

    // Horizontal pass: data → temp
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let mut sum = 0.0;
                for (k, &w) in kernel.iter().enumerate() {
                    let sx = (x as isize + k as isize - half as isize)
                        .max(0)
                        .min((width - 1) as isize) as usize;
                    sum += data[(y * width + sx) * channels + c] as f64 * w;
                }
                temp[(y * width + x) * channels + c] = sum;
            }
        }
    }

    // Vertical pass: temp → data
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let mut sum = 0.0;
                for (k, &w) in kernel.iter().enumerate() {
                    let sy = (y as isize + k as isize - half as isize)
                        .max(0)
                        .min((height - 1) as isize) as usize;
                    sum += temp[(sy * width + x) * channels + c] * w;
                }
                data[(y * width + x) * channels + c] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Downscale an image by integer factor using area averaging.
pub fn downscale(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    scale: usize,
) -> (Vec<u8>, usize, usize) {
    let new_w = width / scale;
    let new_h = height / scale;
    let mut out = vec![0u8; new_w * new_h * channels];

    for y in 0..new_h {
        for x in 0..new_w {
            for c in 0..channels {
                let mut sum = 0u32;
                let mut count = 0u32;
                for dy in 0..scale {
                    for dx in 0..scale {
                        let sy = y * scale + dy;
                        let sx = x * scale + dx;
                        if sy < height && sx < width {
                            sum += data[(sy * width + sx) * channels + c] as u32;
                            count += 1;
                        }
                    }
                }
                out[(y * new_w + x) * channels + c] = (sum / count) as u8;
            }
        }
    }

    (out, new_w, new_h)
}

/// Upscale an image by integer factor using bilinear interpolation.
pub fn upscale(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; target_w * target_h * channels];

    for y in 0..target_h {
        for x in 0..target_w {
            let src_x = x as f64 * (width as f64 - 1.0) / (target_w as f64 - 1.0).max(1.0);
            let src_y = y as f64 * (height as f64 - 1.0) / (target_h as f64 - 1.0).max(1.0);

            let x0 = (src_x.floor() as usize).min(width - 1);
            let x1 = (x0 + 1).min(width - 1);
            let y0 = (src_y.floor() as usize).min(height - 1);
            let y1 = (y0 + 1).min(height - 1);

            let fx = src_x - x0 as f64;
            let fy = src_y - y0 as f64;

            for c in 0..channels {
                let v00 = data[(y0 * width + x0) * channels + c] as f64;
                let v10 = data[(y0 * width + x1) * channels + c] as f64;
                let v01 = data[(y1 * width + x0) * channels + c] as f64;
                let v11 = data[(y1 * width + x1) * channels + c] as f64;

                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                out[(y * target_w + x) * channels + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_sums_to_one() {
        let k = gaussian_kernel_1d(7);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_is_symmetric() {
        let k = gaussian_kernel_1d(7);
        for i in 0..k.len() / 2 {
            assert!((k[i] - k[k.len() - 1 - i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_kernel_center_is_largest() {
        let k = gaussian_kernel_1d(7);
        let center = k[3];
        for (i, &v) in k.iter().enumerate() {
            if i != 3 {
                assert!(center >= v);
            }
        }
    }

    #[test]
    fn test_blur_uniform_image_unchanged() {
        // A uniform image should remain unchanged after blur
        let mut data = vec![128u8; 10 * 10 * 3];
        separable_gaussian_blur(&mut data, 10, 10, 3, 5);
        assert!(data.iter().all(|&v| (v as i32 - 128).abs() <= 1));
    }

    #[test]
    fn test_blur_modifies_high_contrast() {
        // A single bright pixel in a dark image should be spread out
        let mut data = vec![0u8; 10 * 10 * 3];
        // Set center pixel to white
        let cx = 5 * 10 + 5;
        data[cx * 3] = 255;
        data[cx * 3 + 1] = 255;
        data[cx * 3 + 2] = 255;

        let original = data.clone();
        separable_gaussian_blur(&mut data, 10, 10, 3, 5);

        // Center pixel should now be less bright
        assert!(data[cx * 3] < 255);
        // Nearby pixels should now be non-zero
        let neighbor = (5 * 10 + 6) * 3;
        assert!(data[neighbor] > 0);
        // The image should be different from original
        assert_ne!(data, original);
    }

    #[test]
    fn test_kernel_size_1_is_identity() {
        let mut data = vec![42u8; 5 * 5 * 3];
        let original = data.clone();
        separable_gaussian_blur(&mut data, 5, 5, 3, 1);
        assert_eq!(data, original);
    }

    #[test]
    fn test_downscale_upscale_roundtrip() {
        // Uniform image should survive roundtrip
        let data = vec![100u8; 8 * 8 * 3];
        let (small, sw, sh) = downscale(&data, 8, 8, 3, 2);
        assert_eq!(sw, 4);
        assert_eq!(sh, 4);
        let big = upscale(&small, sw, sh, 3, 8, 8);
        assert!(big.iter().all(|&v| (v as i32 - 100).abs() <= 1));
    }
}
