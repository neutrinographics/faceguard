/// HSV histogram-based face grouper.
///
/// A fast fallback grouper that compares face crops by their color
/// distribution using 2D Hue-Saturation histograms with Pearson correlation.
/// No ML model required — useful when ArcFace is unavailable.
use crate::detection::domain::face_grouper::FaceGrouper;
use crate::detection::infrastructure::math;

pub const DEFAULT_THRESHOLD: f64 = 0.7;

const HUE_BINS: usize = 32;
const SAT_BINS: usize = 32;

pub struct HistogramFaceGrouper {
    threshold: f64,
}

impl HistogramFaceGrouper {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Default for HistogramFaceGrouper {
    fn default() -> Self {
        Self::new(DEFAULT_THRESHOLD)
    }
}

impl FaceGrouper for HistogramFaceGrouper {
    fn group(
        &self,
        crops: &[(u32, &[u8], u32, u32)],
    ) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }

        let histograms: Vec<Vec<f64>> = crops
            .iter()
            .map(|(_, data, w, h)| compute_histogram(data, *w, *h))
            .collect();

        let n = crops.len();
        let mut parent: Vec<usize> = (0..n).collect();

        for i in 0..n {
            for j in (i + 1)..n {
                if pearson_correlation(&histograms[i], &histograms[j]) >= self.threshold {
                    math::union(&mut parent, i, j);
                }
            }
        }

        let entries: Vec<(usize, u32)> = crops
            .iter()
            .enumerate()
            .map(|(idx, (track_id, _, _, _))| (idx, *track_id))
            .collect();
        Ok(math::collect_groups(&mut parent, &entries))
    }
}

fn compute_histogram(rgb_data: &[u8], width: u32, height: u32) -> Vec<f64> {
    let num_pixels = (width * height) as usize;
    let mut hist = vec![0.0f64; HUE_BINS * SAT_BINS];
    let mut count = 0usize;

    for i in 0..num_pixels {
        let offset = i * 3;
        if offset + 2 >= rgb_data.len() {
            break;
        }
        let r = rgb_data[offset] as f64 / 255.0;
        let g = rgb_data[offset + 1] as f64 / 255.0;
        let b = rgb_data[offset + 2] as f64 / 255.0;

        let (h, s, _v) = rgb_to_hsv(r, g, b);

        let h_bin = ((h / 360.0) * HUE_BINS as f64).min(HUE_BINS as f64 - 1.0) as usize;
        let s_bin = (s * SAT_BINS as f64).min(SAT_BINS as f64 - 1.0) as usize;

        hist[h_bin * SAT_BINS + s_bin] += 1.0;
        count += 1;
    }

    if count > 0 {
        let total = count as f64;
        for v in &mut hist {
            *v /= total;
        }
    }

    hist
}

fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };

    let h = if delta == 0.0 {
        0.0
    } else if (max - r).abs() < f64::EPSILON {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() < f64::EPSILON {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    (h, s, v)
}

/// Pearson correlation coefficient.
///
/// Returns 1.0 when both inputs have zero variance (identical distributions),
/// and 0.0 when only one has zero variance (undefined, treated as uncorrelated).
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..a.len().min(b.len()) {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < f64::EPSILON {
        return if var_a < f64::EPSILON && var_b < f64::EPSILON {
            1.0
        } else {
            0.0
        };
    }

    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(r: u8, g: u8, b: u8, w: u32, h: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            data.push(r);
            data.push(g);
            data.push(b);
        }
        data
    }

    #[test]
    fn test_empty_input_returns_empty() {
        let grouper = HistogramFaceGrouper::new(0.7);
        assert!(grouper.group(&[]).unwrap().is_empty());
    }

    #[test]
    fn test_single_face_returns_one_group() {
        let grouper = HistogramFaceGrouper::new(0.7);
        let crop = solid_rgb(128, 0, 0, 50, 50);
        let result = grouper.group(&[(1, &crop, 50, 50)]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![1]);
    }

    #[test]
    fn test_identical_images_group_together() {
        let grouper = HistogramFaceGrouper::new(0.7);
        let crop = solid_rgb(200, 50, 50, 50, 50);
        let result = grouper
            .group(&[(1, &crop, 50, 50), (2, &crop, 50, 50)])
            .unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&1));
        assert!(result[0].contains(&2));
    }

    #[test]
    fn test_very_different_images_stay_separate() {
        let grouper = HistogramFaceGrouper::new(0.7);
        let red = solid_rgb(255, 0, 0, 50, 50);
        let blue = solid_rgb(0, 0, 255, 50, 50);
        let result = grouper
            .group(&[(1, &red, 50, 50), (2, &blue, 50, 50)])
            .unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_transitive_grouping() {
        let grouper = HistogramFaceGrouper::new(0.7);
        let crop = solid_rgb(100, 200, 50, 50, 50);
        let result = grouper
            .group(&[(1, &crop, 50, 50), (2, &crop, 50, 50), (3, &crop, 50, 50)])
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_custom_threshold_very_high() {
        let grouper = HistogramFaceGrouper::new(0.999);
        let crop_a = solid_rgb(255, 0, 0, 50, 50);
        let crop_b = solid_rgb(0, 255, 0, 50, 50);
        let result = grouper
            .group(&[(1, &crop_a, 50, 50), (2, &crop_b, 50, 50)])
            .unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_preserves_track_ids() {
        let grouper = HistogramFaceGrouper::new(0.7);
        let crop = solid_rgb(100, 100, 100, 50, 50);
        let result = grouper
            .group(&[(10, &crop, 50, 50), (20, &crop, 50, 50)])
            .unwrap();
        let all_ids: Vec<u32> = result.into_iter().flatten().collect();
        assert!(all_ids.contains(&10));
        assert!(all_ids.contains(&20));
    }

    #[test]
    fn test_rgb_to_hsv_red() {
        let (h, s, v) = rgb_to_hsv(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1.0);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rgb_to_hsv_green() {
        let (h, s, v) = rgb_to_hsv(0.0, 1.0, 0.0);
        assert!((h - 120.0).abs() < 1.0);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rgb_to_hsv_blue() {
        let (h, s, v) = rgb_to_hsv(0.0, 0.0, 1.0);
        assert!((h - 240.0).abs() < 1.0);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rgb_to_hsv_white() {
        let (_h, s, v) = rgb_to_hsv(1.0, 1.0, 1.0);
        assert_eq!(s, 0.0);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pearson_identical() {
        let a = vec![0.1, 0.2, 0.3, 0.4];
        assert!((pearson_correlation(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pearson_uncorrelated() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        assert!((pearson_correlation(&a, &b) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_pearson_both_zero_variance() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        assert!((pearson_correlation(&a, &a) - 1.0).abs() < 1e-9);
    }
}
