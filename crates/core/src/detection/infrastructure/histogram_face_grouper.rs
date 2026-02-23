/// HSV histogram-based face grouper.
///
/// A fast fallback grouper that compares face crops by their color
/// distribution using 2D Hue-Saturation histograms with Pearson correlation.
/// No ML model required.
use crate::detection::domain::face_grouper::FaceGrouper;

/// Default correlation threshold for grouping.
pub const DEFAULT_THRESHOLD: f64 = 0.7;

/// Histogram bins per channel.
const HUE_BINS: usize = 32;
const SAT_BINS: usize = 32;

/// HSV histogram face grouper.
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

        // Compute histogram for each crop
        let histograms: Vec<Vec<f64>> = crops
            .iter()
            .map(|(_, data, w, h)| compute_histogram(data, *w, *h))
            .collect();

        // Union-find clustering
        let n = crops.len();
        let mut parent: Vec<usize> = (0..n).collect();

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = pearson_correlation(&histograms[i], &histograms[j]);
                if sim >= self.threshold {
                    union(&mut parent, i, j);
                }
            }
        }

        // Collect groups
        let mut groups: std::collections::HashMap<usize, Vec<u32>> =
            std::collections::HashMap::new();
        for (idx, (track_id, _, _, _)) in crops.iter().enumerate() {
            let root = find(&mut parent, idx);
            groups.entry(root).or_default().push(*track_id);
        }

        let mut result: Vec<Vec<u32>> = groups.into_values().collect();
        result.sort_by_key(|g| g[0]);
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// HSV conversion + histogram computation
// ---------------------------------------------------------------------------

/// Convert RGB to HSV and compute a normalized 2D Hue-Saturation histogram.
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

        // Hue: [0, 360) → bin [0, HUE_BINS)
        let h_bin = ((h / 360.0) * HUE_BINS as f64).min(HUE_BINS as f64 - 1.0) as usize;
        // Saturation: [0, 1] → bin [0, SAT_BINS)
        let s_bin = (s * SAT_BINS as f64).min(SAT_BINS as f64 - 1.0) as usize;

        hist[h_bin * SAT_BINS + s_bin] += 1.0;
        count += 1;
    }

    // Normalize
    if count > 0 {
        let total = count as f64;
        for v in &mut hist {
            *v /= total;
        }
    }

    hist
}

/// Convert RGB [0,1] to HSV. H in [0,360), S and V in [0,1].
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

// ---------------------------------------------------------------------------
// Pearson correlation
// ---------------------------------------------------------------------------

/// Pearson correlation coefficient between two histograms.
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
        return 0.0;
    }

    cov / denom
}

// ---------------------------------------------------------------------------
// Union-Find
// ---------------------------------------------------------------------------

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]]; // path halving
        i = parent[i];
    }
    i
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let result = grouper.group(&[]).unwrap();
        assert!(result.is_empty());
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
        // With threshold=0.999, distinctly different colors should separate
        let grouper = HistogramFaceGrouper::new(0.999);
        let crop_a = solid_rgb(255, 0, 0, 50, 50); // pure red
        let crop_b = solid_rgb(0, 255, 0, 50, 50); // pure green
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
        assert!((h - 0.0).abs() < 1.0); // ~0 degrees
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
        let (h, s, v) = rgb_to_hsv(1.0, 1.0, 1.0);
        assert_eq!(s, 0.0); // achromatic
        assert!((v - 1.0).abs() < 0.01);
        let _ = h; // hue is undefined for achromatic
    }

    #[test]
    fn test_pearson_identical() {
        let a = vec![0.1, 0.2, 0.3, 0.4];
        let r = pearson_correlation(&a, &a);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pearson_uncorrelated() {
        // Flat vs one-hot — should have low correlation
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let r = pearson_correlation(&a, &b);
        // Flat histogram has zero variance → correlation = 0
        assert!((r - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_union_find_basic() {
        let mut parent = vec![0, 1, 2, 3];
        union(&mut parent, 0, 1);
        assert_eq!(find(&mut parent, 0), find(&mut parent, 1));

        union(&mut parent, 2, 3);
        assert_eq!(find(&mut parent, 2), find(&mut parent, 3));

        assert_ne!(find(&mut parent, 0), find(&mut parent, 2));

        union(&mut parent, 1, 3);
        assert_eq!(find(&mut parent, 0), find(&mut parent, 3));
    }
}
