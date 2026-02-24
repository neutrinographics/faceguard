use std::collections::HashSet;

pub const DEFAULT_IOU_THRESHOLD: f64 = 0.3;

/// A blur target region with edge-aware ellipse rendering support.
///
/// Carries both clamped (visible) and unclamped (full) geometry so
/// ellipses slide off frame edges naturally instead of shrinking.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub track_id: Option<u32>,
    pub full_width: Option<i32>,
    pub full_height: Option<i32>,
    pub unclamped_x: Option<i32>,
    pub unclamped_y: Option<i32>,
}

impl Region {
    /// Filters regions by track ID inclusion/exclusion sets.
    ///
    /// `blur_ids` takes precedence: when set, only matching track IDs pass.
    pub fn filter(
        regions: &[Region],
        blur_ids: Option<&HashSet<u32>>,
        exclude_ids: Option<&HashSet<u32>>,
    ) -> Vec<Region> {
        if let Some(ids) = blur_ids {
            regions
                .iter()
                .filter(|r| r.track_id.is_some_and(|tid| ids.contains(&tid)))
                .cloned()
                .collect()
        } else if let Some(ids) = exclude_ids {
            regions
                .iter()
                .filter(|r| r.track_id.map_or(true, |tid| !ids.contains(&tid)))
                .cloned()
                .collect()
        } else {
            regions.to_vec()
        }
    }

    /// Greedy deduplication: keeps a region only if its IoU with every
    /// previously-kept region is at or below the threshold.
    pub fn deduplicate(regions: &[Region], iou_threshold: f64) -> Vec<Region> {
        if regions.len() <= 1 {
            return regions.to_vec();
        }
        let mut kept: Vec<Region> = Vec::with_capacity(regions.len());
        for r in regions {
            let dominated = kept.iter().any(|k| r.iou(k) > iou_threshold);
            if !dominated {
                kept.push(r.clone());
            }
        }
        kept
    }

    pub fn iou(&self, other: &Region) -> f64 {
        let ix1 = self.x.max(other.x);
        let iy1 = self.y.max(other.y);
        let ix2 = (self.x + self.width).min(other.x + other.width);
        let iy2 = (self.y + self.height).min(other.y + other.height);

        let inter = (ix2 - ix1).max(0) as f64 * (iy2 - iy1).max(0) as f64;
        if inter == 0.0 {
            return 0.0;
        }

        let area_a = self.width as f64 * self.height as f64;
        let area_b = other.width as f64 * other.height as f64;
        inter / (area_a + area_b - inter)
    }

    /// Ellipse center relative to the clamped ROI.
    ///
    /// When clipped at a frame edge, the center shifts so the ellipse
    /// extends naturally off-screen rather than collapsing inward.
    pub fn ellipse_center_in_roi(&self) -> (f64, f64) {
        let (fw, fh, ux, uy) = self.unclamped_geometry();
        let center_x = (fw as f64) / 2.0 - (self.x - ux) as f64;
        let center_y = (fh as f64) / 2.0 - (self.y - uy) as f64;
        (center_x, center_y)
    }

    pub fn ellipse_axes(&self) -> (f64, f64) {
        let (fw, fh, _, _) = self.unclamped_geometry();
        (fw as f64 / 2.0, fh as f64 / 2.0)
    }

    fn unclamped_geometry(&self) -> (i32, i32, i32, i32) {
        (
            self.full_width.unwrap_or(self.width),
            self.full_height.unwrap_or(self.height),
            self.unclamped_x.unwrap_or(self.x),
            self.unclamped_y.unwrap_or(self.y),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

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

    #[allow(clippy::too_many_arguments)]
    fn region_with_unclamped(
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        ux: i32,
        uy: i32,
        fw: i32,
        fh: i32,
    ) -> Region {
        Region {
            unclamped_x: Some(ux),
            unclamped_y: Some(uy),
            full_width: Some(fw),
            full_height: Some(fh),
            ..region(x, y, w, h)
        }
    }

    // ── IoU ──────────────────────────────────────────────────────────

    #[test]
    fn test_iou_identical_regions() {
        let a = region(10, 10, 100, 100);
        assert_relative_eq!(a.iou(&a), 1.0);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = region(0, 0, 50, 50);
        let b = region(100, 100, 50, 50);
        assert_relative_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_iou_partial_overlap() {
        // a: [0,0]-[100,100], b: [50,0]-[150,100]
        // intersection: [50,0]-[100,100] = 50*100 = 5000
        // union: 10000 + 10000 - 5000 = 15000
        let a = region(0, 0, 100, 100);
        let b = region(50, 0, 100, 100);
        assert_relative_eq!(a.iou(&b), 5000.0 / 15000.0);
    }

    #[test]
    fn test_iou_contained() {
        // b fully inside a
        let a = region(0, 0, 100, 100);
        let b = region(25, 25, 50, 50);
        // inter = 2500, union = 10000 + 2500 - 2500 = 10000
        assert_relative_eq!(a.iou(&b), 2500.0 / 10000.0);
    }

    #[test]
    fn test_iou_touching_edges() {
        let a = region(0, 0, 50, 50);
        let b = region(50, 0, 50, 50);
        assert_relative_eq!(a.iou(&b), 0.0);
    }

    // ── Deduplication ────────────────────────────────────────────────

    #[test]
    fn test_deduplicate_empty() {
        let result = Region::deduplicate(&[], DEFAULT_IOU_THRESHOLD);
        assert!(result.is_empty());
    }

    #[test]
    fn test_deduplicate_single() {
        let regions = vec![region(0, 0, 50, 50)];
        let result = Region::deduplicate(&regions, DEFAULT_IOU_THRESHOLD);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_deduplicate_removes_overlapping() {
        let regions = vec![
            region(0, 0, 100, 100),
            region(10, 10, 100, 100), // high IoU with first
        ];
        let result = Region::deduplicate(&regions, DEFAULT_IOU_THRESHOLD);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], regions[0]);
    }

    #[test]
    fn test_deduplicate_keeps_non_overlapping() {
        let regions = vec![
            region(0, 0, 50, 50),
            region(200, 200, 50, 50), // no overlap
        ];
        let result = Region::deduplicate(&regions, DEFAULT_IOU_THRESHOLD);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_deduplicate_default_threshold() {
        // Verify the constant is 0.3
        assert_relative_eq!(DEFAULT_IOU_THRESHOLD, 0.3);
    }

    // ── Ellipse geometry ─────────────────────────────────────────────

    #[test]
    fn test_ellipse_center_no_clamping() {
        // No unclamped values → center is just width/2, height/2
        let r = region(100, 100, 200, 150);
        let (cx, cy) = r.ellipse_center_in_roi();
        assert_relative_eq!(cx, 100.0);
        assert_relative_eq!(cy, 75.0);
    }

    #[test]
    fn test_ellipse_center_clipped_at_left_edge() {
        // Full region: unclamped_x=-50, full_width=200
        // Clamped: x=0, width=150
        // center_x = 200/2 - (0 - (-50)) = 100 - 50 = 50
        let r = region_with_unclamped(0, 100, 150, 200, -50, 100, 200, 200);
        let (cx, _cy) = r.ellipse_center_in_roi();
        assert_relative_eq!(cx, 50.0);
    }

    #[test]
    fn test_ellipse_center_clipped_at_top_edge() {
        // Full region: unclamped_y=-30, full_height=100
        // Clamped: y=0, height=70
        // center_y = 100/2 - (0 - (-30)) = 50 - 30 = 20
        let r = region_with_unclamped(100, 0, 200, 70, 100, -30, 200, 100);
        let (_cx, cy) = r.ellipse_center_in_roi();
        assert_relative_eq!(cy, 20.0);
    }

    #[test]
    fn test_ellipse_axes_no_unclamped() {
        let r = region(0, 0, 200, 150);
        let (ax, ay) = r.ellipse_axes();
        assert_relative_eq!(ax, 100.0);
        assert_relative_eq!(ay, 75.0);
    }

    #[test]
    fn test_ellipse_axes_uses_full_dimensions() {
        // full_width=300, full_height=250 (larger than clamped)
        let r = region_with_unclamped(0, 0, 200, 150, -50, -50, 300, 250);
        let (ax, ay) = r.ellipse_axes();
        assert_relative_eq!(ax, 150.0);
        assert_relative_eq!(ay, 125.0);
    }

    // ── Parametrized IoU edge cases ──────────────────────────────────

    #[rstest]
    #[case::zero_width(region(0, 0, 0, 100), region(0, 0, 50, 50), 0.0)]
    #[case::zero_height(region(0, 0, 100, 0), region(0, 0, 50, 50), 0.0)]
    fn test_iou_degenerate(#[case] a: Region, #[case] b: Region, #[case] expected: f64) {
        assert_relative_eq!(a.iou(&b), expected);
    }

    // ── Filter ────────────────────────────────────────────────────────

    fn tracked_region(track_id: Option<u32>) -> Region {
        Region {
            x: 10,
            y: 10,
            width: 50,
            height: 50,
            track_id,
            full_width: None,
            full_height: None,
            unclamped_x: None,
            unclamped_y: None,
        }
    }

    #[test]
    fn test_filter_no_filters_returns_all() {
        let regions = vec![tracked_region(Some(1)), tracked_region(Some(2))];
        let result = Region::filter(&regions, None, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_filter_blur_ids_keeps_only_matching() {
        let regions = vec![
            tracked_region(Some(1)),
            tracked_region(Some(2)),
            tracked_region(Some(3)),
        ];
        let blur_ids = HashSet::from([1, 3]);
        let result = Region::filter(&regions, Some(&blur_ids), None);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].track_id, Some(1));
        assert_eq!(result[1].track_id, Some(3));
    }

    #[test]
    fn test_filter_exclude_ids_removes_matching() {
        let regions = vec![
            tracked_region(Some(1)),
            tracked_region(Some(2)),
            tracked_region(Some(3)),
        ];
        let exclude_ids = HashSet::from([2]);
        let result = Region::filter(&regions, None, Some(&exclude_ids));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].track_id, Some(1));
        assert_eq!(result[1].track_id, Some(3));
    }

    #[test]
    fn test_filter_blur_ids_takes_precedence_over_exclude_ids() {
        let regions = vec![
            tracked_region(Some(1)),
            tracked_region(Some(2)),
            tracked_region(Some(3)),
        ];
        let blur_ids = HashSet::from([1]);
        let exclude_ids = HashSet::from([3]);
        let result = Region::filter(&regions, Some(&blur_ids), Some(&exclude_ids));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].track_id, Some(1));
    }

    #[test]
    fn test_filter_blur_ids_excludes_none_track_id() {
        let regions = vec![tracked_region(None), tracked_region(Some(1))];
        let blur_ids = HashSet::from([1]);
        let result = Region::filter(&regions, Some(&blur_ids), None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].track_id, Some(1));
    }

    #[test]
    fn test_filter_exclude_ids_includes_none_track_id() {
        let regions = vec![tracked_region(None), tracked_region(Some(1))];
        let exclude_ids = HashSet::from([1]);
        let result = Region::filter(&regions, None, Some(&exclude_ids));
        assert_eq!(result.len(), 1);
        assert!(result[0].track_id.is_none());
    }

    #[test]
    fn test_filter_empty_regions() {
        let result = Region::filter(&[], None, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_empty_blur_ids_excludes_all() {
        let regions = vec![tracked_region(Some(1))];
        let blur_ids = HashSet::new();
        let result = Region::filter(&regions, Some(&blur_ids), None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_empty_exclude_ids_keeps_all() {
        let regions = vec![tracked_region(Some(1)), tracked_region(Some(2))];
        let exclude_ids = HashSet::new();
        let result = Region::filter(&regions, None, Some(&exclude_ids));
        assert_eq!(result.len(), 2);
    }
}
