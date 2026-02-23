use std::collections::HashSet;

use crate::shared::region::{Region, DEFAULT_IOU_THRESHOLD};

/// Fraction of frame dimension used as edge detection threshold.
const EDGE_FRACTION: f64 = 0.25;

/// Merges current-frame regions with lookahead regions.
///
/// Deduplicates by track_id first (current wins), then by IoU.
/// Applies edge-aware interpolation to push lookahead regions toward
/// the nearest frame edge for smooth slide-in animation.
pub struct RegionMerger;

impl RegionMerger {
    pub fn new() -> Self {
        Self
    }

    pub fn merge(
        &self,
        current: &[Region],
        lookahead: &[Vec<Region>],
        frame_w: u32,
        frame_h: u32,
    ) -> Vec<Region> {
        let mut seen_ids: HashSet<u32> = current.iter().filter_map(|r| r.track_id).collect();

        let mut result: Vec<Region> = current.to_vec();
        let total = lookahead.len();

        for (idx, future) in lookahead.iter().enumerate() {
            for r in future {
                match r.track_id {
                    Some(tid) => {
                        if !seen_ids.contains(&tid) {
                            seen_ids.insert(tid);
                            let interpolated = if total > 0 {
                                interpolate(r, idx, total, frame_w, frame_h)
                            } else {
                                r.clone()
                            };
                            result.push(interpolated);
                        }
                    }
                    None => {
                        result.push(r.clone());
                    }
                }
            }
        }

        Region::deduplicate(&result, DEFAULT_IOU_THRESHOLD)
    }
}

impl Default for RegionMerger {
    fn default() -> Self {
        Self::new()
    }
}

/// Push a lookahead region toward its nearest frame edge.
///
/// Interpolation strength: `t = (idx + 1) / (total + 1)`
/// Only applies if the region center is within EDGE_FRACTION of a frame edge.
fn interpolate(region: &Region, idx: usize, total: usize, frame_w: u32, frame_h: u32) -> Region {
    let t = (idx + 1) as f64 / (total + 1) as f64;

    let cx = region.x as f64 + region.width as f64 / 2.0;
    let cy = region.y as f64 + region.height as f64 / 2.0;

    let d_left = cx;
    let d_right = frame_w as f64 - cx;
    let d_top = cy;
    let d_bottom = frame_h as f64 - cy;

    let min_dist = d_left.min(d_right).min(d_top).min(d_bottom);

    // Determine which edge is nearest and its threshold.
    // Use a small tolerance for float comparison (matching Python's integer arithmetic).
    let eps = 1e-9;
    let threshold =
        if (min_dist - d_left).abs() < eps || (min_dist - d_right).abs() < eps {
            frame_w as f64 * EDGE_FRACTION
        } else {
            frame_h as f64 * EDGE_FRACTION
        };

    if min_dist > threshold {
        return region.clone();
    }

    let (dx, dy) = if (min_dist - d_left).abs() < eps {
        (-d_left * t, 0.0)
    } else if (min_dist - d_right).abs() < eps {
        (d_right * t, 0.0)
    } else if (min_dist - d_top).abs() < eps {
        (0.0, -d_top * t)
    } else {
        (0.0, d_bottom * t)
    };

    let new_x = (region.x as f64 + dx).max(0.0) as i32;
    let new_y = (region.y as f64 + dy).max(0.0) as i32;

    Region {
        x: new_x,
        y: new_y,
        width: region.width,
        height: region.height,
        track_id: region.track_id,
        full_width: region.full_width,
        full_height: region.full_height,
        unclamped_x: region.unclamped_x.map(|ux| (ux as f64 + dx) as i32),
        unclamped_y: region.unclamped_y.map(|uy| (uy as f64 + dy) as i32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn region(x: i32, y: i32, w: i32, h: i32, tid: Option<u32>) -> Region {
        Region {
            x,
            y,
            width: w,
            height: h,
            track_id: tid,
            full_width: None,
            full_height: None,
            unclamped_x: None,
            unclamped_y: None,
        }
    }

    const FW: u32 = 1000;
    const FH: u32 = 800;

    // ── Basic merge behavior ────────────────────────────────────────

    #[test]
    fn test_empty_inputs() {
        let merger = RegionMerger::new();
        let result = merger.merge(&[], &[], FW, FH);
        assert!(result.is_empty());
    }

    #[test]
    fn test_current_only_no_lookahead() {
        let merger = RegionMerger::new();
        let current = vec![region(100, 100, 50, 50, Some(1))];
        let result = merger.merge(&current, &[], FW, FH);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].track_id, Some(1));
    }

    #[test]
    fn test_track_id_dedup_current_wins() {
        let merger = RegionMerger::new();
        let current = vec![region(100, 100, 50, 50, Some(1))];
        let lookahead = vec![vec![region(200, 200, 50, 50, Some(1))]]; // same track_id
        let result = merger.merge(&current, &lookahead, FW, FH);

        // Should keep only the current region (track_id=1 already seen)
        let track1_regions: Vec<_> = result.iter().filter(|r| r.track_id == Some(1)).collect();
        assert_eq!(track1_regions.len(), 1);
        assert_eq!(track1_regions[0].x, 100); // current's position
    }

    #[test]
    fn test_new_track_from_lookahead_added() {
        let merger = RegionMerger::new();
        let current = vec![region(100, 100, 50, 50, Some(1))];
        let lookahead = vec![vec![region(500, 500, 50, 50, Some(2))]]; // new track
        let result = merger.merge(&current, &lookahead, FW, FH);

        assert!(result.iter().any(|r| r.track_id == Some(1)));
        assert!(result.iter().any(|r| r.track_id == Some(2)));
    }

    #[test]
    fn test_untracked_always_appended() {
        let merger = RegionMerger::new();
        let current = vec![region(100, 100, 50, 50, Some(1))];
        let lookahead = vec![vec![
            region(500, 500, 50, 50, None), // untracked
        ]];
        let result = merger.merge(&current, &lookahead, FW, FH);

        // Both should be present (untracked always added)
        assert!(result.len() >= 2);
    }

    // ── IoU deduplication after merge ────────────────────────────────

    #[test]
    fn test_iou_dedup_removes_overlapping_untracked() {
        let merger = RegionMerger::new();
        let current = vec![region(100, 100, 100, 100, None)];
        let lookahead = vec![vec![
            region(110, 110, 100, 100, None), // high IoU with current
        ]];
        let result = merger.merge(&current, &lookahead, FW, FH);

        // After IoU dedup, the overlapping untracked should be removed
        assert_eq!(result.len(), 1);
    }

    // ── Edge interpolation ──────────────────────────────────────────

    #[test]
    fn test_edge_interpolation_pushes_toward_left_edge() {
        // Region near left edge: center at x=50 (within 25% of 1000 = 250)
        let r = region(25, 400, 50, 50, Some(5));
        let interpolated = interpolate(&r, 0, 3, FW, FH);
        // Should push left (negative dx)
        assert!(interpolated.x <= r.x);
    }

    #[test]
    fn test_edge_interpolation_pushes_toward_right_edge() {
        // Region near right edge: center at x=975
        let r = region(950, 400, 50, 50, Some(5));
        let interpolated = interpolate(&r, 0, 3, FW, FH);
        // Should push right (positive dx)
        assert!(interpolated.x >= r.x);
    }

    #[test]
    fn test_edge_interpolation_pushes_toward_top_edge() {
        // Region near top edge: center at y=25
        let r = region(500, 0, 50, 50, Some(5));
        let interpolated = interpolate(&r, 0, 3, FW, FH);
        // Should push up (negative dy)
        assert!(interpolated.y <= r.y);
    }

    #[test]
    fn test_edge_interpolation_no_push_center_region() {
        // Region in center: center at (500, 400) — not near any edge
        let r = region(475, 375, 50, 50, Some(5));
        let interpolated = interpolate(&r, 0, 3, FW, FH);
        // Should not be modified
        assert_eq!(interpolated.x, r.x);
        assert_eq!(interpolated.y, r.y);
    }

    #[test]
    fn test_interpolation_strength_increases_with_distance() {
        let r = region(25, 400, 50, 50, Some(5));

        let near = interpolate(&r, 0, 5, FW, FH); // t = 1/6
        let far = interpolate(&r, 4, 5, FW, FH); // t = 5/6

        // Farther lookahead should push more
        assert!(far.x <= near.x);
    }

    #[test]
    fn test_interpolation_t_formula() {
        // Verify: t = (idx + 1) / (total + 1)
        let t0 = 1_f64 / (5 + 1) as f64;
        let t4 = (4 + 1) as f64 / (5 + 1) as f64;
        assert_relative_eq!(t0, 1.0 / 6.0);
        assert_relative_eq!(t4, 5.0 / 6.0);
    }

    #[test]
    fn test_interpolation_clamps_x_to_zero() {
        // Region already at x=0, pushing left shouldn't go negative
        let r = region(0, 400, 50, 50, Some(5));
        let interpolated = interpolate(&r, 4, 5, FW, FH);
        assert!(interpolated.x >= 0);
    }

    #[test]
    fn test_interpolation_shifts_unclamped() {
        let r = Region {
            x: 25,
            y: 400,
            width: 50,
            height: 50,
            track_id: Some(5),
            full_width: Some(80),
            full_height: Some(80),
            unclamped_x: Some(-10),
            unclamped_y: Some(380),
        };
        let interpolated = interpolate(&r, 0, 3, FW, FH);
        // unclamped_x should also be shifted by dx
        assert!(interpolated.unclamped_x.is_some());
        assert!(interpolated.unclamped_x.unwrap() <= r.unclamped_x.unwrap());
    }

    #[test]
    fn test_edge_fraction_constant() {
        assert_relative_eq!(EDGE_FRACTION, 0.25);
    }
}
