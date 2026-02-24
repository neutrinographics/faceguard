use std::collections::HashSet;

use crate::shared::region::{Region, DEFAULT_IOU_THRESHOLD};

const EDGE_FRACTION: f64 = 0.25;

/// Merges current-frame regions with lookahead regions for smooth face entry.
///
/// Lookahead regions that haven't appeared yet are interpolated toward
/// the nearest frame edge, creating a slide-in animation instead of
/// an abrupt pop-in.
pub struct RegionMerger;

impl RegionMerger {
    pub fn new() -> Self {
        Self
    }

    pub fn merge(
        &self,
        current: &[Region],
        lookahead: &[&[Region]],
        frame_w: u32,
        frame_h: u32,
    ) -> Vec<Region> {
        let mut seen_ids: HashSet<u32> = current.iter().filter_map(|r| r.track_id).collect();
        let mut result: Vec<Region> = current.to_vec();
        let total = lookahead.len();

        for (idx, future) in lookahead.iter().enumerate() {
            for r in *future {
                match r.track_id {
                    Some(tid) if seen_ids.contains(&tid) => {}
                    Some(tid) => {
                        seen_ids.insert(tid);
                        let interpolated = if total > 0 {
                            interpolate_toward_edge(r, idx, total, frame_w, frame_h)
                        } else {
                            r.clone()
                        };
                        result.push(interpolated);
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
/// The interpolation strength increases with lookahead distance:
/// `t = (idx + 1) / (total + 1)`. Regions not near an edge are unchanged.
fn interpolate_toward_edge(
    region: &Region,
    idx: usize,
    total: usize,
    frame_w: u32,
    frame_h: u32,
) -> Region {
    let t = (idx + 1) as f64 / (total + 1) as f64;
    let (cx, cy) = region_center(region);

    let (dx, dy) = match nearest_edge_offset(cx, cy, frame_w, frame_h, t) {
        Some(offset) => offset,
        None => return region.clone(),
    };

    Region {
        x: (region.x as f64 + dx).max(0.0) as i32,
        y: (region.y as f64 + dy).max(0.0) as i32,
        width: region.width,
        height: region.height,
        track_id: region.track_id,
        full_width: region.full_width,
        full_height: region.full_height,
        unclamped_x: region.unclamped_x.map(|ux| (ux as f64 + dx) as i32),
        unclamped_y: region.unclamped_y.map(|uy| (uy as f64 + dy) as i32),
    }
}

fn region_center(region: &Region) -> (f64, f64) {
    (
        region.x as f64 + region.width as f64 / 2.0,
        region.y as f64 + region.height as f64 / 2.0,
    )
}

/// Compute the (dx, dy) offset to push toward the nearest edge.
///
/// Returns None if the region center isn't within EDGE_FRACTION of any edge.
fn nearest_edge_offset(
    cx: f64,
    cy: f64,
    frame_w: u32,
    frame_h: u32,
    t: f64,
) -> Option<(f64, f64)> {
    let eps = 1e-9;

    let d_left = cx;
    let d_right = frame_w as f64 - cx;
    let d_top = cy;
    let d_bottom = frame_h as f64 - cy;

    let min_dist = d_left.min(d_right).min(d_top).min(d_bottom);

    let threshold = if (min_dist - d_left).abs() < eps || (min_dist - d_right).abs() < eps {
        frame_w as f64 * EDGE_FRACTION
    } else {
        frame_h as f64 * EDGE_FRACTION
    };

    if min_dist > threshold {
        return None;
    }

    let offset = if (min_dist - d_left).abs() < eps {
        (-d_left * t, 0.0)
    } else if (min_dist - d_right).abs() < eps {
        (d_right * t, 0.0)
    } else if (min_dist - d_top).abs() < eps {
        (0.0, -d_top * t)
    } else {
        (0.0, d_bottom * t)
    };

    Some(offset)
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

    #[test]
    fn test_empty_inputs() {
        let merger = RegionMerger::new();
        assert!(merger.merge(&[], &[], FW, FH).is_empty());
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
        let la0 = vec![region(200, 200, 50, 50, Some(1))];
        let lookahead: Vec<&[Region]> = vec![&la0];
        let result = merger.merge(&current, &lookahead, FW, FH);

        let track1: Vec<_> = result.iter().filter(|r| r.track_id == Some(1)).collect();
        assert_eq!(track1.len(), 1);
        assert_eq!(track1[0].x, 100);
    }

    #[test]
    fn test_new_track_from_lookahead_added() {
        let merger = RegionMerger::new();
        let current = vec![region(100, 100, 50, 50, Some(1))];
        let la0 = vec![region(500, 500, 50, 50, Some(2))];
        let lookahead: Vec<&[Region]> = vec![&la0];
        let result = merger.merge(&current, &lookahead, FW, FH);

        assert!(result.iter().any(|r| r.track_id == Some(1)));
        assert!(result.iter().any(|r| r.track_id == Some(2)));
    }

    #[test]
    fn test_untracked_always_appended() {
        let merger = RegionMerger::new();
        let current = vec![region(100, 100, 50, 50, Some(1))];
        let la0 = vec![region(500, 500, 50, 50, None)];
        let lookahead: Vec<&[Region]> = vec![&la0];
        let result = merger.merge(&current, &lookahead, FW, FH);
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_iou_dedup_removes_overlapping_untracked() {
        let merger = RegionMerger::new();
        let current = vec![region(100, 100, 100, 100, None)];
        let la0 = vec![region(110, 110, 100, 100, None)];
        let lookahead: Vec<&[Region]> = vec![&la0];
        let result = merger.merge(&current, &lookahead, FW, FH);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_edge_interpolation_pushes_toward_left_edge() {
        let r = region(25, 400, 50, 50, Some(5));
        let interpolated = interpolate_toward_edge(&r, 0, 3, FW, FH);
        assert!(interpolated.x <= r.x);
    }

    #[test]
    fn test_edge_interpolation_pushes_toward_right_edge() {
        let r = region(950, 400, 50, 50, Some(5));
        let interpolated = interpolate_toward_edge(&r, 0, 3, FW, FH);
        assert!(interpolated.x >= r.x);
    }

    #[test]
    fn test_edge_interpolation_pushes_toward_top_edge() {
        let r = region(500, 0, 50, 50, Some(5));
        let interpolated = interpolate_toward_edge(&r, 0, 3, FW, FH);
        assert!(interpolated.y <= r.y);
    }

    #[test]
    fn test_edge_interpolation_no_push_center_region() {
        let r = region(475, 375, 50, 50, Some(5));
        let interpolated = interpolate_toward_edge(&r, 0, 3, FW, FH);
        assert_eq!(interpolated.x, r.x);
        assert_eq!(interpolated.y, r.y);
    }

    #[test]
    fn test_interpolation_strength_increases_with_distance() {
        let r = region(25, 400, 50, 50, Some(5));
        let near = interpolate_toward_edge(&r, 0, 5, FW, FH);
        let far = interpolate_toward_edge(&r, 4, 5, FW, FH);
        assert!(far.x <= near.x);
    }

    #[test]
    fn test_interpolation_t_formula() {
        let t0 = 1_f64 / (5 + 1) as f64;
        let t4 = (4 + 1) as f64 / (5 + 1) as f64;
        assert_relative_eq!(t0, 1.0 / 6.0);
        assert_relative_eq!(t4, 5.0 / 6.0);
    }

    #[test]
    fn test_interpolation_clamps_x_to_zero() {
        let r = region(0, 400, 50, 50, Some(5));
        let interpolated = interpolate_toward_edge(&r, 4, 5, FW, FH);
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
        let interpolated = interpolate_toward_edge(&r, 0, 3, FW, FH);
        assert!(interpolated.unclamped_x.is_some());
        assert!(interpolated.unclamped_x.unwrap() <= r.unclamped_x.unwrap());
    }

    #[test]
    fn test_edge_fraction_constant() {
        assert_relative_eq!(EDGE_FRACTION, 0.25);
    }
}
