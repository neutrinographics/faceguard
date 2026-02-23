use std::collections::HashSet;

use crate::shared::region::Region;

/// Filters detected regions by track ID inclusion/exclusion sets.
///
/// **Priority rule:** `blur_ids` takes absolute precedence over `exclude_ids`.
/// - If `blur_ids` is Some: keep only regions with `track_id` in the set.
/// - Else if `exclude_ids` is Some: keep regions with `track_id` NOT in the set.
/// - Else: keep all regions.
///
/// Regions with `track_id = None` are excluded when `blur_ids` is set,
/// and included when `exclude_ids` is set.
pub fn filter_regions(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn region_with_track(track_id: Option<u32>) -> Region {
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
    fn test_no_filters_returns_all() {
        let regions = vec![region_with_track(Some(1)), region_with_track(Some(2))];
        let result = filter_regions(&regions, None, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_blur_ids_keeps_only_matching() {
        let regions = vec![
            region_with_track(Some(1)),
            region_with_track(Some(2)),
            region_with_track(Some(3)),
        ];
        let blur_ids = HashSet::from([1, 3]);
        let result = filter_regions(&regions, Some(&blur_ids), None);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].track_id, Some(1));
        assert_eq!(result[1].track_id, Some(3));
    }

    #[test]
    fn test_exclude_ids_removes_matching() {
        let regions = vec![
            region_with_track(Some(1)),
            region_with_track(Some(2)),
            region_with_track(Some(3)),
        ];
        let exclude_ids = HashSet::from([2]);
        let result = filter_regions(&regions, None, Some(&exclude_ids));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].track_id, Some(1));
        assert_eq!(result[1].track_id, Some(3));
    }

    #[test]
    fn test_blur_ids_takes_precedence_over_exclude_ids() {
        let regions = vec![
            region_with_track(Some(1)),
            region_with_track(Some(2)),
            region_with_track(Some(3)),
        ];
        let blur_ids = HashSet::from([1]);
        let exclude_ids = HashSet::from([3]);
        let result = filter_regions(&regions, Some(&blur_ids), Some(&exclude_ids));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].track_id, Some(1));
    }

    #[test]
    fn test_blur_ids_excludes_none_track_id() {
        let regions = vec![region_with_track(None), region_with_track(Some(1))];
        let blur_ids = HashSet::from([1]);
        let result = filter_regions(&regions, Some(&blur_ids), None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].track_id, Some(1));
    }

    #[test]
    fn test_exclude_ids_includes_none_track_id() {
        let regions = vec![region_with_track(None), region_with_track(Some(1))];
        let exclude_ids = HashSet::from([1]);
        let result = filter_regions(&regions, None, Some(&exclude_ids));
        assert_eq!(result.len(), 1);
        assert!(result[0].track_id.is_none());
    }

    #[test]
    fn test_empty_regions() {
        let result = filter_regions(&[], None, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_empty_blur_ids_excludes_all() {
        let regions = vec![region_with_track(Some(1))];
        let blur_ids = HashSet::new();
        let result = filter_regions(&regions, Some(&blur_ids), None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_empty_exclude_ids_keeps_all() {
        let regions = vec![region_with_track(Some(1)), region_with_track(Some(2))];
        let exclude_ids = HashSet::new();
        let result = filter_regions(&regions, None, Some(&exclude_ids));
        assert_eq!(result.len(), 2);
    }
}
