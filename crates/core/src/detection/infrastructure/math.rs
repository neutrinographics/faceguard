//! Shared math utilities for detection infrastructure.
//!
//! Provides union-find clustering and bounding-box IoU computation
//! used across multiple detection backends.

/// IoU between two bounding boxes represented as `[x1, y1, x2, y2]`.
pub fn bbox_iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    if inter == 0.0 {
        return 0.0;
    }

    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter)
}

/// Find root of element `i` with path halving for amortized near-O(1).
pub fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Merge the sets containing `a` and `b`.
pub fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

/// Collect union-find clusters into sorted groups of associated IDs.
///
/// Takes a union-find `parent` array and an iterator of `(index, id)` pairs.
/// Returns groups sorted by their smallest ID for deterministic output.
pub fn collect_groups(parent: &mut [usize], entries: &[(usize, u32)]) -> Vec<Vec<u32>> {
    let mut groups: std::collections::HashMap<usize, Vec<u32>> =
        std::collections::HashMap::new();
    for &(idx, id) in entries {
        let root = find(parent, idx);
        groups.entry(root).or_default().push(id);
    }

    let mut result: Vec<Vec<u32>> = groups.into_values().collect();
    for g in &mut result {
        g.sort();
    }
    result.sort_by_key(|g| g[0]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_iou_no_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&a, &b), 0.0);
    }

    #[test]
    fn test_bbox_iou_perfect_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert!((bbox_iou(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_iou_partial_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        let expected = 25.0 / 175.0;
        assert!((bbox_iou(&a, &b) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_union_find_transitive() {
        let mut parent = vec![0, 1, 2];
        union(&mut parent, 0, 1);
        union(&mut parent, 1, 2);
        assert_eq!(find(&mut parent, 0), find(&mut parent, 2));
    }

    #[test]
    fn test_union_find_separate() {
        let mut parent = vec![0, 1, 2, 3];
        union(&mut parent, 0, 1);
        union(&mut parent, 2, 3);
        assert_ne!(find(&mut parent, 0), find(&mut parent, 2));
    }

    #[test]
    fn test_collect_groups_basic() {
        let mut parent = vec![0, 1, 2, 3];
        union(&mut parent, 0, 1);
        union(&mut parent, 2, 3);
        let entries: Vec<(usize, u32)> = vec![(0, 10), (1, 20), (2, 30), (3, 40)];
        let groups = collect_groups(&mut parent, &entries);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], vec![10, 20]);
        assert_eq!(groups[1], vec![30, 40]);
    }
}
