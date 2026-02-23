/// Simplified ByteTrack multi-object tracker.
///
/// Maintains persistent track IDs across frames using IoU-based greedy
/// matching with a two-stage association strategy (high-confidence first,
/// then low-confidence for unmatched tracks).
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single detection to be fed into the tracker.
#[derive(Clone, Debug)]
pub struct Detection {
    /// Bounding box as `[x1, y1, x2, y2]` in absolute pixel coordinates.
    pub bbox: [f64; 4],
    /// Detection confidence score.
    pub score: f64,
}

/// A tracked object with a persistent ID.
#[derive(Clone, Debug)]
pub struct Track {
    /// Persistent track identifier.
    pub id: u32,
    /// Current bounding box `[x1, y1, x2, y2]`.
    pub bbox: [f64; 4],
}

/// Confidence threshold separating high and low detections.
const HIGH_THRESH: f64 = 0.5;

/// IoU threshold for matching detections to tracks.
const MATCH_THRESH: f64 = 0.3;

// ---------------------------------------------------------------------------
// Internal track state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TrackState {
    id: u32,
    bbox: [f64; 4],
    frames_lost: usize,
}

// ---------------------------------------------------------------------------
// ByteTracker
// ---------------------------------------------------------------------------

/// Pure-Rust ByteTrack implementation.
///
/// Tracks objects across video frames by matching incoming detections to
/// existing tracks using IoU. Two-stage matching: high-confidence detections
/// first, then low-confidence detections for any remaining unmatched tracks.
pub struct ByteTracker {
    tracks: Vec<TrackState>,
    next_id: u32,
    max_lost: usize,
}

impl ByteTracker {
    /// Create a new tracker.
    ///
    /// `max_lost` is the maximum number of frames a track can be "lost"
    /// (unmatched) before it is removed.
    pub fn new(max_lost: usize) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_lost,
        }
    }

    /// Process a new frame of detections and return active tracks.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        // Split detections into high and low confidence
        let mut high: Vec<(usize, &Detection)> = Vec::new();
        let mut low: Vec<(usize, &Detection)> = Vec::new();
        for (i, det) in detections.iter().enumerate() {
            if det.score >= HIGH_THRESH {
                high.push((i, det));
            } else {
                low.push((i, det));
            }
        }

        let mut matched_track_indices: Vec<bool> = vec![false; self.tracks.len()];
        let mut matched_det_indices: Vec<bool> = vec![false; detections.len()];

        // Stage 1: match high-confidence detections to tracks
        let matches_high = greedy_match(&self.tracks, &high, MATCH_THRESH);
        for (ti, di) in &matches_high {
            self.tracks[*ti].bbox = detections[*di].bbox;
            self.tracks[*ti].frames_lost = 0;
            matched_track_indices[*ti] = true;
            matched_det_indices[*di] = true;
        }

        // Stage 2: match low-confidence detections to remaining tracks
        let unmatched_tracks: Vec<(usize, &TrackState)> = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(i, _)| !matched_track_indices[*i])
            .collect();
        let matches_low = greedy_match_refs(&unmatched_tracks, &low, MATCH_THRESH);
        for (ti, di) in &matches_low {
            self.tracks[*ti].bbox = detections[*di].bbox;
            self.tracks[*ti].frames_lost = 0;
            matched_track_indices[*ti] = true;
            matched_det_indices[*di] = true;
        }

        // Start new tracks for unmatched high-confidence detections
        for (di, _det) in &high {
            if !matched_det_indices[*di] {
                self.tracks.push(TrackState {
                    id: self.next_id,
                    bbox: detections[*di].bbox,
                    frames_lost: 0,
                });
                self.next_id += 1;
                // Grow matched arrays to keep indices valid
                matched_track_indices.push(true);
            }
        }

        // Increment lost counter for unmatched tracks, remove stale ones
        for (i, matched) in matched_track_indices
            .iter()
            .enumerate()
            .take(self.tracks.len())
        {
            if !matched {
                self.tracks[i].frames_lost += 1;
            }
        }
        let max_lost = self.max_lost;
        self.tracks.retain(|t| t.frames_lost <= max_lost);

        // Return all active tracks
        self.tracks
            .iter()
            .map(|t| Track {
                id: t.id,
                bbox: t.bbox,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// IoU + greedy matching
// ---------------------------------------------------------------------------

fn iou_bbox(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter = inter_w * inter_h;

    if inter == 0.0 {
        return 0.0;
    }

    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter)
}

/// Greedy IoU matching between existing tracks and a subset of detections.
/// Returns `Vec<(track_index, detection_original_index)>`.
fn greedy_match(
    tracks: &[TrackState],
    dets: &[(usize, &Detection)],
    thresh: f64,
) -> Vec<(usize, usize)> {
    // Build IoU matrix and sort by descending IoU
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for (ti, track) in tracks.iter().enumerate() {
        for (di, det) in dets {
            let score = iou_bbox(&track.bbox, &det.bbox);
            if score >= thresh {
                pairs.push((ti, *di, score));
            }
        }
    }
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_tracks = vec![false; tracks.len()];
    // detection indices can be sparse, track by original index
    let mut used_dets: HashMap<usize, bool> = HashMap::new();
    let mut matches = Vec::new();

    for (ti, di, _) in &pairs {
        if !used_tracks[*ti] && !used_dets.contains_key(di) {
            used_tracks[*ti] = true;
            used_dets.insert(*di, true);
            matches.push((*ti, *di));
        }
    }
    matches
}

/// Same as `greedy_match` but takes pre-filtered (index, &TrackState) refs.
fn greedy_match_refs(
    tracks: &[(usize, &TrackState)],
    dets: &[(usize, &Detection)],
    thresh: f64,
) -> Vec<(usize, usize)> {
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for (ti, track) in tracks {
        for (di, det) in dets {
            let score = iou_bbox(&track.bbox, &det.bbox);
            if score >= thresh {
                pairs.push((*ti, *di, score));
            }
        }
    }
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_tracks: HashMap<usize, bool> = HashMap::new();
    let mut used_dets: HashMap<usize, bool> = HashMap::new();
    let mut matches = Vec::new();

    for (ti, di, _) in &pairs {
        if !used_tracks.contains_key(ti) && !used_dets.contains_key(di) {
            used_tracks.insert(*ti, true);
            used_dets.insert(*di, true);
            matches.push((*ti, *di));
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f64, y1: f64, x2: f64, y2: f64, score: f64) -> Detection {
        Detection {
            bbox: [x1, y1, x2, y2],
            score,
        }
    }

    #[test]
    fn test_new_detections_get_unique_ids() {
        let mut tracker = ByteTracker::new(5);
        let tracks = tracker.update(&[
            det(0.0, 0.0, 50.0, 50.0, 0.9),
            det(100.0, 100.0, 150.0, 150.0, 0.8),
        ]);
        assert_eq!(tracks.len(), 2);
        assert_ne!(tracks[0].id, tracks[1].id);
    }

    #[test]
    fn test_consistent_id_across_frames() {
        let mut tracker = ByteTracker::new(5);
        let t1 = tracker.update(&[det(10.0, 10.0, 60.0, 60.0, 0.9)]);
        assert_eq!(t1.len(), 1);
        let id = t1[0].id;

        // Same position, should keep same ID
        let t2 = tracker.update(&[det(12.0, 12.0, 62.0, 62.0, 0.9)]);
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].id, id);
    }

    #[test]
    fn test_lost_track_removal() {
        let mut tracker = ByteTracker::new(2);
        tracker.update(&[det(10.0, 10.0, 60.0, 60.0, 0.9)]);

        // 3 empty frames → track should be removed (max_lost=2)
        tracker.update(&[]);
        tracker.update(&[]);
        let t = tracker.update(&[]);
        assert!(t.is_empty());
    }

    #[test]
    fn test_track_survives_within_max_lost() {
        let mut tracker = ByteTracker::new(3);
        let t1 = tracker.update(&[det(10.0, 10.0, 60.0, 60.0, 0.9)]);
        let id = t1[0].id;

        // 2 empty frames (within max_lost=3)
        tracker.update(&[]);
        tracker.update(&[]);

        // Re-appear at same position
        let t2 = tracker.update(&[det(12.0, 12.0, 62.0, 62.0, 0.9)]);
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].id, id);
    }

    #[test]
    fn test_empty_frame() {
        let mut tracker = ByteTracker::new(5);
        let t = tracker.update(&[]);
        assert!(t.is_empty());
    }

    #[test]
    fn test_low_confidence_matches_existing_track() {
        let mut tracker = ByteTracker::new(5);
        // First frame: high-confidence detection starts a track
        let t1 = tracker.update(&[det(10.0, 10.0, 60.0, 60.0, 0.9)]);
        let id = t1[0].id;

        // Second frame: low-confidence detection at same position
        // Should match via stage 2 (low confidence → unmatched tracks)
        let t2 = tracker.update(&[det(12.0, 12.0, 62.0, 62.0, 0.3)]);
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].id, id);
    }

    #[test]
    fn test_low_confidence_does_not_start_new_track() {
        let mut tracker = ByteTracker::new(5);
        // Only low-confidence detections should NOT start new tracks
        let t = tracker.update(&[det(10.0, 10.0, 60.0, 60.0, 0.3)]);
        assert!(t.is_empty());
    }

    #[test]
    fn test_multiple_tracks_independent() {
        let mut tracker = ByteTracker::new(5);
        let t1 = tracker.update(&[
            det(0.0, 0.0, 50.0, 50.0, 0.9),
            det(200.0, 200.0, 250.0, 250.0, 0.9),
        ]);
        assert_eq!(t1.len(), 2);
        let id_a = t1[0].id;
        let id_b = t1[1].id;

        // Move both slightly
        let t2 = tracker.update(&[
            det(2.0, 2.0, 52.0, 52.0, 0.9),
            det(202.0, 202.0, 252.0, 252.0, 0.9),
        ]);
        assert_eq!(t2.len(), 2);

        let ids: Vec<u32> = t2.iter().map(|t| t.id).collect();
        assert!(ids.contains(&id_a));
        assert!(ids.contains(&id_b));
    }

    #[test]
    fn test_iou_bbox_no_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(iou_bbox(&a, &b), 0.0);
    }

    #[test]
    fn test_iou_bbox_perfect_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert!((iou_bbox(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_bbox_partial_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        // Intersection: 5x5=25, Union: 100+100-25=175
        let expected = 25.0 / 175.0;
        assert!((iou_bbox(&a, &b) - expected).abs() < 1e-9);
    }
}
