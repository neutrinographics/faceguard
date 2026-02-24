/// Simplified ByteTrack multi-object tracker.
///
/// Two-stage association strategy: high-confidence detections are matched
/// first, then low-confidence detections fill remaining unmatched tracks.
/// This prevents spurious tracks from weak detections while allowing
/// existing tracks to survive momentary confidence drops.
use std::collections::HashSet;

use super::math::bbox_iou;

#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: [f64; 4],
    pub score: f64,
}

#[derive(Clone, Debug)]
pub struct Track {
    pub id: u32,
    pub bbox: [f64; 4],
    pub det_index: Option<usize>,
}

const HIGH_THRESH: f64 = 0.5;
const MATCH_THRESH: f64 = 0.3;

#[derive(Clone, Debug)]
struct TrackState {
    id: u32,
    bbox: [f64; 4],
    frames_lost: usize,
    matched: bool,
    det_index: Option<usize>,
}

pub struct ByteTracker {
    tracks: Vec<TrackState>,
    next_id: u32,
    max_lost: usize,
}

impl ByteTracker {
    pub fn new(max_lost: usize) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_lost,
        }
    }

    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        let (high, low) = split_by_confidence(detections);

        self.reset_match_flags();
        let num_existing = self.tracks.len();
        let matched_high = self.match_high_confidence(&high, detections);
        self.match_low_confidence(&low, detections);
        self.create_new_tracks(&high, &matched_high, detections);
        self.age_unmatched_tracks(num_existing);

        self.active_tracks()
    }

    fn reset_match_flags(&mut self) {
        for track in &mut self.tracks {
            track.matched = false;
            track.det_index = None;
        }
    }

    fn match_high_confidence(
        &mut self,
        high: &[(usize, &Detection)],
        detections: &[Detection],
    ) -> HashSet<usize> {
        let track_refs: Vec<(usize, [f64; 4])> = self
            .tracks
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.bbox))
            .collect();

        let mut matched_det_indices = HashSet::new();
        for (ti, di) in greedy_match(&track_refs, high, MATCH_THRESH) {
            self.apply_match(ti, di, &detections[di].bbox);
            matched_det_indices.insert(di);
        }
        matched_det_indices
    }

    fn match_low_confidence(&mut self, low: &[(usize, &Detection)], detections: &[Detection]) {
        let unmatched_refs: Vec<(usize, [f64; 4])> = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.matched)
            .map(|(i, t)| (i, t.bbox))
            .collect();

        for (ti, di) in greedy_match(&unmatched_refs, low, MATCH_THRESH) {
            self.apply_match(ti, di, &detections[di].bbox);
        }
    }

    fn apply_match(&mut self, track_idx: usize, det_idx: usize, bbox: &[f64; 4]) {
        self.tracks[track_idx].bbox = *bbox;
        self.tracks[track_idx].frames_lost = 0;
        self.tracks[track_idx].matched = true;
        self.tracks[track_idx].det_index = Some(det_idx);
    }

    fn create_new_tracks(
        &mut self,
        high: &[(usize, &Detection)],
        matched: &HashSet<usize>,
        detections: &[Detection],
    ) {
        for (di, _) in high {
            if !matched.contains(di) {
                self.tracks.push(TrackState {
                    id: self.next_id,
                    bbox: detections[*di].bbox,
                    frames_lost: 0,
                    matched: true,
                    det_index: Some(*di),
                });
                self.next_id += 1;
            }
        }
    }

    fn age_unmatched_tracks(&mut self, num_existing: usize) {
        for track in self.tracks.iter_mut().take(num_existing) {
            if !track.matched {
                track.frames_lost += 1;
            }
        }
        let max_lost = self.max_lost;
        self.tracks.retain(|t| t.frames_lost <= max_lost);
    }

    /// Only matched tracks produce output; lost tracks are kept internally
    /// for re-identification but should not generate blur regions.
    fn active_tracks(&self) -> Vec<Track> {
        self.tracks
            .iter()
            .filter(|t| t.matched)
            .map(|t| Track {
                id: t.id,
                bbox: t.bbox,
                det_index: t.det_index,
            })
            .collect()
    }
}

type IndexedDets<'a> = Vec<(usize, &'a Detection)>;

fn split_by_confidence(detections: &[Detection]) -> (IndexedDets<'_>, IndexedDets<'_>) {
    let mut high = Vec::new();
    let mut low = Vec::new();
    for (i, det) in detections.iter().enumerate() {
        if det.score >= HIGH_THRESH {
            high.push((i, det));
        } else {
            low.push((i, det));
        }
    }
    (high, low)
}

/// Greedy IoU matching: pairs sorted by descending IoU, each track/detection
/// used at most once.
fn greedy_match(
    tracks: &[(usize, [f64; 4])],
    dets: &[(usize, &Detection)],
    thresh: f64,
) -> Vec<(usize, usize)> {
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for (ti, bbox) in tracks {
        for (di, det) in dets {
            let score = bbox_iou(bbox, &det.bbox);
            if score >= thresh {
                pairs.push((*ti, *di, score));
            }
        }
    }
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_tracks = HashSet::new();
    let mut used_dets = HashSet::new();
    let mut matches = Vec::new();

    for (ti, di, _) in &pairs {
        if !used_tracks.contains(ti) && !used_dets.contains(di) {
            used_tracks.insert(*ti);
            used_dets.insert(*di);
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
        let id = t1[0].id;

        let t2 = tracker.update(&[det(12.0, 12.0, 62.0, 62.0, 0.9)]);
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].id, id);
    }

    #[test]
    fn test_lost_track_removal() {
        let mut tracker = ByteTracker::new(2);
        tracker.update(&[det(10.0, 10.0, 60.0, 60.0, 0.9)]);

        tracker.update(&[]);
        tracker.update(&[]);
        assert!(tracker.update(&[]).is_empty());
    }

    #[test]
    fn test_track_survives_within_max_lost() {
        let mut tracker = ByteTracker::new(3);
        let t1 = tracker.update(&[det(10.0, 10.0, 60.0, 60.0, 0.9)]);
        let id = t1[0].id;

        tracker.update(&[]);
        tracker.update(&[]);

        let t2 = tracker.update(&[det(12.0, 12.0, 62.0, 62.0, 0.9)]);
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].id, id);
    }

    #[test]
    fn test_empty_frame() {
        let mut tracker = ByteTracker::new(5);
        assert!(tracker.update(&[]).is_empty());
    }

    #[test]
    fn test_low_confidence_matches_existing_track() {
        let mut tracker = ByteTracker::new(5);
        let t1 = tracker.update(&[det(10.0, 10.0, 60.0, 60.0, 0.9)]);
        let id = t1[0].id;

        let t2 = tracker.update(&[det(12.0, 12.0, 62.0, 62.0, 0.3)]);
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].id, id);
    }

    #[test]
    fn test_low_confidence_does_not_start_new_track() {
        let mut tracker = ByteTracker::new(5);
        assert!(tracker
            .update(&[det(10.0, 10.0, 60.0, 60.0, 0.3)])
            .is_empty());
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
        assert_eq!(bbox_iou(&a, &b), 0.0);
    }

    #[test]
    fn test_iou_bbox_perfect_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert!((bbox_iou(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_bbox_partial_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        let expected = 25.0 / 175.0;
        assert!((bbox_iou(&a, &b) - expected).abs() < 1e-9);
    }
}
