use crate::shared::region::Region;

use super::face_landmarks::FaceLandmarks;
use super::region_smoother::{RegionSmootherInterface, SmoothParams};

/// Default padding factor applied symmetrically around the face.
pub const DEFAULT_PADDING: f64 = 0.4;

/// Minimum width ratio: effective_w >= box_h * MIN_WIDTH_RATIO.
const MIN_WIDTH_RATIO: f64 = 0.8;

/// Bounding box as (x1, y1, x2, y2).
pub type BBox = (f64, f64, f64, f64);

/// Default center offset: no shift.
pub const DEFAULT_CENTER_OFFSET: f64 = 0.0;

/// Converts detection boxes + optional landmarks into blur regions.
///
/// Handles profile-aware sizing, center blending, padding, and
/// minimum width constraints.
pub struct FaceRegionBuilder {
    padding: f64,
    center_offset: f64,
    smoother: Option<Box<dyn RegionSmootherInterface>>,
}

impl FaceRegionBuilder {
    pub fn new(
        padding: f64,
        center_offset: f64,
        smoother: Option<Box<dyn RegionSmootherInterface>>,
    ) -> Self {
        Self {
            padding,
            center_offset,
            smoother,
        }
    }

    pub fn build(
        &mut self,
        bbox: BBox,
        frame_w: u32,
        frame_h: u32,
        landmarks: Option<&FaceLandmarks>,
        track_id: Option<u32>,
    ) -> Region {
        let profile_ratio = match landmarks {
            Some(lm) if lm.has_visible() => lm.profile_ratio(),
            _ => 0.0,
        };

        let (half_w, half_h) = self.compute_half_size(bbox, profile_ratio);
        let (cx, cy) = self.compute_center(bbox, landmarks, profile_ratio, half_w);

        let mut params: SmoothParams = [cx, cy, half_w, half_h];
        if let Some(ref mut smoother) = self.smoother {
            params = smoother.smooth(params, track_id);
        }

        self.params_to_region(params, frame_w, frame_h, track_id)
    }

    fn compute_center(
        &self,
        bbox: BBox,
        landmarks: Option<&FaceLandmarks>,
        profile_ratio: f64,
        half_w: f64,
    ) -> (f64, f64) {
        let box_cx = (bbox.0 + bbox.2) / 2.0;
        let box_cy = (bbox.1 + bbox.3) / 2.0;

        match landmarks {
            Some(lm) if lm.has_visible() => {
                let (face_cx, face_cy) = lm.center().unwrap_or((box_cx, box_cy));
                let mut cx = face_cx + (box_cx - face_cx) * profile_ratio;
                let cy = face_cy + (box_cy - face_cy) * profile_ratio;

                if self.center_offset != 0.0 {
                    let direction = lm.back_of_head_direction();
                    cx += self.center_offset * half_w * direction;
                }

                (cx, cy)
            }
            _ => (box_cx, box_cy),
        }
    }

    fn compute_half_size(&self, bbox: BBox, profile_ratio: f64) -> (f64, f64) {
        let box_w = bbox.2 - bbox.0;
        let box_h = bbox.3 - bbox.1;

        let effective_w = (box_w + (box_h - box_w) * profile_ratio).max(box_h * MIN_WIDTH_RATIO);

        let half_w = effective_w * (1.0 + self.padding) / 2.0;
        let half_h = box_h * (1.0 + self.padding) / 2.0;
        (half_w, half_h)
    }

    fn params_to_region(
        &self,
        params: SmoothParams,
        frame_w: u32,
        frame_h: u32,
        track_id: Option<u32>,
    ) -> Region {
        let [cx, cy, half_w, half_h] = params;

        let ux = (cx - half_w) as i32;
        let uy = (cy - half_h) as i32;
        let full_w = (half_w * 2.0) as i32;
        let full_h = (half_h * 2.0) as i32;

        let x = ux.max(0);
        let y = uy.max(0);
        let w = ((cx + half_w).min(frame_w as f64) as i32 - x).max(0);
        let h = ((cy + half_h).min(frame_h as f64) as i32 - y).max(0);

        Region {
            x,
            y,
            width: w,
            height: h,
            track_id,
            full_width: Some(full_w),
            full_height: Some(full_h),
            unclamped_x: Some(ux),
            unclamped_y: Some(uy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    const FRAME_W: u32 = 1000;
    const FRAME_H: u32 = 1000;
    const PADDING: f64 = 0.4;

    fn builder() -> FaceRegionBuilder {
        FaceRegionBuilder::new(PADDING, 0.0, None)
    }

    fn frontal_box() -> BBox {
        (400.0, 300.0, 600.0, 500.0)
    }

    fn frontal_landmarks() -> FaceLandmarks {
        FaceLandmarks::new([
            (440.0, 350.0),
            (560.0, 350.0),
            (500.0, 420.0),
            (460.0, 470.0),
            (540.0, 470.0),
        ])
    }

    fn region_contains_point(r: &Region, px: f64, py: f64) -> bool {
        r.x as f64 <= px
            && px <= (r.x + r.width) as f64
            && r.y as f64 <= py
            && py <= (r.y + r.height) as f64
    }

    // ── Frontal face ────────────────────────────────────────────────

    #[test]
    fn test_frontal_no_landmarks() {
        let mut b = builder();
        let r = b.build(frontal_box(), FRAME_W, FRAME_H, None, None);
        // Box center is (500, 400), box_w=200, box_h=200
        // half_w = 200 * 1.4 / 2 = 140, half_h = 200 * 1.4 / 2 = 140
        assert!(r.width > 0);
        assert!(r.height > 0);
        assert!(region_contains_point(&r, 500.0, 400.0));
    }

    #[test]
    fn test_frontal_with_landmarks_centers_on_face() {
        let mut b = builder();
        let lm = frontal_landmarks();
        let r = b.build(frontal_box(), FRAME_W, FRAME_H, Some(&lm), None);
        // Frontal face: profile_ratio ≈ 0, so center ≈ landmark center = (500, 400)
        let center_x = r.x as f64 + r.width as f64 / 2.0;
        let center_y = r.y as f64 + r.height as f64 / 2.0;
        assert_relative_eq!(center_x, 500.0, epsilon = 5.0);
        assert_relative_eq!(center_y, 400.0, epsilon = 5.0);
    }

    #[test]
    fn test_frontal_region_covers_box_and_landmarks() {
        let mut b = builder();
        let lm = frontal_landmarks();
        let r = b.build(frontal_box(), FRAME_W, FRAME_H, Some(&lm), None);
        // Should contain all landmark points
        for (x, y) in lm.points().iter() {
            assert!(
                region_contains_point(&r, *x, *y),
                "Region should contain landmark ({}, {})",
                x,
                y
            );
        }
        // Should contain box corners
        let bbox = frontal_box();
        assert!(region_contains_point(&r, bbox.0, bbox.1));
        assert!(region_contains_point(&r, bbox.2, bbox.3));
    }

    // ── Profile face ────────────────────────────────────────────────

    #[test]
    fn test_profile_blends_center_toward_box() {
        let mut b = builder();
        // Left profile: nose shifted way left
        let lm = FaceLandmarks::new([
            (120.0, 350.0),
            (180.0, 350.0),
            (80.0, 420.0),
            (100.0, 470.0),
            (160.0, 470.0),
        ]);
        let bbox: BBox = (100.0, 300.0, 200.0, 500.0);
        let r = b.build(bbox, FRAME_W, FRAME_H, Some(&lm), None);

        // Profile ratio is high, so center should be closer to box center (150)
        // than to landmark center
        let center_x = r.x as f64 + r.width as f64 / 2.0;
        let box_cx = 150.0;
        let face_cx = lm.center().unwrap().0;
        // Center should be between face_cx and box_cx, biased toward box
        assert!(center_x > face_cx || (center_x - face_cx).abs() < 20.0);
        assert!((center_x - box_cx).abs() < (face_cx - box_cx).abs() + 5.0);
    }

    #[test]
    fn test_profile_wider_than_frontal() {
        let mut b = builder();
        let bbox: BBox = (100.0, 300.0, 250.0, 500.0); // w=150, h=200

        // Frontal: uses box_w = 150
        let r_frontal = b.build(bbox, FRAME_W, FRAME_H, None, None);

        // Profile: blends toward box_h = 200
        let lm = FaceLandmarks::new([
            (120.0, 350.0),
            (180.0, 350.0),
            (100.0, 420.0),
            (130.0, 470.0),
            (170.0, 470.0),
        ]);
        let r_profile = b.build(bbox, FRAME_W, FRAME_H, Some(&lm), None);

        // Profile region should be at least as wide (effective_w grows toward box_h)
        assert!(
            r_profile.full_width.unwrap() >= r_frontal.full_width.unwrap(),
            "Profile full_width {} should be >= frontal full_width {}",
            r_profile.full_width.unwrap(),
            r_frontal.full_width.unwrap()
        );
    }

    // ── Minimum width constraint ────────────────────────────────────

    #[test]
    fn test_narrow_box_enforces_min_width() {
        let mut b = builder();
        // Very narrow box: w=20, h=200 → min effective_w = 200 * 0.8 = 160
        let bbox: BBox = (490.0, 300.0, 510.0, 500.0);
        let r = b.build(bbox, FRAME_W, FRAME_H, None, None);
        let expected_min_half_w = 200.0 * MIN_WIDTH_RATIO * (1.0 + PADDING) / 2.0;
        assert!(
            r.full_width.unwrap() as f64 >= expected_min_half_w * 2.0 - 2.0,
            "full_width {} should be >= {} (min width constraint)",
            r.full_width.unwrap(),
            expected_min_half_w * 2.0
        );
    }

    // ── Edge clamping ───────────────────────────────────────────────

    #[test]
    fn test_face_at_left_edge_clamps() {
        let mut b = builder();
        let bbox: BBox = (0.0, 300.0, 100.0, 500.0);
        let r = b.build(bbox, FRAME_W, FRAME_H, None, None);
        assert_eq!(r.x, 0);
        assert!(r.unclamped_x.unwrap() < 0, "unclamped_x should be negative");
        assert!(r.width > 0);
    }

    #[test]
    fn test_face_at_top_edge_clamps() {
        let mut b = builder();
        let bbox: BBox = (400.0, 0.0, 600.0, 100.0);
        let r = b.build(bbox, FRAME_W, FRAME_H, None, None);
        assert_eq!(r.y, 0);
        assert!(r.unclamped_y.unwrap() < 0);
        assert!(r.height > 0);
    }

    #[test]
    fn test_face_at_right_edge_clamps() {
        let mut b = builder();
        let bbox: BBox = (900.0, 300.0, 1000.0, 500.0);
        let r = b.build(bbox, FRAME_W, FRAME_H, None, None);
        assert!(r.x + r.width <= FRAME_W as i32);
    }

    #[test]
    fn test_face_at_bottom_edge_clamps() {
        let mut b = builder();
        let bbox: BBox = (400.0, 900.0, 600.0, 1000.0);
        let r = b.build(bbox, FRAME_W, FRAME_H, None, None);
        assert!(r.y + r.height <= FRAME_H as i32);
    }

    // ── Track ID passthrough ────────────────────────────────────────

    #[test]
    fn test_track_id_preserved() {
        let mut b = builder();
        let r = b.build(frontal_box(), FRAME_W, FRAME_H, None, Some(42));
        assert_eq!(r.track_id, Some(42));
    }

    #[test]
    fn test_no_track_id() {
        let mut b = builder();
        let r = b.build(frontal_box(), FRAME_W, FRAME_H, None, None);
        assert_eq!(r.track_id, None);
    }

    // ── Parametrized sizes ──────────────────────────────────────────

    #[rstest]
    #[case::small_face((900.0, 800.0, 950.0, 850.0))]
    #[case::large_face((100.0, 100.0, 700.0, 700.0))]
    fn test_region_has_positive_dimensions(#[case] bbox: BBox) {
        let mut b = builder();
        let r = b.build(bbox, FRAME_W, FRAME_H, None, None);
        assert!(r.width > 0, "width should be positive");
        assert!(r.height > 0, "height should be positive");
    }

    // ── Unclamped geometry stored ────────────────────────────────────

    #[test]
    fn test_unclamped_geometry_always_present() {
        let mut b = builder();
        let r = b.build(frontal_box(), FRAME_W, FRAME_H, None, None);
        assert!(r.full_width.is_some());
        assert!(r.full_height.is_some());
        assert!(r.unclamped_x.is_some());
        assert!(r.unclamped_y.is_some());
    }

    // ── Center offset ─────────────────────────────────────────────

    #[test]
    fn test_center_offset_zero_no_shift() {
        let mut b = FaceRegionBuilder::new(PADDING, 0.0, None);
        let lm = frontal_landmarks();
        let r = b.build(frontal_box(), FRAME_W, FRAME_H, Some(&lm), None);
        let center_x = r.x as f64 + r.width as f64 / 2.0;
        assert_relative_eq!(center_x, 500.0, epsilon = 5.0);
    }

    #[test]
    fn test_center_offset_shifts_left_profile_rightward() {
        // Left profile: nose is left of eye midpoint → back of head is right
        let lm = FaceLandmarks::new([
            (120.0, 350.0),
            (180.0, 350.0),
            (80.0, 420.0),
            (100.0, 470.0),
            (160.0, 470.0),
        ]);
        let bbox: BBox = (50.0, 300.0, 250.0, 500.0);

        let mut b_no_offset = FaceRegionBuilder::new(PADDING, 0.0, None);
        let r_no = b_no_offset.build(bbox, FRAME_W, FRAME_H, Some(&lm), None);
        let cx_no = r_no.unclamped_x.unwrap() as f64 + r_no.full_width.unwrap() as f64 / 2.0;

        let mut b_offset = FaceRegionBuilder::new(PADDING, 0.3, None);
        let r_off = b_offset.build(bbox, FRAME_W, FRAME_H, Some(&lm), None);
        let cx_off = r_off.unclamped_x.unwrap() as f64 + r_off.full_width.unwrap() as f64 / 2.0;

        // Positive offset + left profile → center shifts right (larger cx)
        assert!(
            cx_off > cx_no,
            "Expected offset center {cx_off} > no-offset center {cx_no}"
        );
    }

    #[test]
    fn test_center_offset_shifts_right_profile_leftward() {
        // Right profile: nose is right of eye midpoint → back of head is left
        let lm = FaceLandmarks::new([
            (530.0, 350.0),
            (590.0, 350.0),
            (610.0, 420.0),
            (550.0, 470.0),
            (580.0, 470.0),
        ]);
        let bbox: BBox = (500.0, 300.0, 700.0, 500.0);

        let mut b_no_offset = FaceRegionBuilder::new(PADDING, 0.0, None);
        let r_no = b_no_offset.build(bbox, FRAME_W, FRAME_H, Some(&lm), None);
        let cx_no = r_no.unclamped_x.unwrap() as f64 + r_no.full_width.unwrap() as f64 / 2.0;

        let mut b_offset = FaceRegionBuilder::new(PADDING, 0.3, None);
        let r_off = b_offset.build(bbox, FRAME_W, FRAME_H, Some(&lm), None);
        let cx_off = r_off.unclamped_x.unwrap() as f64 + r_off.full_width.unwrap() as f64 / 2.0;

        // Positive offset + right profile → center shifts left (smaller cx)
        assert!(
            cx_off < cx_no,
            "Expected offset center {cx_off} < no-offset center {cx_no}"
        );
    }

    #[test]
    fn test_center_offset_frontal_no_shift() {
        // Frontal face → back_of_head_direction returns 0.0 → no shift
        let lm = frontal_landmarks();
        let mut b_no = FaceRegionBuilder::new(PADDING, 0.0, None);
        let r_no = b_no.build(frontal_box(), FRAME_W, FRAME_H, Some(&lm), None);

        let mut b_off = FaceRegionBuilder::new(PADDING, 0.5, None);
        let r_off = b_off.build(frontal_box(), FRAME_W, FRAME_H, Some(&lm), None);

        // Centers should be identical for frontal face regardless of offset
        let cx_no = r_no.unclamped_x.unwrap() as f64 + r_no.full_width.unwrap() as f64 / 2.0;
        let cx_off = r_off.unclamped_x.unwrap() as f64 + r_off.full_width.unwrap() as f64 / 2.0;
        assert_relative_eq!(cx_no, cx_off, epsilon = 1.0);
    }

    #[test]
    fn test_center_offset_no_landmarks_no_shift() {
        let mut b_no = FaceRegionBuilder::new(PADDING, 0.0, None);
        let r_no = b_no.build(frontal_box(), FRAME_W, FRAME_H, None, None);

        let mut b_off = FaceRegionBuilder::new(PADDING, 0.5, None);
        let r_off = b_off.build(frontal_box(), FRAME_W, FRAME_H, None, None);

        assert_eq!(r_no.x, r_off.x);
        assert_eq!(r_no.width, r_off.width);
    }

    // ── Custom padding ────────────────────────────────────────────

    #[test]
    fn test_larger_padding_produces_larger_region() {
        let mut b_small = FaceRegionBuilder::new(0.2, 0.0, None);
        let r_small = b_small.build(frontal_box(), FRAME_W, FRAME_H, None, None);

        let mut b_large = FaceRegionBuilder::new(0.8, 0.0, None);
        let r_large = b_large.build(frontal_box(), FRAME_W, FRAME_H, None, None);

        assert!(
            r_large.full_width.unwrap() > r_small.full_width.unwrap(),
            "Larger padding should produce larger width"
        );
        assert!(
            r_large.full_height.unwrap() > r_small.full_height.unwrap(),
            "Larger padding should produce larger height"
        );
    }

    #[test]
    fn test_zero_padding_matches_min_width_constraint() {
        let mut b = FaceRegionBuilder::new(0.0, 0.0, None);
        let r = b.build(frontal_box(), FRAME_W, FRAME_H, None, None);
        // box is 200x200, padding=0 → half_w = max(200, 200*0.8) * 1.0 / 2 = 100
        // full_w = 200
        assert_eq!(r.full_width.unwrap(), 200);
        assert_eq!(r.full_height.unwrap(), 200);
    }
}
