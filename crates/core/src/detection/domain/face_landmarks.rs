//! 5-point face landmarks with weighted centroid and profile detection.
//!
//! Weights emphasize nose (3x) over eyes (2x) and mouth (1x) for stable centering,
//! since nose position is the most reliable anchor across head rotations.

const LEFT_EYE: usize = 0;
const RIGHT_EYE: usize = 1;
const NOSE: usize = 2;

/// Landmark weights: [left_eye, right_eye, nose, left_mouth, right_mouth].
const WEIGHTS: [f64; 5] = [2.0, 2.0, 3.0, 1.0, 1.0];

#[derive(Clone, Debug, PartialEq)]
pub struct FaceLandmarks {
    /// Points with x <= 0 are treated as invisible.
    points: [(f64, f64); 5],
}

impl FaceLandmarks {
    pub fn new(points: [(f64, f64); 5]) -> Self {
        Self { points }
    }

    pub fn points(&self) -> &[(f64, f64); 5] {
        &self.points
    }

    pub fn has_visible(&self) -> bool {
        self.points.iter().any(|(x, _)| *x > 0.0)
    }

    /// Weighted centroid of visible landmarks (x > 0).
    pub fn center(&self) -> Result<(f64, f64), &'static str> {
        let mut wx_sum = 0.0;
        let mut wy_sum = 0.0;
        let mut w_sum = 0.0;

        for (i, (x, y)) in self.points.iter().enumerate() {
            if *x > 0.0 {
                let w = WEIGHTS[i];
                wx_sum += x * w;
                wy_sum += y * w;
                w_sum += w;
            }
        }

        if w_sum == 0.0 {
            return Err("No visible landmarks");
        }

        Ok((wx_sum / w_sum, wy_sum / w_sum))
    }

    /// How much the face is turned: 0.0 = frontal, 1.0 = full profile.
    ///
    /// Measures nose offset from eye midpoint relative to eye span.
    /// Returns 0.0 when required landmarks are not visible.
    pub fn profile_ratio(&self) -> f64 {
        let nose = self.points[NOSE];
        let left_eye = self.points[LEFT_EYE];
        let right_eye = self.points[RIGHT_EYE];

        if nose.0 <= 0.0 || left_eye.0 <= 0.0 || right_eye.0 <= 0.0 {
            return 0.0;
        }

        let eye_mid_x = (left_eye.0 + right_eye.0) / 2.0;
        let eye_span = (right_eye.0 - left_eye.0).abs();

        if eye_span <= 0.0 {
            return 0.0;
        }

        ((nose.0 - eye_mid_x).abs() / eye_span).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    fn frontal_landmarks() -> FaceLandmarks {
        FaceLandmarks::new([
            (440.0, 350.0), // left_eye
            (560.0, 350.0), // right_eye
            (500.0, 420.0), // nose (centered)
            (460.0, 470.0), // left_mouth
            (540.0, 470.0), // right_mouth
        ])
    }

    fn left_profile_landmarks() -> FaceLandmarks {
        // Nose shifted left of eye midpoint
        FaceLandmarks::new([
            (120.0, 350.0), // left_eye
            (180.0, 350.0), // right_eye
            (100.0, 420.0), // nose (shifted left)
            (130.0, 470.0), // left_mouth
            (170.0, 470.0), // right_mouth
        ])
    }

    fn right_profile_landmarks() -> FaceLandmarks {
        // Nose shifted right of eye midpoint
        FaceLandmarks::new([
            (530.0, 350.0), // left_eye
            (590.0, 350.0), // right_eye
            (610.0, 420.0), // nose (shifted right)
            (550.0, 470.0), // left_mouth
            (580.0, 470.0), // right_mouth
        ])
    }

    // ── has_visible ─────────────────────────────────────────────────

    #[test]
    fn test_has_visible_all_visible() {
        assert!(frontal_landmarks().has_visible());
    }

    #[test]
    fn test_has_visible_none_visible() {
        let lm = FaceLandmarks::new([(0.0, 0.0); 5]);
        assert!(!lm.has_visible());
    }

    #[test]
    fn test_has_visible_one_visible() {
        let mut pts = [(0.0, 0.0); 5];
        pts[NOSE] = (100.0, 200.0);
        let lm = FaceLandmarks::new(pts);
        assert!(lm.has_visible());
    }

    // ── center (weighted centroid) ──────────────────────────────────

    #[test]
    fn test_center_frontal_symmetric() {
        let lm = frontal_landmarks();
        let (cx, cy) = lm.center().unwrap();
        // Nose has highest weight (3x), so center is biased toward nose
        // Weights: [2,2,3,1,1] = total 9
        // cx = (440*2 + 560*2 + 500*3 + 460*1 + 540*1) / 9 = 4500/9 = 500
        assert_relative_eq!(cx, 500.0, epsilon = 0.01);
        // cy = (350*2 + 350*2 + 420*3 + 470*1 + 470*1) / 9
        //    = (700 + 700 + 1260 + 470 + 470) / 9 = 3600/9 = 400
        assert_relative_eq!(cy, 400.0, epsilon = 0.01);
    }

    #[test]
    fn test_center_no_visible_returns_error() {
        let lm = FaceLandmarks::new([(0.0, 0.0); 5]);
        assert!(lm.center().is_err());
    }

    #[test]
    fn test_center_partial_visibility() {
        // Only nose visible
        let mut pts = [(0.0, 0.0); 5];
        pts[NOSE] = (300.0, 400.0);
        let lm = FaceLandmarks::new(pts);
        let (cx, cy) = lm.center().unwrap();
        // Only nose contributes, weight=3
        assert_relative_eq!(cx, 300.0);
        assert_relative_eq!(cy, 400.0);
    }

    #[test]
    fn test_center_nose_weighted_heavier() {
        // Eyes at x=100, nose at x=200 — center should be biased toward nose
        let lm = FaceLandmarks::new([
            (100.0, 100.0), // left_eye (w=2)
            (100.0, 100.0), // right_eye (w=2)
            (200.0, 100.0), // nose (w=3)
            (100.0, 100.0), // left_mouth (w=1)
            (100.0, 100.0), // right_mouth (w=1)
        ]);
        let (cx, _) = lm.center().unwrap();
        // cx = (100*2 + 100*2 + 200*3 + 100*1 + 100*1) / 9 = 1200/9 ≈ 133.33
        assert_relative_eq!(cx, 1200.0 / 9.0, epsilon = 0.01);
    }

    // ── profile_ratio ───────────────────────────────────────────────

    #[test]
    fn test_profile_ratio_frontal() {
        let lm = frontal_landmarks();
        // Nose at 500, eye midpoint at (440+560)/2 = 500, so ratio = 0
        assert_relative_eq!(lm.profile_ratio(), 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_profile_ratio_left_profile() {
        let lm = left_profile_landmarks();
        // Nose at 100, eye midpoint at (120+180)/2 = 150
        // eye_span = |180-120| = 60
        // ratio = |100-150|/60 = 50/60 ≈ 0.833
        assert_relative_eq!(lm.profile_ratio(), 50.0 / 60.0, epsilon = 0.01);
    }

    #[test]
    fn test_profile_ratio_right_profile() {
        let lm = right_profile_landmarks();
        // Nose at 610, eye midpoint at (530+590)/2 = 560
        // eye_span = |590-530| = 60
        // ratio = |610-560|/60 = 50/60 ≈ 0.833
        assert_relative_eq!(lm.profile_ratio(), 50.0 / 60.0, epsilon = 0.01);
    }

    #[test]
    fn test_profile_ratio_clamped_to_one() {
        // Extreme case: nose far beyond eye span
        let lm = FaceLandmarks::new([
            (100.0, 100.0),
            (110.0, 100.0), // eye_span = 10
            (200.0, 100.0), // nose offset = 95 >> eye_span
            (100.0, 100.0),
            (100.0, 100.0),
        ]);
        assert_relative_eq!(lm.profile_ratio(), 1.0);
    }

    #[rstest]
    #[case::nose_invisible([(100.0, 100.0), (200.0, 100.0), (0.0, 0.0), (100.0, 100.0), (100.0, 100.0)])]
    #[case::left_eye_invisible([(0.0, 0.0), (200.0, 100.0), (150.0, 100.0), (100.0, 100.0), (100.0, 100.0)])]
    #[case::right_eye_invisible([(100.0, 100.0), (0.0, 0.0), (150.0, 100.0), (100.0, 100.0), (100.0, 100.0)])]
    fn test_profile_ratio_missing_landmarks_returns_zero(#[case] pts: [(f64, f64); 5]) {
        let lm = FaceLandmarks::new(pts);
        assert_relative_eq!(lm.profile_ratio(), 0.0);
    }

    #[test]
    fn test_profile_ratio_zero_eye_span() {
        // Both eyes at same x position
        let lm = FaceLandmarks::new([
            (100.0, 100.0),
            (100.0, 100.0), // same x as left eye
            (150.0, 100.0),
            (100.0, 100.0),
            (100.0, 100.0),
        ]);
        assert_relative_eq!(lm.profile_ratio(), 0.0);
    }
}
