use std::collections::HashMap;

/// Smoothing parameters: [cx, cy, half_w, half_h].
pub type SmoothParams = [f64; 4];

/// Domain interface for temporal smoothing of region parameters.
pub trait RegionSmootherInterface: Send {
    fn smooth(&mut self, params: SmoothParams, track_id: Option<u32>) -> SmoothParams;
}

/// EMA (Exponential Moving Average) smoother with per-track state.
///
/// Formula: `ema[t] = alpha * current + (1 - alpha) * ema[t-1]`
/// Default alpha: 0.6
pub struct RegionSmoother {
    alpha: f64,
    state: HashMap<u32, SmoothParams>,
}

pub const DEFAULT_ALPHA: f64 = 0.6;

impl RegionSmoother {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            state: HashMap::new(),
        }
    }
}

impl Default for RegionSmoother {
    fn default() -> Self {
        Self::new(DEFAULT_ALPHA)
    }
}

impl RegionSmootherInterface for RegionSmoother {
    fn smooth(&mut self, params: SmoothParams, track_id: Option<u32>) -> SmoothParams {
        let Some(tid) = track_id else {
            return params;
        };

        let smoothed = match self.state.get(&tid) {
            None => params,
            Some(prev) => {
                let mut result = [0.0; 4];
                for i in 0..4 {
                    result[i] = self.alpha * params[i] + (1.0 - self.alpha) * prev[i];
                }
                result
            }
        };

        self.state.insert(tid, smoothed);
        smoothed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_default_alpha() {
        assert_relative_eq!(DEFAULT_ALPHA, 0.6);
    }

    #[test]
    fn test_no_track_id_bypasses_smoothing() {
        let mut smoother = RegionSmoother::default();
        let params = [100.0, 200.0, 50.0, 60.0];
        let result = smoother.smooth(params, None);
        assert_eq!(result, params);
    }

    #[test]
    fn test_first_observation_returns_unchanged() {
        let mut smoother = RegionSmoother::default();
        let params = [100.0, 200.0, 50.0, 60.0];
        let result = smoother.smooth(params, Some(1));
        assert_eq!(result, params);
    }

    #[test]
    fn test_second_observation_applies_ema() {
        let mut smoother = RegionSmoother::new(0.6);
        let first = [100.0, 200.0, 50.0, 60.0];
        smoother.smooth(first, Some(1));

        let second = [110.0, 210.0, 55.0, 65.0];
        let result = smoother.smooth(second, Some(1));

        // ema = 0.6 * current + 0.4 * prev
        assert_relative_eq!(result[0], 0.6 * 110.0 + 0.4 * 100.0); // 106
        assert_relative_eq!(result[1], 0.6 * 210.0 + 0.4 * 200.0); // 206
        assert_relative_eq!(result[2], 0.6 * 55.0 + 0.4 * 50.0); // 53
        assert_relative_eq!(result[3], 0.6 * 65.0 + 0.4 * 60.0); // 63
    }

    #[test]
    fn test_convergence() {
        let mut smoother = RegionSmoother::new(0.6);
        let target = [500.0, 500.0, 100.0, 100.0];

        // Start far away
        smoother.smooth([0.0, 0.0, 0.0, 0.0], Some(1));

        // Feed same target repeatedly — should converge
        let mut result = [0.0; 4];
        for _ in 0..50 {
            result = smoother.smooth(target, Some(1));
        }

        for i in 0..4 {
            assert_relative_eq!(result[i], target[i], epsilon = 0.01);
        }
    }

    #[test]
    fn test_multiple_tracks_independent() {
        let mut smoother = RegionSmoother::new(0.6);

        let a = [100.0, 100.0, 50.0, 50.0];
        let b = [500.0, 500.0, 80.0, 80.0];

        smoother.smooth(a, Some(1));
        smoother.smooth(b, Some(2));

        let a2 = [110.0, 110.0, 55.0, 55.0];
        let b2 = [510.0, 510.0, 85.0, 85.0];

        let ra = smoother.smooth(a2, Some(1));
        let rb = smoother.smooth(b2, Some(2));

        // Track 1 should smooth from a → a2
        assert_relative_eq!(ra[0], 0.6 * 110.0 + 0.4 * 100.0);
        // Track 2 should smooth from b → b2
        assert_relative_eq!(rb[0], 0.6 * 510.0 + 0.4 * 500.0);
    }

    #[test]
    fn test_alpha_zero_keeps_first_value() {
        let mut smoother = RegionSmoother::new(0.0);
        let first = [100.0, 200.0, 50.0, 60.0];
        smoother.smooth(first, Some(1));

        let second = [999.0, 999.0, 999.0, 999.0];
        let result = smoother.smooth(second, Some(1));

        // alpha=0: ema = 0*current + 1*prev = prev
        assert_eq!(result, first);
    }

    #[test]
    fn test_alpha_one_uses_current() {
        let mut smoother = RegionSmoother::new(1.0);
        let first = [100.0, 200.0, 50.0, 60.0];
        smoother.smooth(first, Some(1));

        let second = [999.0, 888.0, 777.0, 666.0];
        let result = smoother.smooth(second, Some(1));

        // alpha=1: ema = 1*current + 0*prev = current
        assert_eq!(result, second);
    }
}
