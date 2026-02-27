#[derive(Clone, Debug, PartialEq)]
pub struct CensorRegion {
    pub start_time: f64,
    pub end_time: f64,
    pub padding: f64,
}

impl CensorRegion {
    pub fn effective_start(&self) -> f64 {
        (self.start_time - self.padding).max(0.0)
    }

    pub fn effective_end(&self) -> f64 {
        self.end_time + self.padding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_censor_region_effective_range() {
        let r = CensorRegion {
            start_time: 1.0,
            end_time: 2.0,
            padding: 0.05,
        };
        assert_relative_eq!(r.effective_start(), 0.95);
        assert_relative_eq!(r.effective_end(), 2.05);
    }

    #[test]
    fn test_censor_region_effective_start_clamps_to_zero() {
        let r = CensorRegion {
            start_time: 0.02,
            end_time: 0.5,
            padding: 0.05,
        };
        assert_relative_eq!(r.effective_start(), 0.0);
    }
}
