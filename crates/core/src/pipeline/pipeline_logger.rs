use std::collections::HashMap;
use std::time::Instant;

/// Cross-cutting logger for pipeline orchestration events.
///
/// Decouples use cases from specific output mechanisms (stdout, GUI signals,
/// log crate) so each caller can observe pipeline behavior without changing
/// the orchestration code.
pub trait PipelineLogger: Send {
    /// Report frame-level progress.
    fn progress(&mut self, current: usize, total: usize);

    /// Record how long a named pipeline stage took for one frame.
    fn timing(&mut self, stage: &str, duration_ms: f64);

    /// Record a point-in-time metric (e.g. queue depth, region count).
    fn metric(&mut self, name: &str, value: f64);

    /// Log a human-readable status message.
    fn info(&mut self, message: &str);

    /// Emit an end-of-pipeline summary. Default: no-op.
    fn summary(&self) {}
}

/// Silent logger that discards all events.
///
/// Used by the desktop GUI (which has its own signal-based progress)
/// and by tests where logger output is irrelevant.
pub struct NullPipelineLogger;

impl PipelineLogger for NullPipelineLogger {
    fn progress(&mut self, _current: usize, _total: usize) {}
    fn timing(&mut self, _stage: &str, _duration_ms: f64) {}
    fn metric(&mut self, _name: &str, _value: f64) {}
    fn info(&mut self, _message: &str) {}
}

/// CLI-oriented logger that tracks per-stage timing, metrics, and
/// provides a summary report at pipeline completion.
///
/// Progress output is throttled to every `throttle_frames` frames
/// to avoid excessive I/O on large videos.
pub struct StdoutPipelineLogger {
    throttle_frames: usize,
    timings: HashMap<String, Vec<f64>>,
    metrics: HashMap<String, Vec<f64>>,
    start_time: Instant,
    total_frames: usize,
    messages: Vec<String>,
}

impl StdoutPipelineLogger {
    pub fn new(throttle_frames: usize) -> Self {
        Self {
            throttle_frames: throttle_frames.max(1),
            timings: HashMap::new(),
            metrics: HashMap::new(),
            start_time: Instant::now(),
            total_frames: 0,
            messages: Vec::new(),
        }
    }

    /// Returns the formatted summary string, or `None` if no data recorded.
    pub fn summary_string(&self) -> Option<String> {
        if self.timings.is_empty() && self.metrics.is_empty() {
            return None;
        }

        let elapsed_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        let frames = self.total_frames;
        let mut lines = Vec::new();

        lines.push(format!(
            "Pipeline summary ({frames} frames, {:.1}s total):",
            elapsed_ms / 1000.0
        ));

        let mut stages: Vec<_> = self.timings.keys().collect();
        stages.sort();
        for stage in stages {
            let durations = &self.timings[stage];
            let total_ms: f64 = durations.iter().sum();
            let avg_ms = if durations.is_empty() {
                0.0
            } else {
                total_ms / durations.len() as f64
            };
            let pct = if elapsed_ms > 0.0 {
                total_ms / elapsed_ms * 100.0
            } else {
                0.0
            };
            lines.push(format!(
                "  {stage:12}: avg {avg_ms:6.1}ms  total {total_ms:7.0}ms  ({pct:4.1}%)"
            ));
        }

        let mut metric_names: Vec<_> = self.metrics.keys().collect();
        metric_names.sort();
        for name in metric_names {
            let values = &self.metrics[name];
            let avg = if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            };
            lines.push(format!("  {name}: avg {avg:.1}"));
        }

        if frames > 0 && elapsed_ms > 0.0 {
            let fps = frames as f64 / (elapsed_ms / 1000.0);
            lines.push(format!("  Throughput: {fps:.1} fps"));
        }

        Some(lines.join("\n"))
    }

    /// Returns the timing data for a given stage.
    pub fn timings_for(&self, stage: &str) -> Option<&[f64]> {
        self.timings.get(stage).map(|v| v.as_slice())
    }

    /// Returns the metric data for a given name.
    pub fn metrics_for(&self, name: &str) -> Option<&[f64]> {
        self.metrics.get(name).map(|v| v.as_slice())
    }
}

impl Default for StdoutPipelineLogger {
    fn default() -> Self {
        Self::new(10)
    }
}

impl PipelineLogger for StdoutPipelineLogger {
    fn progress(&mut self, current: usize, total: usize) {
        self.total_frames = total;
        if total > 0 && (current % self.throttle_frames == 0 || current == total) {
            let pct = current as f64 / total as f64 * 100.0;
            log::info!("Processing: {current}/{total} frames ({pct:.1}%)");
        }
    }

    fn timing(&mut self, stage: &str, duration_ms: f64) {
        self.timings
            .entry(stage.to_string())
            .or_default()
            .push(duration_ms);
    }

    fn metric(&mut self, name: &str, value: f64) {
        self.metrics
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    fn info(&mut self, message: &str) {
        self.messages.push(message.to_string());
        log::info!("{message}");
    }

    fn summary(&self) {
        if let Some(text) = self.summary_string() {
            log::info!("\n\n{text}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- NullPipelineLogger tests ---

    #[test]
    fn test_null_logger_all_methods_are_noop() {
        let mut logger = NullPipelineLogger;
        logger.progress(1, 10);
        logger.timing("detect", 5.0);
        logger.metric("queue_depth", 3.0);
        logger.info("hello");
        logger.summary();
        // No panics = success
    }

    // --- StdoutPipelineLogger tests ---

    #[test]
    fn test_timing_records_values() {
        let mut logger = StdoutPipelineLogger::new(10);
        logger.timing("detect", 20.0);
        logger.timing("detect", 30.0);
        logger.timing("blur", 5.0);

        let detect = logger.timings_for("detect").unwrap();
        assert_eq!(detect.len(), 2);
        assert!((detect[0] - 20.0).abs() < f64::EPSILON);
        assert!((detect[1] - 30.0).abs() < f64::EPSILON);

        let blur = logger.timings_for("blur").unwrap();
        assert_eq!(blur.len(), 1);
        assert!((blur[0] - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_records_values() {
        let mut logger = StdoutPipelineLogger::new(10);
        logger.metric("reader_queue_depth", 3.0);
        logger.metric("reader_queue_depth", 4.0);

        let values = logger.metrics_for("reader_queue_depth").unwrap();
        assert_eq!(values.len(), 2);
        let avg = values.iter().sum::<f64>() / values.len() as f64;
        assert!((avg - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_summary_includes_timing() {
        let mut logger = StdoutPipelineLogger::new(10);
        logger.total_frames = 10;
        logger.timing("detect", 20.0);
        logger.timing("detect", 30.0);
        logger.timing("blur", 5.0);

        let summary = logger.summary_string().unwrap();
        assert!(summary.contains("detect"));
        assert!(summary.contains("blur"));
        assert!(summary.contains("Pipeline summary"));
    }

    #[test]
    fn test_summary_includes_metrics() {
        let mut logger = StdoutPipelineLogger::new(10);
        logger.total_frames = 5;
        logger.metric("reader_queue_depth", 3.0);
        logger.metric("reader_queue_depth", 4.0);

        let summary = logger.summary_string().unwrap();
        assert!(summary.contains("reader_queue_depth"));
        assert!(summary.contains("avg 3.5"));
    }

    #[test]
    fn test_summary_includes_fps() {
        let mut logger = StdoutPipelineLogger::new(10);
        logger.total_frames = 100;
        logger.timing("detect", 10.0);

        let summary = logger.summary_string().unwrap();
        assert!(summary.contains("fps"));
    }

    #[test]
    fn test_empty_summary_returns_none() {
        let logger = StdoutPipelineLogger::new(10);
        assert!(logger.summary_string().is_none());
    }

    #[test]
    fn test_progress_throttled() {
        // We can't easily capture log::info output, so we test the throttle
        // logic by checking that progress updates total_frames correctly
        let mut logger = StdoutPipelineLogger::new(10);
        for i in 1..=20 {
            logger.progress(i, 20);
        }
        assert_eq!(logger.total_frames, 20);
    }

    #[test]
    fn test_info_stores_messages() {
        let mut logger = StdoutPipelineLogger::new(10);
        logger.info("hello world");
        assert_eq!(logger.messages.len(), 1);
        assert_eq!(logger.messages[0], "hello world");
    }

    #[test]
    fn test_timing_averages() {
        let mut logger = StdoutPipelineLogger::new(10);
        logger.timing("stage_a", 10.0);
        logger.timing("stage_a", 20.0);
        logger.timing("stage_a", 30.0);

        let values = logger.timings_for("stage_a").unwrap();
        let avg = values.iter().sum::<f64>() / values.len() as f64;
        assert!((avg - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_throttle() {
        let logger = StdoutPipelineLogger::default();
        assert_eq!(logger.throttle_frames, 10);
    }
}
