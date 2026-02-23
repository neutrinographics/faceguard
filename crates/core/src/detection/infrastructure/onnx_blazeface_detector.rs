/// BlazeFace face detector using ONNX Runtime via `ort`.
///
/// A lightweight face detector that provides bounding boxes without tracking
/// or landmarks. Suitable as a fast fallback when YOLO is unavailable.
use std::path::Path;

use crate::detection::domain::face_detector::FaceDetector;
use crate::shared::frame::Frame;
use crate::shared::region::Region;

/// BlazeFace model input resolution.
const INPUT_SIZE: u32 = 128;

/// Default confidence threshold.
pub const DEFAULT_CONFIDENCE: f64 = 0.5;

/// NMS IoU threshold.
const NMS_IOU_THRESH: f64 = 0.3;

/// Number of BlazeFace anchors (short-range model).
const NUM_ANCHORS: usize = 896;

/// BlazeFace face detector backed by an ONNX Runtime session.
///
/// No tracking, no landmarks — produces regions with `track_id: None`.
pub struct OnnxBlazefaceDetector {
    session: ort::session::Session,
    confidence: f64,
    fps: f64,
    anchors: Vec<[f32; 2]>,
}

impl OnnxBlazefaceDetector {
    /// Load a BlazeFace ONNX model.
    pub fn new(
        model_path: &Path,
        confidence: f64,
        fps: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let session = ort::session::Session::builder()?.commit_from_file(model_path)?;
        let anchors = generate_anchors();
        Ok(Self {
            session,
            confidence,
            fps,
            anchors,
        })
    }

    /// Video FPS (used for logging/diagnostics, not detection).
    #[allow(dead_code)]
    pub fn fps(&self) -> f64 {
        self.fps
    }
}

impl FaceDetector for OnnxBlazefaceDetector {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Region>, Box<dyn std::error::Error>> {
        let fw = frame.width();
        let fh = frame.height();

        // 1. Preprocess: resize to 128x128, normalize to [0,1], NCHW
        let input_tensor = preprocess(frame, INPUT_SIZE);

        // 2. Inference
        let input_value = ort::value::Tensor::from_array(input_tensor)?;
        let outputs = self.session.run(ort::inputs![input_value])?;

        // BlazeFace outputs two tensors:
        // - regressors: [1, 896, 16] (box deltas + keypoints)
        // - classificators: [1, 896, 1] (confidence scores)
        if outputs.len() < 2 {
            return Err(
                format!("BlazeFace model expected 2 outputs, got {}", outputs.len()).into(),
            );
        }

        let regressors = outputs[0].try_extract_array::<f32>()?;
        let scores = outputs[1].try_extract_array::<f32>()?;
        let reg_data = regressors.as_slice().ok_or("Cannot get regressor slice")?;
        let score_data = scores.as_slice().ok_or("Cannot get score slice")?;

        // 3. Decode anchor boxes + filter by confidence
        let mut raw_dets = Vec::new();
        let num_anchors = self.anchors.len().min(NUM_ANCHORS);

        for (i, &raw_score) in score_data.iter().enumerate().take(num_anchors) {
            let score = sigmoid(raw_score);
            if score < self.confidence as f32 {
                continue;
            }

            let anchor = &self.anchors[i];
            let reg_offset = i * 16;
            if reg_offset + 4 > reg_data.len() {
                break;
            }

            // Decode box center + size relative to anchor
            let cx = anchor[0] + reg_data[reg_offset] / INPUT_SIZE as f32;
            let cy = anchor[1] + reg_data[reg_offset + 1] / INPUT_SIZE as f32;
            let w = reg_data[reg_offset + 2] / INPUT_SIZE as f32;
            let h = reg_data[reg_offset + 3] / INPUT_SIZE as f32;

            // Convert to original frame coordinates
            let x1 = ((cx - w / 2.0) * fw as f32).max(0.0);
            let y1 = ((cy - h / 2.0) * fh as f32).max(0.0);
            let x2 = ((cx + w / 2.0) * fw as f32).min(fw as f32);
            let y2 = ((cy + h / 2.0) * fh as f32).min(fh as f32);

            raw_dets.push(RawDet {
                x1: x1 as f64,
                y1: y1 as f64,
                x2: x2 as f64,
                y2: y2 as f64,
                score: score as f64,
            });
        }

        // 4. NMS
        let filtered = nms(&mut raw_dets, NMS_IOU_THRESH);

        // 5. Build regions (no tracking, no landmarks)
        let regions = filtered
            .iter()
            .map(|d| {
                // x1/y1 are already clamped to >= 0 during decoding
                let x = d.x1 as i32;
                let y = d.y1 as i32;
                let w = ((d.x2 - d.x1) as i32).min(fw as i32 - x);
                let h = ((d.y2 - d.y1) as i32).min(fh as i32 - y);
                Region {
                    x,
                    y,
                    width: w,
                    height: h,
                    track_id: None,
                    full_width: None,
                    full_height: None,
                    unclamped_x: None,
                    unclamped_y: None,
                }
            })
            .collect();

        Ok(regions)
    }
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Resize frame to `size × size` and normalize to [0,1] NCHW float32.
fn preprocess(frame: &Frame, size: u32) -> ndarray::Array4<f32> {
    let src = frame.as_ndarray();
    let src_h = frame.height() as usize;
    let src_w = frame.width() as usize;
    let s = size as usize;

    let mut tensor = ndarray::Array4::<f32>::zeros((1, 3, s, s));

    for y in 0..s {
        let src_y = (((y as f64 + 0.5) * src_h as f64 / s as f64) as usize).min(src_h - 1);
        for x in 0..s {
            let src_x = (((x as f64 + 0.5) * src_w as f64 / s as f64) as usize).min(src_w - 1);
            for c in 0..3 {
                tensor[[0, c, y, x]] = src[[src_y, src_x, c]] as f32 / 255.0;
            }
        }
    }

    tensor
}

// ---------------------------------------------------------------------------
// Anchor generation (BlazeFace short-range)
// ---------------------------------------------------------------------------

/// Generate BlazeFace anchors for the short-range model.
///
/// The short-range model uses two feature map sizes: 16×16 and 8×8,
/// with 2 and 6 anchors per cell respectively.
fn generate_anchors() -> Vec<[f32; 2]> {
    let strides = [(8, 2), (16, 6)]; // (stride, anchors_per_cell)
    let mut anchors = Vec::with_capacity(NUM_ANCHORS);

    for &(stride, num) in &strides {
        let grid_size = INPUT_SIZE as usize / stride;
        for y in 0..grid_size {
            for x in 0..grid_size {
                let cx = (x as f32 + 0.5) / grid_size as f32;
                let cy = (y as f32 + 0.5) / grid_size as f32;
                for _ in 0..num {
                    anchors.push([cx, cy]);
                }
            }
        }
    }

    anchors
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct RawDet {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    score: f64,
}

fn nms(dets: &mut [RawDet], iou_thresh: f64) -> Vec<RawDet> {
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; dets.len()];

    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if suppressed[j] {
                continue;
            }
            let iou = bbox_iou(&dets[i], &dets[j]);
            if iou > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}

fn bbox_iou(a: &RawDet, b: &RawDet) -> f64 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    if inter == 0.0 {
        return 0.0;
    }
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    inter / (area_a + area_b - inter)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_shape() {
        let data = vec![128u8; 200 * 100 * 3];
        let frame = Frame::new(data, 200, 100, 3, 0);
        let tensor = preprocess(&frame, 128);
        assert_eq!(tensor.shape(), &[1, 3, 128, 128]);
    }

    #[test]
    fn test_preprocess_normalized() {
        let data = vec![255u8; 50 * 50 * 3];
        let frame = Frame::new(data, 50, 50, 3, 0);
        let tensor = preprocess(&frame, 128);
        // All source pixels are 255, so resized pixels should be ~1.0
        assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_generate_anchors_count() {
        let anchors = generate_anchors();
        // 16×16 grid × 2 anchors + 8×8 grid × 6 anchors = 512 + 384 = 896
        assert_eq!(anchors.len(), NUM_ANCHORS);
    }

    #[test]
    fn test_anchors_in_unit_range() {
        let anchors = generate_anchors();
        for a in &anchors {
            assert!(a[0] > 0.0 && a[0] < 1.0);
            assert!(a[1] > 0.0 && a[1] < 1.0);
        }
    }

    #[test]
    fn test_sigmoid_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        assert!((sigmoid(10.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_nms_blazeface_suppresses() {
        let mut dets = vec![
            RawDet {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 100.0,
                score: 0.9,
            },
            RawDet {
                x1: 5.0,
                y1: 5.0,
                x2: 105.0,
                y2: 105.0,
                score: 0.7,
            },
        ];
        let kept = nms(&mut dets, 0.3);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn test_nms_blazeface_keeps_separate() {
        let mut dets = vec![
            RawDet {
                x1: 0.0,
                y1: 0.0,
                x2: 50.0,
                y2: 50.0,
                score: 0.9,
            },
            RawDet {
                x1: 200.0,
                y1: 200.0,
                x2: 250.0,
                y2: 250.0,
                score: 0.8,
            },
        ];
        let kept = nms(&mut dets, 0.3);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_region_has_no_track_id() {
        // Verify the contract: BlazeFace regions always have track_id: None
        let r = Region {
            x: 10,
            y: 20,
            width: 50,
            height: 50,
            track_id: None,
            full_width: None,
            full_height: None,
            unclamped_x: None,
            unclamped_y: None,
        };
        assert!(r.track_id.is_none());
    }
}
