/// YOLO face detector using ONNX Runtime via `ort`.
///
/// Handles letterbox preprocessing, inference, NMS post-processing, ByteTrack
/// tracking, and region building through the domain's `FaceRegionBuilder`.
use std::path::Path;

use crate::detection::domain::face_detector::FaceDetector;
use crate::detection::domain::face_landmarks::FaceLandmarks;
use crate::detection::domain::face_region_builder::FaceRegionBuilder;
use crate::shared::frame::Frame;
use crate::shared::region::Region;

use super::bytetrack_tracker::{ByteTracker, Detection as TrackerDetection};

/// Fallback YOLO model input resolution when the model doesn't specify dimensions.
const DEFAULT_INPUT_SIZE: u32 = 640;

/// Default confidence threshold for face detection.
pub const DEFAULT_CONFIDENCE: f64 = 0.25;

/// NMS IoU threshold.
const NMS_IOU_THRESH: f64 = 0.45;

/// Number of keypoints per detection (5 landmarks × 3 values each: x, y, conf).
const NUM_KEYPOINT_VALUES: usize = 15;

/// Minimum keypoint confidence to treat a landmark as visible.
const KEYPOINT_CONF_THRESH: f64 = 0.5;

/// YOLO face detector backed by an ONNX Runtime session.
pub struct OnnxYoloDetector {
    session: ort::session::Session,
    region_builder: FaceRegionBuilder,
    tracker: ByteTracker,
    confidence: f64,
    input_size: u32,
}

impl OnnxYoloDetector {
    /// Load a YOLO ONNX model and prepare for inference.
    ///
    /// The input resolution is read from the model's input shape (expecting NCHW).
    /// Falls back to 640 if the shape is dynamic or unreadable.
    pub fn new(
        model_path: &Path,
        region_builder: FaceRegionBuilder,
        tracker: ByteTracker,
        confidence: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let session = ort::session::Session::builder()?.commit_from_file(model_path)?;

        // Try to read input size from model metadata (NCHW: [1, 3, H, W])
        let input_size = session
            .inputs()
            .first()
            .and_then(|input| {
                if let ort::value::ValueType::Tensor { ref shape, .. } = input.dtype() {
                    // shape is [N, C, H, W] — use H (they should be equal for square input)
                    if shape.len() >= 4 && shape[2] > 0 {
                        Some(shape[2] as u32)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .unwrap_or(DEFAULT_INPUT_SIZE);

        Ok(Self {
            session,
            region_builder,
            tracker,
            confidence,
            input_size,
        })
    }
}

impl FaceDetector for OnnxYoloDetector {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Region>, Box<dyn std::error::Error>> {
        let fw = frame.width();
        let fh = frame.height();

        // 1. Preprocess: letterbox + normalize → NCHW float32
        let (input_tensor, scale, pad_x, pad_y) = letterbox(frame, self.input_size);

        // 2. Inference
        let input_value = ort::value::Tensor::from_array(input_tensor)?;
        let outputs = self.session.run(ort::inputs![input_value])?;
        if outputs.len() == 0 {
            return Err("YOLO model produced no outputs".into());
        }
        let tensor = outputs[0].try_extract_array::<f32>()?;
        let shape = tensor.shape();

        // YOLO output shape is [1, num_features, num_detections] (transposed)
        // or [1, num_detections, num_features]. Handle both.
        let (num_dets, num_feats) = if shape.len() == 3 {
            if shape[1] < shape[2] {
                // [1, features, detections] → transpose
                (shape[2], shape[1])
            } else {
                (shape[1], shape[2])
            }
        } else {
            return Err(format!("Unexpected YOLO output shape: {shape:?}").into());
        };

        let data = tensor.as_slice().ok_or("Cannot get tensor slice")?;
        let transposed = shape.len() == 3 && shape[1] < shape[2];

        // 3. Parse detections
        let mut raw_dets = Vec::new();
        for i in 0..num_dets {
            let row = if transposed {
                // Read column i from transposed layout
                (0..num_feats)
                    .map(|f| data[f * num_dets + i])
                    .collect::<Vec<f32>>()
            } else {
                data[i * num_feats..(i + 1) * num_feats].to_vec()
            };

            // row format: [cx, cy, w, h, conf, kp0_x, kp0_y, kp0_conf, ...]
            if row.len() < 5 {
                continue;
            }
            let conf = row[4] as f64;
            if conf < self.confidence {
                continue;
            }

            let cx = row[0] as f64;
            let cy = row[1] as f64;
            let w = row[2] as f64;
            let h = row[3] as f64;

            // Convert from letterbox coords back to original frame coords
            let x1 = ((cx - w / 2.0) - pad_x as f64) / scale;
            let y1 = ((cy - h / 2.0) - pad_y as f64) / scale;
            let x2 = ((cx + w / 2.0) - pad_x as f64) / scale;
            let y2 = ((cy + h / 2.0) - pad_y as f64) / scale;

            // Parse keypoints if available, filtering by confidence
            let keypoints = if row.len() >= 5 + NUM_KEYPOINT_VALUES {
                let mut pts = [(0.0f64, 0.0f64); 5];
                for k in 0..5 {
                    let kconf = row[5 + k * 3 + 2] as f64;
                    if kconf >= KEYPOINT_CONF_THRESH {
                        let kx = row[5 + k * 3] as f64;
                        let ky = row[5 + k * 3 + 1] as f64;
                        // Map keypoints from letterbox coords to original
                        pts[k] = ((kx - pad_x as f64) / scale, (ky - pad_y as f64) / scale);
                    }
                    // else: pts[k] remains (0.0, 0.0), treated as invisible by FaceLandmarks
                }
                Some(pts)
            } else {
                None
            };

            raw_dets.push(RawDetection {
                x1,
                y1,
                x2,
                y2,
                confidence: conf,
                keypoints,
            });
        }

        // 4. NMS
        let filtered = nms(&mut raw_dets, NMS_IOU_THRESH);

        // 5. Track
        let tracker_dets: Vec<TrackerDetection> = filtered
            .iter()
            .map(|d| TrackerDetection {
                bbox: [d.x1, d.y1, d.x2, d.y2],
                score: d.confidence,
            })
            .collect();
        let tracks = self.tracker.update(&tracker_dets);

        // 6. Build regions — match tracks back to detections by IoU
        let mut regions = Vec::new();
        for track in &tracks {
            // Find the detection with highest IoU overlap to this track
            let best_det = filtered.iter().max_by(|a, b| {
                let iou_a = bbox_iou(&[a.x1, a.y1, a.x2, a.y2], &track.bbox);
                let iou_b = bbox_iou(&[b.x1, b.y1, b.x2, b.y2], &track.bbox);
                iou_a
                    .partial_cmp(&iou_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let landmarks = best_det.and_then(|d| d.keypoints).map(FaceLandmarks::new);

            let region = self.region_builder.build(
                (track.bbox[0], track.bbox[1], track.bbox[2], track.bbox[3]),
                fw,
                fh,
                landmarks.as_ref(),
                Some(track.id),
            );
            regions.push(region);
        }

        Ok(regions)
    }
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Letterbox-resize a frame to `target_size` × `target_size`.
///
/// Returns `(NCHW float32 tensor, scale, pad_x, pad_y)`.
fn letterbox(frame: &Frame, target_size: u32) -> (ndarray::Array4<f32>, f64, u32, u32) {
    let fw = frame.width() as f64;
    let fh = frame.height() as f64;
    let target = target_size as f64;

    let scale = (target / fw).min(target / fh);
    let new_w = (fw * scale).round() as u32;
    let new_h = (fh * scale).round() as u32;
    let pad_x = (target_size - new_w) / 2;
    let pad_y = (target_size - new_h) / 2;

    // Build padded image (filled with 114/255 gray, YOLO convention)
    let gray = 114.0f32 / 255.0;
    let mut tensor =
        ndarray::Array4::<f32>::from_elem((1, 3, target_size as usize, target_size as usize), gray);

    let src = frame.as_ndarray(); // [H, W, C] u8
    let src_h = frame.height() as usize;
    let src_w = frame.width() as usize;

    // Nearest-neighbor resize + copy into padded region
    for y in 0..new_h as usize {
        let src_y = ((y as f64 / scale) as usize).min(src_h - 1);
        for x in 0..new_w as usize {
            let src_x = ((x as f64 / scale) as usize).min(src_w - 1);
            let ty = pad_y as usize + y;
            let tx = pad_x as usize + x;
            for c in 0..3 {
                tensor[[0, c, ty, tx]] = src[[src_y, src_x, c]] as f32 / 255.0;
            }
        }
    }

    (tensor, scale, pad_x, pad_y)
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct RawDetection {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    confidence: f64,
    keypoints: Option<[(f64, f64); 5]>,
}

/// Greedy NMS: sort by confidence descending, suppress overlapping boxes.
fn nms(dets: &mut [RawDetection], iou_thresh: f64) -> Vec<RawDetection> {
    dets.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
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
            let iou = bbox_iou(
                &[dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2],
                &[dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2],
            );
            if iou > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}

fn bbox_iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letterbox_preserves_aspect_ratio() {
        // 200x100 frame → letterbox to 640x640
        // Scale = min(640/200, 640/100) = min(3.2, 6.4) = 3.2
        // new_w = 640, new_h = 320
        // pad_x = 0, pad_y = 160
        let data = vec![128u8; 200 * 100 * 3];
        let frame = Frame::new(data, 200, 100, 3, 0);
        let (tensor, scale, pad_x, pad_y) = letterbox(&frame, 640);

        assert_eq!(tensor.shape(), &[1, 3, 640, 640]);
        assert!((scale - 3.2).abs() < 0.01);
        assert_eq!(pad_x, 0);
        assert_eq!(pad_y, 160);
    }

    #[test]
    fn test_letterbox_square_frame() {
        let data = vec![128u8; 100 * 100 * 3];
        let frame = Frame::new(data, 100, 100, 3, 0);
        let (tensor, scale, pad_x, pad_y) = letterbox(&frame, 640);

        assert_eq!(tensor.shape(), &[1, 3, 640, 640]);
        assert!((scale - 6.4).abs() < 0.01);
        assert_eq!(pad_x, 0);
        assert_eq!(pad_y, 0);
    }

    #[test]
    fn test_letterbox_values_normalized() {
        // Use a wide frame so there's vertical padding
        let data = vec![255u8; 100 * 50 * 3];
        let frame = Frame::new(data, 100, 50, 3, 0);
        let (tensor, _, pad_x, pad_y) = letterbox(&frame, 640);

        // Wide frame: scale = 640/100 = 6.4, new_w=640, new_h=320, pad_y=160
        assert_eq!(pad_x, 0);
        assert!(pad_y > 0);

        // Check a pixel in the image region is ~1.0
        let y = pad_y as usize + 1;
        let x = pad_x as usize + 1;
        assert!((tensor[[0, 0, y, x]] - 1.0).abs() < 0.01);

        // Check a pad pixel (top-left, outside image region) is ~114/255
        let pad_val = 114.0 / 255.0;
        assert!((tensor[[0, 0, 0, 0]] - pad_val).abs() < 0.01);
    }

    #[test]
    fn test_nms_suppresses_overlapping() {
        let mut dets = vec![
            RawDetection {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 100.0,
                confidence: 0.9,
                keypoints: None,
            },
            RawDetection {
                x1: 5.0,
                y1: 5.0,
                x2: 105.0,
                y2: 105.0,
                confidence: 0.8,
                keypoints: None,
            },
        ];
        let kept = nms(&mut dets, 0.3);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_nms_keeps_non_overlapping() {
        let mut dets = vec![
            RawDetection {
                x1: 0.0,
                y1: 0.0,
                x2: 50.0,
                y2: 50.0,
                confidence: 0.9,
                keypoints: None,
            },
            RawDetection {
                x1: 200.0,
                y1: 200.0,
                x2: 250.0,
                y2: 250.0,
                confidence: 0.8,
                keypoints: None,
            },
        ];
        let kept = nms(&mut dets, 0.3);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_nms_empty_input() {
        let mut dets: Vec<RawDetection> = Vec::new();
        let kept = nms(&mut dets, 0.3);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_nms_confidence_ordering() {
        let mut dets = vec![
            RawDetection {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 100.0,
                confidence: 0.5,
                keypoints: None,
            },
            RawDetection {
                x1: 2.0,
                y1: 2.0,
                x2: 102.0,
                y2: 102.0,
                confidence: 0.9,
                keypoints: None,
            },
        ];
        let kept = nms(&mut dets, 0.3);
        // Higher confidence (0.9) should win
        assert_eq!(kept.len(), 1);
        assert!((kept[0].confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_iou_no_overlap() {
        assert_eq!(
            bbox_iou(&[0.0, 0.0, 10.0, 10.0], &[20.0, 20.0, 30.0, 30.0]),
            0.0
        );
    }

    #[test]
    fn test_bbox_iou_perfect() {
        let b = [0.0, 0.0, 10.0, 10.0];
        assert!((bbox_iou(&b, &b) - 1.0).abs() < 1e-9);
    }
}
