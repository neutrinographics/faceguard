/// ArcFace embedding-based face grouper using ONNX Runtime.
///
/// Computes face embeddings via an ArcFace ONNX model, then clusters
/// by cosine similarity using union-find.
use std::path::Path;
use std::sync::Mutex;

use crate::detection::domain::face_grouper::FaceGrouper;

/// Default cosine similarity threshold for grouping.
pub const DEFAULT_THRESHOLD: f64 = 0.4;

/// ArcFace model input size.
const INPUT_SIZE: usize = 112;

/// ArcFace pixel normalization: `(pixel - 127.5) / 127.5`.
const NORM_MEAN: f32 = 127.5;
const NORM_STD: f32 = 127.5;

/// ArcFace embedding face grouper.
pub struct EmbeddingFaceGrouper {
    session: Mutex<ort::session::Session>,
    threshold: f64,
}

impl EmbeddingFaceGrouper {
    /// Load an ArcFace ONNX model for embedding computation.
    pub fn new(model_path: &Path, threshold: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let session = ort::session::Session::builder()?.commit_from_file(model_path)?;
        Ok(Self {
            session: Mutex::new(session),
            threshold,
        })
    }

    /// Compute L2-normalized embedding for a single crop.
    fn embed(
        &self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let tensor = preprocess(rgb_data, width, height);
        let input_value = ort::value::Tensor::from_array(tensor)?;
        let mut session = self
            .session
            .lock()
            .map_err(|e| format!("Lock poisoned: {e}"))?;
        let outputs = session.run(ort::inputs![input_value])?;
        let output = &outputs[0];
        let embedding_array = output.try_extract_array::<f32>()?;
        let embedding_slice = embedding_array
            .as_slice()
            .ok_or("Cannot get embedding slice")?;

        // L2 normalize
        let mut embedding = embedding_slice.to_vec();
        l2_normalize(&mut embedding);
        Ok(embedding)
    }
}

impl FaceGrouper for EmbeddingFaceGrouper {
    fn group(
        &self,
        crops: &[(u32, &[u8], u32, u32)],
    ) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }

        // Compute embeddings
        let embeddings: Vec<Vec<f32>> = crops
            .iter()
            .map(|(_, data, w, h)| self.embed(data, *w, *h))
            .collect::<Result<Vec<_>, _>>()?;

        // Compute cosine similarity matrix + union-find
        let n = crops.len();
        let mut parent: Vec<usize> = (0..n).collect();

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
                if sim >= self.threshold {
                    union(&mut parent, i, j);
                }
            }
        }

        // Collect groups
        let mut groups: std::collections::HashMap<usize, Vec<u32>> =
            std::collections::HashMap::new();
        for (idx, (track_id, _, _, _)) in crops.iter().enumerate() {
            let root = find(&mut parent, idx);
            groups.entry(root).or_default().push(*track_id);
        }

        let mut result: Vec<Vec<u32>> = groups.into_values().collect();
        for g in &mut result {
            g.sort();
        }
        result.sort_by_key(|g| g[0]);
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Resize crop to 112×112, normalize to `(pixel - 127.5) / 127.5`, NCHW layout.
fn preprocess(rgb_data: &[u8], width: u32, height: u32) -> ndarray::Array4<f32> {
    let src_w = width as usize;
    let src_h = height as usize;

    let mut tensor = ndarray::Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));

    for y in 0..INPUT_SIZE {
        let src_y = (((y as f64 + 0.5) * src_h as f64 / INPUT_SIZE as f64) as usize).min(src_h - 1);
        for x in 0..INPUT_SIZE {
            let src_x =
                (((x as f64 + 0.5) * src_w as f64 / INPUT_SIZE as f64) as usize).min(src_w - 1);
            let offset = (src_y * src_w + src_x) * 3;
            if offset + 2 < rgb_data.len() {
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (rgb_data[offset + c] as f32 - NORM_MEAN) / NORM_STD;
                }
            }
        }
    }

    tensor
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

/// L2-normalize a vector in-place.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two L2-normalized vectors (= dot product).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

// ---------------------------------------------------------------------------
// Union-Find
// ---------------------------------------------------------------------------

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]]; // path halving
        i = parent[i];
    }
    i
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize_unit_vector() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_already_normalized() {
        let mut v = vec![1.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        // Should remain zero (no division by zero)
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![0.6, 0.8];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_preprocess_shape() {
        let data = vec![128u8; 50 * 50 * 3];
        let tensor = preprocess(&data, 50, 50);
        assert_eq!(tensor.shape(), &[1, 3, 112, 112]);
    }

    #[test]
    fn test_preprocess_normalization() {
        // All 127 → (127 - 127.5) / 127.5 ≈ -0.00392
        let data = vec![127u8; 10 * 10 * 3];
        let tensor = preprocess(&data, 10, 10);
        let val = tensor[[0, 0, 0, 0]];
        let expected = (127.0 - 127.5) / 127.5;
        assert!((val - expected).abs() < 0.01);
    }

    #[test]
    fn test_preprocess_normalization_max() {
        // All 255 → (255 - 127.5) / 127.5 = 1.0
        let data = vec![255u8; 10 * 10 * 3];
        let tensor = preprocess(&data, 10, 10);
        let val = tensor[[0, 0, 0, 0]];
        assert!((val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_preprocess_normalization_min() {
        // All 0 → (0 - 127.5) / 127.5 = -1.0
        let data = vec![0u8; 10 * 10 * 3];
        let tensor = preprocess(&data, 10, 10);
        let val = tensor[[0, 0, 0, 0]];
        assert!((val - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_union_find_transitive() {
        let mut parent = vec![0, 1, 2];
        union(&mut parent, 0, 1);
        union(&mut parent, 1, 2);
        // All should be in the same group
        assert_eq!(find(&mut parent, 0), find(&mut parent, 2));
    }

    #[test]
    fn test_union_find_separate() {
        let mut parent = vec![0, 1, 2, 3];
        union(&mut parent, 0, 1);
        union(&mut parent, 2, 3);
        assert_ne!(find(&mut parent, 0), find(&mut parent, 2));
    }
}
