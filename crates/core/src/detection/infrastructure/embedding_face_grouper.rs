/// ArcFace embedding-based face grouper using ONNX Runtime.
///
/// Clusters faces by cosine similarity of ArcFace embeddings. Preferred
/// over histogram grouping when model availability and latency allow.
use std::path::Path;
use std::sync::Mutex;

use crate::detection::domain::face_grouper::FaceGrouper;
use crate::detection::infrastructure::math;

pub const DEFAULT_THRESHOLD: f64 = 0.4;

const INPUT_SIZE: usize = 112;
const NORM_MEAN: f32 = 127.5;
const NORM_STD: f32 = 127.5;

pub struct EmbeddingFaceGrouper {
    session: Mutex<ort::session::Session>,
    threshold: f64,
}

impl EmbeddingFaceGrouper {
    pub fn new(model_path: &Path, threshold: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let intra_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let session = ort::session::Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_inter_threads(1)?
            .with_intra_threads(intra_threads)?
            .with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;
        Ok(Self {
            session: Mutex::new(session),
            threshold,
        })
    }

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
        let embedding_array = outputs[0].try_extract_array::<f32>()?;
        let embedding_slice = embedding_array
            .as_slice()
            .ok_or("Cannot get embedding slice")?;

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

        let embeddings: Vec<Vec<f32>> = crops
            .iter()
            .map(|(_, data, w, h)| self.embed(data, *w, *h))
            .collect::<Result<Vec<_>, _>>()?;

        let n = crops.len();
        let mut parent: Vec<usize> = (0..n).collect();

        for i in 0..n {
            for j in (i + 1)..n {
                if cosine_similarity(&embeddings[i], &embeddings[j]) >= self.threshold {
                    math::union(&mut parent, i, j);
                }
            }
        }

        let entries: Vec<(usize, u32)> = crops
            .iter()
            .enumerate()
            .map(|(idx, (track_id, _, _, _))| (idx, *track_id))
            .collect();
        Ok(math::collect_groups(&mut parent, &entries))
    }
}

/// Resize crop to 112x112, normalize, NCHW layout.
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

pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Dot product of L2-normalized vectors equals cosine similarity.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

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
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![0.6, 0.8];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_preprocess_shape() {
        let data = vec![128u8; 50 * 50 * 3];
        let tensor = preprocess(&data, 50, 50);
        assert_eq!(tensor.shape(), &[1, 3, 112, 112]);
    }

    #[test]
    fn test_preprocess_normalization() {
        let data = vec![127u8; 10 * 10 * 3];
        let tensor = preprocess(&data, 10, 10);
        let val = tensor[[0, 0, 0, 0]];
        let expected = (127.0 - 127.5) / 127.5;
        assert!((val - expected).abs() < 0.01);
    }

    #[test]
    fn test_preprocess_normalization_max() {
        let data = vec![255u8; 10 * 10 * 3];
        let tensor = preprocess(&data, 10, 10);
        assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_preprocess_normalization_min() {
        let data = vec![0u8; 10 * 10 * 3];
        let tensor = preprocess(&data, 10, 10);
        assert!((tensor[[0, 0, 0, 0]] - (-1.0)).abs() < 0.01);
    }
}
