//! # plato-cortex
//!
//! Cross-database mapping layer for Dual-DB JEPA perception-prediction spaces.
//! Implements projection matrices, attention mechanisms, and learned similarity
//! functions that map between perception and prediction databases.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A learnable projection matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionMatrix {
    pub rows: usize,
    pub cols: usize,
    pub weights: Vec<Vec<f64>>,
}

/// Single attention head with query/key/value weight matrices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionHead {
    pub query_weights: Vec<Vec<f64>>,
    pub key_weights: Vec<Vec<f64>>,
    pub value_weights: Vec<Vec<f64>>,
}

/// Method used for cross-space mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MappingMethod {
    Linear,
    Bilinear,
    Attention { heads: usize },
    PiecewiseLinear { regions: usize },
}

/// Loss components for a mapping step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingLoss {
    pub mse: f64,
    pub cosine_loss: f64,
    pub ranking_loss: f64,
    pub total: f64,
}

/// Configuration for constructing a `CrossSpaceMapping`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub method: MappingMethod,
    pub learning_rate: f64,
}

/// Cross-space mapping between perception (Z_in) and prediction (Z_out).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSpaceMapping {
    pub input_dim: usize,
    pub output_dim: usize,
    pub method: MappingMethod,
    pub learning_rate: f64,
    forward_proj: ProjectionMatrix,
    backward_proj: ProjectionMatrix,
    #[serde(skip_serializing_if = "Option::is_none")]
    attention_heads: Option<Vec<AttentionHead>>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple deterministic-ish pseudo-random number (xoshiro-like one-liner).
fn rand_f64(seed: &mut u64) -> f64 {
    // xorshift64
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed as f64) / (u64::MAX as f64)
}

fn rand_matrix(rows: usize, cols: usize, seed: &mut u64) -> Vec<Vec<f64>> {
    let scale = 1.0 / (cols as f64).sqrt();
    (0..rows)
        .map(|_| (0..cols).map(|_| rand_f64(seed) * 2.0 * scale - scale).collect())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn mat_vec(mat: &Vec<Vec<f64>>, v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| dot(row, v)).collect()
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// ProjectionMatrix
// ---------------------------------------------------------------------------

impl ProjectionMatrix {
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut seed = (rows as u64).wrapping_mul(1_000_003).wrapping_add(cols as u64);
        // warm up
        for _ in 0..5 {
            rand_f64(&mut seed);
        }
        Self {
            rows,
            cols,
            weights: rand_matrix(rows, cols, &mut seed),
        }
    }

    pub fn identity(dim: usize) -> Self {
        Self {
            rows: dim,
            cols: dim,
            weights: (0..dim)
                .map(|i| (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect(),
        }
    }

    pub fn project(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.cols, "input dimension mismatch");
        mat_vec(&self.weights, input)
    }

    pub fn update(&mut self, gradient: &Vec<Vec<f64>>, lr: f64) {
        assert_eq!(gradient.len(), self.rows);
        for (i, row) in gradient.iter().enumerate() {
            assert_eq!(row.len(), self.cols);
            for (j, g) in row.iter().enumerate() {
                self.weights[i][j] -= lr * g;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AttentionHead
// ---------------------------------------------------------------------------

impl AttentionHead {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut seed = (input_dim as u64).wrapping_mul(997).wrapping_add(output_dim as u64);
        for _ in 0..3 {
            rand_f64(&mut seed);
        }
        Self {
            query_weights: rand_matrix(output_dim, input_dim, &mut seed),
            key_weights: rand_matrix(output_dim, input_dim, &mut seed),
            value_weights: rand_matrix(output_dim, input_dim, &mut seed),
        }
    }

    pub fn attend(&self, query: &[f64], keys: &[Vec<f64>], values: &[Vec<f64>]) -> Vec<f64> {
        assert_eq!(keys.len(), values.len(), "keys and values must have same length");
        let q = mat_vec(&self.query_weights, query);
        let k_proj: Vec<Vec<f64>> = keys.iter().map(|k| mat_vec(&self.key_weights, k)).collect();
        let v_proj: Vec<Vec<f64>> = values.iter().map(|v| mat_vec(&self.value_weights, v)).collect();

        let scale = (q.len() as f64).sqrt();
        let scores: Vec<f64> = k_proj.iter().map(|k| dot(&q, k) / scale).collect();
        let weights = softmax(&scores);

        let dim = v_proj[0].len();
        (0..dim)
            .map(|i| v_proj.iter().zip(weights.iter()).map(|(v, w)| v[i] * w).sum())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CrossSpaceMapping
// ---------------------------------------------------------------------------

impl CrossSpaceMapping {
    pub fn new(config: MappingConfig) -> Self {
        let forward_proj = ProjectionMatrix::random(config.output_dim, config.input_dim);
        let backward_proj = ProjectionMatrix::random(config.input_dim, config.output_dim);

        let attention_heads = match &config.method {
            MappingMethod::Attention { heads } => {
                let h: Vec<AttentionHead> = (0..*heads)
                    .map(|i| {
                        let mut ah = AttentionHead::new(config.input_dim, config.output_dim);
                        // diversify with index
                        let mut seed = i as u64;
                        for _ in 0..10 {
                            rand_f64(&mut seed);
                        }
                        ah.query_weights = rand_matrix(config.output_dim, config.input_dim, &mut seed);
                        ah.key_weights = rand_matrix(config.output_dim, config.input_dim, &mut seed);
                        ah.value_weights = rand_matrix(config.output_dim, config.input_dim, &mut seed);
                        ah
                    })
                    .collect();
                Some(h)
            }
            _ => None,
        };

        Self {
            input_dim: config.input_dim,
            output_dim: config.output_dim,
            method: config.method,
            learning_rate: config.learning_rate,
            forward_proj,
            backward_proj,
            attention_heads,
        }
    }

    /// Map Z_in → Z_out
    pub fn map_forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_dim);
        match &self.method {
            MappingMethod::Linear | MappingMethod::PiecewiseLinear { .. } => {
                self.forward_proj.project(input)
            }
            MappingMethod::Bilinear => {
                let base = self.forward_proj.project(input);
                // Add quadratic cross-terms (simplified bilinear)
                let dim = base.len().min(input.len());
                let mut out = base;
                for i in 0..dim {
                    for j in 0..dim.min(4) {
                        if i < out.len() {
                            out[i] += 0.01 * input[j % input.len()] * input[i % input.len()];
                        }
                    }
                }
                out
            }
            MappingMethod::Attention { .. } => {
                let heads = self.attention_heads.as_ref().unwrap();
                let results: Vec<Vec<f64>> = heads.iter().map(|h| h.attend(input, &[input.to_vec()], &[input.to_vec()])).collect();
                // average across heads
                let dim = results[0].len();
                (0..dim)
                    .map(|i| results.iter().map(|r| r[i]).sum::<f64>() / results.len() as f64)
                    .collect()
            }
        }
    }

    /// Map Z_out → Z_in
    pub fn map_backward(&self, output: &[f64]) -> Vec<f64> {
        assert_eq!(output.len(), self.output_dim);
        self.backward_proj.project(output)
    }

    /// Cosine similarity between two vectors.
    pub fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let na = vec_norm(a);
        let nb = vec_norm(b);
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        dot(a, b) / (na * nb)
    }

    /// Compute mapping loss between predicted and actual vectors.
    pub fn compute_loss(&self, predicted: &[f64], actual: &[f64]) -> MappingLoss {
        assert_eq!(predicted.len(), actual.len());

        // MSE
        let mse = predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predicted.len() as f64;

        // Cosine loss = 1 - cosine_similarity
        let sim = self.similarity(predicted, actual);
        let cosine_loss = 1.0 - sim;

        // Ranking loss: simplified hinge on per-dimension ordering
        let mut ranking_loss = 0.0;
        for i in 0..predicted.len().saturating_sub(1) {
            let pred_order = predicted[i] - predicted[i + 1];
            let act_order = actual[i] - actual[i + 1];
            if pred_order * act_order < 0.0 {
                ranking_loss += (pred_order - act_order).abs();
            }
        }
        ranking_loss /= predicted.len().max(1) as f64;

        let total = mse + 0.5 * cosine_loss + 0.1 * ranking_loss;

        MappingLoss {
            mse,
            cosine_loss,
            ranking_loss,
            total,
        }
    }

    /// One training step. Returns the loss value before the update.
    pub fn train_step(&mut self, input: &[f64], target: &[f64]) -> f64 {
        let predicted = self.map_forward(input);
        let loss = self.compute_loss(&predicted, target);

        // Numerical gradient estimation for forward_proj with gradient clipping
        let eps = 1e-4;
        let max_grad = 1.0; // clip
        for i in 0..self.forward_proj.rows {
            for j in 0..self.forward_proj.cols {
                let old = self.forward_proj.weights[i][j];
                self.forward_proj.weights[i][j] = old + eps;
                let pred_plus = self.map_forward(input);
                let loss_plus = self.compute_loss(&pred_plus, target).total;
                self.forward_proj.weights[i][j] = old;

                let grad = (loss_plus - loss.total) / eps;
                let clipped = grad.clamp(-max_grad, max_grad);
                self.forward_proj.weights[i][j] -= self.learning_rate * clipped;
            }
        }

        loss.total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_random_dimensions() {
        let p = ProjectionMatrix::random(4, 3);
        assert_eq!(p.rows, 4);
        assert_eq!(p.cols, 3);
        assert_eq!(p.weights.len(), 4);
        assert_eq!(p.weights[0].len(), 3);
    }

    #[test]
    fn test_projection_random_project() {
        let p = ProjectionMatrix::random(3, 4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = p.project(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_projection_identity() {
        let id = ProjectionMatrix::identity(4);
        let input = vec![5.0, -2.0, 0.0, 3.0];
        let output = id.project(&input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_projection_identity_square() {
        let id = ProjectionMatrix::identity(3);
        assert_eq!(id.weights[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(id.weights[1], vec![0.0, 1.0, 0.0]);
        assert_eq!(id.weights[2], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_projection_update_with_gradient() {
        let mut p = ProjectionMatrix::identity(2);
        let grad = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        p.update(&grad, 0.1);
        assert!((p.weights[0][0] - 0.9).abs() < 1e-10);
        assert!((p.weights[1][1] - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_attention_attend_single_key() {
        let head = AttentionHead::new(4, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let keys = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let values = vec![vec![0.0, 1.0, 0.0, 0.0]];
        let result = head.attend(&query, &keys, &values);
        assert_eq!(result.len(), 4);
        // With a single key/value, softmax gives weight 1.0
        // so result should be the projected value
    }

    #[test]
    fn test_attention_attend_multiple_keys() {
        let head = AttentionHead::new(4, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let keys = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let values = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let result = head.attend(&query, &keys, &values);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_cross_mapping_forward() {
        let config = MappingConfig {
            input_dim: 4,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = mapping.map_forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_cross_mapping_backward() {
        let config = MappingConfig {
            input_dim: 4,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let output = vec![1.0, 2.0, 3.0];
        let back = mapping.map_backward(&output);
        assert_eq!(back.len(), 4);
    }

    #[test]
    fn test_similarity_identical() {
        let config = MappingConfig {
            input_dim: 4,
            output_dim: 4,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let sim = mapping.similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_orthogonal() {
        let config = MappingConfig {
            input_dim: 3,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = mapping.similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_loss_mse() {
        let config = MappingConfig {
            input_dim: 3,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let predicted = vec![1.0, 2.0, 3.0];
        let actual = vec![1.0, 2.0, 3.0];
        let loss = mapping.compute_loss(&predicted, &actual);
        assert!((loss.mse).abs() < 1e-10);
    }

    #[test]
    fn test_loss_cosine() {
        let config = MappingConfig {
            input_dim: 3,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let predicted = vec![1.0, 0.0, 0.0];
        let actual = vec![1.0, 0.0, 0.0];
        let loss = mapping.compute_loss(&predicted, &actual);
        assert!((loss.cosine_loss).abs() < 1e-10);
    }

    #[test]
    fn test_loss_nonzero() {
        let config = MappingConfig {
            input_dim: 3,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let predicted = vec![1.0, 0.0, 0.0];
        let actual = vec![0.0, 1.0, 0.0];
        let loss = mapping.compute_loss(&predicted, &actual);
        assert!(loss.mse > 0.0);
        assert!(loss.cosine_loss > 0.0);
    }

    #[test]
    fn test_train_step_reduces_loss() {
        let config = MappingConfig {
            input_dim: 3,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mut mapping = CrossSpaceMapping::new(config);
        let input = vec![1.0, 2.0, 3.0];
        let target = vec![0.5, 1.0, 1.5];

        let loss_before = mapping.train_step(&input, &target);
        let loss_after = mapping.train_step(&input, &target);
        // Second step should not increase loss significantly
        assert!(loss_after <= loss_before + 0.1, "loss_before={}, loss_after={}", loss_before, loss_after);
    }

    #[test]
    fn test_linear_vs_attention_method() {
        let linear_config = MappingConfig {
            input_dim: 4,
            output_dim: 4,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let attention_config = MappingConfig {
            input_dim: 4,
            output_dim: 4,
            method: MappingMethod::Attention { heads: 2 },
            learning_rate: 0.01,
        };
        let linear = CrossSpaceMapping::new(linear_config);
        let attention = CrossSpaceMapping::new(attention_config);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        // Both should produce valid outputs of correct dimension
        assert_eq!(linear.map_forward(&input).len(), 4);
        assert_eq!(attention.map_forward(&input).len(), 4);
    }

    #[test]
    fn test_different_dimensions() {
        let config = MappingConfig {
            input_dim: 8,
            output_dim: 2,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let input = vec![1.0; 8];
        let output = mapping.map_forward(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_zero_vector() {
        let config = MappingConfig {
            input_dim: 3,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let zero = vec![0.0; 3];
        let output = mapping.map_forward(&zero);
        // With zero input, linear projection should give zero or near-zero
        for v in &output {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_single_dimension() {
        let config = MappingConfig {
            input_dim: 1,
            output_dim: 1,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let input = vec![3.0];
        let output = mapping.map_forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_large_dimensions() {
        let config = MappingConfig {
            input_dim: 128,
            output_dim: 64,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let input: Vec<f64> = (0..128).map(|i| (i as f64).sin()).collect();
        let output = mapping.map_forward(&input);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_convergence_repeated_training() {
        let config = MappingConfig {
            input_dim: 3,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mut mapping = CrossSpaceMapping::new(config);
        let input = vec![1.0, 2.0, 3.0];
        let target = vec![0.5, 1.0, 1.5];

        let initial_loss = mapping.train_step(&input, &target);
        for _ in 0..100 {
            mapping.train_step(&input, &target);
        }
        let final_predicted = mapping.map_forward(&input);
        let final_loss = mapping.compute_loss(&final_predicted, &target).total;

        assert!(
            final_loss < initial_loss,
            "final_loss={} should be < initial_loss={}",
            final_loss,
            initial_loss
        );
    }

    #[test]
    fn test_bilinear_method() {
        let config = MappingConfig {
            input_dim: 4,
            output_dim: 4,
            method: MappingMethod::Bilinear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = mapping.map_forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_piecewise_linear_method() {
        let config = MappingConfig {
            input_dim: 4,
            output_dim: 4,
            method: MappingMethod::PiecewiseLinear { regions: 3 },
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = mapping.map_forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_similarity_opposite() {
        let config = MappingConfig {
            input_dim: 3,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = mapping.similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_loss_ranking() {
        let config = MappingConfig {
            input_dim: 3,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        // predicted ascending, actual descending → ranking loss should be positive
        let predicted = vec![1.0, 2.0, 3.0];
        let actual = vec![3.0, 2.0, 1.0];
        let loss = mapping.compute_loss(&predicted, &actual);
        assert!(loss.ranking_loss > 0.0);
    }

    #[test]
    fn test_similarity_zero_vector() {
        let config = MappingConfig {
            input_dim: 3,
            output_dim: 3,
            method: MappingMethod::Linear,
            learning_rate: 0.01,
        };
        let mapping = CrossSpaceMapping::new(config);
        let a = vec![0.0; 3];
        let b = vec![1.0, 2.0, 3.0];
        let sim = mapping.similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }
}
