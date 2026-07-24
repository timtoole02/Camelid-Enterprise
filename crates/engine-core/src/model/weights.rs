//! Materialized weights.
//!
//! [`LlamaWeights::load`] reads every bound descriptor into a [`CpuTensor`]:
//! norms and RoPE frequencies decode to f32; 2-D Q8_0 linears materialize as
//! plain Q8_0 blocks (empty f32 `data`) so the block-streaming matmul sees the
//! same accumulation order as the reference. The token embedding's shape is
//! reinterpreted to `[vocab, embedding]` for the lookup; a tied output
//! projection is read as a separate tensor from the same bytes, kept in its
//! stored orientation.

use crate::model::binding::LlamaTensorBinding;
use crate::model::config::{DenseLlamaDims, LlamaModelConfig};
use crate::tensor::{CpuTensor, TensorStore};
use crate::{EngineError, Result};

/// Materialized per-layer weights.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaLayerWeights {
    pub attention_norm: CpuTensor,
    pub attention_q: CpuTensor,
    pub attention_k: CpuTensor,
    pub attention_v: CpuTensor,
    pub attention_output: CpuTensor,
    /// Per-head RMSNorm weight (`[head_dim]`, f32) for the Q projection; `Some`
    /// only for QK-norm rows (Qwen3), `None` otherwise.
    pub attention_q_norm: Option<CpuTensor>,
    /// Per-head RMSNorm weight for the K projection, bound in lockstep with
    /// [`Self::attention_q_norm`].
    pub attention_k_norm: Option<CpuTensor>,
    pub ffn_norm: CpuTensor,
    pub ffn_gate: CpuTensor,
    pub ffn_up: CpuTensor,
    pub ffn_down: CpuTensor,
}

/// Materialized top-level weights for a dense decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaWeights {
    pub token_embedding: CpuTensor,
    pub output_norm: CpuTensor,
    /// The output (lm_head) projection. Always present here; for a tied model
    /// it is a distinct tensor read from the token-embedding bytes.
    pub output: Option<CpuTensor>,
    pub rope_freqs: Option<CpuTensor>,
    pub layers: Vec<LlamaLayerWeights>,
}

impl LlamaWeights {
    /// The projection used for the final logits: the loaded output tensor, or
    /// the token embedding if none was loaded.
    pub fn output_projection(&self) -> &CpuTensor {
        self.output.as_ref().unwrap_or(&self.token_embedding)
    }

    pub fn load(store: &TensorStore, binding: &LlamaTensorBinding) -> Result<Self> {
        // Every 2-D Q8_0 linear is retained as plain RAM-resident blocks so the
        // block-streaming CPU matmul reproduces the reference accumulation
        // order; non-Q8_0 or non-2-D tensors fall back to eager f32.
        let load_linear = |name: &str| store.load_q8_0_block_backed_linear(name);

        let token_embedding = normalize_token_embedding_shape(
            load_linear(&binding.token_embedding.name)?,
            &binding.token_embedding.name,
        )?;
        let output_norm = store.load_cpu_f32(&binding.output_norm.name)?;
        let output = if binding.output_is_tied_embedding {
            // Tied projection: read the same token-embedding bytes into a
            // distinct tensor named output.weight, kept in stored orientation
            // (NOT the reinterpreted [vocab, embedding] token embedding).
            Some(store.load_q8_0_block_backed_linear_as(
                &binding.token_embedding.name,
                "output.weight",
            )?)
        } else {
            Some(load_linear(&binding.output.name)?)
        };
        let rope_freqs = binding
            .rope_freqs
            .as_ref()
            .map(|desc| store.load_cpu_f32(&desc.name))
            .transpose()?;

        let mut layers = Vec::with_capacity(binding.layers.len());
        for layer in &binding.layers {
            let attention_q_norm = layer
                .attention_q_norm
                .as_ref()
                .map(|desc| store.load_cpu_f32(&desc.name))
                .transpose()?;
            let attention_k_norm = layer
                .attention_k_norm
                .as_ref()
                .map(|desc| store.load_cpu_f32(&desc.name))
                .transpose()?;
            layers.push(LlamaLayerWeights {
                attention_norm: store.load_cpu_f32(&layer.attention_norm.name)?,
                attention_q: load_linear(&layer.attention_q.name)?,
                attention_k: load_linear(&layer.attention_k.name)?,
                attention_v: load_linear(&layer.attention_v.name)?,
                attention_output: load_linear(&layer.attention_output.name)?,
                attention_q_norm,
                attention_k_norm,
                ffn_norm: store.load_cpu_f32(&layer.ffn_norm.name)?,
                ffn_gate: load_linear(&layer.ffn.gate.name)?,
                ffn_up: load_linear(&layer.ffn.up.name)?,
                ffn_down: load_linear(&layer.ffn.down.name)?,
            });
        }

        Ok(Self {
            token_embedding,
            output_norm,
            output,
            rope_freqs,
            layers,
        })
    }

    /// Re-validate the materialized tensors against the derived dims. The
    /// token embedding is checked in its reinterpreted `[vocab, embedding]`
    /// orientation; linears accept either orientation.
    pub fn validate_dense_shapes(&self, config: &LlamaModelConfig) -> Result<()> {
        let dims = DenseLlamaDims::from_config(config)?;
        require_tensor_shape(
            &self.token_embedding,
            &[dims.vocab_size, dims.embedding_length],
            "token embedding",
        )?;
        require_tensor_shape(&self.output_norm, &[dims.embedding_length], "output norm")?;
        require_matrix_shape(
            self.output_projection(),
            dims.embedding_length,
            dims.vocab_size,
            "output projection",
        )?;
        if let Some(rope_freqs) = &self.rope_freqs {
            let rope_dim = config.rope_dimension_count.unwrap_or(dims.head_dim as u32) as usize;
            validate_rope_frequency_tensor(rope_freqs, rope_dim)?;
        }

        if self.layers.len() != dims.block_count {
            return Err(EngineError::ShapeMismatch(format!(
                "config block count {} does not match loaded layer count {}",
                dims.block_count,
                self.layers.len()
            )));
        }

        for (idx, layer) in self.layers.iter().enumerate() {
            require_tensor_shape(
                &layer.attention_norm,
                &[dims.embedding_length],
                &format!("layer {idx} attention norm"),
            )?;
            require_matrix_shape(
                &layer.attention_q,
                dims.embedding_length,
                dims.q_width,
                &format!("layer {idx} attention q"),
            )?;
            require_matrix_shape(
                &layer.attention_k,
                dims.embedding_length,
                dims.kv_width,
                &format!("layer {idx} attention k"),
            )?;
            require_matrix_shape(
                &layer.attention_v,
                dims.embedding_length,
                dims.kv_width,
                &format!("layer {idx} attention v"),
            )?;
            require_matrix_shape(
                &layer.attention_output,
                dims.q_width,
                dims.embedding_length,
                &format!("layer {idx} attention output"),
            )?;
            match (&layer.attention_q_norm, &layer.attention_k_norm) {
                (Some(q_norm), Some(k_norm)) => {
                    require_tensor_shape(
                        q_norm,
                        &[dims.head_dim],
                        &format!("layer {idx} attention q_norm"),
                    )?;
                    require_tensor_shape(
                        k_norm,
                        &[dims.head_dim],
                        &format!("layer {idx} attention k_norm"),
                    )?;
                }
                (None, None) => {}
                _ => {
                    return Err(EngineError::ShapeMismatch(format!(
                        "layer {idx} has exactly one of attn_q_norm/attn_k_norm loaded; QK-norm \
                         weights must be present as a pair"
                    )));
                }
            }
            require_tensor_shape(
                &layer.ffn_norm,
                &[dims.embedding_length],
                &format!("layer {idx} ffn norm"),
            )?;
            require_matrix_shape(
                &layer.ffn_gate,
                dims.embedding_length,
                dims.feed_forward_length,
                &format!("layer {idx} ffn gate"),
            )?;
            require_matrix_shape(
                &layer.ffn_up,
                dims.embedding_length,
                dims.feed_forward_length,
                &format!("layer {idx} ffn up"),
            )?;
            require_matrix_shape(
                &layer.ffn_down,
                dims.feed_forward_length,
                dims.embedding_length,
                &format!("layer {idx} ffn down"),
            )?;
        }

        Ok(())
    }
}

/// Reinterpret the token embedding from the stored `[embedding, vocab]` to the
/// runtime `[vocab, embedding]` the lookup expects. The bytes are already
/// token-major, so this swaps the declared dims only — never a numerical
/// transpose — and only when `dims[0] < dims[1]`.
fn normalize_token_embedding_shape(mut tensor: CpuTensor, name: &str) -> Result<CpuTensor> {
    if tensor.rank() != 2 {
        return Err(EngineError::ShapeMismatch(format!(
            "token embedding tensor {name} expected rank 2, got {:?}",
            tensor.shape.dims
        )));
    }
    if tensor.shape.dims[0] < tensor.shape.dims[1] {
        tensor.shape.dims.swap(0, 1);
    }
    Ok(tensor)
}

fn require_tensor_shape(tensor: &CpuTensor, expected: &[usize], role: &str) -> Result<()> {
    if tensor.shape.dims != expected {
        return Err(EngineError::ShapeMismatch(format!(
            "{role} tensor {} expected shape {:?}, got {:?}",
            tensor.name, expected, tensor.shape.dims
        )));
    }
    Ok(())
}

fn require_matrix_shape(
    tensor: &CpuTensor,
    input_width: usize,
    output_width: usize,
    role: &str,
) -> Result<()> {
    let direct = [input_width, output_width];
    let transposed = [output_width, input_width];
    if tensor.shape.dims.as_slice() != direct && tensor.shape.dims.as_slice() != transposed {
        return Err(EngineError::ShapeMismatch(format!(
            "{role} tensor {} expected shape {:?} or {:?}, got {:?}",
            tensor.name, direct, transposed, tensor.shape.dims
        )));
    }
    Ok(())
}

fn validate_rope_frequency_tensor(rope_freqs: &CpuTensor, rope_dim: usize) -> Result<()> {
    if rope_dim == 0 || !rope_dim.is_multiple_of(2) {
        return Err(EngineError::InvalidModelMetadata(format!(
            "RoPE dimension count {rope_dim} must be even and greater than zero"
        )));
    }
    let expected_count = rope_dim / 2;
    if rope_freqs.shape.dims != [expected_count] {
        return Err(EngineError::InvalidModelMetadata(format!(
            "rope_freqs.weight expected shape [{expected_count}], got {:?}",
            rope_freqs.shape.dims
        )));
    }
    if let Some((idx, frequency)) = rope_freqs
        .data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, frequency)| *frequency <= 0.0 || !frequency.is_finite())
    {
        return Err(EngineError::InvalidModelMetadata(format!(
            "rope_freqs.weight[{idx}] frequency factor {frequency} must be finite and positive"
        )));
    }
    Ok(())
}
