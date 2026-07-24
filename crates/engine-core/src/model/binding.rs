//! Descriptor-level weight binding.
//!
//! [`LlamaTensorBinding`] resolves the dense decoder's tensors by their
//! canonical GGUF names and validates every descriptor's declared shape
//! against the derived [`DenseLlamaDims`] before a single byte of tensor data
//! is read. Weight orientation is never canonicalized here: a 2-D linear may
//! be stored `[in, out]` or `[out, in]`, and both are accepted, so the forward
//! matmul detects the orientation at runtime.

use serde::Serialize;

use crate::gguf::{GgufFile, TensorDescriptor};
use crate::model::config::{architecture_key, DenseLlamaDims, LlamaModelConfig};
use crate::{EngineError, Result};

/// Per-layer descriptor set for one dense decoder block.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaLayerTensors {
    pub attention_norm: TensorDescriptor,
    pub attention_q: TensorDescriptor,
    pub attention_k: TensorDescriptor,
    pub attention_v: TensorDescriptor,
    pub attention_output: TensorDescriptor,
    /// Per-head RMSNorm applied to the Q projection after reshape-to-heads and
    /// before RoPE. `Some` only for architectures that use QK-norm (Qwen3);
    /// `None` for plain Llama-family rows. Shape `[head_dim]` when present.
    pub attention_q_norm: Option<TensorDescriptor>,
    /// Per-head RMSNorm for the K projection; bound in lockstep with
    /// [`Self::attention_q_norm`] (both `Some` or both `None`).
    pub attention_k_norm: Option<TensorDescriptor>,
    pub ffn_norm: TensorDescriptor,
    pub ffn: LlamaFfnTensors,
}

/// Dense feed-forward tensors: gate and up are two separate projections of the
/// same normalized input, and down projects the activated intermediate back to
/// the hidden width.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaFfnTensors {
    pub gate: TensorDescriptor,
    pub up: TensorDescriptor,
    pub down: TensorDescriptor,
}

/// Top-level bound descriptor set for a dense decoder.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaTensorBinding {
    pub token_embedding: TensorDescriptor,
    pub output_norm: TensorDescriptor,
    /// The output (lm_head) projection descriptor. When the model ships no
    /// `output.weight`, this holds the token-embedding descriptor and
    /// [`Self::output_is_tied_embedding`] is set.
    pub output: TensorDescriptor,
    pub output_is_tied_embedding: bool,
    pub rope_freqs: Option<TensorDescriptor>,
    pub layers: Vec<LlamaLayerTensors>,
}

impl LlamaTensorBinding {
    pub fn bind(gguf: &GgufFile, config: &LlamaModelConfig) -> Result<Self> {
        let token_embedding = required_tensor(gguf, "token_embd.weight")?;
        let output_norm = required_tensor(gguf, "output_norm.weight")?;
        // Tied-embedding detection: an absent output.weight means the model
        // reuses the token embedding as the output projection.
        let (output, output_is_tied_embedding) = match find_tensor(gguf, "output.weight") {
            Some(desc) => (desc.clone(), false),
            None => (token_embedding.clone(), true),
        };
        let rope_freqs = find_tensor(gguf, "rope_freqs.weight").cloned();

        // Per-architecture QK-norm classification. Qwen3 applies a per-head
        // RMSNorm to Q and K after the projections and before RoPE; the plain
        // Llama-family rows do not. Classify every architecture that reaches
        // this binder so a model can never be silently mis-bound in either
        // direction (carrying QK-norm weights the forward path would drop, or
        // fabricating them where none exist).
        let architecture = gguf.architecture().unwrap_or_default();
        let expects_qk_norm = architecture == "qwen3";
        let forbids_qk_norm = matches!(architecture, "llama" | "mistral" | "qwen2");

        // Qwen3 sets the per-head dim explicitly via attention.key_length /
        // attention.value_length; the engine assumes a single per-head
        // dimension for K and V, so require they agree.
        if expects_qk_norm {
            let key_length =
                gguf.metadata_u32(&architecture_key(architecture, "attention.key_length"));
            let value_length =
                gguf.metadata_u32(&architecture_key(architecture, "attention.value_length"));
            if let (Some(k), Some(v)) = (key_length, value_length) {
                if k != v {
                    return Err(EngineError::UnsupportedModelArchitecture(format!(
                        "qwen3 attention.key_length={k} != value_length={v}; the engine assumes a \
                         single per-head dimension for K and V, so this row fails closed"
                    )));
                }
            }
        }

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for layer_idx in 0..config.block_count {
            let q_norm_name = format!("blk.{layer_idx}.attn_q_norm.weight");
            let k_norm_name = format!("blk.{layer_idx}.attn_k_norm.weight");
            let (attention_q_norm, attention_k_norm) = if expects_qk_norm {
                // Required for Qwen3: both must be present, or fail closed.
                let q = find_tensor(gguf, &q_norm_name).cloned();
                let k = find_tensor(gguf, &k_norm_name).cloned();
                if q.is_none() || k.is_none() {
                    return Err(EngineError::UnsupportedModelArchitecture(format!(
                        "qwen3 layer {layer_idx} is missing QK-norm tensors \
                         (attn_q_norm present: {}, attn_k_norm present: {}); Qwen3 applies \
                         per-head RMSNorm to Q and K and cannot run correctly without them",
                        q.is_some(),
                        k.is_some()
                    )));
                }
                (q, k)
            } else {
                // Forbidden for the plain Llama-family rows: a GGUF that
                // unexpectedly carries QK-norm tensors would have them silently
                // dropped by the forward path, so fail closed instead.
                if forbids_qk_norm
                    && (find_tensor(gguf, &q_norm_name).is_some()
                        || find_tensor(gguf, &k_norm_name).is_some())
                {
                    return Err(EngineError::UnsupportedModelArchitecture(format!(
                        "architecture {architecture:?} unexpectedly carries QK-norm tensors at \
                         layer {layer_idx} (attn_q_norm/attn_k_norm); the Llama-family forward \
                         path does not apply them, so binding fails closed rather than running a \
                         model whose weights it would silently ignore"
                    )));
                }
                (None, None)
            };
            layers.push(LlamaLayerTensors {
                attention_q_norm,
                attention_k_norm,
                attention_norm: required_tensor(
                    gguf,
                    &format!("blk.{layer_idx}.attn_norm.weight"),
                )?,
                attention_q: required_tensor(gguf, &format!("blk.{layer_idx}.attn_q.weight"))?,
                attention_k: required_tensor(gguf, &format!("blk.{layer_idx}.attn_k.weight"))?,
                attention_v: required_tensor(gguf, &format!("blk.{layer_idx}.attn_v.weight"))?,
                attention_output: required_tensor(
                    gguf,
                    &format!("blk.{layer_idx}.attn_output.weight"),
                )?,
                ffn_norm: required_tensor(gguf, &format!("blk.{layer_idx}.ffn_norm.weight"))?,
                ffn: LlamaFfnTensors {
                    gate: required_tensor(gguf, &format!("blk.{layer_idx}.ffn_gate.weight"))?,
                    up: required_tensor(gguf, &format!("blk.{layer_idx}.ffn_up.weight"))?,
                    down: required_tensor(gguf, &format!("blk.{layer_idx}.ffn_down.weight"))?,
                },
            });
        }

        let binding = Self {
            token_embedding,
            output_norm,
            output,
            output_is_tied_embedding,
            rope_freqs,
            layers,
        };
        binding.validate_dense_shapes(config)?;
        Ok(binding)
    }

    pub fn validate_dense_shapes(&self, config: &LlamaModelConfig) -> Result<()> {
        let dims = DenseLlamaDims::from_config(config)?;
        require_descriptor_matrix_shape(
            &self.token_embedding,
            dims.embedding_length,
            dims.vocab_size,
            "token embedding",
        )?;
        require_descriptor_shape(&self.output_norm, &[dims.embedding_length], "output norm")?;
        require_descriptor_matrix_shape(
            &self.output,
            dims.embedding_length,
            dims.vocab_size,
            "output projection",
        )?;
        validate_output_projection_storage_layout(
            &self.output,
            dims.embedding_length,
            dims.vocab_size,
        )?;
        if let Some(rope_freqs) = &self.rope_freqs {
            let rope_dim = config.rope_dimension_count.unwrap_or(dims.head_dim as u32) as usize;
            if rope_dim == 0 || rope_dim > dims.head_dim || !rope_dim.is_multiple_of(2) {
                return Err(EngineError::InvalidModelMetadata(format!(
                    "RoPE dimension count {rope_dim} must be even and within head dimension {}",
                    dims.head_dim
                )));
            }
            require_descriptor_shape(rope_freqs, &[rope_dim / 2], "rope frequencies")?;
        }

        if self.layers.len() != dims.block_count {
            return Err(EngineError::InvalidModelMetadata(format!(
                "config block count {} does not match bound layer count {}",
                dims.block_count,
                self.layers.len()
            )));
        }

        for (idx, layer) in self.layers.iter().enumerate() {
            require_descriptor_shape(
                &layer.attention_norm,
                &[dims.embedding_length],
                &format!("layer {idx} attention norm"),
            )?;
            require_descriptor_matrix_shape(
                &layer.attention_q,
                dims.embedding_length,
                dims.q_width,
                &format!("layer {idx} attention q"),
            )?;
            require_descriptor_matrix_shape(
                &layer.attention_k,
                dims.embedding_length,
                dims.kv_width,
                &format!("layer {idx} attention k"),
            )?;
            require_descriptor_matrix_shape(
                &layer.attention_v,
                dims.embedding_length,
                dims.kv_width,
                &format!("layer {idx} attention v"),
            )?;
            require_descriptor_matrix_shape(
                &layer.attention_output,
                dims.q_width,
                dims.embedding_length,
                &format!("layer {idx} attention output"),
            )?;
            // QK-norm (Qwen3) — per-head RMSNorm over the head dim. Present as a
            // pair or not at all.
            match (&layer.attention_q_norm, &layer.attention_k_norm) {
                (Some(q_norm), Some(k_norm)) => {
                    require_descriptor_shape(
                        q_norm,
                        &[dims.head_dim],
                        &format!("layer {idx} attention q_norm"),
                    )?;
                    require_descriptor_shape(
                        k_norm,
                        &[dims.head_dim],
                        &format!("layer {idx} attention k_norm"),
                    )?;
                }
                (None, None) => {}
                _ => {
                    return Err(EngineError::InvalidModelMetadata(format!(
                        "layer {idx} has exactly one of attn_q_norm/attn_k_norm bound; QK-norm \
                         weights must be present as a pair"
                    )));
                }
            }
            require_descriptor_shape(
                &layer.ffn_norm,
                &[dims.embedding_length],
                &format!("layer {idx} ffn norm"),
            )?;
            require_descriptor_matrix_shape(
                &layer.ffn.gate,
                dims.embedding_length,
                dims.feed_forward_length,
                &format!("layer {idx} ffn gate"),
            )?;
            require_descriptor_matrix_shape(
                &layer.ffn.up,
                dims.embedding_length,
                dims.feed_forward_length,
                &format!("layer {idx} ffn up"),
            )?;
            require_descriptor_matrix_shape(
                &layer.ffn.down,
                dims.feed_forward_length,
                dims.embedding_length,
                &format!("layer {idx} ffn down"),
            )?;
        }

        Ok(())
    }
}

/// Reject fused dense projections. Some conversions ship a single `attn_qkv`
/// (Q‖K‖V stacked by output row) or a single `ffn_up` carrying the gate‖up
/// halves instead of the separate `attn_q/attn_k/attn_v` and `ffn_gate/ffn_up`
/// tensors this engine binds. Splitting them into name-addressable byte
/// sub-ranges is not implemented here, so a fused row fails closed. For an
/// already-split model this is a no-op and must run before [`LlamaTensorBinding::bind`].
pub fn expand_fused_dense_tensors(gguf: &GgufFile, config: &LlamaModelConfig) -> Result<()> {
    let head_count = config.attention_head_count.max(1);
    let head_dim = config
        .attention_key_length
        .unwrap_or(config.embedding_length / head_count);
    let q_rows = u64::from(head_dim) * u64::from(config.attention_head_count);
    let kv_rows = u64::from(head_dim) * u64::from(config.attention_head_count_kv);
    let ffn = u64::from(config.feed_forward_length);

    for layer in 0..config.block_count {
        if find_tensor(gguf, &format!("blk.{layer}.attn_q.weight")).is_none() {
            if let Some(qkv) = find_tensor(gguf, &format!("blk.{layer}.attn_qkv.weight")) {
                if qkv.dimensions.len() == 2 && qkv.dimensions[1] == q_rows + 2 * kv_rows {
                    return Err(EngineError::UnsupportedGguf(format!(
                        "layer {layer} carries a fused attn_qkv projection; splitting fused \
                         attention projections is not implemented on this path"
                    )));
                }
            }
        }
        if find_tensor(gguf, &format!("blk.{layer}.ffn_gate.weight")).is_none() {
            if let Some(up) = find_tensor(gguf, &format!("blk.{layer}.ffn_up.weight")) {
                if up.dimensions.len() == 2 && up.dimensions[1] == 2 * ffn {
                    return Err(EngineError::UnsupportedGguf(format!(
                        "layer {layer} carries a fused gate-up projection; splitting fused \
                         feed-forward projections is not implemented on this path"
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn required_tensor(gguf: &GgufFile, name: &str) -> Result<TensorDescriptor> {
    find_tensor(gguf, name)
        .cloned()
        .ok_or_else(|| EngineError::TensorNotFound(name.to_string()))
}

pub(crate) fn find_tensor<'a>(gguf: &'a GgufFile, name: &str) -> Option<&'a TensorDescriptor> {
    gguf.tensors.iter().find(|tensor| tensor.name == name)
}

fn require_descriptor_shape(
    tensor: &TensorDescriptor,
    expected: &[usize],
    role: &str,
) -> Result<()> {
    let actual = descriptor_dims(tensor)?;
    if actual != expected {
        return Err(EngineError::InvalidModelMetadata(format!(
            "{role} tensor {} expected descriptor shape {:?}, got {:?}",
            tensor.name, expected, actual
        )));
    }
    Ok(())
}

/// Accept a 2-D linear stored in either orientation: `[input, output]` or
/// `[output, input]`. Binding never canonicalizes orientation; the forward
/// matmul resolves it from the dims at runtime.
fn require_descriptor_matrix_shape(
    tensor: &TensorDescriptor,
    input_width: usize,
    output_width: usize,
    role: &str,
) -> Result<()> {
    let actual = descriptor_dims(tensor)?;
    let direct = [input_width, output_width];
    let transposed = [output_width, input_width];
    if actual.as_slice() != direct && actual.as_slice() != transposed {
        return Err(EngineError::InvalidModelMetadata(format!(
            "{role} tensor {} expected descriptor shape {:?} or {:?}, got {:?}",
            tensor.name, direct, transposed, actual
        )));
    }
    Ok(())
}

/// Integrity guard for the output projection: for a token-major quantized
/// layout the declared byte count must equal the exact per-row block byte size
/// times the row count. Catches a truncated or mis-sized output tensor before
/// any data is read.
fn validate_output_projection_storage_layout(
    tensor: &TensorDescriptor,
    hidden_width: usize,
    vocab_size: usize,
) -> Result<()> {
    let actual = descriptor_dims(tensor)?;
    let (row_values, row_count) = match actual.as_slice() {
        [hidden, vocab] if *hidden == hidden_width && *vocab == vocab_size => (*hidden, *vocab),
        [vocab, hidden] if *hidden == hidden_width && *vocab == vocab_size => (*hidden, *vocab),
        _ => return Ok(()),
    };

    let (block_size, type_size_bytes) = tensor.tensor_type.layout().ok_or_else(|| {
        EngineError::InvalidModelMetadata(format!(
            "output projection tensor {} has unsupported storage type {:?} for token-row validation",
            tensor.name, tensor.tensor_type
        ))
    })?;
    let row_values = u64::try_from(row_values).map_err(|_| {
        EngineError::InvalidModelMetadata(format!(
            "output projection tensor {} token-row width {row_values} does not fit u64",
            tensor.name
        ))
    })?;
    let row_count = u64::try_from(row_count).map_err(|_| {
        EngineError::InvalidModelMetadata(format!(
            "output projection tensor {} token-row count {row_count} does not fit u64",
            tensor.name
        ))
    })?;
    if !row_values.is_multiple_of(block_size) {
        return Err(EngineError::InvalidModelMetadata(format!(
            "output projection tensor {} token-row width {row_values} is not divisible by {:?} block size {block_size}",
            tensor.name, tensor.tensor_type
        )));
    }

    let row_size_bytes = row_values
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size_bytes))
        .ok_or_else(|| {
            EngineError::InvalidModelMetadata(format!(
                "output projection tensor {} token-row byte size overflow",
                tensor.name
            ))
        })?;
    let expected_bytes = row_size_bytes.checked_mul(row_count).ok_or_else(|| {
        EngineError::InvalidModelMetadata(format!(
            "output projection tensor {} token-row byte count overflow",
            tensor.name
        ))
    })?;

    if tensor.n_bytes != expected_bytes {
        return Err(EngineError::InvalidModelMetadata(format!(
            "output projection tensor {} token-major storage validation failed: row_values={row_values}, row_count={row_count}, row_size_bytes={row_size_bytes}, expected_n_bytes={expected_bytes}, actual_n_bytes={}",
            tensor.name, tensor.n_bytes
        )));
    }

    Ok(())
}

fn descriptor_dims(tensor: &TensorDescriptor) -> Result<Vec<usize>> {
    tensor
        .dimensions
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                EngineError::InvalidModelMetadata(format!(
                    "tensor {} dimension {dim} does not fit usize",
                    tensor.name
                ))
            })
        })
        .collect()
}
