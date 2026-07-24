//! The dense decoder forward pass: per-layer attention + FFN, the per-token
//! decode loop, and greedy argmax sampling.
//!
//! One transformer layer runs a fixed six-step spine — attention RMS norm,
//! attention block, attention residual add, post-attention RMS norm, dense
//! SwiGLU FFN, FFN residual add — over the primitives in
//! [`crate::tensor`]. A whole prompt is decoded one token at a time; feeding a
//! prompt token by token is numerically identical to a batched prefill because
//! each token attends causally over the positions already written.

use crate::forward::kv_cache::{LlamaKvCache, LlamaKvCachePlan};
use crate::forward::rope::apply_rope;
use crate::model::config::DenseLlamaDims;
use crate::model::{LlamaModelConfig, LlamaLayerWeights, LlamaWeights};
use crate::tensor::{dot_product, q8_0_dot_rows, quantize_q8_0_blocks, CpuTensor, Q8DotRows};
use crate::{EngineError, Result};

/// A single row-major linear projection: `out[o] = dot(input, weight_row_o)`.
///
/// `in_width` is taken from `input.len()`, so the projection is independent of
/// whether the weight tensor's declared shape is `[in, out]` or `[out, in]` —
/// the weight rows are always `in_width` elements each, row-major. Q8_0 weights
/// quantize the input row once and dot it against every weight row through the
/// supplied `q8_dot` — the single seam a platform crate accelerates by passing
/// its own bit-identical kernel; [`q8_0_dot_rows`] is the portable default.
/// Plain-f32 weights fall back to the portable [`dot_product`].
fn linear_row(weight: &CpuTensor, input: &[f32], q8_dot: Q8DotRows) -> Result<Vec<f32>> {
    let in_width = input.len();
    if in_width == 0 {
        return Err(EngineError::ShapeMismatch(
            "linear projection input row is empty".to_string(),
        ));
    }
    if let Some(blocks) = weight.q8_0_blocks.as_deref() {
        if !in_width.is_multiple_of(32) {
            return Err(EngineError::ShapeMismatch(format!(
                "Q8_0 linear input width {in_width} is not block aligned (multiple of 32)"
            )));
        }
        let blocks_per_row = in_width / 32;
        if blocks_per_row == 0 || !blocks.len().is_multiple_of(blocks_per_row) {
            return Err(EngineError::ShapeMismatch(format!(
                "Q8_0 linear weight {} block count {} is not a multiple of {blocks_per_row} blocks per row",
                weight.name,
                blocks.len()
            )));
        }
        let out_width = blocks.len() / blocks_per_row;
        let input_blocks = quantize_q8_0_blocks(input);
        let mut out = vec![0.0_f32; out_width];
        for (o, out_value) in out.iter_mut().enumerate() {
            let row = &blocks[o * blocks_per_row..(o + 1) * blocks_per_row];
            *out_value = q8_dot(row, &input_blocks);
        }
        Ok(out)
    } else if !weight.data.is_empty() {
        if !weight.data.len().is_multiple_of(in_width) {
            return Err(EngineError::ShapeMismatch(format!(
                "f32 linear weight {} data length {} is not a multiple of input width {in_width}",
                weight.name,
                weight.data.len()
            )));
        }
        let out_width = weight.data.len() / in_width;
        let mut out = vec![0.0_f32; out_width];
        for (o, out_value) in out.iter_mut().enumerate() {
            let row = &weight.data[o * in_width..(o + 1) * in_width];
            *out_value = dot_product(row, input);
        }
        Ok(out)
    } else {
        Err(EngineError::InvalidTensorData(format!(
            "linear weight {} has neither Q8_0 blocks nor row-major f32 data",
            weight.name
        )))
    }
}

/// The dense decoder, holding the materialized weights, the config, and the KV
/// cache. Tokens are fed through [`Self::forward_token`]; the cache advances one
/// position per token.
pub struct Decoder<'w> {
    weights: &'w LlamaWeights,
    config: &'w LlamaModelConfig,
    dims: DenseLlamaDims,
    attention_head_count: usize,
    kv: LlamaKvCache,
    q8_dot: Q8DotRows,
}

impl<'w> Decoder<'w> {
    /// Build a decoder using the portable Q8_0 dot. Results are identical to
    /// any [`Self::with_q8_dot`] kernel, since accelerated kernels are required
    /// to be bit-identical to the portable reference.
    pub fn new(config: &'w LlamaModelConfig, weights: &'w LlamaWeights) -> Result<Self> {
        Self::with_q8_dot(config, weights, q8_0_dot_rows)
    }

    /// Build a decoder with a caller-supplied Q8_0 dot kernel — the seam a
    /// platform crate uses to accelerate the projections. The kernel must be
    /// bit-identical to [`q8_0_dot_rows`]; only speed changes, never output.
    pub fn with_q8_dot(
        config: &'w LlamaModelConfig,
        weights: &'w LlamaWeights,
        q8_dot: Q8DotRows,
    ) -> Result<Self> {
        let dims = DenseLlamaDims::from_config(config)?;
        let kv = LlamaKvCache::new(LlamaKvCachePlan::from_config(config)?)?;
        Ok(Self {
            weights,
            config,
            dims,
            attention_head_count: config.attention_head_count as usize,
            kv,
            q8_dot,
        })
    }

    /// The current sequence length (next write position).
    pub fn position(&self) -> usize {
        self.kv.position()
    }

    /// Decode one token. When `compute_logits` is set, the final norm and
    /// lm_head projection run and the `[1, vocab]` logits are returned;
    /// otherwise only the KV cache is advanced (prefill). The cache position
    /// advances by one after every layer has written its KV.
    pub fn forward_token(
        &mut self,
        token_id: u32,
        compute_logits: bool,
    ) -> Result<Option<CpuTensor>> {
        if !self.kv.can_append() {
            return Err(EngineError::ShapeMismatch(format!(
                "KV cache is full at context length {}",
                self.kv.plan.max_sequence_length
            )));
        }
        let position = self.kv.position();
        self.kv.ensure_position_capacity(position + 1)?;

        let mut hidden =
            self.weights
                .token_embedding
                .embedding_lookup(&[token_id], "hidden")?;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            hidden = self.forward_layer(&hidden, layer, layer_idx, position)?;
        }

        let logits = if compute_logits {
            let normed = if self.weights.output_norm.shape.dims == [0] {
                hidden
            } else {
                hidden.rms_norm(
                    &self.weights.output_norm,
                    self.config.rms_norm_epsilon,
                    "output_norm",
                )?
            };
            let vocab = self.dims.vocab_size;
            let data = linear_row(self.weights.output_projection(), &normed.data, self.q8_dot)?;
            Some(CpuTensor::from_f32("logits", vec![1, vocab], data)?)
        } else {
            None
        };

        self.kv.advance_position();
        Ok(logits)
    }

    /// Greedily decode `max_new` tokens continuing `prompt`.
    ///
    /// The prompt is prefilled token by token (logits computed only on the
    /// final prompt token); each step then takes the argmax of the latest
    /// logits, emits it, and feeds it back. Returns the generated token ids,
    /// not including the prompt. Decoding is greedy and carries no sampling
    /// state, so the result is a pure function of the prompt and the weights.
    pub fn generate(&mut self, prompt: &[u32], max_new: usize) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(EngineError::ShapeMismatch(
                "generate requires a non-empty prompt".to_string(),
            ));
        }
        for &token in &prompt[..prompt.len() - 1] {
            self.forward_token(token, false)?;
        }
        let mut generated = Vec::with_capacity(max_new);
        if max_new == 0 {
            return Ok(generated);
        }
        let mut logits = self
            .forward_token(prompt[prompt.len() - 1], true)?
            .expect("logits requested on the final prompt token");
        loop {
            let next = argmax(&logits)?;
            generated.push(next);
            if generated.len() == max_new {
                return Ok(generated);
            }
            logits = self
                .forward_token(next, true)?
                .expect("logits requested during decode");
        }
    }

    fn forward_layer(
        &mut self,
        hidden: &CpuTensor,
        layer: &LlamaLayerWeights,
        layer_idx: usize,
        position: usize,
    ) -> Result<CpuTensor> {
        let eps = self.config.rms_norm_epsilon;
        let head_dim = self.dims.head_dim;
        let kv_head_count = self.dims.attention_head_count_kv;
        let attention_head_count = self.attention_head_count;

        // (1) attention RMS norm on the layer input.
        let attn_norm = hidden.rms_norm(&layer.attention_norm, eps, "attn_norm")?;

        // (2) Q/K/V projections.
        let mut q = linear_row(&layer.attention_q, &attn_norm.data, self.q8_dot)?;
        let mut k = linear_row(&layer.attention_k, &attn_norm.data, self.q8_dot)?;
        let v = linear_row(&layer.attention_v, &attn_norm.data, self.q8_dot)?;

        // Optional per-head QK RMS norm (Qwen3), applied BEFORE RoPE.
        if let (Some(q_norm), Some(k_norm)) = (&layer.attention_q_norm, &layer.attention_k_norm) {
            let q_tensor = CpuTensor::from_f32("q", vec![1, q.len()], q)?;
            q = q_tensor
                .per_head_rms_norm(q_norm, attention_head_count, eps, "q_norm")?
                .data;
            let k_tensor = CpuTensor::from_f32("k", vec![1, k.len()], k)?;
            k = k_tensor
                .per_head_rms_norm(k_norm, kv_head_count, eps, "k_norm")?
                .data;
        }

        // (3) RoPE on Q (query heads) and K (kv heads).
        let rope_freqs = self.weights.rope_freqs.as_ref();
        apply_rope(&mut q, position, attention_head_count, self.config, rope_freqs)?;
        apply_rope(&mut k, position, kv_head_count, self.config, rope_freqs)?;

        // (4) Write RoPE'd K and raw V into the cache (one kv-head row at a time).
        for kv_head in 0..kv_head_count {
            let base = kv_head * head_dim;
            self.kv.store_kv_head_row(
                layer_idx,
                position,
                kv_head,
                &k[base..base + head_dim],
                &v[base..base + head_dim],
            );
        }

        // (5) Causal attention context.
        let context = attention_context(
            &q,
            &self.kv,
            layer_idx,
            position,
            head_dim,
            kv_head_count,
            attention_head_count,
        )?;

        // (6) Output projection and attention residual add.
        let attn_out_data = linear_row(&layer.attention_output, &context, self.q8_dot)?;
        let attn_out = CpuTensor::from_f32("attn_out", vec![1, attn_out_data.len()], attn_out_data)?;
        let residual = hidden.add(&attn_out, "residual")?;

        // (7) Post-attention RMS norm on the residual (NOT the layer input).
        let ffn_norm = residual.rms_norm(&layer.ffn_norm, eps, "ffn_norm")?;

        // (8) Dense SwiGLU FFN: gate/up (separate weights) -> silu(gate)*up -> down.
        let gate = linear_row(&layer.ffn_gate, &ffn_norm.data, self.q8_dot)?;
        let up = linear_row(&layer.ffn_up, &ffn_norm.data, self.q8_dot)?;
        let gate_tensor = CpuTensor::from_f32("gate", vec![1, gate.len()], gate)?;
        let up_tensor = CpuTensor::from_f32("up", vec![1, up.len()], up)?;
        let activated = gate_tensor.silu_mul(&up_tensor, "swiglu")?;
        let ffn_out_data = linear_row(&layer.ffn_down, &activated.data, self.q8_dot)?;
        let ffn_out = CpuTensor::from_f32("ffn_out", vec![1, ffn_out_data.len()], ffn_out_data)?;

        // (9) FFN residual add (base is the post-attention residual).
        residual.add(&ffn_out, "layer_out")
    }
}

/// The fused per-head causal attention: for each query head, map to its kv head
/// (GQA), score every cached key with `dot(q, k) * (1/sqrt(head_dim))`, run an
/// online softmax (running max, in-place exp, running sum), then normalize by
/// multiply-by-reciprocal folded into a sequential axpy over the cached V rows
/// (`out[d] += prob * value[d]`) in position order. This is the fused form, not
/// a generic softmax-then-matmul; the accumulation order is load-bearing for
/// numeric fidelity. The `position_count == 1` first token skips the softmax and
/// copies the single mapped V row directly.
///
/// `layer_idx`/`position` address the [`LlamaKvCache`], which already holds the
/// RoPE'd keys and raw values; `head_dim`, `kv_head_count`, and
/// `attention_head_count` describe the head geometry (query heads are an exact
/// multiple of kv heads).
fn attention_context(
    q: &[f32],
    kv: &LlamaKvCache,
    layer_idx: usize,
    position: usize,
    head_dim: usize,
    kv_head_count: usize,
    attention_head_count: usize,
) -> Result<Vec<f32>> {
    let repeats = attention_head_count / kv_head_count;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let position_count = position + 1;

    let mut context = vec![0.0_f32; attention_head_count * head_dim];
    for head in 0..attention_head_count {
        let kv_head = head / repeats;
        let q_head = &q[head * head_dim..head * head_dim + head_dim];
        let out = &mut context[head * head_dim..head * head_dim + head_dim];

        if position_count == 1 {
            out.copy_from_slice(kv.value_row(layer_idx, 0, kv_head));
            continue;
        }

        // Scores: score[p] = dot(q_head, key_row_p) * scale.
        let mut scores = vec![0.0_f32; position_count];
        for (p, score) in scores.iter_mut().enumerate() {
            let key = kv.key_row(layer_idx, p, kv_head);
            *score = dot_product(q_head, key) * scale;
        }
        // Online softmax: max, in-place exp, sum, reciprocal.
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for score in scores.iter_mut() {
            *score = (*score - max).exp();
            sum += *score;
        }
        if sum == 0.0 || !sum.is_finite() {
            return Err(EngineError::ShapeMismatch(
                "attention softmax produced an invalid normalization sum".to_string(),
            ));
        }
        let inv_sum = 1.0_f32 / sum;
        // Weighted V sum: multiply by reciprocal, sequential over positions.
        for (p, score) in scores.iter().enumerate() {
            let probability = *score * inv_sum;
            let value = kv.value_row(layer_idx, p, kv_head);
            for (out_value, &v) in out.iter_mut().zip(value) {
                *out_value += probability * v;
            }
        }
    }
    Ok(context)
}

/// Greedy argmax over a `[1, vocab]` logits tensor: the first index achieving
/// the strict maximum wins, so ties resolve to the lowest token id. Non-finite
/// logits and a non-`[1, vocab]` shape are rejected before the scan.
pub fn argmax(logits: &CpuTensor) -> Result<u32> {
    if logits.rank() != 2 || logits.shape.dims[0] != 1 {
        return Err(EngineError::ShapeMismatch(format!(
            "greedy sampling expects logits shape [1, vocab], got {:?}",
            logits.shape.dims
        )));
    }
    if logits.data.is_empty() {
        return Err(EngineError::ShapeMismatch(
            "greedy sampling requires a non-empty logits row".to_string(),
        ));
    }
    if let Some(idx) = logits.data.iter().position(|v| !v.is_finite()) {
        return Err(EngineError::ShapeMismatch(format!(
            "greedy sampling rejects non-finite logit at index {idx}"
        )));
    }
    let mut best_idx = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, &value) in logits.data.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_idx = idx;
        }
    }
    u32::try_from(best_idx).map_err(|_| {
        EngineError::ShapeMismatch(format!("argmax index {best_idx} does not fit u32"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::kv_cache::LlamaKvCachePlan;
    use crate::tensor::{q8_0_dot_rows as core_dot, quantize_q8_0_blocks as core_quant};

    /// Independent reference for the fused per-head kernel, recomputed from the
    /// rounded rows already in the cache. It mirrors the exact operation order —
    /// [`dot_product`] scores, `1/sqrt(head_dim)` scale, fold-max, in-place exp,
    /// running sum, multiply-by-reciprocal axpy over V — so a bit-for-bit match
    /// pins the kernel's arithmetic, not just its shape.
    fn reference_context(
        q: &[f32],
        kv: &LlamaKvCache,
        layer_idx: usize,
        position: usize,
        head_dim: usize,
        kv_head_count: usize,
        attention_head_count: usize,
    ) -> Vec<f32> {
        let repeats = attention_head_count / kv_head_count;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let position_count = position + 1;
        let mut out = vec![0.0_f32; attention_head_count * head_dim];
        for head in 0..attention_head_count {
            let kv_head = head / repeats;
            let q_head = &q[head * head_dim..head * head_dim + head_dim];
            let dst = &mut out[head * head_dim..head * head_dim + head_dim];
            if position_count == 1 {
                dst.copy_from_slice(kv.value_row(layer_idx, 0, kv_head));
                continue;
            }
            let mut scores = vec![0.0_f32; position_count];
            for (p, score) in scores.iter_mut().enumerate() {
                *score = dot_product(q_head, kv.key_row(layer_idx, p, kv_head)) * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for score in scores.iter_mut() {
                *score = (*score - max).exp();
                sum += *score;
            }
            let inv_sum = 1.0_f32 / sum;
            for (p, score) in scores.iter().enumerate() {
                let probability = *score * inv_sum;
                let value = kv.value_row(layer_idx, p, kv_head);
                for (d, &v) in dst.iter_mut().zip(value) {
                    *d += probability * v;
                }
            }
        }
        out
    }

    /// A single query head over two cached positions (no GQA): the fused kernel
    /// must match a hand-mirrored reference loop bit for bit, and the
    /// `position_count == 1` shortcut must return the mapped V row unchanged.
    #[test]
    fn attention_single_head_matches_reference_bits() {
        let head_dim = 4;
        let plan = LlamaKvCachePlan {
            max_sequence_length: 16,
            layer_count: 1,
            kv_head_count: 1,
            head_dim,
        };
        let mut kv = LlamaKvCache::new(plan).unwrap();
        kv.ensure_position_capacity(2).unwrap();

        // Position 0: key/value rows written through the f16-rounding store.
        kv.store_kv_head_row(0, 0, 0, &[0.5, -1.0, 0.25, 2.0], &[1.0, 0.0, -0.5, 0.75]);
        // Position 1.
        kv.store_kv_head_row(0, 1, 0, &[-0.75, 0.5, 1.5, -0.25], &[-1.0, 0.5, 0.25, -2.0]);

        let q = [0.3_f32, -0.6, 0.9, 0.1];

        // First-token shortcut: only position 0 exists -> context is V row 0.
        let ctx0 = attention_context(&q, &kv, 0, 0, head_dim, 1, 1).unwrap();
        assert_eq!(ctx0, kv.value_row(0, 0, 0).to_vec());

        // Two positions: fused online-softmax + axpy over V.
        let ctx = attention_context(&q, &kv, 0, 1, head_dim, 1, 1).unwrap();
        let expected = reference_context(&q, &kv, 0, 1, head_dim, 1, 1);
        for (i, (&got, &want)) in ctx.iter().zip(&expected).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "context lane {i} bit mismatch");
        }
    }

    /// Grouped-query attention: four query heads over two kv heads (repeats = 2)
    /// across three cached positions. Heads 0/1 must read kv head 0 and heads 2/3
    /// kv head 1; the per-head output must match the reference bit for bit, and a
    /// query head reading kv head 1 must differ from one reading kv head 0.
    #[test]
    fn attention_gqa_head_mapping_matches_reference_bits() {
        let head_dim = 4;
        let kv_head_count = 2;
        let attention_head_count = 4;
        let plan = LlamaKvCachePlan {
            max_sequence_length: 16,
            layer_count: 1,
            kv_head_count,
            head_dim,
        };
        let mut kv = LlamaKvCache::new(plan).unwrap();
        kv.ensure_position_capacity(3).unwrap();

        // Distinct rows per (position, kv_head) so a mis-mapped head is visible.
        for p in 0..3 {
            let base = p as f32;
            kv.store_kv_head_row(
                0,
                p,
                0,
                &[0.1 + base, -0.2, 0.3, 0.4 - base],
                &[1.0 + base, -0.5, 0.25, 0.0],
            );
            kv.store_kv_head_row(
                0,
                p,
                1,
                &[-0.4, 0.2 + base, -0.1, 0.5],
                &[-1.0, 0.75, base, 0.5],
            );
        }

        // Four query heads with distinct content.
        let mut q = vec![0.0_f32; attention_head_count * head_dim];
        for (i, slot) in q.iter_mut().enumerate() {
            *slot = ((i as f32) * 0.13) - 0.4;
        }

        let ctx =
            attention_context(&q, &kv, 0, 2, head_dim, kv_head_count, attention_head_count).unwrap();
        let expected =
            reference_context(&q, &kv, 0, 2, head_dim, kv_head_count, attention_head_count);
        assert_eq!(ctx.len(), attention_head_count * head_dim);
        for (i, (&got, &want)) in ctx.iter().zip(&expected).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "gqa context lane {i} bit mismatch");
        }

        // A query head mapped to kv head 1 must generally differ from the same
        // query mapped to kv head 0, confirming the head->kv_head split is live.
        let head0 = &ctx[0..head_dim]; // head 0 -> kv head 0
        let head2 = &ctx[2 * head_dim..3 * head_dim]; // head 2 -> kv head 1
        assert_ne!(
            head0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            head2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "distinct kv heads must yield distinct context"
        );
    }

    #[test]
    fn linear_row_q8_matches_manual_block_dot() {
        // Two output rows, in_width 32 (one block per row).
        let mut flat = Vec::new();
        for o in 0..2 {
            for k in 0..32 {
                flat.push(((o * 3 + k) % 7) as f32 - 3.0);
            }
        }
        let blocks = core_quant(&flat);
        let weight = CpuTensor::from_q8_0_blocks(
            "w",
            crate::tensor::TensorShape { dims: vec![32, 2] },
            blocks.clone(),
        )
        .unwrap();
        let input: Vec<f32> = (0..32).map(|k| (k as f32 * 0.1) - 1.0).collect();

        let out = linear_row(&weight, &input, core_dot).unwrap();
        let input_blocks = core_quant(&input);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].to_bits(), core_dot(&blocks[0..1], &input_blocks).to_bits());
        assert_eq!(out[1].to_bits(), core_dot(&blocks[1..2], &input_blocks).to_bits());
    }

    #[test]
    fn linear_row_f32_matches_dot_product() {
        let weight = CpuTensor::from_f32(
            "w",
            vec![2, 3],
            vec![1.0, -2.0, 0.5, 3.0, 0.0, -1.0],
        )
        .unwrap();
        let input = [2.0_f32, 1.0, -4.0];
        let out = linear_row(&weight, &input, core_dot).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], dot_product(&weight.data[0..3], &input));
        assert_eq!(out[1], dot_product(&weight.data[3..6], &input));
    }

    #[test]
    fn argmax_lowest_index_on_ties_and_rejects_nonfinite() {
        let logits = CpuTensor::from_f32("logits", vec![1, 4], vec![1.0, 3.0, 3.0, 2.0]).unwrap();
        // Strict '>' keeps the FIRST max: index 1, not 2.
        assert_eq!(argmax(&logits).unwrap(), 1);

        let bad = CpuTensor::from_f32("logits", vec![1, 3], vec![1.0, f32::NAN, 2.0]).unwrap();
        assert!(argmax(&bad).is_err());

        let wrong_shape = CpuTensor::from_f32("logits", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(argmax(&wrong_shape).is_err());
    }
}
