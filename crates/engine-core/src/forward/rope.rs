//! Rotary position embedding for the dense decoder spine.
//!
//! [`apply_rope`] rotates the first `rope_dim` dimensions of each head of a Q or
//! K row in place. The per-pair frequency, the optional `rope_freqs` divisor,
//! and the four frequency-scaling kinds (none / linear / llama3 / YaRN) are all
//! ported faithfully; the target model uses `freq_base = 500000` with scaling
//! `none`, but every kind executes when a model selects it.
//!
//! Pairing is decided by the model config: adjacent even/odd (permuted
//! LLaMA-family weights) or NEOX split-half (unpermuted qwen3/phi3 weights).
//! Rotation is always forward and positions are zero-based.

use crate::model::LlamaModelConfig;
use crate::tensor::CpuTensor;
use crate::{EngineError, Result};

/// How a rotated pair's two dimensions are chosen within a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopePairing {
    /// LLaMA-family: pair `p` rotates dims `2p` and `2p+1`.
    AdjacentEvenOdd,
    /// NEOX: pair `p` rotates dims `p` and `p + rope_dim/2`.
    SplitHalf,
}

/// Frequency-scaling kind read from `<arch>.rope.scaling.type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RopeScalingKind {
    None,
    Linear,
    Llama3,
    Yarn,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RopeScaling {
    kind: RopeScalingKind,
    factor: f32,
    original_context_length: Option<u32>,
    low_freq_factor: Option<f32>,
    high_freq_factor: Option<f32>,
}

/// NEOX split-half pairing for the unpermuted conversions; adjacent even/odd
/// otherwise.
pub fn rope_pairing_for_config(config: &LlamaModelConfig) -> RopePairing {
    if config.rope_neox_pairing {
        RopePairing::SplitHalf
    } else {
        RopePairing::AdjacentEvenOdd
    }
}

// YaRN (NTK-by-parts) defaults: beta_fast=32, beta_slow=1, ext_factor=1,
// attn_factor=1.
const YARN_BETA_FAST: f32 = 32.0;
const YARN_BETA_SLOW: f32 = 1.0;

fn yarn_corr_dim(rope_dim: usize, n_ctx_orig: f32, n_rot: f32, base: f32) -> f32 {
    (rope_dim as f32) * (n_ctx_orig / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
}

fn yarn_corr_dims(rope_dim: usize, n_ctx_orig: u32, base: f32) -> (f32, f32) {
    let start = yarn_corr_dim(rope_dim, n_ctx_orig as f32, YARN_BETA_FAST, base).floor();
    let end = yarn_corr_dim(rope_dim, n_ctx_orig as f32, YARN_BETA_SLOW, base).ceil();
    (start.max(0.0), end.min(rope_dim as f32 - 1.0))
}

fn yarn_ramp(low: f32, high: f32, pair_idx: usize) -> f32 {
    let y = ((pair_idx as f32) - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// cos/sin magnitude scaling: 1.0 unless YaRN, where it is `1 + 0.1*ln(factor)`
/// (ext_factor=attn_factor=1).
fn rope_magnitude_scale(scaling: RopeScaling) -> f32 {
    match scaling.kind {
        RopeScalingKind::Yarn => 1.0 + 0.1 * scaling.factor.ln(),
        _ => 1.0,
    }
}

fn rope_scaling_from_config(config: &LlamaModelConfig) -> Result<RopeScaling> {
    let kind = match config.rope_scaling_type.as_deref().map(str::trim) {
        None | Some("") | Some("none") => RopeScalingKind::None,
        Some("linear") => RopeScalingKind::Linear,
        Some("llama3") => RopeScalingKind::Llama3,
        Some("yarn") => RopeScalingKind::Yarn,
        Some(other) => {
            return Err(EngineError::InvalidModelMetadata(format!(
                "unsupported rope.scaling.type {other:?}; expected none, linear, llama3, or yarn"
            )))
        }
    };
    let factor = config.rope_scaling_factor.unwrap_or(1.0);
    if factor <= 0.0 || !factor.is_finite() {
        return Err(EngineError::InvalidModelMetadata(format!(
            "RoPE scaling factor {factor} must be finite and positive"
        )));
    }
    match kind {
        RopeScalingKind::None => Ok(RopeScaling {
            kind,
            factor: 1.0,
            original_context_length: None,
            low_freq_factor: None,
            high_freq_factor: None,
        }),
        RopeScalingKind::Linear => Ok(RopeScaling {
            kind,
            factor,
            original_context_length: None,
            low_freq_factor: None,
            high_freq_factor: None,
        }),
        RopeScalingKind::Llama3 => {
            let original_context_length =
                config.rope_scaling_original_context_length.unwrap_or(8_192);
            if original_context_length == 0 {
                return Err(EngineError::InvalidModelMetadata(
                    "llama3 RoPE scaling original context length must be greater than zero"
                        .to_string(),
                ));
            }
            let low_freq_factor = config.rope_scaling_low_freq_factor.unwrap_or(1.0);
            let high_freq_factor = config.rope_scaling_high_freq_factor.unwrap_or(4.0);
            if low_freq_factor <= 0.0
                || high_freq_factor <= 0.0
                || !low_freq_factor.is_finite()
                || !high_freq_factor.is_finite()
                || high_freq_factor <= low_freq_factor
            {
                return Err(EngineError::InvalidModelMetadata(format!(
                    "llama3 RoPE scaling frequency factors must be finite, positive, and high > low; got low={low_freq_factor}, high={high_freq_factor}"
                )));
            }
            Ok(RopeScaling {
                kind,
                factor,
                original_context_length: Some(original_context_length),
                low_freq_factor: Some(low_freq_factor),
                high_freq_factor: Some(high_freq_factor),
            })
        }
        RopeScalingKind::Yarn => {
            let original_context_length =
                config.rope_scaling_original_context_length.unwrap_or(8_192);
            if original_context_length == 0 {
                return Err(EngineError::InvalidModelMetadata(
                    "yarn RoPE scaling original context length must be greater than zero"
                        .to_string(),
                ));
            }
            Ok(RopeScaling {
                kind,
                factor,
                original_context_length: Some(original_context_length),
                low_freq_factor: None,
                high_freq_factor: None,
            })
        }
    }
}

fn llama3_scaled_rope_frequency(frequency: f32, scaling: RopeScaling) -> f32 {
    let original_context_length = scaling
        .original_context_length
        .expect("validated llama3 scaling has original context length")
        as f32;
    let low_freq_factor = scaling
        .low_freq_factor
        .expect("validated llama3 scaling has low freq factor");
    let high_freq_factor = scaling
        .high_freq_factor
        .expect("validated llama3 scaling has high freq factor");

    let wavelength = (2.0 * std::f32::consts::PI) / frequency;
    let low_freq_wavelength = original_context_length / low_freq_factor;
    let high_freq_wavelength = original_context_length / high_freq_factor;
    if wavelength < high_freq_wavelength {
        frequency
    } else if wavelength > low_freq_wavelength {
        frequency / scaling.factor
    } else {
        let smooth = (original_context_length / wavelength - low_freq_factor)
            / (high_freq_factor - low_freq_factor);
        ((1.0 - smooth) * frequency / scaling.factor) + (smooth * frequency)
    }
}

#[derive(Debug, Clone, Copy)]
struct RopeParams<'a> {
    head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    pairing: RopePairing,
    scaling: RopeScaling,
    rope_freqs: Option<&'a [f32]>,
}

fn rope_pair_frequency(pair_idx: usize, params: &RopeParams<'_>) -> f32 {
    let base_frequency = params
        .freq_base
        .powf(-(pair_idx as f32 * 2.0) / params.rope_dim as f32);
    // The `rope_freqs` table carries per-pair frequency factors: the stored
    // value DIVIDES the base frequency for the pair, it does not replace it.
    let effective_base_frequency = if let Some(rope_freqs) = params.rope_freqs {
        base_frequency / rope_freqs[pair_idx]
    } else {
        base_frequency
    };
    match params.scaling.kind {
        RopeScalingKind::None => effective_base_frequency,
        RopeScalingKind::Linear => effective_base_frequency / params.scaling.factor,
        RopeScalingKind::Llama3 => {
            llama3_scaled_rope_frequency(effective_base_frequency, params.scaling)
        }
        RopeScalingKind::Yarn => {
            let n_ctx_orig = params.scaling.original_context_length.unwrap_or(8_192);
            let (low, high) = yarn_corr_dims(params.rope_dim, n_ctx_orig, params.freq_base);
            let ramp_mix = yarn_ramp(low, high, pair_idx);
            let theta_extrap = effective_base_frequency;
            let theta_interp = theta_extrap / params.scaling.factor;
            theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix
        }
    }
}

fn apply_rope_to_row(data: &mut [f32], position: usize, params: &RopeParams<'_>) {
    let half_rope_dim = params.rope_dim / 2;
    let mscale = rope_magnitude_scale(params.scaling);
    for pair_idx in 0..half_rope_dim {
        let theta = rope_pair_frequency(pair_idx, params);
        let angle = position as f32 * theta;
        let (mut sin, mut cos) = angle.sin_cos();
        sin *= mscale;
        cos *= mscale;
        for head in 0..params.head_count {
            let head_start = head * params.head_dim;
            let (dim0, dim1) = match params.pairing {
                RopePairing::AdjacentEvenOdd => {
                    let dim0 = head_start + (pair_idx * 2);
                    (dim0, dim0 + 1)
                }
                RopePairing::SplitHalf => {
                    (head_start + pair_idx, head_start + pair_idx + half_rope_dim)
                }
            };
            let x0 = data[dim0];
            let x1 = data[dim1];
            // Forward rotation.
            data[dim0] = (x0 * cos) - (x1 * sin);
            data[dim1] = (x0 * sin) + (x1 * cos);
        }
    }
}

/// Validate a `rope_freqs` frequency-factor tensor for `rope_dim`.
fn validate_rope_frequency_tensor(rope_freqs: &CpuTensor, rope_dim: usize) -> Result<&[f32]> {
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
    Ok(&rope_freqs.data)
}

/// Apply RoPE in place to a `[1, width]` Q or K row. `head_count` is the query
/// head count for Q and the KV head count for K; `width` must be a multiple of
/// it. Only the first `rope_dim` dims of each head are rotated; the rest pass
/// through unchanged.
pub fn apply_rope(
    row: &mut [f32],
    position: usize,
    head_count: usize,
    config: &LlamaModelConfig,
    rope_freqs: Option<&CpuTensor>,
) -> Result<()> {
    if head_count == 0 {
        return Err(EngineError::ShapeMismatch(
            "RoPE head count must be greater than zero".to_string(),
        ));
    }
    let width = row.len();
    if !width.is_multiple_of(head_count) {
        return Err(EngineError::ShapeMismatch(format!(
            "RoPE input width {width} is not divisible by head count {head_count}"
        )));
    }
    let head_dim = width / head_count;
    let rope_dim = config.rope_dimension_count.unwrap_or(head_dim as u32) as usize;
    if rope_dim == 0 || rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
        return Err(EngineError::InvalidModelMetadata(format!(
            "RoPE dimension count {rope_dim} must be even and within head dimension {head_dim}"
        )));
    }
    let freq_base = config.rope_freq_base.unwrap_or(10_000.0);
    if freq_base <= 0.0 || !freq_base.is_finite() {
        return Err(EngineError::InvalidModelMetadata(format!(
            "RoPE frequency base {freq_base} must be finite and positive"
        )));
    }
    let scaling = rope_scaling_from_config(config)?;
    let rope_freqs = rope_freqs
        .map(|freqs| validate_rope_frequency_tensor(freqs, rope_dim))
        .transpose()?;
    let params = RopeParams {
        head_count,
        head_dim,
        rope_dim,
        freq_base,
        pairing: rope_pairing_for_config(config),
        scaling,
        rope_freqs,
    };
    apply_rope_to_row(row, position, &params);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> LlamaModelConfig {
        LlamaModelConfig {
            context_length: 2048,
            embedding_length: 4,
            block_count: 1,
            feed_forward_length: 8,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: Some(4),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1e-5,
            vocab_size: Some(8),
            file_type: None,
            attention_key_length: None,
            rope_neox_pairing: false,
        }
    }

    #[test]
    fn position_zero_is_identity() {
        let config = base_config();
        let original = [1.0_f32, -2.0, 3.5, 0.25];
        let mut row = original;
        apply_rope(&mut row, 0, 1, &config, None).unwrap();
        // angle = 0 => cos = 1, sin = 0 => every element unchanged, bit for bit.
        for (i, &v) in original.iter().enumerate() {
            assert_eq!(row[i].to_bits(), v.to_bits());
        }
    }

    #[test]
    fn adjacent_even_odd_known_vector_exact_bits() {
        let config = base_config(); // adjacent even/odd
        let mut row = [1.0_f32, -2.0, 3.5, 0.25];
        apply_rope(&mut row, 2, 1, &config, None).unwrap();

        // Reproduce the exact left-associated arithmetic the kernel uses.
        let freq_base = 10_000.0_f32;
        let rope_dim = 4.0_f32;
        // pair 0 -> dims (0,1)
        let theta0 = freq_base.powf(-(0.0 * 2.0) / rope_dim);
        let (sin0, cos0) = (2.0_f32 * theta0).sin_cos();
        let e0 = (1.0_f32 * cos0) - (-2.0_f32 * sin0);
        let e1 = (1.0_f32 * sin0) + (-2.0_f32 * cos0);
        // pair 1 -> dims (2,3)
        let theta1 = freq_base.powf(-(1.0 * 2.0) / rope_dim);
        let (sin1, cos1) = (2.0_f32 * theta1).sin_cos();
        let e2 = (3.5_f32 * cos1) - (0.25_f32 * sin1);
        let e3 = (3.5_f32 * sin1) + (0.25_f32 * cos1);

        assert_eq!(row[0].to_bits(), e0.to_bits());
        assert_eq!(row[1].to_bits(), e1.to_bits());
        assert_eq!(row[2].to_bits(), e2.to_bits());
        assert_eq!(row[3].to_bits(), e3.to_bits());
    }

    #[test]
    fn neox_split_half_known_vector_exact_bits() {
        let mut config = base_config();
        config.rope_neox_pairing = true; // NEOX split-half
        let mut row = [1.0_f32, -2.0, 3.5, 0.25];
        apply_rope(&mut row, 3, 1, &config, None).unwrap();

        let freq_base = 10_000.0_f32;
        let rope_dim = 4.0_f32;
        // pair 0 -> dims (0, 2); pair 1 -> dims (1, 3)
        let theta0 = freq_base.powf(-(0.0 * 2.0) / rope_dim);
        let (sin0, cos0) = (3.0_f32 * theta0).sin_cos();
        let d0 = (1.0_f32 * cos0) - (3.5_f32 * sin0);
        let d2 = (1.0_f32 * sin0) + (3.5_f32 * cos0);
        let theta1 = freq_base.powf(-(1.0 * 2.0) / rope_dim);
        let (sin1, cos1) = (3.0_f32 * theta1).sin_cos();
        let d1 = (-2.0_f32 * cos1) - (0.25_f32 * sin1);
        let d3 = (-2.0_f32 * sin1) + (0.25_f32 * cos1);

        assert_eq!(row[0].to_bits(), d0.to_bits());
        assert_eq!(row[2].to_bits(), d2.to_bits());
        assert_eq!(row[1].to_bits(), d1.to_bits());
        assert_eq!(row[3].to_bits(), d3.to_bits());
    }

    #[test]
    fn rope_freqs_divides_base_frequency() {
        let config = base_config();
        let mut plain = [1.0_f32, -2.0, 3.5, 0.25];
        let mut divided = plain;
        apply_rope(&mut plain, 2, 1, &config, None).unwrap();

        // A frequency-factor of 2.0 on each pair halves the effective frequency.
        let freqs = CpuTensor::from_f32("rope_freqs.weight", vec![2], vec![2.0, 2.0]).unwrap();
        apply_rope(&mut divided, 2, 1, &config, Some(&freqs)).unwrap();
        assert_ne!(plain[0].to_bits(), divided[0].to_bits());

        // Reproduce dim 0/1 (pair 0) with the divided frequency.
        let freq_base = 10_000.0_f32;
        let theta0 = freq_base.powf(-(0.0 * 2.0) / 4.0) / 2.0;
        let (sin0, cos0) = (2.0_f32 * theta0).sin_cos();
        let e0 = (1.0_f32 * cos0) - (-2.0_f32 * sin0);
        assert_eq!(divided[0].to_bits(), e0.to_bits());
    }

    #[test]
    fn dims_beyond_rope_dim_pass_through() {
        let mut config = base_config();
        config.rope_dimension_count = Some(2); // only first 2 of 4 dims rotate
        let mut row = [1.0_f32, -2.0, 3.5, 0.25];
        apply_rope(&mut row, 5, 1, &config, None).unwrap();
        // dims 2 and 3 are beyond rope_dim, untouched.
        assert_eq!(row[2].to_bits(), 3.5_f32.to_bits());
        assert_eq!(row[3].to_bits(), 0.25_f32.to_bits());
    }
}
