//! Key/value cache for the dense decoder spine.
//!
//! The cache holds RoPE'd keys and raw values for every decoded position in a
//! flat `[position, layer, kv_head, head_dim]` (position-major) f32 buffer.
//! Every stored element is rounded through IEEE binary16 and back before it
//! lands in the buffer, so the cache holds exactly-f16-representable values at
//! f32 width — the rounding is load-bearing for numeric fidelity, not an
//! optimization.

use crate::model::{DenseLlamaDims, LlamaModelConfig};
use crate::{EngineError, Result};

/// Reference f32 -> f16 (IEEE binary16) with round-to-nearest-even. Overflow
/// (including the 65520 tie) saturates to `±inf`; subnormals round RNE; every
/// NaN is canonicalized to `sign | 0x7E00`. This is the exact conversion the
/// KV write path applies to every stored key and value; it differs from the
/// general-purpose [`crate::tensor::f32_to_f16_bits`] only on NaN inputs
/// (which canonicalize here, keep their payload there), so the KV lane carries
/// its own copy rather than reusing that helper.
pub fn f32_to_f16_kv(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x0080_0000;
        let shift = 14 - half_exp;
        let mut half_mant = (mantissa >> shift) as u16;
        let round_bit = 1_u32 << (shift - 1);
        if (mantissa & round_bit) != 0 && ((mantissa & (round_bit - 1)) != 0 || (half_mant & 1) != 0)
        {
            half_mant = half_mant.wrapping_add(1);
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    if (mant & 0x0000_1000) != 0 && ((mant & 0x0000_0fff) != 0 || (half & 1) != 0) {
        half = half.wrapping_add(1);
    }
    half
}

/// Reference f16 -> f32 (exact expansion). Signalling NaNs are quieted (the
/// f32 quiet bit is forced) as `vcvtph2ps` does; the canonical NaN the store
/// produces is already quiet, so this only matters for completeness.
pub fn f16_to_f32_kv(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = u32::from(bits & 0x03ff);
    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut e = -14_i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                let exp32 = (e + 127) as u32;
                sign | (exp32 << 23) | (mant << 13)
            }
        }
        0x1f => {
            if frac == 0 {
                sign | 0x7f80_0000
            } else {
                sign | 0x7f80_0000 | 0x0040_0000 | (frac << 13)
            }
        }
        _ => {
            let exp32 = u32::from(exp) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(out)
}

/// The f16 round-trip applied to every stored key/value element.
#[inline]
fn kv_round(value: f32) -> f32 {
    f16_to_f32_kv(f32_to_f16_kv(value))
}

/// Immutable cache geometry derived from the model config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaKvCachePlan {
    pub max_sequence_length: usize,
    pub layer_count: usize,
    /// GQA key/value head count (`attention_head_count_kv`), NOT the query-head
    /// count — the cache is sized by KV heads.
    pub kv_head_count: usize,
    pub head_dim: usize,
}

impl LlamaKvCachePlan {
    pub fn from_config(config: &LlamaModelConfig) -> Result<Self> {
        let dims = DenseLlamaDims::from_config(config)?;
        Ok(Self {
            max_sequence_length: config.context_length as usize,
            layer_count: dims.block_count,
            kv_head_count: dims.attention_head_count_kv,
            head_dim: dims.head_dim,
        })
    }
}

/// Position-major f32 key/value cache. `keys` and `values` are flat buffers
/// addressed by [`LlamaKvCache::offset`]; `position` is the next write index
/// (and the current sequence length). Buffers grow on demand up to
/// `max_sequence_length`.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaKvCache {
    pub plan: LlamaKvCachePlan,
    keys: Vec<f32>,
    values: Vec<f32>,
    allocated_sequence_length: usize,
    position: usize,
}

impl LlamaKvCache {
    pub fn new(plan: LlamaKvCachePlan) -> Result<Self> {
        if plan.layer_count == 0 || plan.kv_head_count == 0 || plan.head_dim == 0 {
            return Err(EngineError::InvalidModelMetadata(
                "KV cache plan requires non-zero layer, kv-head, and head-dim counts".to_string(),
            ));
        }
        Ok(Self {
            plan,
            keys: Vec::new(),
            values: Vec::new(),
            allocated_sequence_length: 0,
            position: 0,
        })
    }

    /// The next write index / current sequence length.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether another token can be appended without exceeding the context.
    pub fn can_append(&self) -> bool {
        self.position < self.plan.max_sequence_length
    }

    /// Element count one token occupies across all layers and kv heads.
    fn position_stride(&self) -> usize {
        self.plan.layer_count * self.plan.kv_head_count * self.plan.head_dim
    }

    /// Element base for one `(layer, position, kv_head)` row in position-major
    /// order: `(((position*layer_count)+layer)*kv_head_count+kv_head)*head_dim`.
    pub fn offset(&self, layer_idx: usize, position: usize, kv_head: usize) -> usize {
        (((position * self.plan.layer_count) + layer_idx) * self.plan.kv_head_count + kv_head)
            * self.plan.head_dim
    }

    /// Ensure the buffers hold at least `required_sequence_length` positions,
    /// growing in chunks (never past `max_sequence_length`). Growth is a plain
    /// zero-fill resize; position-major means positions are outermost, so no
    /// relayout is needed.
    pub fn ensure_position_capacity(&mut self, required_sequence_length: usize) -> Result<()> {
        if required_sequence_length > self.plan.max_sequence_length {
            return Err(EngineError::ShapeMismatch(format!(
                "KV cache position {required_sequence_length} exceeds context length {}",
                self.plan.max_sequence_length
            )));
        }
        if required_sequence_length <= self.allocated_sequence_length {
            return Ok(());
        }
        // Grow with headroom to keep append amortized O(1), capped at the context.
        let target = required_sequence_length
            .max(self.allocated_sequence_length.saturating_mul(2))
            .max(GROW_CHUNK)
            .min(self.plan.max_sequence_length);
        let elements = target
            .checked_mul(self.position_stride())
            .ok_or_else(|| {
                EngineError::ShapeMismatch("KV cache element count overflow".to_string())
            })?;
        self.keys.resize(elements, 0.0);
        self.values.resize(elements, 0.0);
        self.allocated_sequence_length = target;
        Ok(())
    }

    /// Store one `(layer, position, kv_head)` key and value row, rounding every
    /// element through f16. The RoPE'd key and the raw value are written here;
    /// callers must have reserved capacity for `position`.
    pub fn store_kv_head_row(
        &mut self,
        layer_idx: usize,
        position: usize,
        kv_head: usize,
        key_row: &[f32],
        value_row: &[f32],
    ) {
        let head_dim = self.plan.head_dim;
        debug_assert_eq!(key_row.len(), head_dim);
        debug_assert_eq!(value_row.len(), head_dim);
        let dst = self.offset(layer_idx, position, kv_head);
        for (slot, &value) in self.keys[dst..dst + head_dim].iter_mut().zip(key_row) {
            *slot = kv_round(value);
        }
        for (slot, &value) in self.values[dst..dst + head_dim].iter_mut().zip(value_row) {
            *slot = kv_round(value);
        }
    }

    /// One stored key row as an f32 slice.
    pub fn key_row(&self, layer_idx: usize, position: usize, kv_head: usize) -> &[f32] {
        let src = self.offset(layer_idx, position, kv_head);
        &self.keys[src..src + self.plan.head_dim]
    }

    /// One stored value row as an f32 slice.
    pub fn value_row(&self, layer_idx: usize, position: usize, kv_head: usize) -> &[f32] {
        let src = self.offset(layer_idx, position, kv_head);
        &self.values[src..src + self.plan.head_dim]
    }

    /// Advance the write index by one token. Called once per decoded token,
    /// AFTER every layer has written its KV for that token.
    pub fn advance_position(&mut self) {
        self.position += 1;
    }
}

/// Positions added per growth step, so appends stay amortized instead of
/// resizing the flat buffers every token.
const GROW_CHUNK: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> LlamaKvCachePlan {
        LlamaKvCachePlan {
            max_sequence_length: 2048,
            layer_count: 2,
            kv_head_count: 2,
            head_dim: 4,
        }
    }

    #[test]
    fn offset_is_position_major() {
        let cache = LlamaKvCache::new(plan()).unwrap();
        // (((pos*layers)+layer)*kv_heads+kv_head)*head_dim, layers=kv_heads=2, head_dim=4.
        assert_eq!(cache.offset(0, 0, 0), 0);
        // next layer: (1*2+0)*4 = 8 (one layer = kv_heads*head_dim elements).
        assert_eq!(cache.offset(1, 0, 0), 8);
        // next kv head within a layer: one head_dim over.
        assert_eq!(cache.offset(0, 0, 1), 4);
        // next position: a full token stride (layers*kv_heads*head_dim = 16).
        assert_eq!(cache.offset(0, 1, 0), 16);
    }

    #[test]
    fn store_read_round_trip_with_f16_rounding() {
        let mut cache = LlamaKvCache::new(plan()).unwrap();
        cache.ensure_position_capacity(1).unwrap();

        // 0.1 is not representable in f16, so the stored value differs from the
        // input but reads back exactly as the rounded value.
        let key = [0.1_f32, 0.5, 1.0 / 3.0, 2.0];
        // 0.5 and 2.0 ARE representable and survive unchanged.
        let value = [0.5_f32, 0.25, 0.1, 100.0];
        cache.store_kv_head_row(1, 0, 1, &key, &value);

        let read_key = cache.key_row(1, 0, 1);
        let read_value = cache.value_row(1, 0, 1);

        // The rounding visibly changed 0.1 and 1/3.
        assert_ne!(read_key[0], 0.1_f32, "0.1 must be perturbed by f16 rounding");
        assert_ne!(read_key[2], 1.0 / 3.0, "1/3 must be perturbed by f16 rounding");
        // Reads return exactly the rounded value, bit for bit.
        for (i, &raw) in key.iter().enumerate() {
            assert_eq!(read_key[i].to_bits(), kv_round(raw).to_bits());
        }
        for (i, &raw) in value.iter().enumerate() {
            assert_eq!(read_value[i].to_bits(), kv_round(raw).to_bits());
        }
        // Representable values are unchanged.
        assert_eq!(read_value[0], 0.5_f32);
        assert_eq!(read_key[1], 0.5_f32);

        // A different (layer, position, kv_head) is untouched (still zero).
        assert_eq!(cache.key_row(0, 0, 0), &[0.0; 4]);
    }

    #[test]
    fn f16_round_trip_is_idempotent_and_canonicalizes_nan() {
        // Round-tripping an already-rounded value is a fixed point.
        let once = kv_round(0.1);
        assert_eq!(kv_round(once).to_bits(), once.to_bits());
        // NaN canonicalizes to a quiet NaN (sign|0x7E00) then expands to quiet.
        assert_eq!(f32_to_f16_kv(f32::NAN) & 0x7fff, 0x7e00);
        assert!(kv_round(f32::NAN).is_nan());
        // Overflow saturates to infinity.
        assert!(kv_round(70_000.0).is_infinite());
    }

    #[test]
    fn capacity_growth_and_bounds() {
        let mut cache = LlamaKvCache::new(plan()).unwrap();
        assert!(cache.can_append());
        cache.ensure_position_capacity(1).unwrap();
        // Grows in chunks with headroom.
        assert!(cache.allocated_sequence_length >= 1);
        // Exceeding the context is refused.
        let err = cache.ensure_position_capacity(4096).unwrap_err();
        assert!(matches!(err, EngineError::ShapeMismatch(_)));
    }

    #[test]
    fn position_advances_after_writes() {
        let mut cache = LlamaKvCache::new(plan()).unwrap();
        assert_eq!(cache.position(), 0);
        cache.advance_position();
        assert_eq!(cache.position(), 1);
    }
}
