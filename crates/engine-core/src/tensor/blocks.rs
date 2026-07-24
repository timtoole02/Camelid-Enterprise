//! Quantized block formats and scalar dequantization.
//!
//! Each `*Block` struct mirrors one GGUF wire block layout byte for byte and
//! decodes to f32 with a fixed, unfused arithmetic order: every dequant
//! expression is plain f32 multiply/add evaluated left to right, so decoded
//! values are bit-deterministic across runs and hosts.

use crate::{EngineError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}

impl TensorShape {
    pub fn from_gguf_dims(dims: &[u64]) -> Result<Self> {
        let dims = dims
            .iter()
            .map(|dim| {
                usize::try_from(*dim).map_err(|_| {
                    EngineError::InvalidTensorData(format!("dimension {dim} does not fit usize"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { dims })
    }

    pub fn element_count(&self) -> Result<usize> {
        self.dims.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                EngineError::InvalidTensorData("tensor element count overflow".to_string())
            })
        })
    }
}

/// Runtime element type after decode. Every quantized format decodes to f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDType {
    F32,
}

/// One decoded Q8_0 block: the wire f16 scale already widened to f32, plus the
/// 32 signed byte quants.
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0Block {
    pub scale: f32,
    pub quants: [i8; 32],
}

/// f32 -> IEEE f16 bits with round-to-nearest-even.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        // Inf / NaN (preserve a NaN payload bit so NaN stays NaN).
        let nan = if frac != 0 {
            0x0200 | ((frac >> 13) as u16 & 0x03ff)
        } else {
            0
        };
        return sign | 0x7c00 | nan;
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00; // overflow -> +/-inf
    }
    if half_exp <= 0 {
        // Subnormal half (or zero): shift the implicit-1 mantissa down.
        if half_exp < -10 {
            return sign; // underflow -> +/-0
        }
        let mant = frac | 0x0080_0000;
        let shift = (14 - half_exp) as u32; // 14..=24
        let mut half = (mant >> shift) as u16;
        let rem = mant & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }
    let mut half = ((half_exp as u32) << 10) | (frac >> 13);
    let rem = frac & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1; // mantissa carry propagates into the exponent correctly
    }
    sign | half as u16
}

/// IEEE f16 bits -> f32, exact for all 65536 inputs (subnormals expanded
/// without rounding; NaN payloads and signed zero preserved).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = u32::from(bits & 0x03ff);

    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut e = -14i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                let exp32 = u32::try_from(e + 127).expect("subnormal f16 exponent in range");
                sign | (exp32 << 23) | (mant << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = u32::from(exp) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(out)
}

// Quantization Constants
pub const Q8_BLOCK_SIZE: usize = 32;
/// Number of f32 values quantized into one Q8_0 block. Single source of truth
/// aliased to [`Q8_BLOCK_SIZE`] so the two names can never diverge.
pub const Q8_0_BLOCK_VALUES: usize = Q8_BLOCK_SIZE;
pub const Q4_0_BLOCK_BYTES: usize = 2 + (Q8_BLOCK_SIZE / 2);
pub const Q4_1_BLOCK_BYTES: usize = 4 + (Q8_BLOCK_SIZE / 2);
pub const Q5_0_BLOCK_BYTES: usize = 2 + 4 + (Q8_BLOCK_SIZE / 2);
pub const Q5_1_BLOCK_BYTES: usize = 4 + 4 + (Q8_BLOCK_SIZE / 2);
pub const QK_K_BLOCK_SIZE: usize = 256;
pub const Q2_K_BLOCK_BYTES: usize = 16 + 64 + 4;
pub const Q3_K_BLOCK_BYTES: usize = 32 + 64 + 12 + 2;
pub const Q4_K_BLOCK_BYTES: usize = 4 + 12 + 128;
pub const Q5_K_BLOCK_BYTES: usize = 4 + 12 + 32 + 128;
pub const Q6_K_BLOCK_BYTES: usize = 128 + 64 + 16 + 2;
pub const Q8_K_BLOCK_BYTES: usize = 292;
pub const IQ4_NL_BLOCK_BYTES: usize = 18;
// block_iq4_xs = f16 d(2) + scales_h u16(2) + scales_l[QK_K/64]=4 + qs[QK_K/2]=128 = 136 (4.25 bpw)
pub const IQ4_XS_BLOCK_BYTES: usize = 2 + 2 + (QK_K_BLOCK_SIZE / 64) + (QK_K_BLOCK_SIZE / 2);

/// The 16-entry non-linear codebook shared by the IQ4_NL and IQ4_XS formats,
/// as signed integers (the quantized weight magnitudes). Single source of truth.
pub const KVALUES_IQ4NL_I8: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// f32 view of [`KVALUES_IQ4NL_I8`], derived at compile time so the two can never diverge.
/// Used by the block decoders; an integer dot path uses the i8 table directly.
pub const KVALUES_IQ4NL: [f32; 16] = {
    let mut out = [0.0_f32; 16];
    let mut i = 0;
    while i < 16 {
        out[i] = KVALUES_IQ4NL_I8[i] as f32;
        i += 1;
    }
    out
};

#[inline(always)]
pub fn fast_f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = u32::from(bits & 0x7c00) >> 10;
    let fraction = u32::from(bits & 0x03ff);

    if exponent == 0 {
        if fraction == 0 {
            return f32::from_bits(sign);
        }
        f16_bits_to_f32(bits)
    } else if exponent == 0x1f {
        f32::from_bits(sign | 0x7f80_0000 | (fraction << 13))
    } else {
        f32::from_bits(sign | ((exponent + 112) << 23) | (fraction << 13))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q4_0Block {
    scale_bits: u16,
    values: [u8; Q8_BLOCK_SIZE / 2],
}

impl Q4_0Block {
    pub fn from_bytes(bytes: &[u8; Q4_0_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        values.copy_from_slice(&bytes[2..]);
        Self { scale_bits, values }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn unpack_values(&self) -> [i8; Q8_BLOCK_SIZE] {
        let mut out = [0_i8; Q8_BLOCK_SIZE];
        for (idx, &byte) in self.values.iter().enumerate() {
            out[idx] = ((byte & 0x0f) as i8) - 8;
            out[idx + 16] = ((byte >> 4) as i8) - 8;
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q4_1Block {
    scale_bits: u16,
    min_bits: u16,
    values: [u8; Q8_BLOCK_SIZE / 2],
}

impl Q4_1Block {
    pub fn from_bytes(bytes: &[u8; Q4_1_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let min_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        values.copy_from_slice(&bytes[4..]);
        Self {
            scale_bits,
            min_bits,
            values,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn min_f32(&self) -> f32 {
        fast_f16_to_f32(self.min_bits)
    }

    pub fn unpack_values(&self) -> [u8; Q8_BLOCK_SIZE] {
        let mut out = [0_u8; Q8_BLOCK_SIZE];
        for (idx, &byte) in self.values.iter().enumerate() {
            out[idx] = byte & 0x0f;
            out[idx + 16] = byte >> 4;
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q5_0Block {
    scale_bits: u16,
    high_bits: u32,
    values: [u8; Q8_BLOCK_SIZE / 2],
}

impl Q5_0Block {
    pub fn from_bytes(bytes: &[u8; Q5_0_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let high_bits = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        values.copy_from_slice(&bytes[6..]);
        Self {
            scale_bits,
            high_bits,
            values,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn unpack_values(&self) -> [i8; Q8_BLOCK_SIZE] {
        let mut out = [0_i8; Q8_BLOCK_SIZE];
        for (idx, &byte) in self.values.iter().enumerate() {
            let low_high = (((self.high_bits >> idx) & 1) as u8) << 4;
            let high_high = (((self.high_bits >> (idx + 16)) & 1) as u8) << 4;
            out[idx] = ((byte & 0x0f) | low_high) as i8 - 16;
            out[idx + 16] = ((byte >> 4) | high_high) as i8 - 16;
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q5_1Block {
    scale_bits: u16,
    min_bits: u16,
    high_bits: u32,
    values: [u8; Q8_BLOCK_SIZE / 2],
}

impl Q5_1Block {
    pub fn from_bytes(bytes: &[u8; Q5_1_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let min_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let high_bits = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        values.copy_from_slice(&bytes[8..]);
        Self {
            scale_bits,
            min_bits,
            high_bits,
            values,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn min_f32(&self) -> f32 {
        fast_f16_to_f32(self.min_bits)
    }

    pub fn unpack_values(&self) -> [u8; Q8_BLOCK_SIZE] {
        let mut out = [0_u8; Q8_BLOCK_SIZE];
        for (idx, &byte) in self.values.iter().enumerate() {
            let low_high = (((self.high_bits >> idx) & 1) as u8) << 4;
            let high_high = (((self.high_bits >> (idx + 16)) & 1) as u8) << 4;
            out[idx] = (byte & 0x0f) | low_high;
            out[idx + 16] = (byte >> 4) | high_high;
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q2KBlock {
    scales: [u8; QK_K_BLOCK_SIZE / 16],
    values: [u8; QK_K_BLOCK_SIZE / 4],
    scale_bits: u16,
    min_bits: u16,
}

impl Q2KBlock {
    pub fn from_bytes(bytes: &[u8; Q2_K_BLOCK_BYTES]) -> Self {
        let mut scales = [0_u8; QK_K_BLOCK_SIZE / 16];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 4];
        scales.copy_from_slice(&bytes[0..16]);
        values.copy_from_slice(&bytes[16..80]);
        let scale_bits = u16::from_le_bytes([bytes[80], bytes[81]]);
        let min_bits = u16::from_le_bytes([bytes[82], bytes[83]]);
        Self {
            scales,
            values,
            scale_bits,
            min_bits,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn min_f32(&self) -> f32 {
        fast_f16_to_f32(self.min_bits)
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let d_min = self.min_f32();
        let mut scale_idx = 0;

        for super_idx in 0..2 {
            let value_base = super_idx * 32;
            let out_base = super_idx * 128;
            let mut shift = 0;
            for group_idx in 0..4 {
                let low_scale = self.scales[scale_idx];
                scale_idx += 1;
                let low_d = d * (low_scale & 0x0f) as f32;
                let low_min = d_min * (low_scale >> 4) as f32;
                for l in 0..16 {
                    out[out_base + group_idx * 32 + l] =
                        low_d * ((self.values[value_base + l] >> shift) & 3) as f32 - low_min;
                }

                let high_scale = self.scales[scale_idx];
                scale_idx += 1;
                let high_d = d * (high_scale & 0x0f) as f32;
                let high_min = d_min * (high_scale >> 4) as f32;
                for l in 0..16 {
                    out[out_base + group_idx * 32 + 16 + l] = high_d
                        * ((self.values[value_base + 16 + l] >> shift) & 3) as f32
                        - high_min;
                }

                shift += 2;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q3KBlock {
    high_bits: [u8; QK_K_BLOCK_SIZE / 8],
    values: [u8; QK_K_BLOCK_SIZE / 4],
    scales: [u8; 12],
    scale_bits: u16,
}

impl Q3KBlock {
    pub fn from_bytes(bytes: &[u8; Q3_K_BLOCK_BYTES]) -> Self {
        let mut high_bits = [0_u8; QK_K_BLOCK_SIZE / 8];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 4];
        let mut scales = [0_u8; 12];
        high_bits.copy_from_slice(&bytes[0..32]);
        values.copy_from_slice(&bytes[32..96]);
        scales.copy_from_slice(&bytes[96..108]);
        let scale_bits = u16::from_le_bytes([bytes[108], bytes[109]]);
        Self {
            high_bits,
            values,
            scales,
            scale_bits,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    /// Expand the 12 packed scale bytes into 16 signed 6-bit sub-block scales
    /// (still carrying the +32 storage bias; the caller subtracts it).
    fn expanded_scales(&self) -> [i8; 16] {
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;

        let mut aux = [
            u32::from_le_bytes([
                self.scales[0],
                self.scales[1],
                self.scales[2],
                self.scales[3],
            ]),
            u32::from_le_bytes([
                self.scales[4],
                self.scales[5],
                self.scales[6],
                self.scales[7],
            ]),
            u32::from_le_bytes([
                self.scales[8],
                self.scales[9],
                self.scales[10],
                self.scales[11],
            ]),
            0,
        ];

        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

        let mut out = [0_i8; 16];
        for (chunk_idx, chunk) in aux.iter().enumerate() {
            for (byte_idx, byte) in chunk.to_le_bytes().iter().enumerate() {
                out[chunk_idx * 4 + byte_idx] = i8::from_le_bytes([*byte]);
            }
        }
        out
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let scales = self.expanded_scales();
        let mut scale_idx = 0;
        let mut high_mask = 1_u8;

        for super_idx in 0..2 {
            let value_base = super_idx * 32;
            let out_base = super_idx * 128;
            let mut shift = 0;
            for group_idx in 0..4 {
                let low_d = d * (scales[scale_idx] - 32) as f32;
                scale_idx += 1;
                for l in 0..16 {
                    // High-bit polarity is inverted: a SET mask bit means the
                    // element keeps its 2-bit value, a CLEAR bit subtracts 4.
                    let high = if self.high_bits[l] & high_mask != 0 {
                        0
                    } else {
                        4
                    };
                    let value = ((self.values[value_base + l] >> shift) & 3) as i8 - high;
                    out[out_base + group_idx * 32 + l] = low_d * value as f32;
                }

                let high_d = d * (scales[scale_idx] - 32) as f32;
                scale_idx += 1;
                for l in 0..16 {
                    let idx = 16 + l;
                    let high = if self.high_bits[idx] & high_mask != 0 {
                        0
                    } else {
                        4
                    };
                    let value = ((self.values[value_base + idx] >> shift) & 3) as i8 - high;
                    out[out_base + group_idx * 32 + 16 + l] = high_d * value as f32;
                }

                shift += 2;
                high_mask <<= 1;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q4KBlock {
    scale_bits: u16,
    min_bits: u16,
    scales: [u8; 12],
    values: [u8; QK_K_BLOCK_SIZE / 2],
}

impl Q4KBlock {
    pub fn from_bytes(bytes: &[u8; Q4_K_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let min_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales = [0_u8; 12];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 2];
        scales.copy_from_slice(&bytes[4..16]);
        values.copy_from_slice(&bytes[16..]);
        Self {
            scale_bits,
            min_bits,
            scales,
            values,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn min_f32(&self) -> f32 {
        fast_f16_to_f32(self.min_bits)
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let d_min = self.min_f32();
        for pair_idx in 0..4 {
            let low_scale_idx = pair_idx * 2;
            let high_scale_idx = low_scale_idx + 1;
            let (low_scale, low_min) = q4_k_scale_min(low_scale_idx, &self.scales);
            let (high_scale, high_min) = q4_k_scale_min(high_scale_idx, &self.scales);
            let low_scale = d * low_scale as f32;
            let high_scale = d * high_scale as f32;
            let low_min = d_min * low_min as f32;
            let high_min = d_min * high_min as f32;
            let value_base = pair_idx * 32;
            let out_base = pair_idx * 64;

            for l in 0..32 {
                let byte = self.values[value_base + l];
                out[out_base + l] = low_scale * (byte & 0x0f) as f32 - low_min;
                out[out_base + 32 + l] = high_scale * (byte >> 4) as f32 - high_min;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q5KBlock {
    scale_bits: u16,
    min_bits: u16,
    scales: [u8; 12],
    high_bits: [u8; QK_K_BLOCK_SIZE / 8],
    values: [u8; QK_K_BLOCK_SIZE / 2],
}

impl Q5KBlock {
    pub fn from_bytes(bytes: &[u8; Q5_K_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let min_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales = [0_u8; 12];
        let mut high_bits = [0_u8; QK_K_BLOCK_SIZE / 8];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 2];
        scales.copy_from_slice(&bytes[4..16]);
        high_bits.copy_from_slice(&bytes[16..48]);
        values.copy_from_slice(&bytes[48..]);
        Self {
            scale_bits,
            min_bits,
            scales,
            high_bits,
            values,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn min_f32(&self) -> f32 {
        fast_f16_to_f32(self.min_bits)
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let d_min = self.min_f32();
        let mut u1 = 1_u8;
        let mut u2 = 2_u8;

        for pair_idx in 0..4 {
            let low_scale_idx = pair_idx * 2;
            let high_scale_idx = low_scale_idx + 1;
            let (low_scale, low_min) = q4_k_scale_min(low_scale_idx, &self.scales);
            let (high_scale, high_min) = q4_k_scale_min(high_scale_idx, &self.scales);
            let low_scale = d * low_scale as f32;
            let high_scale = d * high_scale as f32;
            let low_min = d_min * low_min as f32;
            let high_min = d_min * high_min as f32;
            let value_base = pair_idx * 32;
            let out_base = pair_idx * 64;

            for l in 0..32 {
                let byte = self.values[value_base + l];
                let qh = self.high_bits[l];
                let low = (byte & 0x0f) + if qh & u1 != 0 { 16 } else { 0 };
                let high = (byte >> 4) + if qh & u2 != 0 { 16 } else { 0 };
                out[out_base + l] = low_scale * low as f32 - low_min;
                out[out_base + 32 + l] = high_scale * high as f32 - high_min;
            }

            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// 6-bit (scale, min) extraction shared by Q4_K and Q5_K, `idx` in 0..8. For
/// idx >= 4 the low nibble of `scales[idx+4]` is the scale and the high nibble
/// the min, each topped with the 2 spilled high bits of `scales[idx-4]` /
/// `scales[idx]`.
#[inline]
fn q4_k_scale_min(idx: usize, scales: &[u8; 12]) -> (u8, u8) {
    if idx < 4 {
        (scales[idx] & 63, scales[idx + 4] & 63)
    } else {
        (
            (scales[idx + 4] & 0x0f) | ((scales[idx - 4] >> 6) << 4),
            (scales[idx + 4] >> 4) | ((scales[idx] >> 6) << 4),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q6KBlock {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
    scale_bits: u16,
}

impl Q6KBlock {
    pub fn from_bytes(bytes: &[u8; Q6_K_BLOCK_BYTES]) -> Self {
        let mut ql = [0_u8; 128];
        let mut qh = [0_u8; 64];
        let mut scales = [0_i8; 16];
        ql.copy_from_slice(&bytes[0..128]);
        qh.copy_from_slice(&bytes[128..192]);
        for (scale, &byte) in scales.iter_mut().zip(&bytes[192..208]) {
            *scale = i8::from_le_bytes([byte]);
        }
        let scale_bits = u16::from_le_bytes([bytes[208], bytes[209]]);
        Self {
            ql,
            qh,
            scales,
            scale_bits,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let mut ql_offset = 0;
        let mut qh_offset = 0;
        let mut scale_offset = 0;

        for n in (0..QK_K_BLOCK_SIZE).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let qh = self.qh[qh_offset + l];
                let q1 = ((self.ql[ql_offset + l] & 0x0f) | ((qh & 0x03) << 4)) as i8 - 32;
                let q2 =
                    ((self.ql[ql_offset + l + 32] & 0x0f) | (((qh >> 2) & 0x03) << 4)) as i8 - 32;
                let q3 = ((self.ql[ql_offset + l] >> 4) | (((qh >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 =
                    ((self.ql[ql_offset + l + 32] >> 4) | (((qh >> 6) & 0x03) << 4)) as i8 - 32;

                out[n + l] = d * self.scales[scale_offset + is] as f32 * q1 as f32;
                out[n + l + 32] = d * self.scales[scale_offset + is + 2] as f32 * q2 as f32;
                out[n + l + 64] = d * self.scales[scale_offset + is + 4] as f32 * q3 as f32;
                out[n + l + 96] = d * self.scales[scale_offset + is + 6] as f32 * q4 as f32;
            }

            ql_offset += 64;
            qh_offset += 32;
            scale_offset += 8;
        }
    }
}

/// Q8_K block: unlike every other block format the scale `d` is a raw LE f32
/// on the wire, not f16. `bsums` are per-16-element quant sums retained for
/// integer dot paths; `dequantize` does not use them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q8KBlock {
    d: f32,
    qs: [i8; QK_K_BLOCK_SIZE],
    bsums: [i16; QK_K_BLOCK_SIZE / 16],
}

impl Q8KBlock {
    pub fn from_bytes(bytes: &[u8; Q8_K_BLOCK_BYTES]) -> Self {
        let d = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let mut qs = [0_i8; 256];
        for (i, &byte) in qs.iter_mut().zip(&bytes[4..260]) {
            *i = byte as i8;
        }
        let mut bsums = [0_i16; 16];
        for (i, bsum) in bsums.iter_mut().enumerate() {
            let offset = 260 + i * 2;
            *bsum = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        }
        Self { d, qs, bsums }
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.d;
        for (out_value, &q) in out.iter_mut().zip(&self.qs) {
            *out_value = d * q as f32;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IQ4NLBlock {
    d: u16,
    qs: [u8; 16],
}

impl IQ4NLBlock {
    pub fn from_bytes(bytes: &[u8; IQ4_NL_BLOCK_BYTES]) -> Self {
        let d = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut qs = [0_u8; 16];
        qs.copy_from_slice(&bytes[2..18]);
        Self { d, qs }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.d)
    }

    pub fn dequantize(&self, out: &mut [f32; 32]) {
        let d = self.scale_f32();
        for j in 0..16 {
            let byte = self.qs[j];
            out[j] = d * KVALUES_IQ4NL[(byte & 0x0F) as usize];
            out[j + 16] = d * KVALUES_IQ4NL[(byte >> 4) as usize];
        }
    }
}

/// IQ4_XS super-block (256 weights in 136 bytes, 4.25 bpw). One f16 super-block scale, eight
/// 6-bit sub-block scales (biased by -32, low nibble in `scales_l`, high 2 bits in `scales_h`),
/// and 128 bytes of 4-bit codebook indices into [`KVALUES_IQ4NL`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IQ4XSBlock {
    d: u16,
    scales_h: u16,
    scales_l: [u8; QK_K_BLOCK_SIZE / 64],
    qs: [u8; QK_K_BLOCK_SIZE / 2],
}

impl IQ4XSBlock {
    pub fn from_bytes(bytes: &[u8; IQ4_XS_BLOCK_BYTES]) -> Self {
        let d = u16::from_le_bytes([bytes[0], bytes[1]]);
        let scales_h = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales_l = [0_u8; QK_K_BLOCK_SIZE / 64];
        scales_l.copy_from_slice(&bytes[4..4 + QK_K_BLOCK_SIZE / 64]);
        let mut qs = [0_u8; QK_K_BLOCK_SIZE / 2];
        qs.copy_from_slice(&bytes[4 + QK_K_BLOCK_SIZE / 64..IQ4_XS_BLOCK_BYTES]);
        Self {
            d,
            scales_h,
            scales_l,
            qs,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.d)
    }

    /// Effective f32 scale of sub-block `ib` (0..8): the 6-bit scale is the low nibble from
    /// `scales_l[ib/2]` (even/odd nibble) OR'd with the high 2 bits from `scales_h`, biased -32.
    #[inline]
    fn sub_block_scale(&self, ib: usize) -> f32 {
        self.scale_f32() * self.sub_block_scale_int(ib) as f32
    }

    /// Integer part of sub-block `ib`'s scale: `ls - 32` (before the f16 super-scale). An
    /// integer dot path multiplies this by the super-scale and the Q8_K activation scale.
    #[inline]
    pub fn sub_block_scale_int(&self, ib: usize) -> i32 {
        let low = (self.scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
        let high = ((self.scales_h >> (2 * ib)) & 0x3) as u8;
        i32::from(low | (high << 4)) - 32
    }

    /// The 128 raw 4-bit codebook-index bytes (two indices per byte).
    #[inline]
    pub fn qs(&self) -> &[u8; QK_K_BLOCK_SIZE / 2] {
        &self.qs
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        for ib in 0..QK_K_BLOCK_SIZE / 32 {
            let dl = self.sub_block_scale(ib);
            let qs = &self.qs[ib * 16..ib * 16 + 16];
            let base = ib * 32;
            for j in 0..16 {
                out[base + j] = dl * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize];
                out[base + j + 16] = dl * KVALUES_IQ4NL[(qs[j] >> 4) as usize];
            }
        }
    }
}

// Decoding Block Helpers
pub fn decode_q4_0_blocks(bytes: &[u8]) -> Result<Vec<Q4_0Block>> {
    if !bytes.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "Q4_0 byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q4_0_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q4_0_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q4_0_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q4_0Block::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q4_1_blocks(bytes: &[u8]) -> Result<Vec<Q4_1Block>> {
    if !bytes.len().is_multiple_of(Q4_1_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "Q4_1 byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q4_1_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q4_1_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q4_1_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q4_1Block::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q5_0_blocks(bytes: &[u8]) -> Result<Vec<Q5_0Block>> {
    if !bytes.len().is_multiple_of(Q5_0_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "Q5_0 byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q5_0_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q5_0_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q5_0_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q5_0Block::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q5_1_blocks(bytes: &[u8]) -> Result<Vec<Q5_1Block>> {
    if !bytes.len().is_multiple_of(Q5_1_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "Q5_1 byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q5_1_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q5_1_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q5_1_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q5_1Block::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q2_k_blocks(bytes: &[u8]) -> Result<Vec<Q2KBlock>> {
    if !bytes.len().is_multiple_of(Q2_K_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "Q2_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q2_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q2_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q2_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q2KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q3_k_blocks(bytes: &[u8]) -> Result<Vec<Q3KBlock>> {
    if !bytes.len().is_multiple_of(Q3_K_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "Q3_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q3_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q3_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q3_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q3KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q4_k_blocks(bytes: &[u8]) -> Result<Vec<Q4KBlock>> {
    if !bytes.len().is_multiple_of(Q4_K_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "Q4_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q4_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q4_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q4_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q4KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q5_k_blocks(bytes: &[u8]) -> Result<Vec<Q5KBlock>> {
    if !bytes.len().is_multiple_of(Q5_K_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "Q5_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q5_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q5_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q5_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q5KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q6_k_blocks(bytes: &[u8]) -> Result<Vec<Q6KBlock>> {
    if !bytes.len().is_multiple_of(Q6_K_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "Q6_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q6_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q6_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q6_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q6KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q8_k_blocks(bytes: &[u8]) -> Result<Vec<Q8KBlock>> {
    if !bytes.len().is_multiple_of(Q8_K_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "Q8_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q8_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q8_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q8_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q8KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_iq4_nl_blocks(bytes: &[u8]) -> Result<Vec<IQ4NLBlock>> {
    if !bytes.len().is_multiple_of(IQ4_NL_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "IQ4_NL byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            IQ4_NL_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(IQ4_NL_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; IQ4_NL_BLOCK_BYTES] = chunk.try_into().unwrap();
            IQ4NLBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_iq4_xs_blocks(bytes: &[u8]) -> Result<Vec<IQ4XSBlock>> {
    if !bytes.len().is_multiple_of(IQ4_XS_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "IQ4_XS byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            IQ4_XS_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(IQ4_XS_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; IQ4_XS_BLOCK_BYTES] = chunk.try_into().unwrap();
            IQ4XSBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

// Flat dequantization to f32 helpers
pub fn decode_f32_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if bytes.len() != expected_elements * 4 {
        return Err(EngineError::InvalidTensorData(format!(
            "tensor {name} f32 byte length {} does not match expected {}",
            bytes.len(),
            expected_elements * 4
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("exact chunk length")))
        .collect())
}

pub fn decode_f16_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if bytes.len() != expected_elements * 2 {
        return Err(EngineError::InvalidTensorData(format!(
            "tensor {name} f16 byte length {} does not match expected {}",
            bytes.len(),
            expected_elements * 2
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            f16_bits_to_f32(u16::from_le_bytes(
                chunk.try_into().expect("exact chunk length"),
            ))
        })
        .collect())
}

/// bf16 -> f32 is the exact bit-widening `f32::from_bits(u32::from(u16) << 16)`:
/// bf16 stores the high 16 bits of the IEEE-754 f32 encoding, so widening
/// appends 16 zero low bits — lossless, no rounding, bit-deterministic.
pub fn decode_bf16_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if bytes.len() != expected_elements * 2 {
        return Err(EngineError::InvalidTensorData(format!(
            "tensor {name} bf16 byte length {} does not match expected {}",
            bytes.len(),
            expected_elements * 2
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            f32::from_bits(
                u32::from(u16::from_le_bytes(
                    chunk.try_into().expect("exact chunk length"),
                )) << 16,
            )
        })
        .collect())
}

pub fn decode_q8_0_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q8_0_blocks(name, bytes, expected_elements)?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        for q in block.quants {
            out.push(block.scale * f32::from(q));
        }
    }
    Ok(out)
}

pub fn decode_q8_0_blocks(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<Q8_0Block>> {
    const BLOCK_VALUES: usize = 32;
    const BLOCK_BYTES: usize = 34;
    if !expected_elements.is_multiple_of(BLOCK_VALUES) {
        return Err(EngineError::InvalidTensorData(format!(
            "tensor {name} q8_0 element count {expected_elements} is not divisible by {BLOCK_VALUES}"
        )));
    }
    let expected_bytes = expected_elements / BLOCK_VALUES * BLOCK_BYTES;
    if bytes.len() != expected_bytes {
        return Err(EngineError::InvalidTensorData(format!(
            "tensor {name} q8_0 byte length {} does not match expected {expected_bytes}",
            bytes.len()
        )));
    }
    let mut blocks = Vec::with_capacity(expected_elements / BLOCK_VALUES);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut quants = [0_i8; BLOCK_VALUES];
        for (idx, q) in block[2..].iter().enumerate() {
            quants[idx] = *q as i8;
        }
        blocks.push(Q8_0Block { scale, quants });
    }
    Ok(blocks)
}

pub fn decode_q4_0_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q4_0_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let scale = block.scale_f32();
        for val in block.unpack_values() {
            out.push(val as f32 * scale);
        }
    }
    Ok(out)
}

// ---- Ternary TQ1_0 / TQ2_0 flat dequantization to f32 ----
// The element ORDER and the u8-truncating base-3 decode are load-bearing: both
// must match the wire producer bit for bit.
const TQ1_0_BLOCK_BYTES: usize = 54; // qs[48] + qh[4] + f16 d  (1.69 bpw over 256 weights)
const TQ2_0_BLOCK_BYTES: usize = 66; // qs[64] + f16 d          (2.06 bpw over 256 weights)

pub fn decode_tq2_0_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(TQ2_0_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "{name}: TQ2_0 byte length {} is not aligned to {TQ2_0_BLOCK_BYTES}-byte blocks",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(expected_elements);
    for block in bytes.chunks_exact(TQ2_0_BLOCK_BYTES) {
        let qs = &block[0..64];
        let d = f16_bits_to_f32(u16::from_le_bytes([block[64], block[65]]));
        // Plane-major order: for j in {0,32}; for l in 0..4; for m in 0..32:
        // q = (qs[j+m] >> (l*2)) & 3; emit (q-1)*d.
        let mut j = 0usize;
        while j < 64 {
            for l in 0..4 {
                for m in 0..32 {
                    let q = ((qs[j + m] >> (l * 2)) & 3) as i32;
                    out.push((q - 1) as f32 * d);
                }
            }
            j += 32;
        }
    }
    Ok(out)
}

pub fn decode_tq1_0_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(TQ1_0_BLOCK_BYTES) {
        return Err(EngineError::InvalidTensorData(format!(
            "{name}: TQ1_0 byte length {} is not aligned to {TQ1_0_BLOCK_BYTES}-byte blocks",
            bytes.len()
        )));
    }
    // pow3[n] for the base-3 digit extraction: trit_n = ((u8(qs*pow3[n]) * 3) >> 8) - 1.
    const POW3: [u32; 5] = [1, 3, 9, 27, 81];
    let mut out = Vec::with_capacity(expected_elements);
    for block in bytes.chunks_exact(TQ1_0_BLOCK_BYTES) {
        let qs = &block[0..48];
        let qh = &block[48..52];
        let d = f16_bits_to_f32(u16::from_le_bytes([block[52], block[53]]));
        // part 1: j=0 (qs[0..32]), 5 trit planes x 32
        for &pw in POW3.iter() {
            #[allow(clippy::needless_range_loop)]
            for m in 0..32 {
                let q = (qs[m] as u32).wrapping_mul(pw) as u8;
                let xi = (((q as u16) * 3) >> 8) as i32;
                out.push((xi - 1) as f32 * d);
            }
        }
        // part 2: j=32 (qs[32..48]), 5 trit planes x 16
        for &pw in POW3.iter() {
            for m in 0..16 {
                let q = (qs[32 + m] as u32).wrapping_mul(pw) as u8;
                let xi = (((q as u16) * 3) >> 8) as i32;
                out.push((xi - 1) as f32 * d);
            }
        }
        // part 3: qh (4 bytes), 4 trit planes x 4
        for &pw in POW3.iter().take(4) {
            #[allow(clippy::needless_range_loop)]
            for jj in 0..4 {
                let q = (qh[jj] as u32).wrapping_mul(pw) as u8;
                let xi = (((q as u16) * 3) >> 8) as i32;
                out.push((xi - 1) as f32 * d);
            }
        }
    }
    Ok(out)
}

pub fn decode_q4_1_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q4_1_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let scale = block.scale_f32();
        let min = block.min_f32();
        for val in block.unpack_values() {
            out.push(val as f32 * scale + min);
        }
    }
    Ok(out)
}

pub fn decode_q5_0_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q5_0_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let scale = block.scale_f32();
        for val in block.unpack_values() {
            out.push(val as f32 * scale);
        }
    }
    Ok(out)
}

pub fn decode_q5_1_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q5_1_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let scale = block.scale_f32();
        let min = block.min_f32();
        for val in block.unpack_values() {
            out.push(val as f32 * scale + min);
        }
    }
    Ok(out)
}

pub fn decode_q2_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q2_k_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub fn decode_q3_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q3_k_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub fn decode_q4_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q4_k_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub fn decode_q5_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q5_k_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub fn decode_q6_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q6_k_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub fn decode_q8_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q8_k_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub fn decode_iq4_nl_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_iq4_nl_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; 32];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub fn decode_iq4_xs_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_iq4_xs_blocks(bytes)
        .map_err(|e| EngineError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

// ---- NVFP4 format core -------------------------------------------------------------------
//
// Wire layout (type id 40): 36 bytes per 64 elements — `d[4]` UE4M3 sub-block scales (one
// per 16 elements) FIRST, then `qs[32]` packed E2M1 nibbles. Sub-block `s` (0..3) owns
// `qs[s*8 .. s*8+7]`; the LOW nibble of byte `s*8+j` is element `s*16+j`, the HIGH nibble
// element `s*16+8+j` (a half/half split within the sub-block, not adjacent pairing).
//
// WHY TWO DECODE SEMANTICS COEXIST:
// - Per-block dequant ([`nvfp4_wire_block_dequant`] and the block loop inside
//   [`decode_nvfp4_tensor`]) matches the format's reference CPU dequantizer bit for bit:
//   scale byte 0x7F is the NaN sentinel and FLUSHES to d = 0.0 silently. This keeps every
//   decoded value bit-identical to the reference, whatever the bytes say.
// - The LOAD path ([`decode_nvfp4_tensor`]) FAILS CLOSED first: it scans every block's
//   `d[4]` and refuses tensors carrying a NaN-sentinel scale byte (0x7F or 0xFF) with a
//   machine-readable error, because a sentinel in a weight file means the quantizer saw
//   garbage and silently zeroing 16 weights per hit would be quiet model corruption.
//   Files that pass admission therefore never contain sentinels, so both semantics agree
//   on every tensor that actually runs; the bit-exact flush only ever fires in
//   fixture/parity harnesses that feed crafted blocks below the load path.
//
// Sentinel subtlety the golden fixtures lock in (`nvfp4_ue4m3_table.json`): the reference
// CPU decode checks the RAW byte (`x == 0x7F`), so 0xFF is NOT flushed — it decodes through
// exp/man extraction to 240.0. The reference GPU mirror flushes both 0x7F and 0xFF, which
// is exactly why the load path refuses both bytes: the two reference backends disagree on
// 0xFF, and refusing at admission avoids that ambiguity entirely.

/// NVFP4 values per wire super-block.
pub const NVFP4_VALUES_PER_BLOCK: usize = 64;

/// NVFP4 wire bytes per super-block: `d[4]` UE4M3 scales then `qs[32]` nibbles.
pub const NVFP4_WIRE_BYTES_PER_BLOCK: usize = 36;

/// NVFP4 sub-block width: one UE4M3 scale byte per 16 elements.
pub const NVFP4_SUB_BLOCK_VALUES: usize = 16;

/// The E2M1 element magnitudes DOUBLED (true magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6);
/// nibble bit 3 selects the sign half.
/// THE PAIR RULE: this doubling is paired with the extra 0.5 factor baked into
/// [`UE4M3_TO_F32`] — the two conventions must always travel together, or every
/// decoded value is off by 2x or 0.5x.
pub const KVALUES_MXFP4_I8: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// f32 view of [`KVALUES_MXFP4_I8`], derived at compile time so the two can never
/// diverge (same idiom as [`KVALUES_IQ4NL`]).
pub const KVALUES_MXFP4: [f32; 16] = {
    let mut out = [0.0_f32; 16];
    let mut i = 0;
    while i < 16 {
        out[i] = KVALUES_MXFP4_I8[i] as f32;
        i += 1;
    }
    out
};

/// One UE4M3 scale byte -> f32, bit-for-bit: raw bytes 0x00 and 0x7F return 0.0
/// (0x7F is the NaN sentinel, FLUSHED — and the check is on the raw byte, so 0xFF
/// is NOT flushed and decodes to 240.0; see the module comment above). Otherwise
/// exp = bits 6..3 (bias 7), man = bits 2..0; exp == 0 is subnormal `man * 2^-9`,
/// else `(1 + man/8) * 2^(exp-7)`; the result carries the extra 0.5 pair-rule
/// factor. Every step multiplies exact values by powers of two, so const
/// evaluation cannot round.
const fn ue4m3_to_f32_const(byte: u8) -> f32 {
    if byte == 0x00 || byte == 0x7F {
        return 0.0;
    }
    let exp = ((byte >> 3) & 0xF) as i32;
    let man = (byte & 0x7) as f32;
    let raw = if exp == 0 {
        // subnormal: man * 2^-9
        man * f32::from_bits((127 - 9) << 23)
    } else {
        // normal: (1 + man/8) * 2^(exp-7); exp-7 in -6..=8 so the power is a
        // normal f32 built directly from its biased exponent
        (1.0 + man / 8.0) * f32::from_bits(((exp - 7 + 127) as u32) << 23)
    };
    raw * 0.5
}

/// Precomputed 256-entry UE4M3 decode table (see [`ue4m3_to_f32_const`]); anchored
/// bit-exactly to `tests/fixtures/dequant/nvfp4_ue4m3_table.json`.
pub const UE4M3_TO_F32: [f32; 256] = {
    let mut out = [0.0_f32; 256];
    let mut b = 0usize;
    while b < 256 {
        out[b] = ue4m3_to_f32_const(b as u8);
        b += 1;
    }
    out
};

/// Scan NVFP4 wire bytes for NaN-sentinel UE4M3 scale bytes (0x7F / 0xFF) in any
/// block's `d[4]`, returning the FIRST offending block index. This is the single
/// definition of load-time sentinel refusal. Scans whole 36-byte blocks only;
/// callers validate total length separately.
pub fn nvfp4_find_nan_scale(bytes: &[u8]) -> Option<usize> {
    bytes
        .chunks_exact(NVFP4_WIRE_BYTES_PER_BLOCK)
        .position(|block| block[..4].iter().any(|&b| b == 0x7F || b == 0xFF))
}

/// Decode one 36-byte NVFP4 wire block into 64 f32 values: per sub-block `s`,
/// `d = UE4M3_TO_F32[d[s]]`, low nibble of `qs[s*8+j]` -> element `s*16+j`, high
/// nibble -> element `s*16+8+j`, value = `KVALUES_MXFP4[nibble] * d`. Negative
/// codes (9..15) under a zero scale produce -0.0 (the i8-derived f32 sign
/// survives the multiply).
fn nvfp4_block_decode_into(out: &mut [f32], block: &[u8]) {
    debug_assert_eq!(block.len(), NVFP4_WIRE_BYTES_PER_BLOCK);
    debug_assert_eq!(out.len(), NVFP4_VALUES_PER_BLOCK);
    for s in 0..4 {
        let d = UE4M3_TO_F32[block[s] as usize];
        for j in 0..8 {
            let byte = block[4 + s * 8 + j];
            out[s * NVFP4_SUB_BLOCK_VALUES + j] = KVALUES_MXFP4[(byte & 0x0F) as usize] * d;
            out[s * NVFP4_SUB_BLOCK_VALUES + 8 + j] = KVALUES_MXFP4[(byte >> 4) as usize] * d;
        }
    }
}

/// Dequantize a single NVFP4 wire super-block into 64 f32 values, bit-exact for
/// ANY byte pattern. Deliberately NO NaN-sentinel scan here: scale byte 0x7F
/// flushes to d = 0.0 (and raw 0xFF decodes to 240.0); the LOAD path
/// ([`decode_nvfp4_tensor`]) is the seam that fails closed on sentinel bytes —
/// see the module comment above for why the two semantics coexist.
pub fn nvfp4_wire_block_dequant(block_bytes: &[u8]) -> [f32; NVFP4_VALUES_PER_BLOCK] {
    debug_assert_eq!(block_bytes.len(), NVFP4_WIRE_BYTES_PER_BLOCK);
    let mut out = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
    nvfp4_block_decode_into(&mut out, block_bytes);
    out
}

/// Flat NVFP4 tensor dequantization for the LOAD path — mirrors
/// [`decode_q4_k_tensor`]'s shape, plus the fail-closed sentinel scan (see the
/// module comment above for why this deliberately diverges from the bit-exact
/// per-block seam on sentinel-bearing bytes).
pub fn decode_nvfp4_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if !expected_elements.is_multiple_of(NVFP4_VALUES_PER_BLOCK) {
        return Err(EngineError::InvalidTensorData(format!(
            "{name}: NVFP4 element count {expected_elements} is not a multiple of \
             {NVFP4_VALUES_PER_BLOCK}"
        )));
    }
    let blocks = expected_elements / NVFP4_VALUES_PER_BLOCK;
    let expected_bytes = blocks * NVFP4_WIRE_BYTES_PER_BLOCK;
    if bytes.len() != expected_bytes {
        return Err(EngineError::InvalidTensorData(format!(
            "{name}: NVFP4 wire length {} != {blocks} blocks * {NVFP4_WIRE_BYTES_PER_BLOCK} \
             bytes = {expected_bytes}",
            bytes.len()
        )));
    }
    // Fail closed: refuse NaN-sentinel scale bytes at load. A file that admits
    // never reaches the bit-exact flush below.
    if let Some(block_idx) = nvfp4_find_nan_scale(bytes) {
        return Err(EngineError::InvalidTensorData(format!(
            "{name}: NVFP4 block {block_idx} carries a NaN-sentinel UE4M3 scale byte \
             (0x7F/0xFF); refusing at load (per-block dequant stays bit-exact)"
        )));
    }
    let mut out = vec![0.0_f32; expected_elements];
    for (i, block) in bytes.chunks_exact(NVFP4_WIRE_BYTES_PER_BLOCK).enumerate() {
        nvfp4_block_decode_into(
            &mut out[i * NVFP4_VALUES_PER_BLOCK..(i + 1) * NVFP4_VALUES_PER_BLOCK],
            block,
        );
    }
    Ok(out)
}

/// f32 -> UE4M3 scale byte — TEST-ANCHORING ONLY. This crate is consume-side;
/// the encoder exists so the encode golden vectors and round-trip property
/// tests can reproduce the reference quantizer's wire bytes, and is not an
/// exported quantizer surface. Semantics locked by `nvfp4_encode_vectors.json`:
/// NaN/<=0 -> 0x00; input domain clamps at 448.0; normal path rounds HALF-UP on
/// the 4th mantissa bit (with carry into the exponent); exp >= 15 saturates to
/// 0x7E (the encoder never emits 0x78..0x7D, 0x7F, or any byte with bit 7 set);
/// subnormal path rounds half-up via `(x * 512 + 0.5)` truncation.
#[cfg(test)]
pub(crate) fn fp32_to_ue4m3(x: f32) -> u8 {
    // NaN and every x <= 0 return 0.
    if x.is_nan() || x <= 0.0 {
        return 0;
    }
    let x = if x > 448.0 { 448.0 } else { x };
    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x7) as i32;
    let mut ue_exp = fp32_exp + 7;
    if ue_exp <= 0 {
        // subnormal: round-half-up on man * 512 (truncation of positive value)
        let man = ((x * 512.0 + 0.5) as i32).min(7);
        if man < 1 {
            return 0;
        }
        return man as u8;
    }
    if ue_exp >= 15 {
        return 0x7E; // saturate to max finite code
    }
    let round_bit = ((bits >> 19) & 1) as i32;
    let mut ue_man = fp32_man + round_bit;
    if ue_man > 7 {
        ue_man = 0;
        ue_exp += 1;
        if ue_exp >= 15 {
            return 0x7E;
        }
    }
    ((ue_exp << 3) | ue_man) as u8
}

/// Nearest-code search — TEST-ANCHORING ONLY (see [`fp32_to_ue4m3`]).
/// Exhaustive nearest search over `KVALUES_MXFP4[i] * d`, strict `<` so the FIRST
/// index wins exact ties (scan order 0..15) — not IEEE round-nearest-even. NaN
/// inputs never beat the initial candidate, so they quantize to code 0.
#[cfg(test)]
fn nvfp4_best_index(x: f32, d: f32) -> u8 {
    let mut best_index = 0usize;
    let mut best_err = (KVALUES_MXFP4[0] * d - x).abs();
    for (i, kv) in KVALUES_MXFP4.iter().enumerate().skip(1) {
        let err = (kv * d - x).abs();
        if err < best_err {
            best_index = i;
            best_err = err;
        }
    }
    best_index as u8
}

/// Encode one 64-element row into a 36-byte NVFP4 wire block — TEST-ANCHORING
/// ONLY (this crate is consume-side; the golden quantizer is external). Per
/// sub-block: amax via a NaN-insensitive `<` comparison (all-NaN rows therefore
/// encode to an all-zero wire, and +/-Inf rows saturate the scale to 0x7E while
/// the nearest-code search leaves every element at code 0), scale byte =
/// `fp32_to_ue4m3(amax / 6)`, elements quantized against the DECODED scale via
/// first-wins nearest-LUT search.
#[cfg(test)]
pub(crate) fn encode_nvfp4_block(
    x: &[f32; NVFP4_VALUES_PER_BLOCK],
) -> [u8; NVFP4_WIRE_BYTES_PER_BLOCK] {
    let mut wire = [0u8; NVFP4_WIRE_BYTES_PER_BLOCK];
    for s in 0..4 {
        let xb = &x[s * NVFP4_SUB_BLOCK_VALUES..(s + 1) * NVFP4_SUB_BLOCK_VALUES];
        let mut amax = 0.0_f32;
        for &v in xb {
            if amax < v.abs() {
                amax = v.abs();
            }
        }
        let ue = fp32_to_ue4m3(amax / 6.0);
        wire[s] = ue;
        let d = UE4M3_TO_F32[ue as usize];
        for j in 0..8 {
            let lo = nvfp4_best_index(xb[j], d);
            let hi = nvfp4_best_index(xb[8 + j], d);
            wire[4 + s * 8 + j] = lo | (hi << 4);
        }
    }
    wire
}

/// NVFP4 encode-side anchoring + property loops. The decode-side golden suites
/// (ue4m3 table, 4096-pair decode table, nibble probes, 10k random + real GGUF
/// blocks, fail-closed seam) live in `tests/nvfp4_format.rs`; these unit tests
/// stay inline because [`encode_nvfp4_block`] / [`fp32_to_ue4m3`] are
/// deliberately not exported (this crate is consume-side; quantizer ownership
/// is external).
#[cfg(test)]
mod nvfp4_tests {
    use super::{
        decode_nvfp4_tensor, encode_nvfp4_block, fp32_to_ue4m3, nvfp4_block_decode_into,
        KVALUES_MXFP4, NVFP4_VALUES_PER_BLOCK, NVFP4_WIRE_BYTES_PER_BLOCK, UE4M3_TO_F32,
    };

    fn fixture_json(name: &str) -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("dequant")
            .join(name);
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
        let v: serde_json::Value =
            serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{name} parses: {e}"));
        v
    }

    fn hex_u32(h: &str) -> u32 {
        u32::from_str_radix(h, 16).unwrap_or_else(|e| panic!("bad hex u32 {h:?}: {e}"))
    }

    fn hex_row_bits(h: &str) -> Vec<u32> {
        assert!(h.len().is_multiple_of(8));
        (0..h.len())
            .step_by(8)
            .map(|i| hex_u32(&h[i..i + 8]))
            .collect()
    }

    /// Minimal RFC 4648 base64 decoder (fixtures only; no base64 dependency).
    fn b64_decode(s: &str) -> Vec<u8> {
        let mut table = [255u8; 256];
        for (i, c) in (b'A'..=b'Z').enumerate() {
            table[c as usize] = i as u8;
        }
        for (i, c) in (b'a'..=b'z').enumerate() {
            table[c as usize] = 26 + i as u8;
        }
        for (i, c) in (b'0'..=b'9').enumerate() {
            table[c as usize] = 52 + i as u8;
        }
        table[b'+' as usize] = 62;
        table[b'/' as usize] = 63;
        let bytes: Vec<u8> = s.bytes().filter(|&b| b != b'=').collect();
        let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
        for chunk in bytes.chunks(4) {
            let mut acc = 0u32;
            for (k, &b) in chunk.iter().enumerate() {
                let v = table[b as usize];
                assert_ne!(v, 255, "bad base64 byte {b}");
                acc |= u32::from(v) << (18 - 6 * k);
            }
            out.push((acc >> 16) as u8);
            if chunk.len() > 2 {
                out.push((acc >> 8) as u8);
            }
            if chunk.len() > 3 {
                out.push(acc as u8);
            }
        }
        out
    }

    fn decode_block_via_tensor_path(wire: &[u8]) -> [f32; NVFP4_VALUES_PER_BLOCK] {
        let out = decode_nvfp4_tensor("nvfp4-unit-test", wire, NVFP4_VALUES_PER_BLOCK)
            .expect("clean block decodes");
        let mut arr = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
        arr.copy_from_slice(&out);
        arr
    }

    fn assert_bits(got: &[f32], want_bits: &[u32], ctx: &str) {
        assert_eq!(got.len(), want_bits.len(), "{ctx}: length");
        for (j, (g, w)) in got.iter().zip(want_bits.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                *w,
                "{ctx}: element {j} got {:#010x} want {w:#010x}",
                g.to_bits()
            );
        }
    }

    /// All 27 reference-generated encode vectors reproduce byte-exactly,
    /// including the pathological rows (golden truth, not judged): all-NaN
    /// input -> all-zero wire; +/-Inf rows -> scale 0x7E with every element
    /// code 0 (decode +0.0); exact LUT midpoints -> LOWER index; -0.0;
    /// subnormals; saturation.
    #[test]
    fn encode_vectors_reproduce_golden_wire_bytes_and_dequant() {
        let fx = fixture_json("nvfp4_encode_vectors.json");
        let vectors = fx["vectors"].as_array().expect("vectors");
        assert_eq!(vectors.len(), 27);
        let mut seen_spotlock_tags = std::collections::BTreeSet::new();
        for vec in vectors {
            let tag = vec["tag"].as_str().expect("tag");
            let input_hex = vec["input"].as_array().expect("input");
            assert_eq!(input_hex.len(), NVFP4_VALUES_PER_BLOCK, "{tag}: input len");
            let mut x = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
            for (j, h) in input_hex.iter().enumerate() {
                x[j] = f32::from_bits(hex_u32(h.as_str().expect("hex")));
            }
            let want_wire = b64_decode(vec["wire"].as_str().expect("wire"));
            assert_eq!(
                want_wire.len(),
                NVFP4_WIRE_BYTES_PER_BLOCK,
                "{tag}: wire len"
            );
            let got_wire = encode_nvfp4_block(&x);
            assert_eq!(
                got_wire.as_slice(),
                want_wire.as_slice(),
                "{tag}: wire bytes"
            );

            let want_bits = hex_row_bits(vec["dequant"].as_str().expect("dequant"));
            let got = decode_block_via_tensor_path(&got_wire);
            assert_bits(&got, &want_bits, tag);

            // Spot-lock the pathological semantics by tag so a future regression
            // fails with a readable message, not just a byte diff. Every tag the
            // arms name must actually occur in the fixture — a silent `_` arm
            // would let a fixture-tag rename disable these locks unnoticed.
            match tag {
                "path-all-qnan" | "path-all-neg-qnan" | "path-all-negzero" | "negzero-single" => {
                    assert_eq!(got_wire, [0u8; 36], "{tag}: expected all-zero wire");
                    seen_spotlock_tags.insert(tag.to_string());
                }
                "path-all-pinf" | "path-all-ninf" | "path-inf-alt" => {
                    assert!(
                        got_wire[..4].iter().all(|&b| b == 0x7E),
                        "{tag}: scale 0x7E"
                    );
                    assert!(got_wire[4..].iter().all(|&b| b == 0), "{tag}: all code 0");
                    seen_spotlock_tags.insert(tag.to_string());
                }
                "sat-exact-448" | "sat-448-plus-ulp" | "sat-1e4" | "sat-fltmax" => {
                    assert!(
                        got_wire[..4].iter().all(|&b| b == 0x7E),
                        "{tag}: scale 0x7E"
                    );
                    seen_spotlock_tags.insert(tag.to_string());
                }
                _ => {}
            }
        }
        for expected in [
            "path-all-qnan",
            "path-all-neg-qnan",
            "path-all-negzero",
            "negzero-single",
            "path-all-pinf",
            "path-all-ninf",
            "path-inf-alt",
            "sat-exact-448",
            "sat-448-plus-ulp",
            "sat-1e4",
            "sat-fltmax",
        ] {
            assert!(
                seen_spotlock_tags.contains(expected),
                "fixture is missing spot-lock tag {expected}: the semantic lock never ran"
            );
        }
    }

    /// Every reference-quantized PRNG/edge input row reproduces the reference
    /// wire bytes through this encoder — 10031 diverse encode anchors on top
    /// of the 27 curated vectors.
    #[test]
    fn random_blocks_encode_reproduces_golden_wire_bytes() {
        let fx = fixture_json("nvfp4_random_blocks.json");
        let blocks = fx["blocks"].as_array().expect("blocks");
        assert_eq!(blocks.len(), 10031);
        for (i, blk) in blocks.iter().enumerate() {
            let tag = blk["tag"].as_str().expect("tag");
            let input = b64_decode(blk["i"].as_str().expect("input"));
            assert_eq!(input.len(), 256, "block {i} ({tag}): input bytes");
            let mut x = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
            for (j, chunk) in input.chunks_exact(4).enumerate() {
                x[j] = f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let want_wire = b64_decode(blk["w"].as_str().expect("wire"));
            let got_wire = encode_nvfp4_block(&x);
            assert_eq!(
                got_wire.as_slice(),
                want_wire.as_slice(),
                "block {i} ({tag}): wire bytes"
            );
        }
    }

    /// Representable-value round-trip: x = KVALUES_MXFP4[c] * UE4M3_TO_F32[s] for
    /// every scale byte with a nonzero decoded scale and all 16 codes. For every
    /// ENCODER-REACHABLE scale (masked 0x01..=0x77 and 0x7E) the round trip is
    /// bit-exact and the stored scale byte equals the masked input byte. Masked
    /// 0x78..0x7D (raw >= 256 saturates to 0x7E before mantissa rounding) and 0xFF
    /// (raw 480 exceeds the 448 input clamp) are unreachable from the encoder
    /// BY DESIGN — for those the encoder must emit 0x7E, and the quantized VALUE
    /// set must be a fixed point of a second quantize->dequantize pass. (The WIRE
    /// is deliberately not asserted stable: at masked 0x78 the reference itself
    /// re-tightens the scale on a second pass — amax of the first-pass output
    /// drops to 1344, whose amax/6 leaves the exp>=15 saturation region and
    /// re-encodes as 0x76 — while the decoded values stay bit-identical.)
    #[test]
    fn representable_value_round_trip_all_scales_all_codes() {
        for (s, &d) in UE4M3_TO_F32.iter().enumerate() {
            if d.to_bits() == 0 {
                continue; // 0x00 (zero), 0x7F (sentinel flush), 0x80 (masked zero)
            }
            let mut x = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
            for sub in 0..4 {
                for (c, kv) in KVALUES_MXFP4.iter().enumerate() {
                    x[sub * 16 + c] = kv * d;
                }
            }
            let wire = encode_nvfp4_block(&x);
            let masked = (s as u8) & 0x7F;
            let encodable = (0x01..=0x77).contains(&masked) || masked == 0x7E;
            if encodable {
                assert!(
                    wire[..4].iter().all(|&b| b == masked),
                    "scale {s:#04x}: stored scale byte {:#04x} != masked {masked:#04x}",
                    wire[0]
                );
                let y = decode_block_via_tensor_path(&wire);
                for (j, (got, want)) in y.iter().zip(x.iter()).enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "scale {s:#04x} element {j}: round trip not bit-exact"
                    );
                }
            } else {
                assert!(
                    wire[..4].iter().all(|&b| b == 0x7E),
                    "scale {s:#04x}: unreachable scale must saturate to 0x7E, got {:#04x}",
                    wire[0]
                );
                // Value-level fixed point: re-quantizing the quantized values
                // reproduces them bit-exactly (even where the wire scale byte
                // legitimately re-tightens, e.g. masked 0x78).
                let y = decode_block_via_tensor_path(&wire);
                let y2 = decode_block_via_tensor_path(&encode_nvfp4_block(&y));
                for (j, (a, b)) in y.iter().zip(y2.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "scale {s:#04x} element {j}: quantized values not a fixed point"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_block_round_trip_and_negative_zero_encode() {
        // All +0.0: zero wire, all +0.0 back.
        let x = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
        let wire = encode_nvfp4_block(&x);
        assert_eq!(wire, [0u8; NVFP4_WIRE_BYTES_PER_BLOCK]);
        let y = decode_block_via_tensor_path(&wire);
        for v in &y {
            assert_eq!(v.to_bits(), 0x0000_0000);
        }
        // All -0.0 encodes IDENTICALLY (amax stays 0 because `0.0 < |-0.0|` is
        // false, and the nearest-code search's initial candidate 0 survives
        // every tie) — the -0.0 sign does NOT survive the encode side. Sign
        // survival on the DECODE side is covered below.
        let neg = [f32::from_bits(0x8000_0000); NVFP4_VALUES_PER_BLOCK];
        let wire = encode_nvfp4_block(&neg);
        assert_eq!(wire, [0u8; NVFP4_WIRE_BYTES_PER_BLOCK]);
    }

    /// Decode-side -0.0 sign survival: negative codes (9..15) under a ZERO decoded
    /// scale (byte 0x00, and the flushed sentinel 0x7F) multiply to -0.0
    /// (bit pattern 0x80000000), positive codes and code 8 to +0.0 — matching the
    /// golden decode-table rows bit-for-bit.
    #[test]
    fn negative_codes_times_zero_scale_decode_to_negative_zero() {
        let mut wire = [0u8; NVFP4_WIRE_BYTES_PER_BLOCK];
        wire[..4].copy_from_slice(&[0x00, 0x7F, 0x00, 0x7F]);
        // sub 0: code 9 everywhere; sub 1: code 15; sub 2: code 0; sub 3: code 8.
        wire[4..12].fill(0x99);
        wire[12..20].fill(0xFF);
        wire[20..28].fill(0x00);
        wire[28..36].fill(0x88);
        let mut out = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
        nvfp4_block_decode_into(&mut out, &wire);
        for j in 0..16 {
            assert_eq!(out[j].to_bits(), 0x8000_0000, "sub 0 (code 9 x 0.0): -0.0");
            assert_eq!(
                out[16 + j].to_bits(),
                0x8000_0000,
                "sub 1 (code 15 x 0x7F): -0.0"
            );
            assert_eq!(out[32 + j].to_bits(), 0x0000_0000, "sub 2 (code 0): +0.0");
            assert_eq!(out[48 + j].to_bits(), 0x0000_0000, "sub 3 (code 8): +0.0");
        }
    }

    /// First-wins ties: at d = 0.5 (scale byte 0x38, anchored by a 6.0 element)
    /// every exact midpoint between adjacent representable magnitudes resolves to
    /// the LOWER LUT index, for both signs — strict `<` in the nearest search, not
    /// round-nearest-even. Expected codes are hand-derived from the LUT scan order
    /// and cross-checked against the `tie-mid-d0.5` golden vectors.
    #[test]
    fn exact_midpoint_ties_resolve_to_first_lut_index() {
        // Representable true values at d=0.5: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
        let sub: [f32; 16] = [
            6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, // low nibbles
            -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 0.0, // high nibbles
        ];
        let mut x = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
        for s in 0..4 {
            x[s * 16..(s + 1) * 16].copy_from_slice(&sub);
        }
        let wire = encode_nvfp4_block(&x);
        assert!(wire[..4].iter().all(|&b| b == 0x38), "anchor scale 0.5");
        // Expected codes: lows [7,0,1,2,3,4,5,6]; highs [0,9,10,11,12,13,14,0].
        // (-0.25 ties +0.0 at index 0 BEFORE -0.5 at index 9, so it goes positive-zero.)
        let expected_qs: [u8; 8] = [0x07, 0x90, 0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0x06];
        for s in 0..4 {
            assert_eq!(
                &wire[4 + s * 8..4 + (s + 1) * 8],
                &expected_qs,
                "sub-block {s} tie codes"
            );
        }
    }

    /// Scale saturation boundary around 448 x 6: the largest encoder-reachable
    /// sub-block scale is 0x7E (decoded 224, raw 448); the largest NON-saturating
    /// scale is 0x77 (decoded 120, raw 240). amax = 2688 = 12 x 224 = 448 x 6 hits
    /// 0x7E exactly and round-trips; one ULP either side stays at 0x7E (clamp /
    /// exponent saturation); amax = 6 x 248 carries past raw 240 into saturation.
    #[test]
    fn scale_saturation_boundary_at_448_by_6() {
        let cases: [(f32, u8); 5] = [
            (2688.0, 0x7E),                 // amax/6 == 448 exactly
            (2688.0_f32.next_up(), 0x7E),   // just over: clamps to 448
            (2688.0_f32.next_down(), 0x7E), // just under: exp path still saturates
            (6.0 * 240.0, 0x77),            // largest non-saturating: raw 240
            (6.0 * 248.0, 0x7E),            // round-half-up carry into exp 15
        ];
        for (amax, want_scale) in cases {
            let x = [amax; NVFP4_VALUES_PER_BLOCK];
            let wire = encode_nvfp4_block(&x);
            assert!(
                wire[..4].iter().all(|&b| b == want_scale),
                "amax {amax}: scale {:#04x} want {want_scale:#04x}",
                wire[0]
            );
        }
        // Exact round trip at the boundary: 2688 = code 7 x 224.
        let x = [2688.0_f32; NVFP4_VALUES_PER_BLOCK];
        let y = decode_block_via_tensor_path(&encode_nvfp4_block(&x));
        for v in &y {
            assert_eq!(v.to_bits(), 2688.0_f32.to_bits());
        }
        // 1440 = code 7 x 120 round-trips through the last non-saturating scale.
        let x = [1440.0_f32; NVFP4_VALUES_PER_BLOCK];
        let y = decode_block_via_tensor_path(&encode_nvfp4_block(&x));
        for v in &y {
            assert_eq!(v.to_bits(), 1440.0_f32.to_bits());
        }
    }

    /// The UE4M3 encoder in isolation: exact grid values, half-up rounding, the
    /// subnormal path, the 448 clamp, and the NaN/non-positive zero returns.
    #[test]
    fn fp32_to_ue4m3_semantics() {
        assert_eq!(fp32_to_ue4m3(f32::NAN), 0x00);
        assert_eq!(fp32_to_ue4m3(0.0), 0x00);
        assert_eq!(fp32_to_ue4m3(-1.0), 0x00);
        assert_eq!(fp32_to_ue4m3(f32::from_bits(0x8000_0000)), 0x00); // -0.0
        assert_eq!(fp32_to_ue4m3(f32::INFINITY), 0x7E); // clamp then saturate
        assert_eq!(fp32_to_ue4m3(448.0), 0x7E);
        assert_eq!(fp32_to_ue4m3(1.0), 0x38); // raw 1.0 -> exp 7, man 0
        assert_eq!(fp32_to_ue4m3(240.0), 0x77); // largest non-saturating grid point
        assert_eq!(fp32_to_ue4m3(248.0), 0x7E); // half-up carry into exp 15
        assert_eq!(fp32_to_ue4m3(1.0 / 512.0), 0x01); // subnormal grid
        assert_eq!(fp32_to_ue4m3(0.9 / 512.0), 0x01); // rounds half-up to man 1
        assert_eq!(fp32_to_ue4m3(0.4 / 512.0), 0x00); // below the subnormal floor
                                                      // Every encoder-reachable byte decodes back to a value that re-encodes to
                                                      // itself (grid fixed points).
        for b in 0x01..=0x77u8 {
            let raw = UE4M3_TO_F32[b as usize] * 2.0; // undo the pair-rule half
            assert_eq!(fp32_to_ue4m3(raw), b, "grid fixed point {b:#04x}");
        }
        let raw_7e = UE4M3_TO_F32[0x7E] * 2.0;
        assert_eq!(fp32_to_ue4m3(raw_7e), 0x7E);
    }
}

/// BF16 dequant-parity gate. bf16 -> f32 is the exact bit-widening
/// `f32::from_bits(u32::from(u16) << 16)`: bf16 stores the high 16 bits of the
/// IEEE-754 f32 encoding, so widening appends 16 zero low bits — lossless, no
/// rounding, bit-deterministic. The committed fixture
/// `tests/fixtures/dequant/bf16_exact.json` carries the LE wire bytes plus the
/// reference f32 outputs as u32 bit patterns; every comparison here is on
/// `f32::to_bits()` (so +0.0/-0.0 and NaN payloads are distinguished exactly).
#[cfg(test)]
mod bf16_dequant_parity_tests {
    use super::decode_bf16_tensor;

    fn fixture() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("dequant")
            .join("bf16_exact.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("bf16_exact.json parses: {e}"))
    }

    fn hex_u32(s: &str) -> u32 {
        u32::from_str_radix(s.trim_start_matches("0x"), 16)
            .unwrap_or_else(|e| panic!("hex {s:?}: {e}"))
    }

    fn hex_bytes(h: &str) -> Vec<u8> {
        assert!(h.len().is_multiple_of(2), "odd hex length {}", h.len());
        (0..h.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&h[i..i + 2], 16).expect("hex byte"))
            .collect()
    }

    /// The golden reference bits must BE the exact widening `bf16_u16 << 16` —
    /// a definitional self-check so the fixture can never silently encode a
    /// wrong value.
    #[test]
    fn bf16_fixture_reference_is_the_exact_widening() {
        let fx = fixture();
        assert_eq!(fx["qtype"].as_str(), Some("BF16"));
        let u16s = fx["bf16_u16_hex"].as_array().expect("bf16_u16_hex");
        let refs = fx["ref_f32_bits"].as_array().expect("ref_f32_bits");
        assert_eq!(u16s.len(), refs.len(), "u16 vs ref length");
        for (u, r) in u16s.iter().zip(refs.iter()) {
            let bf16 = hex_u32(u.as_str().expect("u16 hex"));
            let want = hex_u32(r.as_str().expect("ref hex"));
            assert_eq!(
                bf16 << 16,
                want,
                "reference bits must be the exact widening (bf16 {bf16:#06x} << 16)"
            );
        }
    }

    /// `decode_bf16_tensor` reproduces the golden reference bit-for-bit.
    #[test]
    fn decode_bf16_tensor_matches_golden_bit_exact() {
        let fx = fixture();
        let n = fx["n_elements"].as_u64().expect("n_elements") as usize;
        let bytes = hex_bytes(fx["quant_hex"].as_str().expect("quant_hex"));
        assert_eq!(bytes.len(), n * 2, "wire byte length");
        let refs: Vec<u32> = fx["ref_f32_bits"]
            .as_array()
            .expect("ref_f32_bits")
            .iter()
            .map(|r| hex_u32(r.as_str().expect("ref hex")))
            .collect();
        assert_eq!(refs.len(), n, "ref count");

        let out =
            decode_bf16_tensor("fixture:bf16_exact", &bytes, n).expect("bf16 decode must succeed");
        assert_eq!(out.len(), n, "decoded length");
        for (i, (got, want)) in out.iter().zip(refs.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                *want,
                "element {i}: got {:#010x} want {want:#010x}",
                got.to_bits()
            );
        }
    }

    /// Wrong wire length fails closed (the lane never pads or truncates silently).
    #[test]
    fn decode_bf16_tensor_wrong_length_fails_closed() {
        let err = decode_bf16_tensor("t", &[0u8; 6], 2).expect_err("length mismatch must refuse");
        assert!(matches!(
            err,
            crate::error::EngineError::InvalidTensorData(_)
        ));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn f32_f16_roundtrip_matches_ieee_rne() {
        use super::{f16_bits_to_f32, f32_to_f16_bits};
        // Exact halves roundtrip exactly.
        for v in [0.0f32, 1.0, -1.0, 0.5, -0.25, 65504.0, -65504.0] {
            assert_eq!(f16_bits_to_f32(f32_to_f16_bits(v)), v);
        }
        // Observed reference roundings (f32 value -> f16-stored value).
        let cases = [
            (-0.2714f32, -0.27148438f32),
            (-0.6571, -0.65722656),
            (0.0809, 0.08087158),
        ];
        for (input, expect) in cases {
            let got = f16_bits_to_f32(f32_to_f16_bits(input));
            assert!(
                (got - expect).abs() < 2e-6,
                "{input} -> {got}, want {expect}"
            );
        }
        // Round-to-nearest-EVEN tie: 1 + 2^-11 is exactly halfway between
        // half(1.0) and half(1.0009766); RNE picks the even mantissa (1.0).
        let tie = 1.0f32 + 2.0f32.powi(-11);
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(tie)), 1.0);
        // Just above the tie rounds up.
        let above = 1.0f32 + 2.0f32.powi(-11) + 2.0f32.powi(-20);
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(above)), 1.0009766);
        // Overflow saturates to inf; tiny values flush toward subnormals/zero.
        assert_eq!(f32_to_f16_bits(1.0e6) & 0x7fff, 0x7c00);
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(1.0e-8)), 0.0);
    }

    /// Independent, spec-literal reference for IQ4_XS dequant (a second
    /// implementation used only to cross-check the optimized
    /// [`super::IQ4XSBlock`] decoder). Mirrors the wire layout field for field.
    fn iq4_xs_reference_dequant(block: &[u8; super::IQ4_XS_BLOCK_BYTES]) -> [f32; 256] {
        use super::{f16_bits_to_f32, KVALUES_IQ4NL};
        let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];
        let mut out = [0.0_f32; 256];
        for ib in 0..8usize {
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
            let high = ((scales_h >> (2 * ib)) & 0x3) as u8;
            let ls = (low | (high << 4)) as i32;
            let dl = d * (ls - 32) as f32;
            for j in 0..16usize {
                let byte = qs[ib * 16 + j];
                out[ib * 32 + j] = dl * KVALUES_IQ4NL[(byte & 0x0F) as usize];
                out[ib * 32 + j + 16] = dl * KVALUES_IQ4NL[(byte >> 4) as usize];
            }
        }
        out
    }

    #[test]
    fn iq4_xs_block_dequant_matches_hand_computed_golden() {
        use super::{IQ4XSBlock, IQ4_XS_BLOCK_BYTES};
        // d = 1.0 (f16 0x3C00). Sub-block scales chosen so dl = ls - 32 is exact per sub-block:
        //   ib:  0   1   2    3    4    5   6    7
        //   ls: 33  32  63    0   24    2  36   26
        //   dl:  1   0  31  -32   -8  -30   4   -6
        // scales_l nibbles (even ib -> low, odd ib -> high of scales_l[ib/2]):
        //   [0]=(ib0 low=1, ib1 high=0)=0x01  [1]=(ib2=15, ib3=0)=0x0F
        //   [2]=(ib4=8,  ib5 high=2)=0x28     [3]=(ib6=4,  ib7 high=10)=0xA4
        // scales_h (2 bits/sub-block): highs 2,2,3,0,1,0,2,1 -> 0x613A.
        let mut bytes = [0_u8; IQ4_XS_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&0x613Au16.to_le_bytes());
        bytes[4..8].copy_from_slice(&[0x01, 0x0F, 0x28, 0xA4]);
        // Every quant byte = 0x80: low nibble 0 -> kv[0]=-127, high nibble 8 -> kv[8]=1.
        for b in bytes[8..136].iter_mut() {
            *b = 0x80;
        }
        let mut out = [0.0_f32; 256];
        IQ4XSBlock::from_bytes(&bytes).dequantize(&mut out);

        let dl = [1.0, 0.0, 31.0, -32.0, -8.0, -30.0, 4.0, -6.0];
        for (ib, &d) in dl.iter().enumerate() {
            for j in 0..16 {
                assert_eq!(out[ib * 32 + j], d * -127.0, "ib{ib} j{j} low half");
                assert_eq!(out[ib * 32 + j + 16], d * 1.0, "ib{ib} j{j} high half");
            }
        }
        // And the optimized decoder equals the spec-literal reference on this block.
        assert_eq!(out, iq4_xs_reference_dequant(&bytes));
    }

    #[test]
    fn iq4_xs_block_dequant_matches_reference_over_deterministic_blocks() {
        use super::{IQ4XSBlock, IQ4_XS_BLOCK_BYTES};
        // Deterministic LCG fills exercise every codebook index, scale split, and nibble.
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for _ in 0..64 {
            let mut bytes = [0_u8; IQ4_XS_BLOCK_BYTES];
            for b in bytes.iter_mut() {
                *b = next();
            }
            let mut out = [0.0_f32; 256];
            IQ4XSBlock::from_bytes(&bytes).dequantize(&mut out);
            let reference = iq4_xs_reference_dequant(&bytes);
            // Bit-for-bit: random f16 `d` bytes can encode NaN/Inf, so compare raw bits
            // (NaN != NaN under `==`). The two implementations run identical float ops, so
            // every lane — finite or not — must match exactly.
            for i in 0..256 {
                assert_eq!(
                    out[i].to_bits(),
                    reference[i].to_bits(),
                    "lane {i} differs for block {bytes:02x?}"
                );
            }
        }
    }

    #[test]
    fn iq4_xs_sub_block_scale_unpacks_low_high_and_minus32_bias() {
        use super::{IQ4XSBlock, IQ4_XS_BLOCK_BYTES};
        // d = 1.0. ib0: scales_l low nibble 5 + scales_h high bits 3 -> ls = 5|48 = 53 -> dl = 21.
        // ib1: low 0 + high 0 -> ls = 0 -> dl = -32. qs byte 0x00 -> both nibbles index kv[0].
        let mut bytes = [0_u8; IQ4_XS_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&0x0003u16.to_le_bytes());
        bytes[4] = 0x05;
        bytes[8] = 0x00;
        let block = IQ4XSBlock::from_bytes(&bytes);
        assert_eq!(block.sub_block_scale(0), 21.0);
        assert_eq!(block.sub_block_scale(1), -32.0);
        let mut out = [0.0_f32; 256];
        block.dequantize(&mut out);
        assert_eq!(out[0], 21.0 * -127.0); // ib0, kv[0]
        assert_eq!(out[32], -32.0 * -127.0); // ib1, kv[0]
    }

    #[test]
    fn iq4_xs_tensor_decode_spans_multiple_blocks_and_rejects_misalignment() {
        use super::{decode_iq4_xs_tensor, IQ4_XS_BLOCK_BYTES};
        // Two full super-blocks of distinct constant bytes.
        let mut bytes = Vec::new();
        for fill in [0x11u8, 0x22u8] {
            let mut blk = vec![0u8; IQ4_XS_BLOCK_BYTES];
            blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
            for b in blk[2..].iter_mut() {
                *b = fill;
            }
            bytes.extend_from_slice(&blk);
        }
        let decoded = decode_iq4_xs_tensor("blk.iq4xs", &bytes, 512).unwrap();
        assert_eq!(decoded.len(), 512);
        let mut b0 = [0u8; IQ4_XS_BLOCK_BYTES];
        b0.copy_from_slice(&bytes[0..IQ4_XS_BLOCK_BYTES]);
        assert_eq!(&decoded[0..256], &iq4_xs_reference_dequant(&b0)[..]);

        // One byte short of a block boundary must fail closed, not truncate silently.
        let err = decode_iq4_xs_tensor("blk.bad", &bytes[..bytes.len() - 1], 512).unwrap_err();
        assert!(format!("{err}").contains("not aligned"), "got: {err}");
    }

    #[test]
    fn iq4_nl_and_iq4_xs_share_the_same_codebook() {
        use super::{IQ4NLBlock, IQ4_NL_BLOCK_BYTES, KVALUES_IQ4NL};
        // The shared const carries the exact codebook table.
        assert_eq!(
            KVALUES_IQ4NL,
            [
                -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0,
                53.0, 69.0, 89.0, 113.0
            ]
        );
        // IQ4_NL still indexes that same table (d = 1.0, qs byte 0xF0 -> kv[0] then kv[15]).
        let mut bytes = [0u8; IQ4_NL_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        bytes[2] = 0xF0;
        let mut out = [0.0_f32; 32];
        IQ4NLBlock::from_bytes(&bytes).dequantize(&mut out);
        assert_eq!(out[0], KVALUES_IQ4NL[0]);
        assert_eq!(out[16], KVALUES_IQ4NL[15]);
    }

    #[test]
    fn converts_f16_bits_to_f32() {
        use super::f16_bits_to_f32;
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xc000), -2.0);
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
    }
}
