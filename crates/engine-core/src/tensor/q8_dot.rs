//! Portable Q8_0 quantization and the Q8_0 x Q8_0 row dot.
//!
//! This module is the reference definition of the Q8_0 matmul leaf: a plain
//! scalar quantizer and a plain integer per-block dot. It contains no host
//! dispatch and no SIMD; the per-platform kernel crates are tested for
//! bit-identity against these functions.
//!
//! Quantization stores an f16-rounded scale but divides input values by the
//! un-rounded scale, matching the on-wire Q8_0 `d` semantics. The dot computes
//! each block's product sum in exact i32 arithmetic, then reduces the per-block
//! f32 terms over a single accumulator in ascending block order with no fused
//! multiply-add and no reassociation, so results are bit-deterministic.

use super::blocks::{f16_bits_to_f32, f32_to_f16_bits, Q8_0Block, Q8_0_BLOCK_VALUES};

/// Quantize one chunk of exactly [`Q8_0_BLOCK_VALUES`] f32 values into a
/// [`Q8_0Block`].
///
/// The stored scale is the un-rounded scale (`max_abs / 127`) round-tripped
/// through f16; the quants are produced by dividing each value by the
/// *un-rounded* scale, rounding half away from zero, and clamping to the i8
/// range `[-128, 127]`.
pub fn quantize_q8_0_block(block: &[f32]) -> Q8_0Block {
    debug_assert_eq!(block.len(), Q8_0_BLOCK_VALUES);
    let max_abs = block
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    let unrounded_scale = max_abs / 127.0;
    let scale_bits = f32_to_f16_bits(unrounded_scale);
    let scale = f16_bits_to_f32(scale_bits);
    let inv_scale = if unrounded_scale == 0.0 {
        0.0
    } else {
        1.0 / unrounded_scale
    };
    let mut quants = [0_i8; Q8_0_BLOCK_VALUES];
    for (idx, value) in block.iter().enumerate() {
        quants[idx] = (value * inv_scale).round().clamp(-128.0, 127.0) as i8;
    }
    Q8_0Block { scale, quants }
}

/// Quantize a flat f32 slice into Q8_0 blocks, one per [`Q8_0_BLOCK_VALUES`]
/// chunk. A trailing partial chunk is silently dropped; callers pass lengths
/// that are multiples of the block size.
pub fn quantize_q8_0_blocks(input: &[f32]) -> Vec<Q8_0Block> {
    debug_assert!(input.len().is_multiple_of(Q8_0_BLOCK_VALUES));
    input
        .chunks_exact(Q8_0_BLOCK_VALUES)
        .map(quantize_q8_0_block)
        .collect()
}

/// Exact i32 dot of one block: sum over 32 lanes of `weight[k] * input[k]`,
/// each product widened to i32 before summing. The magnitude is at most
/// `32 * 127 * 127 = 516128`, far under `i32::MAX`, so no overflow occurs and
/// the grouping order does not affect the result.
fn block_int_dot(weight: &[i8; Q8_0_BLOCK_VALUES], input: &[i8; Q8_0_BLOCK_VALUES]) -> i32 {
    let group4 = |start: usize| -> i32 {
        i32::from(weight[start]) * i32::from(input[start])
            + i32::from(weight[start + 1]) * i32::from(input[start + 1])
            + i32::from(weight[start + 2]) * i32::from(input[start + 2])
            + i32::from(weight[start + 3]) * i32::from(input[start + 3])
    };
    let lanes = [
        group4(0) + group4(16),
        group4(4) + group4(20),
        group4(8) + group4(24),
        group4(12) + group4(28),
    ];
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// Dot two rows of Q8_0 blocks.
///
/// Each block contributes `(int_sum as f32) * weight.scale * input.scale`,
/// left-associated, and the terms are summed into a single f32 accumulator in
/// ascending block order (`Iterator::sum` is a left fold from `0.0`). The
/// integer block dot is exact; the only rounding is in the per-block scale
/// multiplies and the running sum.
pub fn q8_0_dot_rows(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
    weight
        .iter()
        .zip(input)
        .map(|(weight_block, input_block)| {
            let int_sum = block_int_dot(&weight_block.quants, &input_block.quants);
            int_sum as f32 * weight_block.scale * input_block.scale
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_known_vector_exact_bytes_and_scale() {
        // A single 1.0 in an otherwise-zero block: max_abs = 1.0,
        // unrounded_scale = 1/127, inv_scale = 127, so quant[0] = 127.
        let mut block = [0.0_f32; Q8_0_BLOCK_VALUES];
        block[0] = 1.0;
        let out = quantize_q8_0_block(&block);

        // 1/127 round-tripped through f16 is exactly 0.00787353515625.
        assert_eq!(out.scale, 0.007_873_535_156_25_f32);
        assert_eq!(out.scale.to_bits(), 0x3C01_0000);

        let mut expected = [0_i8; Q8_0_BLOCK_VALUES];
        expected[0] = 127;
        assert_eq!(out.quants, expected);
    }

    #[test]
    fn scale_inv_scale_asymmetry() {
        // The stored scale is f16-rounded, but quantization divides by the
        // un-rounded scale. With max_abs = 100 the two differ, and using the
        // stored (rounded) scale to reconstruct the quant would drift.
        let mut block = [0.0_f32; Q8_0_BLOCK_VALUES];
        block[0] = 100.0;
        block[1] = 50.0;
        let out = quantize_q8_0_block(&block);

        let unrounded = 100.0_f32 / 127.0;
        let stored = f16_bits_to_f32(f32_to_f16_bits(unrounded));
        // Stored scale is genuinely rounded away from the un-rounded value, and
        // it is the value written into the block.
        assert_ne!(stored, unrounded);
        assert_eq!(out.scale, stored);

        // Quants are produced by the UN-rounded inv_scale (1/unrounded), which
        // differs from the reciprocal of the stored scale. A port that reused
        // the stored scale to build inv_scale would divide by a different
        // number.
        let inv_scale = 1.0_f32 / unrounded;
        let stored_inv = 1.0_f32 / stored;
        assert_ne!(inv_scale, stored_inv);
        let expected0 = (100.0_f32 * inv_scale).round().clamp(-128.0, 127.0) as i8;
        let expected1 = (50.0_f32 * inv_scale).round().clamp(-128.0, 127.0) as i8;
        assert_eq!(out.quants[0], expected0);
        assert_eq!(out.quants[1], expected1);

        // A concrete block where the choice of divisor flips a quant: with
        // max 0.13 the un-rounded inv_scale yields 23 for this probe value,
        // the stored-scale reciprocal would yield 24.
        let mut probe = [0.0_f32; Q8_0_BLOCK_VALUES];
        probe[0] = 0.13;
        probe[1] = 185.0 * 0.001 * 0.13;
        let pout = quantize_q8_0_block(&probe);
        let pu = 0.13_f32 / 127.0;
        let ps = f16_bits_to_f32(f32_to_f16_bits(pu));
        assert_eq!(pout.quants[1], (probe[1] * (1.0 / pu)).round() as i8);
        assert_ne!(
            pout.quants[1],
            (probe[1] * (1.0 / ps)).round().clamp(-128.0, 127.0) as i8,
        );
    }

    #[test]
    fn zero_input_block_has_zero_scale_and_quants() {
        let block = [0.0_f32; Q8_0_BLOCK_VALUES];
        let out = quantize_q8_0_block(&block);
        // max_abs = 0 -> unrounded_scale = 0 -> guard yields inv_scale = 0.
        assert_eq!(out.scale, 0.0);
        assert_eq!(out.quants, [0_i8; Q8_0_BLOCK_VALUES]);
        // A -0.0-laden block behaves identically (abs clears the sign).
        let neg = [-0.0_f32; Q8_0_BLOCK_VALUES];
        assert_eq!(quantize_q8_0_block(&neg).quants, [0_i8; Q8_0_BLOCK_VALUES]);
    }

    #[test]
    fn saturation_at_range_endpoints_and_infinite_input() {
        // The block max saturates the format at the range ends: the largest
        // positive value maps to +127 and the largest-magnitude negative value
        // maps to -127 (never -128 for finite inputs), and every other quant
        // stays inside [-128, 127].
        let mut block = [0.0_f32; Q8_0_BLOCK_VALUES];
        block[0] = 5.0; // positive max -> +127
        block[1] = -5.0; // negative extreme of same magnitude -> -127
        block[2] = 2.5; // interior value
        block[3] = -4.9999; // just under the max magnitude
        let out = quantize_q8_0_block(&block);
        assert_eq!(out.quants[0], 127);
        assert_eq!(out.quants[1], -127);
        assert!(out.quants.iter().all(|&q| (-128..=127).contains(&(q as i32))));
        assert!(out.quants[3] > -128 && out.quants[3] < 0);

        // Infinite input exercises the clamp/NaN-through-cast edge: max_abs is
        // inf, inv_scale = 1/inf = +0, every product is 0 or inf*0 = NaN, and
        // `NaN as i8` saturates to 0. Scale is the f16 image of inf.
        let mut inf_block = [1.0_f32; Q8_0_BLOCK_VALUES];
        inf_block[7] = f32::INFINITY;
        let out_inf = quantize_q8_0_block(&inf_block);
        assert!(out_inf.scale.is_infinite());
        assert_eq!(out_inf.quants, [0_i8; Q8_0_BLOCK_VALUES]);
    }

    #[test]
    fn quantize_then_dot_round_trip_exact() {
        // Two known blocks, one quant nonzero pattern each, exact expected f32.
        let mut wv = [0.0_f32; Q8_0_BLOCK_VALUES];
        let mut iv = [0.0_f32; Q8_0_BLOCK_VALUES];
        wv[0] = 1.0;
        wv[1] = 0.5;
        iv[0] = 2.0;
        iv[1] = 1.0;

        let w = quantize_q8_0_blocks(&wv);
        let i = quantize_q8_0_blocks(&iv);
        assert_eq!(w.len(), 1);
        assert_eq!(i.len(), 1);

        // Weight block: max 1.0, inv_scale 127 -> quants[0]=127, quants[1]=64
        // (0.5*127=63.5 rounds half away to 64). Scale = f16(1/127).
        assert_eq!(w[0].quants[0], 127);
        assert_eq!(w[0].quants[1], 64);
        // Input block: max 2.0, inv_scale 63.5 -> quants[0]=127, quants[1]=64
        // (1.0*63.5=63.5 rounds half away to 64). Scale = f16(2/127).
        assert_eq!(i[0].quants[0], 127);
        assert_eq!(i[0].quants[1], 64);

        let int_sum = 127 * 127 + 64 * 64; // = 16129 + 4096 = 20225
        assert_eq!(block_int_dot(&w[0].quants, &i[0].quants), int_sum);

        let expected = int_sum as f32 * w[0].scale * i[0].scale;
        let got = q8_0_dot_rows(&w, &i);
        assert_eq!(got, expected);
        assert_eq!(got.to_bits(), expected.to_bits());
    }

    #[test]
    fn dot_accumulates_ascending_single_accumulator() {
        // Multiple blocks: result equals the explicit left fold from 0.0.
        let mut flat_w = Vec::new();
        let mut flat_i = Vec::new();
        for b in 0..3 {
            for k in 0..Q8_0_BLOCK_VALUES {
                flat_w.push(((b + k) % 5) as f32 - 2.0);
                flat_i.push(((b * 2 + k) % 7) as f32 - 3.0);
            }
        }
        let w = quantize_q8_0_blocks(&flat_w);
        let i = quantize_q8_0_blocks(&flat_i);

        let mut acc = 0.0_f32;
        for (wb, ib) in w.iter().zip(&i) {
            let int_sum = block_int_dot(&wb.quants, &ib.quants);
            acc += int_sum as f32 * wb.scale * ib.scale;
        }
        assert_eq!(q8_0_dot_rows(&w, &i).to_bits(), acc.to_bits());
    }
}
