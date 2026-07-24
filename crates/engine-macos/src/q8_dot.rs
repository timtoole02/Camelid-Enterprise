//! Apple-Silicon NEON acceleration for the Q8_0 matmul leaf.
//!
//! The public entry points ([`q8_0_dot_rows`], [`quantize_q8_0_block`],
//! [`quantize_q8_0_blocks`]) mirror the portable reference in
//! [`engine_core::tensor`]. On `aarch64` the dot uses the ARMv8.2 dot-product
//! instruction (SDOT) when the host advertises it, and the NEON widening
//! multiply (`vmull`) otherwise; the quantizer uses the NEON convert path. On
//! every other target these delegate to the portable reference.
//!
//! The NEON kernels are the accelerated definition of the same numeric
//! contract. The dot is bit-identical to the portable reference for every
//! input: the integer block product is exact in i32 (order-independent) and the
//! per-block f32 terms are reduced over a single accumulator in ascending block
//! order. The quantizer agrees with the portable reference on every input that
//! yields a nonzero stored scale — i.e. every input that can contribute to an
//! output; the one exception is documented on [`quantize_q8_0_block`] and is
//! numerically inert.

use engine_core::tensor::{Q8_0Block, Q8_0_BLOCK_VALUES};

/// Dot two rows of Q8_0 blocks.
///
/// On `aarch64` this runs the SDOT kernel when the host supports the
/// dot-product feature, and the `vmull` widening kernel otherwise; both produce
/// the same exact i32 block sums and the same single-accumulator f32 reduction
/// as the portable reference. On other targets it delegates to
/// [`engine_core::tensor::q8_0_dot_rows`].
pub fn q8_0_dot_rows(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::aarch64_dotprod_enabled() {
            // SAFETY: `aarch64_dotprod_enabled` returned true, so the host
            // supports the dot-product instructions this kernel issues.
            return unsafe { neon::q8_0_dot_rows_neon_dotprod(weight, input) };
        }
        // SAFETY: NEON is baseline on aarch64; the kernel only reads the 32
        // contiguous quant bytes each block owns.
        return unsafe { neon::q8_0_dot_rows_neon_mul(weight, input) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        engine_core::tensor::q8_0_dot_rows(weight, input)
    }
}

/// Quantize one chunk of exactly [`Q8_0_BLOCK_VALUES`] f32 values into a
/// [`Q8_0Block`].
///
/// On `aarch64` this uses the NEON convert path; on other targets it delegates
/// to [`engine_core::tensor::quantize_q8_0_block`]. The two paths produce the
/// identical [`Q8_0Block`] whenever the block's stored scale is nonzero, which
/// covers every input that can affect a dot result. The sole exception is a
/// block whose largest magnitude is so small the f16 scale rounds to zero:
/// there the NEON and scalar convert paths may choose different quant bytes,
/// but the zero scale makes both dequantize to all-zero, so the result is
/// unaffected (see the `degenerate_scale_band_is_numerically_inert` test).
pub fn quantize_q8_0_block(block: &[f32]) -> Q8_0Block {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; the kernel loads exactly the 32
        // values the debug assertion requires the block to hold.
        return unsafe { neon::quantize_q8_0_block(block) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        engine_core::tensor::quantize_q8_0_block(block)
    }
}

/// Quantize a flat f32 slice into Q8_0 blocks, one per [`Q8_0_BLOCK_VALUES`]
/// chunk. A trailing partial chunk is silently dropped.
pub fn quantize_q8_0_blocks(input: &[f32]) -> Vec<Q8_0Block> {
    debug_assert!(input.len().is_multiple_of(Q8_0_BLOCK_VALUES));
    input
        .chunks_exact(Q8_0_BLOCK_VALUES)
        .map(quantize_q8_0_block)
        .collect()
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use engine_core::tensor::{f16_bits_to_f32, f32_to_f16_bits, Q8_0Block, Q8_0_BLOCK_VALUES};
    use std::arch::aarch64::int32x4_t;

    /// Whether the host advertises the ARMv8.2 dot-product feature. Cached for
    /// the process lifetime: the answer is fixed hardware and the SDOT path is
    /// a numeric no-op relative to the `vmull` path.
    pub(super) fn aarch64_dotprod_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
    }

    /// Sum the four i32 lanes of `acc` into a single i32. Order-independent and
    /// overflow-free for Q8_0 block sums (magnitude at most `32 * 128 * 128`).
    #[inline(always)]
    pub(super) fn horizontal_sum_i32x4(acc: int32x4_t) -> i32 {
        // SAFETY: `vaddvq_s32` is a baseline NEON instruction on aarch64.
        unsafe { std::arch::aarch64::vaddvq_s32(acc) }
    }

    /// Exact i32 dot of one 32-lane block via SDOT.
    ///
    /// Loads the low and high 16 bytes of each row and issues two `sdot`
    /// instructions into a 4-lane i32 accumulator, then sums the lanes. SDOT
    /// does not saturate; each lane accumulates eight signed-byte products, so
    /// the accumulator stays far inside i32 range.
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn q8_0_i8_block_dotprod(weight: *const i8, input: *const i8) -> i32 {
        use std::arch::aarch64::{vdupq_n_s32, vld1q_s8};
        use std::arch::asm;

        // SAFETY: callers pass pointers to at least 32 contiguous i8 values.
        let weight_lo = unsafe { vld1q_s8(weight) };
        let input_lo = unsafe { vld1q_s8(input) };
        let weight_hi = unsafe { vld1q_s8(weight.add(16)) };
        let input_hi = unsafe { vld1q_s8(input.add(16)) };

        let mut acc = vdupq_n_s32(0);
        // SAFETY: target_feature(dotprod) enables SDOT for this function. The
        // operands are full 128-bit vector registers loaded above, and the
        // instruction only updates `acc`.
        unsafe {
            asm!(
                "sdot {acc:v}.4s, {weight_lo:v}.16b, {input_lo:v}.16b",
                "sdot {acc:v}.4s, {weight_hi:v}.16b, {input_hi:v}.16b",
                acc = inout(vreg) acc,
                weight_lo = in(vreg) weight_lo,
                input_lo = in(vreg) input_lo,
                weight_hi = in(vreg) weight_hi,
                input_hi = in(vreg) input_hi,
                options(nostack, preserves_flags)
            );
        }
        horizontal_sum_i32x4(acc)
    }

    /// Exact i32 dot of one 32-lane block via the NEON widening multiply.
    ///
    /// Each `vmull_s8` produces exact i16 products (single-product magnitude at
    /// most `128 * 128 = 16384`, inside i16 range); `vpaddlq_s16` widens to i32
    /// before accumulation, so no intermediate overflows.
    #[inline(always)]
    pub(super) unsafe fn q8_0_i8_block_neon_mul(weight: *const i8, input: *const i8) -> i32 {
        use std::arch::aarch64::{
            vaddq_s32, vdupq_n_s32, vget_high_s8, vget_low_s8, vld1q_s8, vmull_s8, vpaddlq_s16,
        };

        // SAFETY: callers pass pointers to at least 32 contiguous i8 values.
        let weight_lo = unsafe { vld1q_s8(weight) };
        let input_lo = unsafe { vld1q_s8(input) };
        let weight_hi = unsafe { vld1q_s8(weight.add(16)) };
        let input_hi = unsafe { vld1q_s8(input.add(16)) };

        // SAFETY: all operands below are full NEON vectors loaded above; these
        // are baseline aarch64 instructions with no memory access.
        unsafe {
            let mut acc = vdupq_n_s32(0);
            acc = vaddq_s32(
                acc,
                vpaddlq_s16(vmull_s8(vget_low_s8(weight_lo), vget_low_s8(input_lo))),
            );
            acc = vaddq_s32(
                acc,
                vpaddlq_s16(vmull_s8(vget_high_s8(weight_lo), vget_high_s8(input_lo))),
            );
            acc = vaddq_s32(
                acc,
                vpaddlq_s16(vmull_s8(vget_low_s8(weight_hi), vget_low_s8(input_hi))),
            );
            acc = vaddq_s32(
                acc,
                vpaddlq_s16(vmull_s8(vget_high_s8(weight_hi), vget_high_s8(input_hi))),
            );
            horizontal_sum_i32x4(acc)
        }
    }

    /// Dot two rows of Q8_0 blocks using the SDOT block kernel, reducing the
    /// per-block f32 terms into one accumulator in ascending block order.
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn q8_0_dot_rows_neon_dotprod(
        weight: &[Q8_0Block],
        input: &[Q8_0Block],
    ) -> f32 {
        let mut total_sum = 0.0_f32;
        for (w_block, i_block) in weight.iter().zip(input) {
            // SAFETY: each block owns 32 contiguous quant bytes, and the
            // dot-product feature is enabled for this function.
            let int_sum =
                unsafe { q8_0_i8_block_dotprod(w_block.quants.as_ptr(), i_block.quants.as_ptr()) };
            total_sum += int_sum as f32 * w_block.scale * i_block.scale;
        }
        total_sum
    }

    /// Dot two rows of Q8_0 blocks using the `vmull` block kernel, reducing the
    /// per-block f32 terms into one accumulator in ascending block order.
    pub(super) unsafe fn q8_0_dot_rows_neon_mul(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
        let mut total_sum = 0.0_f32;
        for (w_block, i_block) in weight.iter().zip(input) {
            // SAFETY: each block owns 32 contiguous quant bytes.
            let int_sum =
                unsafe { q8_0_i8_block_neon_mul(w_block.quants.as_ptr(), i_block.quants.as_ptr()) };
            total_sum += int_sum as f32 * w_block.scale * i_block.scale;
        }
        total_sum
    }

    /// NEON quantizer for one 32-value block. The absolute-max reduction is a
    /// lane-wise `vmaxq` tree collapsed to a scalar; the un-rounded scale is
    /// `max_abs / 127`, the stored scale is that value round-tripped through
    /// f16, and each element is `round-half-away(value / unrounded_scale)`
    /// saturated to the i8 range by the narrowing store chain.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn quantize_q8_0_block(block: &[f32]) -> Q8_0Block {
        use std::arch::aarch64::{
            vabsq_f32, vcombine_s16, vcombine_s8, vcvtaq_s32_f32, vdupq_n_f32, vget_high_f32,
            vget_lane_f32, vget_low_f32, vld1q_f32, vmax_f32, vmaxq_f32, vmovn_s32, vmulq_f32,
            vqmovn_s16, vst1q_s8,
        };

        debug_assert_eq!(block.len(), Q8_0_BLOCK_VALUES);

        // SAFETY: the block holds exactly 32 values, so loading 8 consecutive
        // 4-float vectors and storing 32 i8 quants stays in bounds.
        unsafe {
            let v0 = vld1q_f32(block.as_ptr());
            let v1 = vld1q_f32(block.as_ptr().add(4));
            let v2 = vld1q_f32(block.as_ptr().add(8));
            let v3 = vld1q_f32(block.as_ptr().add(12));
            let v4 = vld1q_f32(block.as_ptr().add(16));
            let v5 = vld1q_f32(block.as_ptr().add(20));
            let v6 = vld1q_f32(block.as_ptr().add(24));
            let v7 = vld1q_f32(block.as_ptr().add(28));

            let abs0 = vabsq_f32(v0);
            let abs1 = vabsq_f32(v1);
            let abs2 = vabsq_f32(v2);
            let abs3 = vabsq_f32(v3);
            let abs4 = vabsq_f32(v4);
            let abs5 = vabsq_f32(v5);
            let abs6 = vabsq_f32(v6);
            let abs7 = vabsq_f32(v7);

            let max01 = vmaxq_f32(abs0, abs1);
            let max23 = vmaxq_f32(abs2, abs3);
            let max45 = vmaxq_f32(abs4, abs5);
            let max67 = vmaxq_f32(abs6, abs7);

            let max03 = vmaxq_f32(max01, max23);
            let max47 = vmaxq_f32(max45, max67);

            let max_vec = vmaxq_f32(max03, max47);
            let max_half = vmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
            let max_abs = vget_lane_f32::<0>(max_half).max(vget_lane_f32::<1>(max_half));

            let unrounded_scale = max_abs / 127.0;
            let scale_bits = f32_to_f16_bits(unrounded_scale);
            let scale = f16_bits_to_f32(scale_bits);

            let inv_scale = if unrounded_scale == 0.0 {
                0.0
            } else {
                1.0 / unrounded_scale
            };

            let v_inv_scale = vdupq_n_f32(inv_scale);
            let scaled0 = vmulq_f32(v0, v_inv_scale);
            let scaled1 = vmulq_f32(v1, v_inv_scale);
            let scaled2 = vmulq_f32(v2, v_inv_scale);
            let scaled3 = vmulq_f32(v3, v_inv_scale);
            let scaled4 = vmulq_f32(v4, v_inv_scale);
            let scaled5 = vmulq_f32(v5, v_inv_scale);
            let scaled6 = vmulq_f32(v6, v_inv_scale);
            let scaled7 = vmulq_f32(v7, v_inv_scale);

            let int0 = vcvtaq_s32_f32(scaled0);
            let int1 = vcvtaq_s32_f32(scaled1);
            let int2 = vcvtaq_s32_f32(scaled2);
            let int3 = vcvtaq_s32_f32(scaled3);
            let int4 = vcvtaq_s32_f32(scaled4);
            let int5 = vcvtaq_s32_f32(scaled5);
            let int6 = vcvtaq_s32_f32(scaled6);
            let int7 = vcvtaq_s32_f32(scaled7);

            let i16_0 = vmovn_s32(int0);
            let i16_1 = vmovn_s32(int1);
            let i16_01 = vcombine_s16(i16_0, i16_1);

            let i16_2 = vmovn_s32(int2);
            let i16_3 = vmovn_s32(int3);
            let i16_23 = vcombine_s16(i16_2, i16_3);

            let i16_4 = vmovn_s32(int4);
            let i16_5 = vmovn_s32(int5);
            let i16_45 = vcombine_s16(i16_4, i16_5);

            let i16_6 = vmovn_s32(int6);
            let i16_7 = vmovn_s32(int7);
            let i16_67 = vcombine_s16(i16_6, i16_7);

            let i8_01 = vqmovn_s16(i16_01);
            let i8_23 = vqmovn_s16(i16_23);
            let i8_45 = vqmovn_s16(i16_45);
            let i8_67 = vqmovn_s16(i16_67);

            let i8_03 = vcombine_s8(i8_01, i8_23);
            let i8_47 = vcombine_s8(i8_45, i8_67);

            let mut quants = [0_i8; Q8_0_BLOCK_VALUES];
            vst1q_s8(quants.as_mut_ptr(), i8_03);
            vst1q_s8(quants.as_mut_ptr().add(16), i8_47);

            Q8_0Block { scale, quants }
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use engine_core::tensor::Q8_0Block;

    /// Deterministic 64-bit LCG (constants from Knuth's MMIX) so the fuzz
    /// blocks are reproducible without a `rand` dependency.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }

        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }

        /// A signed byte spanning the full i8 range, biased so the saturation
        /// endpoints and zero appear often.
        fn next_quant(&mut self) -> i8 {
            match self.next_u32() % 8 {
                0 => 127,
                1 => -127,
                2 => -128,
                3 => 0,
                _ => (self.next_u32() % 256) as i8, // wraps into -128..=127
            }
        }

        /// A finite, non-NaN f32 scale in a realistic magnitude range.
        fn next_scale(&mut self) -> f32 {
            let unit = self.next_u32() as f32 / u32::MAX as f32;
            // Span roughly 1e-4 .. ~2.5e-1 without ever hitting 0/NaN/inf.
            unit * 0.25 + 1e-4
        }

        /// A finite f32 element for the quantizer, mixed signs, occasionally
        /// large enough to drive saturation, never NaN or infinite.
        fn next_elem(&mut self) -> f32 {
            let sign = if self.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
            match self.next_u32() % 6 {
                0 => 0.0,
                1 => sign * 5.0, // ties/saturation driver
                _ => {
                    let unit = self.next_u32() as f32 / u32::MAX as f32;
                    sign * unit * 10.0
                }
            }
        }
    }

    fn random_block(rng: &mut Lcg) -> Q8_0Block {
        let mut quants = [0_i8; 32];
        for q in &mut quants {
            *q = rng.next_quant();
        }
        Q8_0Block {
            scale: rng.next_scale(),
            quants,
        }
    }

    #[test]
    fn neon_dot_is_bit_identical_to_portable_reference() {
        let dotprod = neon::aarch64_dotprod_enabled();
        let mut rng = Lcg::new(0x0BAD_C0DE_1234_5678);

        // Many rows of varying length, each row a fresh set of random blocks
        // including saturation, zeros, and mixed signs in the quants.
        for row in 0..512 {
            let blocks = (row % 6) + 1;
            let weight: Vec<Q8_0Block> = (0..blocks).map(|_| random_block(&mut rng)).collect();
            let input: Vec<Q8_0Block> = (0..blocks).map(|_| random_block(&mut rng)).collect();

            let reference = engine_core::tensor::q8_0_dot_rows(&weight, &input);

            // SAFETY: NEON is baseline; dotprod path guarded by feature detect.
            let vmull = unsafe { neon::q8_0_dot_rows_neon_mul(&weight, &input) };
            assert_eq!(
                vmull.to_bits(),
                reference.to_bits(),
                "vmull row {row} diverged from portable reference"
            );

            if dotprod {
                // SAFETY: the host advertises the dot-product feature.
                let sdot = unsafe { neon::q8_0_dot_rows_neon_dotprod(&weight, &input) };
                assert_eq!(
                    sdot.to_bits(),
                    reference.to_bits(),
                    "sdot row {row} diverged from portable reference"
                );
            }
        }

        // This host must actually exercise SDOT for the slice to have run its
        // hardware path.
        assert!(dotprod, "expected the aarch64 test host to advertise dotprod");
    }

    #[test]
    fn neon_quantize_is_bit_identical_to_portable_reference() {
        let mut rng = Lcg::new(0xFEED_FACE_DEAD_BEEF);

        for case in 0..1024 {
            let mut block = [0.0_f32; 32];
            for v in &mut block {
                *v = rng.next_elem();
            }

            let reference = engine_core::tensor::quantize_q8_0_block(&block);
            // SAFETY: NEON is baseline on aarch64; block holds exactly 32 values.
            let neon = unsafe { neon::quantize_q8_0_block(&block) };

            assert_eq!(
                neon.scale.to_bits(),
                reference.scale.to_bits(),
                "quantize case {case}: scale bits diverged"
            );
            assert_eq!(
                neon.quants, reference.quants,
                "quantize case {case}: quants diverged"
            );
        }

        // Explicit saturation blocks: every lane at the positive and negative
        // magnitude extremes must land on +127 / -127 identically.
        let mut pos = [5.0_f32; 32];
        pos[0] = 5.0;
        let mut neg = [-5.0_f32; 32];
        neg[0] = -5.0;
        for block in [&pos, &neg] {
            let reference = engine_core::tensor::quantize_q8_0_block(block);
            // SAFETY: block holds exactly 32 values.
            let neon = unsafe { neon::quantize_q8_0_block(block) };
            assert_eq!(neon.scale.to_bits(), reference.scale.to_bits());
            assert_eq!(neon.quants, reference.quants);
        }
    }

    #[test]
    fn public_dispatchers_match_reference() {
        let mut rng = Lcg::new(0x1357_9BDF_2468_ACE0);
        let flat: Vec<f32> = (0..(32 * 4)).map(|_| rng.next_elem()).collect();

        let ref_blocks = engine_core::tensor::quantize_q8_0_blocks(&flat);
        let neon_blocks = quantize_q8_0_blocks(&flat);
        assert_eq!(neon_blocks, ref_blocks);

        let reference = engine_core::tensor::q8_0_dot_rows(&ref_blocks, &ref_blocks);
        let dispatched = q8_0_dot_rows(&neon_blocks, &neon_blocks);
        assert_eq!(dispatched.to_bits(), reference.to_bits());
    }

    /// A block whose largest magnitude is far below the smallest representable
    /// f16 scale is the one case where the NEON convert path and the portable
    /// scalar path pick different quant bytes. The NEON `1/unrounded_scale`
    /// overflows to infinity, so the float-to-int convert saturates and narrows
    /// differently from the scalar `(v * inv).round().clamp(..)`. This is inert:
    /// the stored scale rounds to exactly 0.0 in both paths, so every quant byte
    /// dequantizes to 0 and the block contributes exactly 0.0 to any dot. The
    /// NEON path is kept byte-for-byte as the accelerated convert rather than
    /// "corrected" to match the scalar bytes, because that reproduces what the
    /// same convert does on this hardware — the property the deterministic lane
    /// actually depends on. Pinned here so the divergence stays understood.
    #[test]
    fn degenerate_scale_band_is_numerically_inert() {
        let block = [1e-40_f32; 32];
        let neon = quantize_q8_0_block(&block);
        let scalar = engine_core::tensor::quantize_q8_0_block(&block);

        // Both stores round the scale to exactly zero.
        assert_eq!(neon.scale.to_bits(), 0);
        assert_eq!(scalar.scale.to_bits(), 0);
        // The quant bytes are allowed to differ in this band...
        assert_ne!(neon.quants, scalar.quants);

        // ...but a zero scale makes any dot contribution exactly zero, so the
        // observable result is identical regardless of the byte choice.
        let weight =
            quantize_q8_0_blocks(&(0..32).map(|i| (i as f32) - 15.0).collect::<Vec<_>>());
        let d_neon = q8_0_dot_rows(&weight, std::slice::from_ref(&neon));
        let d_scalar = engine_core::tensor::q8_0_dot_rows(&weight, std::slice::from_ref(&scalar));
        assert_eq!(d_neon.to_bits(), 0);
        assert_eq!(d_scalar.to_bits(), 0);
    }
}
