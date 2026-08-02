//! x86-64 AVX2 acceleration for the Q8_0 matmul leaf.
//!
//! The public entry points ([`q8_0_dot_rows`], [`quantize_q8_0_block`],
//! [`quantize_q8_0_blocks`]) mirror the portable reference in
//! [`engine_core::tensor`]. On an `x86_64` host that advertises AVX2 they run
//! the kernels in the private `x86` module below; everywhere else — including an
//! x86-64 host without AVX2 — they delegate to the portable reference itself.
//! The fallback is the reference rather than a second scalar transcription of
//! it, so a dispatch decision here can only change speed.
//!
//! Both kernels are bit-identical to the portable reference for every input,
//! with no exception. Two places invite one, and neither is taken here.
//!
//! The first is the quantizer's degenerate band, where the f16 scale rounds to
//! zero: the macOS port documents a divergence there, because the ARM convert
//! saturates before anything can clamp it. The AVX2 quantizer clamps in f32
//! *before* converting, which costs two instructions per vector and removes the
//! band entirely (see `degenerate_scale_band_matches_byte_for_byte`). A port
//! only gets to keep an exception it cannot close.
//!
//! The second is the sign of zero. std's float `Sum` folds from `-0.0`, not
//! `0.0`, so a kernel that seeds its accumulator at `0.0` reproduces the
//! reference on every row with a nonzero term and diverges on the rows where
//! every term is `-0.0` — two empty rows, or a row pairing zero integer sums
//! with negative scales. This kernel seeds `-0.0`; see [`q8_0_dot_rows`] and
//! `an_empty_reduction_keeps_the_reference_sign_of_zero`.
//!
//! Three choices below are load-bearing and would each be silently wrong if
//! "optimized" later.
//!
//! **The block dot widens to i16 and uses `VPMADDWD`, not `VPMADDUBSW`.** The
//! `_mm256_maddubs_epi16` + `_mm256_sign_epi8` idiom is the standard Q8_0 kernel
//! and is roughly twice as cheap here. It is also wrong on this input domain,
//! and a `-128` quant is what makes it wrong. `VPSIGNB` cannot negate `-128` —
//! `-(-128)` is not representable in i8 — so where the idiom moves the weight's
//! sign onto the input byte, a `-128 * -128` product comes back as `-16384`
//! instead of `+16384`.
//!
//! `VPMADDUBSW`'s saturating i16 pairwise sum is a second hazard on that same
//! input and on no other. Two `-128 * -128` products are the only pair this
//! domain can sum to `+32768`, one past `i16::MAX`; a pair of `127 * 127` sums
//! to `32258` and fits, and every mixed pair is smaller still. The saturation
//! never actually fires, because the sign failure has already turned that pair
//! into `-32768` — exactly representable — before saturation could reach it. So
//! the two are independent properties of two instructions that land on a single
//! input, and only the sign one is observable.
//!
//! Upstream kernels get away with the idiom because their own quantizers never
//! emit `-128`; this crate's weights are decoded from GGUF bytes, where `0x80`
//! is an ordinary byte. `VPMADDWD` after a sign-extending widen is exact for the
//! whole i8 range with 4096x of i32 headroom.
//! `extreme_quants_are_exact_where_the_saturating_idiom_is_not` runs a scalar
//! model of the idiom against the reference rather than only describing it.
//!
//! **The f32 reduction stays scalar and sequential.** The integer block sums are
//! exact, so any lane order or horizontal-reduction shape reproduces them — that
//! is the port's one free dimension. The per-block f32 term is not free: it is
//! two roundings, left-associated, folded into one accumulator in ascending
//! block order. Folding the two scales together, accumulating into vector lanes,
//! unrolling into several accumulators, or letting an FMA collapse the multiply
//! and the add each change the answer by a few ulps — enough to flip a greedy
//! token, and invisible to anything but a bit-comparison.
//!
//! **There is no AVX-512 tier.** `avx512f`, `avx512bw` and `avx512vnni` are
//! already in this crate's reported vocabulary, so adding one later costs no
//! identity churn. It is left out because no runner this project has pins a CPU
//! model: a VNNI bit-identity test would compile everywhere and be *executed*
//! nowhere the project can point at, which is the shape of check this crate's
//! module docs already argue against. A Q8_0 block is 32 bytes — a quarter of a
//! ZMM — and the per-block f32 scale forces a horizontal reduce regardless of
//! register width, so the win is not large enough to buy an unverifiable claim.

use engine_core::tensor::{Q8_0Block, Q8_0_BLOCK_VALUES};

/// The accelerated [`engine_core::tensor::Q8Projection`]: every row of one
/// projection, through [`q8_0_dot_rows`].
///
/// Built on `engine_core`'s generic `project_rows` rather than a hand-rolled
/// loop, so the row-splitting policy and its threshold live in one place for
/// every host. Because that helper is generic over the leaf kernel, this
/// monomorphizes with [`q8_0_dot_rows`] inlined, which is the point of the seam
/// being per projection: the AVX2 dot is small enough that an indirect call per
/// output row would cost more than the kernel saves.
///
/// The dispatch inside [`q8_0_dot_rows`] is resolved per row rather than hoisted
/// out of the projection. That is deliberate and costs nothing measurable: the
/// AVX2 check is a cached `OnceLock` read the optimiser hoists out of the loop
/// on its own, and hoisting it by hand would mean two monomorphized copies of
/// the row loop for a branch that is constant for the process's lifetime.
pub fn q8_0_project_rows(
    weight_blocks: &[Q8_0Block],
    blocks_per_row: usize,
    input_blocks: &[Q8_0Block],
    out: &mut [f32],
) {
    engine_core::tensor::project_rows(
        weight_blocks,
        blocks_per_row,
        input_blocks,
        out,
        q8_0_dot_rows,
    )
}

/// Dot two rows of Q8_0 blocks.
///
/// On an AVX2 host this runs the AVX2 kernel, which produces the same exact i32
/// block sums and the same single-accumulator f32 reduction as
/// [`engine_core::tensor::q8_0_dot_rows`]; otherwise it *is* that function.
///
/// Rows of unequal length dot only their common prefix — the reference zips
/// rather than indexes, and a caller may rely on it.
///
/// Two empty rows give `-0.0`, not `+0.0`. The reference reduces with
/// `Iterator::sum`, whose float impl folds from `-0.0`, and this kernel seeds
/// its accumulator the same way, so the sign bit is carried rather than
/// invented. `-0.0` and `+0.0` compare equal and print the same, so a caller
/// only sees the difference through `to_bits` — which is exactly what this
/// module claims to preserve, and what
/// `an_empty_reduction_keeps_the_reference_sign_of_zero` pins.
pub fn q8_0_dot_rows(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x86::avx2_enabled() {
        // SAFETY: `avx2_enabled` returned true, so this host executes the AVX2
        // instructions the kernel issues. The kernel reads only the 32 quant
        // bytes each block owns, through unaligned loads.
        return unsafe { x86::q8_0_dot_rows_avx2(weight, input) };
    }
    engine_core::tensor::q8_0_dot_rows(weight, input)
}

/// Quantize one chunk of exactly [`Q8_0_BLOCK_VALUES`] f32 values into a
/// [`Q8_0Block`].
///
/// Bit-identical to [`engine_core::tensor::quantize_q8_0_block`] for every
/// input, including NaN, both infinities, exact ties, and the band where the
/// stored f16 scale rounds to zero. There is no divergence to document.
///
/// The length check is a real branch, not only a debug assertion. The portable
/// reference is driven by the slice — a short block yields zeros in the tail
/// rather than reading past the end — while the AVX2 path loads 32 floats
/// unconditionally. Delegating the wrong-length case keeps this function
/// memory-safe in release for any slice, which the `debug_assert` alone would
/// not: it is compiled out exactly where it would matter. engine-macos's NEON
/// quantizer has that shape, and in release a short slice reads out of bounds
/// from safe code there.
///
/// No test covers the branch, and none can: the reference debug-asserts the same
/// length, so a debug test build panics inside engine-core before the fallback's
/// result can be compared to anything. What holds it up is construction — the
/// accelerated path is entered only at exactly [`Q8_0_BLOCK_VALUES`] — not a
/// green assertion.
pub fn quantize_q8_0_block(block: &[f32]) -> Q8_0Block {
    debug_assert_eq!(block.len(), Q8_0_BLOCK_VALUES);
    #[cfg(target_arch = "x86_64")]
    if block.len() == Q8_0_BLOCK_VALUES && x86::avx2_enabled() {
        // SAFETY: the length is exactly the 32 values the kernel loads, and
        // `avx2_enabled` returned true for the instructions it issues.
        return unsafe { x86::quantize_q8_0_block_avx2(block) };
    }
    engine_core::tensor::quantize_q8_0_block(block)
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

#[cfg(target_arch = "x86_64")]
mod x86 {
    use engine_core::tensor::{f16_bits_to_f32, f32_to_f16_bits, Q8_0Block, Q8_0_BLOCK_VALUES};
    use std::arch::x86_64::{
        __m128, __m128i, __m256, __m256i, _mm256_add_epi32, _mm256_add_ps, _mm256_and_ps,
        _mm256_andnot_ps, _mm256_castps256_ps128, _mm256_castsi256_si128, _mm256_cmp_ps,
        _mm256_cvtepi8_epi16, _mm256_cvttps_epi32, _mm256_extractf128_ps, _mm256_extracti128_si256,
        _mm256_loadu_ps, _mm256_madd_epi16, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps,
        _mm256_or_ps, _mm256_packs_epi16, _mm256_packs_epi32, _mm256_permutevar8x32_epi32,
        _mm256_round_ps, _mm256_set1_ps, _mm256_setr_epi32, _mm256_setzero_ps, _mm256_storeu_si256,
        _mm256_sub_ps, _mm_add_epi32, _mm_cvtsi128_si32, _mm_cvtss_f32, _mm_loadu_si128,
        _mm_max_ps, _mm_shuffle_epi32, _mm_shuffle_ps, _CMP_EQ_OQ, _MM_FROUND_NO_EXC,
        _MM_FROUND_TO_NEAREST_INT,
    };

    /// Whether the host advertises AVX2. Cached for the process lifetime: the
    /// answer is fixed hardware, and the kernel runs on rayon workers rather
    /// than only the calling thread, so the probe has to be process-wide state
    /// and not something a caller threads through. The seam is a bare `fn`
    /// pointer, so there is nowhere else to put it.
    pub(super) fn avx2_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
    }

    /// Sum the eight i32 lanes of `acc` into a single i32.
    ///
    /// Order-independent and overflow-free for Q8_0 block sums: each lane holds
    /// at most `4 * 128 * 128` in magnitude and the whole sum at most
    /// `32 * 128 * 128 = 524288`, so no intermediate approaches `i32::MAX`.
    ///
    /// # Safety
    ///
    /// The caller must be running on a host with AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn horizontal_sum_i32x8(acc: __m256i) -> i32 {
        // No `unsafe` block: every operation here is register-to-register, and
        // an intrinsic that touches no memory is a safe call inside a function
        // that already enables its target feature. The `unsafe fn` remains
        // because calling it on a host without AVX2 is what is unsound.
        let hi = _mm256_extracti128_si256::<1>(acc);
        let lo = _mm256_castsi256_si128(acc);
        let sum = _mm_add_epi32(lo, hi);
        // 0x4E swaps the 64-bit halves, 0xB1 the adjacent 32-bit lanes, so two
        // rounds collapse four lanes into lane 0.
        let sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0x4E>(sum));
        let sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0xB1>(sum));
        _mm_cvtsi128_si32(sum)
    }

    /// Exact i32 dot of one 32-lane block.
    ///
    /// Sign-extends both operands to i16 (`VPMOVSXBW`, exact for every i8
    /// including `-128`) and multiplies-and-adds adjacent pairs into i32
    /// (`VPMADDWD`, which does not saturate for |operand| <= 128: a product is
    /// at most `16384`, a pair sum at most `32768`). See the module docs for why
    /// the cheaper `VPMADDUBSW` idiom is not usable here.
    ///
    /// # Safety
    ///
    /// The caller must be running on a host with AVX2, and `weight` and `input`
    /// must each point at 32 readable i8 values.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn q8_0_i8_block_avx2(weight: *const i8, input: *const i8) -> i32 {
        // SAFETY: the caller guarantees 32 readable bytes behind each pointer.
        // The loads are the unaligned form because `Q8_0Block` is 4-byte
        // aligned and its `quants` sit at offset 4 of a 36-byte `#[repr(C)]`
        // struct, so a 16-byte-aligned load would fault.
        unsafe {
            let w_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(weight.cast::<__m128i>()));
            let w_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(weight.add(16).cast::<__m128i>()));
            let i_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(input.cast::<__m128i>()));
            let i_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(input.add(16).cast::<__m128i>()));

            let acc =
                _mm256_add_epi32(_mm256_madd_epi16(w_lo, i_lo), _mm256_madd_epi16(w_hi, i_hi));
            horizontal_sum_i32x8(acc)
        }
    }

    /// Dot two rows of Q8_0 blocks using the AVX2 block kernel, reducing the
    /// per-block f32 terms into one accumulator in ascending block order.
    ///
    /// The f32 arithmetic here is plain scalar Rust on purpose; see the module
    /// docs. `zip` rather than an index loop, because the reference truncates to
    /// the shorter row instead of panicking and callers may pass one.
    ///
    /// The accumulator is seeded at `-0.0`, which is not a typo and not a
    /// stylistic choice. The reference reduces with `Iterator::sum`, and std's
    /// float `Sum` folds from `-0.0` rather than `0.0` — `-0.0` is the true
    /// additive identity for f32, because `-0.0 + -0.0` is `-0.0` while
    /// `0.0 + -0.0` is `+0.0`. Seeding at `0.0` reproduces the reference on every
    /// row that has at least one nonzero term, and differs on exactly the rows
    /// where every term is `-0.0`: two empty rows, or a row whose blocks all pair
    /// a zero integer sum with a negative scale (a sign bit the wire format
    /// permits). The result differs only in its sign bit, which compares equal
    /// and prints the same, and is why this needs an explicit test rather than a
    /// fuzz to catch.
    ///
    /// # Safety
    ///
    /// The caller must be running on a host with AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn q8_0_dot_rows_avx2(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
        let mut total_sum = -0.0_f32;
        for (w_block, i_block) in weight.iter().zip(input) {
            // SAFETY: each block owns exactly 32 contiguous quant bytes, and
            // AVX2 is enabled for this function.
            let int_sum =
                unsafe { q8_0_i8_block_avx2(w_block.quants.as_ptr(), i_block.quants.as_ptr()) };
            total_sum += int_sum as f32 * w_block.scale * i_block.scale;
        }
        total_sum
    }

    /// `|v|` with NaN lanes replaced by `+0.0`.
    ///
    /// This is what makes the SIMD abs-max reduction agree with the reference,
    /// and the direction of the disagreement is the opposite of what the
    /// instruction names suggest. Rust's `f32::max` is IEEE `maxNum`: it returns
    /// the *non*-NaN operand, so folding from `0.0` silently ignores NaN inputs
    /// and `max_abs` is the max over the finite lanes. x86 `VMAXPS` is
    /// `(SRC1 > SRC2) ? SRC1 : SRC2`, which returns SRC2 whenever either operand
    /// is NaN — so a naive max tree both propagates NaN and depends on operand
    /// order. Scrubbing to `+0.0` first is exact rather than merely careful:
    /// the fold is seeded at `+0.0` and every other lane is `>= +0.0`, so a
    /// `+0.0` lane cannot change the maximum.
    ///
    /// # Safety
    ///
    /// The caller must be running on a host with AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn abs_with_nan_as_zero(v: __m256) -> __m256 {
        let ordered = _mm256_cmp_ps::<_CMP_EQ_OQ>(v, v);
        _mm256_and_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0), v), ordered)
    }

    /// `x.round().clamp(-128.0, 127.0)` for eight lanes, exactly.
    ///
    /// Three properties have to survive here, and each one rules out the obvious
    /// instruction.
    ///
    /// *NaN becomes `+0.0` first.* The reference path is `NaN.round()` -> NaN,
    /// `NaN.clamp(..)` -> NaN (both of `clamp`'s comparisons are false), then
    /// `NaN as i8` -> 0 by Rust's saturating float cast. `VCVTTPS2DQ` instead
    /// yields the integer-indefinite `0x80000000`, which narrows to `-128`. The
    /// scrub must happen here and not on the input vector: the NaN the
    /// reference's own test exercises is manufactured *inside* the kernel by
    /// `inf * 0.0`, so a mask taken from the pre-multiply values would miss it.
    /// It must also happen before the clamp, because `_mm256_min_ps(NaN, 127.0)`
    /// returns `127.0` and would launder a NaN into a saturated value.
    ///
    /// *The clamp runs in f32, before the convert.* Both bounds are integers and
    /// rounding is monotone with `round(L) = L`, so clamp and round commute and
    /// doing the clamp first is free. What it buys: every lane entering the
    /// convert is an integral f32 in `[-128, 127]`, so the convert is exact and
    /// the two narrowing packs provably never saturate. It is also what closes
    /// the degenerate-scale band, where `inv_scale` overflows to infinity and
    /// the products are infinite — `min(+inf, 127.0)` is `127.0`, matching
    /// `f32::clamp`, where a saturating convert would have produced `-128`.
    ///
    /// *Rounding is half-away-from-zero, which x86 does not have.* Both
    /// `_MM_FROUND_TO_NEAREST_INT` and `VCVTPS2DQ` are ties-to-even, and the
    /// popular `trunc(x + copysign(0.5, x))` fix-up is wrong at exactly one
    /// magnitude: for `x = 0x3EFFFFFF` (the largest f32 below `0.5`) the sum
    /// `x + 0.5` is the exact midpoint between `1 - 2^-24` and `1.0`, ties to
    /// `1.0`, and truncates to `1` where `f32::round` gives `0`. Instead this
    /// rounds to nearest-even and then adds `copysign(1.0, x)` on the lanes
    /// where `x - r == copysign(0.5, x)`. `x - r` is exact for every input
    /// (`|d| <= 0.5`; `r = 0` makes the subtraction trivial, and every lane where
    /// `r` is nonzero has `|x| >= 0.5`, which is where Sterbenz applies —
    /// `|r|/2 <= |x| <= 2|r|` holds for every such lane, `x = 0.75` with `r = 1`
    /// included), and that equality is precisely the case where ties-to-even
    /// rounded toward zero — the only case the two rules disagree on.
    ///
    /// The bound is `0.5` and not `1`: an inv_scale of exactly `1.0` — a block
    /// whose max magnitude is exactly `127.0` — delivers lanes in `(0.5, 1)` to
    /// this function, and those round to `±1` rather than to `0`. Stating it as
    /// `|x| >= 1` would leave that interval covered by neither half.
    ///
    /// `_MM_FROUND_TO_NEAREST_INT` also has the "use MXCSR" bit clear, so the
    /// mode is architectural rather than inherited from whatever a host process
    /// left in the control word.
    ///
    /// # Safety
    ///
    /// The caller must be running on a host with AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn round_half_away_clamped_to_i32(x: __m256) -> __m256i {
        let ordered = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, x);
        let x = _mm256_and_ps(x, ordered);
        let x = _mm256_min_ps(
            _mm256_max_ps(x, _mm256_set1_ps(-128.0)),
            _mm256_set1_ps(127.0),
        );

        let sign_mask = _mm256_set1_ps(-0.0);
        let nearest = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(x);
        let delta = _mm256_sub_ps(x, nearest);
        let sign = _mm256_and_ps(x, sign_mask);
        let signed_half = _mm256_or_ps(sign, _mm256_set1_ps(0.5));
        let is_tie = _mm256_cmp_ps::<_CMP_EQ_OQ>(delta, signed_half);
        let signed_one = _mm256_or_ps(sign, _mm256_set1_ps(1.0));
        let rounded = _mm256_add_ps(nearest, _mm256_and_ps(is_tie, signed_one));

        _mm256_cvttps_epi32(rounded)
    }

    /// AVX2 quantizer for one 32-value block.
    ///
    /// The abs-max reduction, the per-lane scale-and-round and the narrowing
    /// store are vectorized; the scale tail is scalar Rust calling engine-core's
    /// own f16 helpers. That split is deliberate. F16C's `VCVTPS2PH` is
    /// available and would be one instruction, but it honours `MXCSR.FTZ` and
    /// canonicalizes NaN, where [`f32_to_f16_bits`] never flushes a subnormal
    /// half and deliberately preserves a NaN payload bit — so the fast
    /// instruction is a different function, on inputs this kernel can be handed.
    /// Reusing engine-core's leaves nothing to prove about the scale at all.
    ///
    /// The divisor asymmetry is the other thing carried over verbatim: quants
    /// are divided by the *un-rounded* scale even though the f16-rounded one is
    /// what gets stored, and the division is a precomputed reciprocal multiply
    /// (two roundings), not `_mm256_div_ps` (one) and certainly not
    /// `_mm256_rcp_ps` (twelve bits).
    ///
    /// # Safety
    ///
    /// The caller must be running on a host with AVX2, and `block` must hold
    /// exactly [`Q8_0_BLOCK_VALUES`] values — this reads all 32 unconditionally.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn quantize_q8_0_block_avx2(block: &[f32]) -> Q8_0Block {
        debug_assert_eq!(block.len(), Q8_0_BLOCK_VALUES);

        // SAFETY: the caller guarantees exactly 32 values, so four 8-wide
        // unaligned loads and one 32-byte unaligned store stay in bounds.
        unsafe {
            let v0 = _mm256_loadu_ps(block.as_ptr());
            let v1 = _mm256_loadu_ps(block.as_ptr().add(8));
            let v2 = _mm256_loadu_ps(block.as_ptr().add(16));
            let v3 = _mm256_loadu_ps(block.as_ptr().add(24));

            let mut max_vec = _mm256_setzero_ps();
            for v in [v0, v1, v2, v3] {
                max_vec = _mm256_max_ps(max_vec, abs_with_nan_as_zero(v));
            }
            let max_abs = horizontal_max_f32x8(max_vec);

            let unrounded_scale = max_abs / 127.0;
            let scale_bits = f32_to_f16_bits(unrounded_scale);
            let scale = f16_bits_to_f32(scale_bits);
            let inv_scale = if unrounded_scale == 0.0 {
                0.0
            } else {
                1.0 / unrounded_scale
            };

            let inv = _mm256_set1_ps(inv_scale);
            let i0 = round_half_away_clamped_to_i32(_mm256_mul_ps(v0, inv));
            let i1 = round_half_away_clamped_to_i32(_mm256_mul_ps(v1, inv));
            let i2 = round_half_away_clamped_to_i32(_mm256_mul_ps(v2, inv));
            let i3 = round_half_away_clamped_to_i32(_mm256_mul_ps(v3, inv));

            // AVX2's packs are per-128-bit-lane, so the 32 bytes land in the
            // dword order 0,4,1,5,2,6,3,7 relative to what the block wants. One
            // lane-crossing VPERMD puts them back; the values themselves are
            // already in `[-128, 127]`, so neither pack saturates.
            let packed = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
            let ordered =
                _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

            let mut quants = [0_i8; Q8_0_BLOCK_VALUES];
            _mm256_storeu_si256(quants.as_mut_ptr().cast::<__m256i>(), ordered);

            Q8_0Block { scale, quants }
        }
    }

    /// Reduce eight lanes to their maximum. Only ever called on a vector whose
    /// lanes are already non-NaN and `>= +0.0`, so the reduction order does not
    /// matter and `VMAXPS`'s NaN behaviour never comes up.
    ///
    /// # Safety
    ///
    /// The caller must be running on a host with AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_max_f32x8(v: __m256) -> f32 {
        let hi: __m128 = _mm256_extractf128_ps::<1>(v);
        let lo: __m128 = _mm256_castps256_ps128(v);
        let m = _mm_max_ps(lo, hi);
        let m = _mm_max_ps(m, _mm_shuffle_ps::<0x4E>(m, m));
        let m = _mm_max_ps(m, _mm_shuffle_ps::<0xB1>(m, m));
        _mm_cvtss_f32(m)
    }
}

/// Dispatcher-level checks, ungated so they run on every host this crate is
/// built for.
///
/// What they prove: the public entry points agree with the portable reference on
/// whatever arm this host actually takes, and the non-x86-64 arm is wired to the
/// reference rather than to nothing. What they do not prove: that the AVX2
/// kernels are correct — on an aarch64 or non-AVX2 host these compare the
/// reference against itself and pass by construction. That claim belongs to the
/// arch-gated module below, which asserts the host advertises AVX2 rather than
/// skipping quietly.
#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic 64-bit LCG (constants from Knuth's MMIX) so the fuzz blocks
    /// are reproducible without a `rand` dependency.
    pub(super) struct Lcg(u64);

    impl Lcg {
        pub(super) fn new(seed: u64) -> Self {
            Lcg(seed)
        }

        pub(super) fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }

        /// A signed byte spanning the full i8 range, biased so the saturation
        /// endpoints and zero appear often. `-128` is the value that breaks
        /// every `VPMADDUBSW`-based kernel, and no test in engine-core ever
        /// feeds one to the reference dot, so generating it here is the point.
        pub(super) fn next_quant(&mut self) -> i8 {
            match self.next_u32() % 8 {
                0 => 127,
                1 => -127,
                2 => -128,
                3 => 0,
                _ => (self.next_u32() % 256) as i8, // wraps into -128..=127
            }
        }

        /// A finite, non-NaN f32 scale in a realistic magnitude range.
        pub(super) fn next_scale(&mut self) -> f32 {
            let unit = self.next_u32() as f32 / u32::MAX as f32;
            unit * 0.25 + 1e-4
        }

        /// A finite f32 element for the quantizer: mixed signs, sometimes large
        /// enough to drive saturation, sometimes a half-integer so the scaled
        /// value lands on a rounding tie. Never NaN or infinite — those have
        /// their own table, because a fuzz that produced them would be asserting
        /// the interesting cases at a rate nobody can read off the seed.
        pub(super) fn next_elem(&mut self) -> f32 {
            let sign = if self.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
            match self.next_u32() % 8 {
                0 => 0.0,
                1 => sign * 5.0,
                2 => sign * ((self.next_u32() % 64) as f32 + 0.5),
                3 => sign * (self.next_u32() % 128) as f32,
                _ => {
                    let unit = self.next_u32() as f32 / u32::MAX as f32;
                    sign * unit * 10.0
                }
            }
        }
    }

    pub(super) fn random_block(rng: &mut Lcg) -> Q8_0Block {
        let mut quants = [0_i8; Q8_0_BLOCK_VALUES];
        for q in &mut quants {
            *q = rng.next_quant();
        }
        Q8_0Block {
            scale: rng.next_scale(),
            quants,
        }
    }

    /// Bit equality, not `PartialEq`. `Q8_0Block` derives `PartialEq`, which
    /// compares `scale` with f32 `==` — under which `+0.0` equals `-0.0` and a
    /// NaN scale never equals itself. Comparing the way the derive does would
    /// weaken exactly the claim this port exists to make.
    pub(super) fn assert_block_bits_eq(got: &Q8_0Block, want: &Q8_0Block, case: &str) {
        assert_eq!(
            got.scale.to_bits(),
            want.scale.to_bits(),
            "{case}: scale bits diverged"
        );
        assert_eq!(got.quants, want.quants, "{case}: quant bytes diverged");
    }

    /// The projection agrees with engine-core's portable projection, on both
    /// sides of the threshold where the row loop becomes parallel.
    ///
    /// The row dot is covered above; what this adds is the seam itself. Row
    /// counts are chosen either side of `PARALLEL_LINEAR_MIN_OUTPUTS` because
    /// that is where `project_rows` switches to rayon, and a projection is only
    /// order-independent while nothing accumulates across rows. If a future
    /// kernel ever split a single row's fold to widen a vector, this is the test
    /// that would go red, on the wide case only.
    #[test]
    fn projection_matches_the_portable_projection_across_the_parallel_threshold() {
        use engine_core::tensor::PARALLEL_LINEAR_MIN_OUTPUTS;

        let mut rng = Lcg::new(0x5EED_0F17_C0DE_1234);
        let blocks_per_row = 4;

        for rows in [1, 7, PARALLEL_LINEAR_MIN_OUTPUTS - 1, PARALLEL_LINEAR_MIN_OUTPUTS + 3] {
            let weight: Vec<Q8_0Block> =
                (0..rows * blocks_per_row).map(|_| random_block(&mut rng)).collect();
            let input: Vec<Q8_0Block> =
                (0..blocks_per_row).map(|_| random_block(&mut rng)).collect();

            let mut got = vec![0.0_f32; rows];
            q8_0_project_rows(&weight, blocks_per_row, &input, &mut got);

            let mut want = vec![0.0_f32; rows];
            engine_core::tensor::q8_0_project_rows(&weight, blocks_per_row, &input, &mut want);

            for (row, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "projection of {rows} rows diverged at row {row}"
                );
            }
        }
    }

    #[test]
    fn dispatched_quantizer_matches_the_portable_reference() {
        let mut rng = Lcg::new(0x1357_9BDF_2468_ACE0);
        let flat: Vec<f32> = (0..(Q8_0_BLOCK_VALUES * 16))
            .map(|_| rng.next_elem())
            .collect();

        let reference = engine_core::tensor::quantize_q8_0_blocks(&flat);
        let dispatched = quantize_q8_0_blocks(&flat);
        assert_eq!(dispatched.len(), reference.len());
        for (idx, (got, want)) in dispatched.iter().zip(&reference).enumerate() {
            assert_block_bits_eq(got, want, &format!("dispatched block {idx}"));
        }

        let want = engine_core::tensor::q8_0_dot_rows(&reference, &reference);
        let got = q8_0_dot_rows(&dispatched, &dispatched);
        assert_eq!(got.to_bits(), want.to_bits());
    }

    #[test]
    fn dispatched_dot_matches_the_portable_reference_on_random_blocks() {
        let mut rng = Lcg::new(0x00C0_FFEE_0BAD_F00D);
        for row in 0..128 {
            let blocks = (row % 9) + 1;
            let weight: Vec<Q8_0Block> = (0..blocks).map(|_| random_block(&mut rng)).collect();
            let input: Vec<Q8_0Block> = (0..blocks).map(|_| random_block(&mut rng)).collect();
            assert_eq!(
                q8_0_dot_rows(&weight, &input).to_bits(),
                engine_core::tensor::q8_0_dot_rows(&weight, &input).to_bits(),
                "dispatched row {row} diverged from the portable reference"
            );
        }
    }

    /// The reference zips, so a length mismatch dots the common prefix instead
    /// of panicking. A kernel that took its trip count from `weight.len()` would
    /// panic here, or in unsafe code read past the end of the shorter row.
    #[test]
    fn dispatched_dot_truncates_to_the_shorter_row() {
        let mut rng = Lcg::new(0xDEAD_10CC_5EED_0001);
        let long: Vec<Q8_0Block> = (0..7).map(|_| random_block(&mut rng)).collect();
        let short: Vec<Q8_0Block> = (0..3).map(|_| random_block(&mut rng)).collect();

        for (w, i) in [(&long, &short), (&short, &long)] {
            assert_eq!(
                q8_0_dot_rows(w, i).to_bits(),
                engine_core::tensor::q8_0_dot_rows(w, i).to_bits(),
            );
        }
    }

    /// An empty reduction is `-0.0`, not `+0.0`, and the two differ in bits
    /// while comparing equal and printing the same.
    ///
    /// The reference reduces with `Iterator::sum`, whose float impl folds from
    /// `-0.0` — the true additive identity, since `0.0 + -0.0` is `+0.0`. A
    /// kernel that seeds its accumulator at `0.0` matches on every row with a
    /// nonzero term and diverges on the all-`-0.0` rows: two empty slices, and a
    /// row whose blocks all pair a zero integer sum with a negative scale, which
    /// the wire format's sign bit permits. Asserted against a literal rather
    /// than against the reference so the claim is legible without running both.
    #[test]
    fn an_empty_reduction_keeps_the_reference_sign_of_zero() {
        assert_eq!(q8_0_dot_rows(&[], &[]).to_bits(), (-0.0_f32).to_bits());

        let mut rng = Lcg::new(0xDEAD_10CC_5EED_0002);
        let row: Vec<Q8_0Block> = (0..5).map(|_| random_block(&mut rng)).collect();
        assert_eq!(
            q8_0_dot_rows(&row, &[]).to_bits(),
            engine_core::tensor::q8_0_dot_rows(&row, &[]).to_bits(),
        );

        // Every term `-0.0`: a zero integer sum against a negative scale.
        let zeroed = |scale: f32| Q8_0Block {
            scale,
            quants: [0_i8; Q8_0_BLOCK_VALUES],
        };
        let weight = [zeroed(-1.0), zeroed(-2.0)];
        let input = [zeroed(1.0), zeroed(1.0)];
        let reference = engine_core::tensor::q8_0_dot_rows(&weight, &input);
        assert_eq!(reference.to_bits(), (-0.0_f32).to_bits());
        assert_eq!(
            q8_0_dot_rows(&weight, &input).to_bits(),
            reference.to_bits()
        );
    }
}

/// The AVX2 kernels against the portable reference, on the hardware.
///
/// Gated on `target_arch` rather than `target_os`, which is what gets these run
/// on the Linux runner as well as the Windows one — this crate's tests are
/// deliberately not `target_os`-gated, and the numeric claim is about the
/// instruction set, not the operating system. The gate is still required: the
/// aarch64-pc-windows-msvc cross-check compiles `--all-targets`, where none of
/// these intrinsics exist.
///
/// Every test here returns early without AVX2, so
/// `the_x86_64_test_host_advertises_avx2` is the vacuity guard: without it a
/// hypothetical pre-Haswell runner would turn this whole module green while
/// executing none of it.
#[cfg(all(test, target_arch = "x86_64"))]
mod avx2_kernel {
    use super::tests::{assert_block_bits_eq, random_block, Lcg};
    use super::*;
    use engine_core::tensor::{f16_bits_to_f32, f32_to_f16_bits};

    /// Safe wrappers around the two `unsafe` kernels.
    ///
    /// The guard is inside the wrapper, not delegated to a caller convention.
    /// Every test below already opens with `if !x86::avx2_enabled() { return; }`,
    /// so on a host without AVX2 the assertion is never reached and the skip
    /// behaviour is unchanged — but the prologue is four lines repeated eight
    /// times, which is the shape that gets copy-edited away. A test that loses
    /// it now fails on a named assertion instead of taking SIGILL from a call
    /// site the compiler accepted without `unsafe`.
    fn avx2_dot(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
        assert!(
            x86::avx2_enabled(),
            "avx2_dot called on a host without AVX2"
        );
        // SAFETY: the assertion above dominates this call, so the host executes
        // the instructions the kernel issues.
        unsafe { x86::q8_0_dot_rows_avx2(weight, input) }
    }

    fn avx2_quantize(block: &[f32; Q8_0_BLOCK_VALUES]) -> Q8_0Block {
        assert!(
            x86::avx2_enabled(),
            "avx2_quantize called on a host without AVX2"
        );
        // SAFETY: the assertion above dominates this call, and the block is
        // exactly the 32 values the kernel loads.
        unsafe { x86::quantize_q8_0_block_avx2(block) }
    }

    /// The vacuity guard, and a deliberate trade-off rather than an oversight.
    ///
    /// Every other test here returns early without AVX2, so without this one a
    /// pre-Haswell Intel, a pre-Excavator AMD, an x86-64 guest with AVX2 masked
    /// off, or an x86-64 build under Rosetta 2 would turn the whole module green
    /// while executing none of it. The cost is that such a host fails exactly
    /// this test while behaving entirely correctly — it takes the portable
    /// fallback arm the module docs promise, and the dispatcher tests above
    /// still check that arm against the reference. A green suite that proves
    /// nothing is the worse failure, so the assertion stays and says so.
    ///
    /// No runner this project has is affected: both `ubuntu-latest` and
    /// `windows-latest` advertise AVX2, and the aarch64 cross-check compiles
    /// this module out.
    #[test]
    fn the_x86_64_test_host_advertises_avx2() {
        assert!(
            x86::avx2_enabled(),
            "this x86-64 host does not advertise AVX2, so every bit-identity \
             test in this module returned before executing a single kernel \
             instruction. The suite is green and proves nothing."
        );
    }

    #[test]
    fn avx2_dot_is_bit_identical_to_the_portable_reference() {
        if !x86::avx2_enabled() {
            return;
        }
        let mut rng = Lcg::new(0x0BAD_C0DE_1234_5678);

        // Row lengths sweep 0..=33 so the corpus crosses every plausible unroll
        // factor. The kernel does not unroll today; a later one that did, into
        // several f32 accumulators, would reassociate the reduction and pass a
        // three-block test while drifting on real rows of hundreds of blocks.
        for row in 0..1024_usize {
            let blocks = row % 34;
            let weight: Vec<Q8_0Block> = (0..blocks).map(|_| random_block(&mut rng)).collect();
            let input: Vec<Q8_0Block> = (0..blocks).map(|_| random_block(&mut rng)).collect();

            let reference = engine_core::tensor::q8_0_dot_rows(&weight, &input);
            assert_eq!(
                avx2_dot(&weight, &input).to_bits(),
                reference.to_bits(),
                "AVX2 row {row} ({blocks} blocks) diverged from the portable reference"
            );
        }

        // Mismatched lengths in both directions: the reference zips.
        let long: Vec<Q8_0Block> = (0..11).map(|_| random_block(&mut rng)).collect();
        let short: Vec<Q8_0Block> = (0..4).map(|_| random_block(&mut rng)).collect();
        for (w, i) in [(&long, &short), (&short, &long)] {
            assert_eq!(
                avx2_dot(w, i).to_bits(),
                engine_core::tensor::q8_0_dot_rows(w, i).to_bits(),
            );
        }
    }

    /// Scalar model of the `VPSIGNB` + `VPMADDUBSW` idiom this kernel rejects,
    /// so the module docs' claim about it is executed rather than asserted in
    /// prose. One block, both rows constant, which is the only shape the table
    /// below uses.
    ///
    /// `ax = _mm256_sign_epi8(w, w)` read as *unsigned* bytes is `|w|` for every
    /// i8 including `-128` (whose wrapping negation, `0x80`, reads as `128`).
    /// `sy = _mm256_sign_epi8(i, w)` is where it breaks: for `w < 0` it wants
    /// `-i`, and `-(-128)` wraps back to `-128`. `VPMADDUBSW` then multiplies
    /// the unsigned byte by the signed one and adds adjacent pairs with signed
    /// i16 saturation.
    fn maddubs_sign_idiom_block(w: i8, i: i8) -> i32 {
        let ax = i32::from(w.unsigned_abs());
        let sy = i32::from(match w.signum() {
            0 => 0,
            1 => i,
            _ => i.wrapping_neg(),
        });
        // Both lanes of every pair are identical here, so one pair stands for
        // all sixteen.
        let pair = (ax * sy) * 2;
        pair.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) * 16
    }

    /// The one input class that separates this kernel from the standard
    /// `VPMADDUBSW` + `VPSIGNB` Q8_0 kernel, and the class no engine-core test
    /// reaches: `-128` quants, which the reference quantizer cannot emit but
    /// `decode_q8_0_blocks` produces from any `0x80` byte on the wire.
    ///
    /// `-128 x -128` is the whole story, and the table says which rows are and
    /// are not witnesses. Under the sign trick that product comes back as
    /// `-16384` instead of `+16384`, so the row diverges. It is also the only
    /// row whose correct pair sum, `+32768`, is out of i16 range at all — the
    /// `127 x 127` pair is `32258` and fits, so that row is a control, not a
    /// saturation witness. The saturating add never fires even on the `-128`
    /// row, because the sign failure has already turned `+32768` into `-32768`,
    /// which is representable.
    ///
    /// So the idiom is checked here, not just described: it must reproduce the
    /// reference on the three control rows and miss it on the one witness. A row
    /// pruned as "redundant" then fails on its own label.
    #[test]
    fn extreme_quants_are_exact_where_the_saturating_idiom_is_not() {
        let avx2 = x86::avx2_enabled();
        let unit = |q: i8| Q8_0Block {
            scale: 1.0,
            quants: [q; Q8_0_BLOCK_VALUES],
        };

        for (w, i, want, idiom_agrees) in [
            // 32 * 16384: the largest product, the only pair sum past
            // `i16::MAX`, and the only row the idiom gets wrong.
            (-128_i8, -128_i8, 524288.0_f32, false),
            (-128, 127, -520192.0, true), // 32 * -16256, pair -32512, fits
            (127, 127, 516128.0, true),   // 32 * 16129, pair 32258, fits
            (-128, 0, 0.0, true),
        ] {
            let weight = [unit(w)];
            let input = [unit(i)];
            let reference = engine_core::tensor::q8_0_dot_rows(&weight, &input);
            assert_eq!(reference.to_bits(), want.to_bits(), "reference {w} x {i}");

            // Scale is 1.0 on both rows, so the f32 term is the integer sum.
            let idiom = maddubs_sign_idiom_block(w, i) as f32;
            if idiom_agrees {
                assert_eq!(
                    idiom.to_bits(),
                    want.to_bits(),
                    "the {w} x {i} row is labelled a control but the idiom misses it"
                );
            } else {
                assert_ne!(
                    idiom.to_bits(),
                    want.to_bits(),
                    "the {w} x {i} row is labelled the witness but the idiom reproduces it"
                );
            }

            if avx2 {
                assert_eq!(
                    avx2_dot(&weight, &input).to_bits(),
                    want.to_bits(),
                    "AVX2 {w} x {i}"
                );
            }
        }
    }

    #[test]
    fn avx2_quantize_is_bit_identical_to_the_portable_reference() {
        if !x86::avx2_enabled() {
            return;
        }
        let mut rng = Lcg::new(0xFEED_FACE_DEAD_BEEF);

        for case in 0..4096 {
            let mut block = [0.0_f32; Q8_0_BLOCK_VALUES];
            for v in &mut block {
                *v = rng.next_elem();
            }
            let reference = engine_core::tensor::quantize_q8_0_block(&block);
            assert_block_bits_eq(&avx2_quantize(&block), &reference, &format!("case {case}"));
        }

        // Every lane at a magnitude extreme, and both signed zeros: the
        // reference's abs-fold clears the sign, and `-0.0` must not reach the
        // convert as a `-1`.
        for block in [
            [5.0_f32; Q8_0_BLOCK_VALUES],
            [-5.0_f32; Q8_0_BLOCK_VALUES],
            [0.0_f32; Q8_0_BLOCK_VALUES],
            [-0.0_f32; Q8_0_BLOCK_VALUES],
            [f32::MAX; Q8_0_BLOCK_VALUES],
            [f32::MIN_POSITIVE; Q8_0_BLOCK_VALUES],
        ] {
            let reference = engine_core::tensor::quantize_q8_0_block(&block);
            assert_block_bits_eq(&avx2_quantize(&block), &reference, "extreme block");
        }
    }

    /// Rounding is half-away-from-zero, and this is the table that says so.
    ///
    /// The block's largest magnitude is exactly `127.0`, which makes
    /// `unrounded_scale` exactly `1.0` and `inv_scale` exactly `1.0` — so the
    /// values in the block *are* the pre-round values and the expected quants
    /// can be read off rather than reverse-engineered.
    ///
    /// The ties chosen matter, and the rule is one line: a tie discriminates
    /// exactly when its away-from-zero answer is *odd*, because that is when
    /// ties-to-even must go the other way. `1.5 -> 2`, `3.5 -> 4` and
    /// `63.5 -> 64` are ties with an even away answer, so ties-to-even agrees
    /// and a `VCVTPS2DQ` kernel passes a test built only from those —
    /// engine-core's own round-trip test is built only from `63.5`. The
    /// discriminating ties here are `0.5 -> 1`, `2.5 -> 3`, `62.5 -> 63` and
    /// `126.5 -> 127` with their negatives; `0.5` is one of them, not a
    /// makeweight, since ties-to-even sends it to `0`. And `0x3EFFFFFF -> 0` is
    /// the value that separates a correct rounder from the
    /// `trunc(x + copysign(0.5, x))` shortcut, which returns 1 there.
    ///
    /// That classification is asserted below rather than left as prose, so the
    /// table cannot be re-labelled or pruned on a wrong reading of it.
    #[test]
    fn rounding_is_half_away_from_zero_at_the_discriminating_ties() {
        if !x86::avx2_enabled() {
            return;
        }
        const JUST_UNDER_HALF: f32 = f32::from_bits(0x3EFF_FFFF);

        let mut block = [0.0_f32; Q8_0_BLOCK_VALUES];
        let values: [f32; 16] = [
            127.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            -2.5,
            3.5,
            -3.5,
            62.5,
            -62.5,
            63.5,
            -63.5,
            126.5,
            JUST_UNDER_HALF,
            -JUST_UNDER_HALF,
        ];
        block[..values.len()].copy_from_slice(&values);

        let reference = engine_core::tensor::quantize_q8_0_block(&block);
        let got = avx2_quantize(&block);
        assert_block_bits_eq(&got, &reference, "tie table");

        // The construction claim, so a change to it fails here rather than
        // silently making the table meaningless.
        assert_eq!(reference.scale.to_bits(), 1.0_f32.to_bits());

        let expected: [i8; 16] = [127, 1, -1, 2, -2, 3, -3, 4, -4, 63, -63, 64, -64, 127, 0, 0];
        assert_eq!(&got.quants[..expected.len()], &expected[..]);

        // The classification, executed. A ties-to-even kernel — plain
        // `VCVTPS2DQ`, or `_MM_FROUND_TO_NEAREST_INT` without the fix-up —
        // answers `round_ties_even` here, and every entry where that differs
        // from `f32::round` is an entry this table would catch it on.
        let mut discriminating = 0;
        for (idx, v) in values.iter().enumerate() {
            assert_eq!(
                i32::from(got.quants[idx]),
                v.round() as i32,
                "entry {idx} ({v}) is not the half-away answer"
            );
            if v.round_ties_even() != v.round() {
                discriminating += 1;
                assert_ne!(
                    i32::from(got.quants[idx]),
                    v.round_ties_even() as i32,
                    "entry {idx} ({v}) is labelled discriminating but ties-to-even agrees"
                );
            }
        }
        assert_eq!(
            discriminating, 7,
            "the table is meant to carry seven entries a ties-to-even kernel fails: \
             +/-0.5, +/-2.5, +/-62.5 and 126.5"
        );
    }

    /// NaN and both infinities, which the reference handles through a chain no
    /// x86 convert reproduces: `clamp` passes NaN through (both of its
    /// comparisons are false), then `as i8` saturates NaN to `0`, `+inf` to
    /// `127` and `-inf` to `-128`. `VCVTTPS2DQ` maps all three to
    /// `0x80000000`, which narrows to `-128`.
    ///
    /// The one-infinity case is the trap: `max_abs` is infinite, so `inv_scale`
    /// is `1/inf = +0.0` and every product is `0.0` or `inf * 0.0 = NaN`. The
    /// NaN is manufactured inside the kernel — the input vector holds none — so
    /// a scrub taken from the inputs would miss it.
    #[test]
    fn nan_and_infinite_inputs_quantize_like_the_reference() {
        if !x86::avx2_enabled() {
            return;
        }
        let mut cases: Vec<(&str, [f32; Q8_0_BLOCK_VALUES])> = Vec::new();

        let mut one_inf = [1.0_f32; Q8_0_BLOCK_VALUES];
        one_inf[7] = f32::INFINITY;
        cases.push(("one +inf", one_inf));

        let mut one_neg_inf = [1.0_f32; Q8_0_BLOCK_VALUES];
        one_neg_inf[19] = f32::NEG_INFINITY;
        cases.push(("one -inf", one_neg_inf));

        cases.push(("all +inf", [f32::INFINITY; Q8_0_BLOCK_VALUES]));
        cases.push(("all -inf", [f32::NEG_INFINITY; Q8_0_BLOCK_VALUES]));

        let mut one_nan = [1.0_f32; Q8_0_BLOCK_VALUES];
        one_nan[3] = f32::NAN;
        cases.push(("one NaN", one_nan));

        cases.push(("all NaN", [f32::NAN; Q8_0_BLOCK_VALUES]));

        let mut mixed = [0.25_f32; Q8_0_BLOCK_VALUES];
        mixed[1] = f32::NAN;
        mixed[2] = f32::INFINITY;
        mixed[3] = f32::NEG_INFINITY;
        mixed[4] = -f32::NAN;
        mixed[5] = -0.0;
        cases.push(("mixed", mixed));

        for (name, block) in cases {
            let reference = engine_core::tensor::quantize_q8_0_block(&block);
            assert_block_bits_eq(&avx2_quantize(&block), &reference, name);
        }

        // The specific value engine-core pins, restated so a regression names
        // itself: an infinity-bearing block quantizes to all zeros, not to a
        // `-128` in the infinite lane.
        let reference = engine_core::tensor::quantize_q8_0_block(&one_inf);
        assert!(reference.scale.is_infinite());
        assert_eq!(avx2_quantize(&one_inf).quants, [0_i8; Q8_0_BLOCK_VALUES]);
    }

    /// The band where the stored f16 scale rounds to exactly zero while the
    /// quants do not.
    ///
    /// This is where the macOS NEON quantizer has its one documented
    /// divergence: `1/unrounded_scale` overflows to infinity, ARM's
    /// float-to-int convert saturates to `i32::MAX`, and the truncating narrow
    /// turns that into `-1` where the scalar path clamps to `127`. It is inert
    /// there — a zero scale makes every contribution `0.0` — but it is still an
    /// exception the reader has to carry.
    ///
    /// The AVX2 path has no such exception, and the reason is one instruction:
    /// clamping in f32 before the convert turns `+inf` into `127.0` and `-inf`
    /// into `-128.0`, which is exactly what `f32::clamp` does. So this test
    /// asserts *equality* where the macOS one asserts inequality. The inertness
    /// is still checked, because it is what would make a future divergence
    /// survivable rather than a bug.
    #[test]
    fn degenerate_scale_band_matches_byte_for_byte() {
        if !x86::avx2_enabled() {
            return;
        }
        let mut mixed = [1e-40_f32; Q8_0_BLOCK_VALUES];
        for (idx, v) in mixed.iter_mut().enumerate() {
            if idx % 3 == 0 {
                *v = -1e-40;
            }
        }

        for block in [[1e-40_f32; Q8_0_BLOCK_VALUES], mixed] {
            let reference = engine_core::tensor::quantize_q8_0_block(&block);
            let got = avx2_quantize(&block);
            assert_eq!(
                reference.scale.to_bits(),
                0,
                "the band's premise: scale is 0"
            );
            assert_block_bits_eq(&got, &reference, "degenerate scale band");

            // Inert either way: a zero scale zeroes the whole term.
            let weight = quantize_q8_0_blocks(
                &(0..Q8_0_BLOCK_VALUES)
                    .map(|i| (i as f32) - 15.0)
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                q8_0_dot_rows(&weight, std::slice::from_ref(&got)).to_bits(),
                0
            );
        }
    }

    /// The stored scale is f16-rounded; the divisor is not. Three ways to get
    /// this wrong — reusing the stored scale's reciprocal, dividing instead of
    /// multiplying by a precomputed reciprocal, or reaching for `VRCPPS` — and
    /// this block separates the first from the truth by one count, so it is a
    /// witness rather than a restatement.
    #[test]
    fn quants_divide_by_the_unrounded_scale_not_the_stored_one() {
        if !x86::avx2_enabled() {
            return;
        }
        let mut probe = [0.0_f32; Q8_0_BLOCK_VALUES];
        probe[0] = 0.13;
        probe[1] = 185.0 * 0.001 * 0.13;

        let got = avx2_quantize(&probe);
        let reference = engine_core::tensor::quantize_q8_0_block(&probe);
        assert_block_bits_eq(&got, &reference, "scale asymmetry probe");

        let unrounded = 0.13_f32 / 127.0;
        let stored = f16_bits_to_f32(f32_to_f16_bits(unrounded));
        assert_ne!(stored, unrounded);
        assert_eq!(got.scale.to_bits(), stored.to_bits());
        assert_eq!(got.quants[1], (probe[1] * (1.0 / unrounded)).round() as i8);
        assert_ne!(
            got.quants[1],
            (probe[1] * (1.0 / stored)).round().clamp(-128.0, 127.0) as i8,
            "the stored-scale reciprocal would have produced a different quant here, \
             so this block genuinely discriminates between the two divisors"
        );
    }

    /// Quantize-then-dot end to end on the accelerated path, against the
    /// reference doing the same. Both halves of the kernel in one expression is
    /// what the forward pass actually asks for.
    #[test]
    fn quantize_then_dot_agrees_end_to_end() {
        if !x86::avx2_enabled() {
            return;
        }
        let mut rng = Lcg::new(0x5EED_1234_ABCD_9876);
        for row in 0..64_usize {
            let width = Q8_0_BLOCK_VALUES * ((row % 7) + 1);
            let flat_w: Vec<f32> = (0..width).map(|_| rng.next_elem()).collect();
            let flat_i: Vec<f32> = (0..width).map(|_| rng.next_elem()).collect();

            let w_ref = engine_core::tensor::quantize_q8_0_blocks(&flat_w);
            let i_ref = engine_core::tensor::quantize_q8_0_blocks(&flat_i);
            let w_avx = quantize_q8_0_blocks(&flat_w);
            let i_avx = quantize_q8_0_blocks(&flat_i);

            assert_eq!(
                q8_0_dot_rows(&w_avx, &i_avx).to_bits(),
                engine_core::tensor::q8_0_dot_rows(&w_ref, &i_ref).to_bits(),
                "row {row}"
            );
        }
    }
}
