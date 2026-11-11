//! Research only: bounded exact widening and certified coarse reciprocals.
//! Domain: 2^-125 <= abs(x) <= 2^125, RNE, gradual underflow.
#![allow(dead_code)] // Different study binaries instantiate different variants.

#[inline(always)]
fn hardware_fma<T: archmage::SimdToken>() -> bool {
    let id = core::any::TypeId::of::<T>();
    #[cfg(target_arch = "x86_64")]
    {
        if id == core::any::TypeId::of::<archmage::X64V3Token>() {
            return true;
        }
        #[cfg(feature = "avx512")]
        if id == core::any::TypeId::of::<archmage::X64V4Token>()
            || id == core::any::TypeId::of::<archmage::X64V4xToken>()
        {
            return true;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if id == core::any::TypeId::of::<archmage::NeonToken>() {
        return true;
    }
    let _ = id;
    false
}

/// The products of the fixed-seed Newton graph stay near one. Their exact
/// binary32 products use at most 48 bits; subtracting from two fits in binary64.
/// This is deliberately NOT a general-purpose FMA replacement.
#[inline(always)]
fn correction<const TWO: bool, T: archmage::SimdToken, const N: usize>(
    t: T,
    x: [f32; N],
    y: [f32; N],
) -> [f32; N] {
    #[cfg(target_arch = "wasm32")]
    if let Some(proof) = (&t as &dyn core::any::Any).downcast_ref::<archmage::Wasm128Token>() {
        return wasm::lanes::<TWO, N>(*proof, x, y);
    }
    let _ = t;
    let c = if TWO { 2.0_f64 } else { 1.0_f64 };
    core::array::from_fn(|i| (c - f64::from(x[i]) * f64::from(y[i])) as f32)
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use core::arch::wasm32 as w;
    /// Explicit demotion avoids scalar extract/demote/insert sequences observed
    /// in the automatic vectorizer. Arithmetic is unchanged and exact in f64.
    #[archmage::rite]
    pub(super) fn lanes<const TWO: bool, const N: usize>(
        t: archmage::Wasm128Token,
        x: [f32; N],
        y: [f32; N],
    ) -> [f32; N] {
        assert_eq!(N % 4, 0);
        let mut out = [0.; N];
        for ((x, y), out) in x
            .as_chunks::<4>()
            .0
            .iter()
            .zip(y.as_chunks::<4>().0)
            .zip(out.as_chunks_mut::<4>().0)
        {
            let a = w::f32x4(x[0], x[1], x[2], x[3]);
            let b = w::f32x4(y[0], y[1], y[2], y[3]);
            let c = w::f64x2_splat(if TWO { 2.0 } else { 1.0 });
            let lo = w::f64x2_sub(
                c,
                w::f64x2_mul(w::f64x2_promote_low_f32x4(a), w::f64x2_promote_low_f32x4(b)),
            );
            let ah = w::i32x4_shuffle::<2, 3, 2, 3>(a, a);
            let bh = w::i32x4_shuffle::<2, 3, 2, 3>(b, b);
            let hi = w::f64x2_sub(
                c,
                w::f64x2_mul(
                    w::f64x2_promote_low_f32x4(ah),
                    w::f64x2_promote_low_f32x4(bh),
                ),
            );
            let r = w::i32x4_shuffle::<0, 1, 4, 5>(
                w::f32x4_demote_f64x2_zero(lo),
                w::f32x4_demote_f64x2_zero(hi),
            );
            *out = [
                w::f32x4_extract_lane::<0>(r),
                w::f32x4_extract_lane::<1>(r),
                w::f32x4_extract_lane::<2>(r),
                w::f32x4_extract_lane::<3>(r),
            ];
        }
        out
    }
}

macro_rules! kernels {
    ($bounded:ident,$canonical:ident,$v:ident,$i:ident,$bound:ident,$n:literal) => {
        #[inline(always)]
        pub(crate) fn $bounded<const STEPS: usize, T: magetypes::simd::backends::$bound>(
            t: T,
            x: magetypes::simd::generic::$v<T>,
        ) -> magetypes::simd::generic::$v<T> {
            const {
                assert!(STEPS == 1 || STEPS == 2);
            }
            use magetypes::simd::generic::{$i as I, $v as V};
            let mut y = (I::splat_t(t, 0x7ef127ea) - x.bitcast_to_i32()).bitcast_to_f32();
            for _ in 0..STEPS {
                y = y * V::from_array_t(t, correction::<true, _, _>(t, x.to_array(), y.to_array()));
            }
            y
        }

        /// MODE 0: certify every native proposal; 1: certify native proposals
        /// within 16 binary32 encoding units of a coarse midpoint; 2: division
        /// proposals, certifying only double-rounding midpoints. Same target bits.
        #[inline(always)]
        pub(crate) fn $canonical<
            const P: u32,
            const MODE: u8,
            T: magetypes::simd::backends::$bound,
        >(
            t: T,
            x: magetypes::simd::generic::$v<T>,
        ) -> magetypes::simd::generic::$v<T> {
            const {
                assert!(P == 8 || P == 14);
                assert!(MODE <= 2);
            }
            use magetypes::simd::generic::{$i as I, $v as V};
            let sp = |x| V::splat_t(t, x);
            let ip = |x| I::splat_t(t, x);
            let proposal = if MODE == 2 { sp(1.0) / x } else { x.recip() };
            let bits = proposal.bitcast_to_i32();
            let abs = bits & ip(i32::MAX);
            let drop = 24 - P;
            let half = 1 << (drop - 1);
            let mask = (1 << drop) - 1;
            // Integer RNE of the positive encoding; restore the original sign.
            let round = (abs + ip(half - 1) + (abs.shr_logical_uniform(drop) & ip(1))) & ip(!mask);
            let result = round | (bits & ip(i32::MIN));
            if MODE != 0 {
                let low = abs & ip(mask);
                let uncertain = if MODE == 2 {
                    low.simd_eq(ip(half))
                } else {
                    (low - ip(half)).abs().simd_le(ip(16))
                };
                if !uncertain.any_true() {
                    return result.bitcast_to_f32();
                }
            }
            // Only normalize in the correction path. This avoids subnormal
            // midpoint deltas at the large-magnitude edge of the input domain.
            let input = x.bitcast_to_i32() & ip(i32::MAX);
            let shift = ((input.shr_logical_const::<23>() & ip(255)) - ip(127)).shl_const::<23>();
            let xn = ((input & ip(0x7fffff)) | ip(0x3f800000)).bitcast_to_f32();
            let qn = (round + shift).bitcast_to_f32();
            let scale = (1_u32 << P) as f32;
            let k = (qn * sp(scale)).to_i32();
            let lo = (k.to_f32() - sp(0.5)) * sp(1.0 / scale);
            let hi = (k.to_f32() + sp(0.5)) * sp(1.0 / scale);
            let (a, b) = if hardware_fma::<T>() {
                (
                    (-xn).mul_add_relaxed(lo, sp(1.0)),
                    (-xn).mul_add_relaxed(hi, sp(1.0)),
                )
            } else {
                // A 24-bit significand times a <=16-bit midpoint fits exactly
                // in f64. Subtraction from one is exact too; no software FMA.
                let (x, lo, hi) = (xn.to_array(), lo.to_array(), hi.to_array());
                (
                    V::from_array_t(t, correction::<false, _, _>(t, x, lo)),
                    V::from_array_t(t, correction::<false, _, _>(t, x, hi)),
                )
            };
            let odd = k & ip(1);
            let down = (a.simd_lt(sp(0.0)).bitcast_to_i32() & ip(1))
                | (a.simd_eq(sp(0.0)).bitcast_to_i32() & odd);
            let up = (b.simd_gt(sp(0.0)).bitcast_to_i32() & ip(1))
                | (b.simd_eq(sp(0.0)).bitcast_to_i32() & odd);
            let repaired = ((k + up - down).to_f32() * sp(1.0 / scale)).bitcast_to_i32();
            ((repaired - shift) | (bits & ip(i32::MIN))).bitcast_to_f32()
        }
    };
}
kernels!(bounded4, canonical4, f32x4, i32x4, F32x4Convert, 4);
kernels!(bounded8, canonical8, f32x8, i32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
kernels!(bounded16, canonical16, f32x16, i32x16, F32x16Convert, 16);
