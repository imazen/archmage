use archmage::prelude::*;

// ---- A1: the README gain kernel ----
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a1_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat_t(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x8::load_t(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

// ---- A2: two f32x8 per iteration ----
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a2_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat_t(token, gain);
    let (chunks, tail) = plane.as_chunks_mut::<16>();
    for chunk in chunks {
        let (halves, _) = chunk.as_chunks_mut::<8>();
        let [a, b] = halves else { unreachable!() };
        let x = f32x8::load_t(token, a) * factor;
        let y = f32x8::load_t(token, b) * factor;
        x.store(a);
        y.store(b);
    }
    for value in tail {
        *value *= gain;
    }
}

// ---- A3: four f32x8 per iteration ----
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a3_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat_t(token, gain);
    let (chunks, tail) = plane.as_chunks_mut::<32>();
    for chunk in chunks {
        let (parts, _) = chunk.as_chunks_mut::<8>();
        let [a, b, c, d] = parts else { unreachable!() };
        let w = f32x8::load_t(token, a) * factor;
        let x = f32x8::load_t(token, b) * factor;
        let y = f32x8::load_t(token, c) * factor;
        let z = f32x8::load_t(token, d) * factor;
        w.store(a);
        x.store(b);
        y.store(c);
        z.store(d);
    }
    for value in tail {
        *value *= gain;
    }
}

// ---- A4: compare and select ----
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a4_impl(token: Token, x: &mut [f32], t: f32, scale: f32, fill: f32) {
    let thr = f32x8::splat_t(token, t);
    let s = f32x8::splat_t(token, scale);
    let f = f32x8::splat_t(token, fill);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, x);
    for chunk in chunks {
        let v = f32x8::load_t(token, chunk);
        f32x8::blend(v.simd_lt(thr), v * s, f).store(chunk);
    }
    for value in tail {
        *value = if *value < t { *value * scale } else { fill };
    }
}

// ---- A5: abs, negation, three-input bitwise select ----
// out = (|x| & m) | (-x & !m), m taken from the bit pattern of `m`.
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a5_impl(token: Token, x: &[f32], m: &[f32], out: &mut [f32]) {
    let n = x.len().min(m.len()).min(out.len());
    let (x, m, out) = (&x[..n], &m[..n], &mut out[..n]);
    let (xc, xt) = f32x8::partition_slice_t(token, x);
    let (mc, mt) = f32x8::partition_slice_t(token, m);
    let (oc, ot) = f32x8::partition_slice_mut_t(token, out);
    for ((xa, ma), oa) in xc.iter().zip(mc).zip(oc) {
        let v = f32x8::load_t(token, xa);
        let mask = f32x8::load_t(token, ma);
        ((v.abs() & mask) | ((-v) & mask.not())).store(oa);
    }
    for ((xv, mv), ov) in xt.iter().zip(mt).zip(ot) {
        let mb = mv.to_bits();
        *ov = f32::from_bits((xv.abs().to_bits() & mb) | ((-xv).to_bits() & !mb));
    }
}

// ---- A6: degree-8 Horner with mul_add, nine constants ----
pub const A6_COEF: [f32; 9] = [
    1.0, 0.99999, 0.5, 0.166_67, 0.041_667, 0.008_333, 0.001_389, 0.000_198, 0.000_025,
];
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a6_impl(token: Token, x: &mut [f32]) {
    let c = A6_COEF.map(|v| f32x8::splat_t(token, v));
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, x);
    for chunk in chunks {
        let v = f32x8::load_t(token, chunk);
        let mut acc = c[8];
        acc = acc.mul_add(v, c[7]);
        acc = acc.mul_add(v, c[6]);
        acc = acc.mul_add(v, c[5]);
        acc = acc.mul_add(v, c[4]);
        acc = acc.mul_add(v, c[3]);
        acc = acc.mul_add(v, c[2]);
        acc = acc.mul_add(v, c[1]);
        acc = acc.mul_add(v, c[0]);
        acc.store(chunk);
    }
    for value in tail {
        let v = *value;
        let mut acc = A6_COEF[8];
        for k in (0..8).rev() {
            acc = acc * v + A6_COEF[k];
        }
        *value = acc;
    }
}

// ---- A7: 20-tap FIR, 20 hoisted coefficient vectors ----
pub const A7_TAPS: usize = 20;
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a7_impl(token: Token, input: &[f32], coef: &[f32; A7_TAPS], out: &mut [f32]) {
    let c = coef.map(|v| f32x8::splat_t(token, v));
    let n = out.len().min(input.len().saturating_sub(A7_TAPS - 1));
    let out = &mut out[..n];
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, out);
    let mut i = 0;
    for chunk in chunks {
        // One bounds check per chunk; the 20 tap windows index a fixed-size array.
        let win: &[f32; A7_TAPS + 7] = input[i..i + A7_TAPS + 7].try_into().unwrap();
        let mut acc = f32x8::zero_t(token);
        for k in 0..A7_TAPS {
            acc = f32x8::from_slice_t(token, &win[k..k + 8]).mul_add(c[k], acc);
        }
        acc.store(chunk);
        i += 8;
    }
    for value in tail {
        let mut acc = 0.0f32;
        for k in 0..A7_TAPS {
            acc = input[i + k].mul_add(coef[k], acc);
        }
        *value = acc;
        i += 1;
    }
}

// ---- A8: rounding, conversions ----
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a8_round_impl(token: Token, x: &mut [f32]) {
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, x);
    for chunk in chunks {
        let v = f32x8::load_t(token, chunk);
        (v.round() + v.floor()).store(chunk);
    }
    for value in tail {
        *value = value.round_ties_even() + value.floor();
    }
}

#[magetypes(define(f32x8), v3, scalar)]
pub fn a8_sat_impl(token: Token, x: &[f32], out: &mut [i32]) {
    let n = x.len().min(out.len());
    let (xc, _) = f32x8::partition_slice_t(token, &x[..n]);
    let (oc, _) = out[..n].as_chunks_mut::<8>();
    for (xa, oa) in xc.iter().zip(oc) {
        f32x8::load_t(token, xa).to_i32_saturating().store(oa);
    }
}

#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a8_u8_impl(token: Token, x: &[f32], out: &mut [u8]) {
    let n = x.len().min(out.len());
    let (xc, _) = f32x8::partition_slice_t(token, &x[..n]);
    let (oc, _) = out[..n].as_chunks_mut::<8>();
    for (xa, oa) in xc.iter().zip(oc) {
        *oa = f32x8::load_t(token, xa).to_u8();
    }
}

// ---- A9: reductions over a slice ----
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a9_sum_impl(token: Token, x: &[f32]) -> f32 {
    let (chunks, tail) = f32x8::partition_slice_t(token, x);
    let mut acc = f32x8::zero_t(token);
    for chunk in chunks {
        acc += f32x8::load_t(token, chunk);
    }
    let mut s = acc.reduce_add();
    for v in tail {
        s += *v;
    }
    s
}

#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a9_max_impl(token: Token, x: &[f32]) -> f32 {
    let (chunks, tail) = f32x8::partition_slice_t(token, x);
    let mut acc = f32x8::splat_t(token, f32::MIN);
    for chunk in chunks {
        acc = acc.max(f32x8::load_t(token, chunk));
    }
    let mut s = acc.reduce_max();
    for v in tail {
        s = s.max(*v);
    }
    s
}

// ---- A10: recip and rsqrt ----
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a10_recip_impl(token: Token, x: &mut [f32]) {
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, x);
    for chunk in chunks {
        f32x8::load_t(token, chunk).recip().store(chunk);
    }
    for v in tail {
        *v = 1.0 / *v;
    }
}

#[magetypes(define(f32x8), v4(cfg(avx512)), v3, scalar)]
pub fn a10_rsqrt_impl(token: Token, x: &mut [f32]) {
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, x);
    for chunk in chunks {
        f32x8::load_t(token, chunk).rsqrt().store(chunk);
    }
    for v in tail {
        *v = 1.0 / v.sqrt();
    }
}

// ---- A11: transcendentals ----
#[magetypes(define(f32x8), v3, scalar)]
pub fn a11_exp2_impl(token: Token, x: &mut [f32]) {
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, x);
    for chunk in chunks {
        f32x8::load_t(token, chunk).exp2_midp().store(chunk);
    }
    // Tail through the same vector op on a padded copy (no libm calls).
    if !tail.is_empty() {
        let mut pad = [1.0f32; 8];
        pad[..tail.len()].copy_from_slice(tail);
        f32x8::load_t(token, &pad).exp2_midp().store(&mut pad);
        tail.copy_from_slice(&pad[..tail.len()]);
    }
}

#[magetypes(define(f32x8), v3, scalar)]
pub fn a11_ln_impl(token: Token, x: &mut [f32]) {
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, x);
    for chunk in chunks {
        f32x8::load_t(token, chunk).ln_midp().store(chunk);
    }
    // Tail through the same vector op on a padded copy (no libm calls).
    if !tail.is_empty() {
        let mut pad = [1.0f32; 8];
        pad[..tail.len()].copy_from_slice(tail);
        f32x8::load_t(token, &pad).ln_midp().store(&mut pad);
        tail.copy_from_slice(&pad[..tail.len()]);
    }
}

// ---- A12: A1 with f32x16 ----
#[magetypes(define(f32x16), v4(cfg(avx512)), v3, scalar)]
pub fn a12_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x16::splat_t(token, gain);
    let (chunks, tail) = f32x16::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x16::load_t(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

// ---- V4 variants of A8-sat / A11 reached through the V3 downcast ----
// `f32x8::<X64V4Token>` has no `to_i32_saturating`, `exp2_midp` or `ln_midp`:
// `X64V4Token: F32x8Convert` is not satisfied (E0599), so the `#[magetypes]`
// kernels above carry no v4 tier. These bodies are the same kernels with
// `token.v3()` supplying the token; they are compiled in the V4 feature context.
#[cfg(feature = "avx512")]
mod via_v3 {
    use archmage::prelude::*;
    use magetypes::simd::generic::f32x8;

    #[inline(never)]
    pub fn a8_sat_via_v3(token: X64V4Token, x: &[f32], out: &mut [i32]) {
        #[arcane]
        fn inner(token: X64V4Token, x: &[f32], out: &mut [i32]) {
            let t = token.v3();
            let n = x.len().min(out.len());
            let (xc, _) = f32x8::partition_slice_t(t, &x[..n]);
            let (oc, _) = out[..n].as_chunks_mut::<8>();
            for (xa, oa) in xc.iter().zip(oc) {
                f32x8::load_t(t, xa).to_i32_saturating().store(oa);
            }
        }
        inner(token, x, out)
    }

    #[inline(never)]
    pub fn a11_exp2_via_v3(token: X64V4Token, x: &mut [f32]) {
        #[arcane]
        fn inner(token: X64V4Token, x: &mut [f32]) {
            let t = token.v3();
            let (chunks, tail) = f32x8::partition_slice_mut_t(t, x);
            for chunk in chunks {
                f32x8::load_t(t, chunk).exp2_midp().store(chunk);
            }
            if !tail.is_empty() {
                let mut pad = [1.0f32; 8];
                pad[..tail.len()].copy_from_slice(tail);
                f32x8::load_t(t, &pad).exp2_midp().store(&mut pad);
                tail.copy_from_slice(&pad[..tail.len()]);
            }
        }
        inner(token, x)
    }

    #[inline(never)]
    pub fn a11_ln_via_v3(token: X64V4Token, x: &mut [f32]) {
        #[arcane]
        fn inner(token: X64V4Token, x: &mut [f32]) {
            let t = token.v3();
            let (chunks, tail) = f32x8::partition_slice_mut_t(t, x);
            for chunk in chunks {
                f32x8::load_t(t, chunk).ln_midp().store(chunk);
            }
            if !tail.is_empty() {
                let mut pad = [1.0f32; 8];
                pad[..tail.len()].copy_from_slice(tail);
                f32x8::load_t(t, &pad).ln_midp().store(&mut pad);
                tail.copy_from_slice(&pad[..tail.len()]);
            }
        }
        inner(token, x)
    }
}
#[cfg(feature = "avx512")]
pub use via_v3::*;
