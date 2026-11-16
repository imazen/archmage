//! `#[attune]` family and dispatcher shapes from zenmetrics, butteraugli and
//! a private metrics crate.

use archmage::{ScalarToken, attune};

// ---------------------------------------------------------------------------
// Most common modern shape (38 sites): a token-first vector family with an
// opt-in AVX-512 tier, dispatched by `attuned!` with the full tier list.
// Mirrors zenmetrics `cvvdp/src/color.rs:27` and its dispatcher at `:132`.
// ---------------------------------------------------------------------------

/// 3x3 matrix applied to planar f32 channels (a color-space conversion).
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn matrix3_planes_inner(
    token: Token,
    m: &[[f32; 3]; 3],
    a: &mut [f32],
    b: &mut [f32],
    c: &mut [f32],
) {
    let n = a.len().min(b.len()).min(c.len());
    let s = |v: f32| f32x8::splat_t(token, v);
    let (m00, m01, m02) = (s(m[0][0]), s(m[0][1]), s(m[0][2]));
    let (m10, m11, m12) = (s(m[1][0]), s(m[1][1]), s(m[1][2]));
    let (m20, m21, m22) = (s(m[2][0]), s(m[2][1]), s(m[2][2]));
    let mut i = 0;
    while i + 8 <= n {
        let x = f32x8::load_t(token, a[i..i + 8].try_into().unwrap());
        let y = f32x8::load_t(token, b[i..i + 8].try_into().unwrap());
        let z = f32x8::load_t(token, c[i..i + 8].try_into().unwrap());
        (m00 * x + m01 * y + m02 * z).store((&mut a[i..i + 8]).try_into().unwrap());
        (m10 * x + m11 * y + m12 * z).store((&mut b[i..i + 8]).try_into().unwrap());
        (m20 * x + m21 * y + m22 * z).store((&mut c[i..i + 8]).try_into().unwrap());
        i += 8;
    }
    for j in i..n {
        let (x, y, z) = (a[j], b[j], c[j]);
        a[j] = m[0][0] * x + m[0][1] * y + m[0][2] * z;
        b[j] = m[1][0] * x + m[1][1] * y + m[1][2] * z;
        c[j] = m[2][0] * x + m[2][1] * y + m[2][2] * z;
    }
}

/// Dispatched entry: explicit tier list, as at every zen `attuned!` site.
pub fn matrix3_planes(m: &[[f32; 3]; 3], a: &mut [f32], b: &mut [f32], c: &mut [f32]) {
    archmage::attuned!(
        matrix3_planes_inner(m, a, b, c),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// ---------------------------------------------------------------------------
// Per-tier width split (zenmetrics `brisque/src/kernels.rs:110-170`): the
// portable family drops scalar/wasm, a same-named f32x16 sibling serves v4x,
// and a hand-written `_scalar_t` function is the guaranteed arm.
// ---------------------------------------------------------------------------

#[attune(_*_t, -_scalar_t, -_wasm128_t, define(f32x8))]
fn scale_add(token: Token, x: &[f32], k: f32, out: &mut [f32]) {
    let (chunks, tail) = f32x8::partition_slice_t(token, x);
    let (out_chunks, out_tail) = f32x8::partition_slice_mut_t(token, out);
    let kv = f32x8::splat_t(token, k);
    for (o, i) in out_chunks.iter_mut().zip(chunks) {
        (f32x8::load_t(token, i) * kv + f32x8::load_t(token, o)).store(o);
    }
    for (o, &v) in out_tail.iter_mut().zip(tail) {
        *o += v * k;
    }
}

/// 16-wide variant for the opt-in AVX-512 `v4x` tier.
#[attune(_v4x_t(cfg(avx512)), define(f32x16))]
fn scale_add(token: Token, x: &[f32], k: f32, out: &mut [f32]) {
    let (chunks, tail) = f32x16::partition_slice_t(token, x);
    let (out_chunks, out_tail) = f32x16::partition_slice_mut_t(token, out);
    let kv = f32x16::splat_t(token, k);
    for (o, i) in out_chunks.iter_mut().zip(chunks) {
        (f32x16::load_t(token, i) * kv + f32x16::load_t(token, o)).store(o);
    }
    for (o, &v) in out_tail.iter_mut().zip(tail) {
        *o += v * k;
    }
}

/// `_scalar` arm for `attuned!`.
fn scale_add_scalar_t(_t: ScalarToken, x: &[f32], k: f32, out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += v * k;
    }
}

pub fn scale_add_dispatch(x: &[f32], k: f32, out: &mut [f32]) {
    archmage::attuned!(
        scale_add(x, k, out),
        [_v4x(cfg(avx512)), _v3, _neon, _scalar]
    );
}

// ---------------------------------------------------------------------------
// Ordinary callable dispatchers (zenmetrics `hdrvdp/src/fft.rs:379`,
// aic4 `kernels/elem.rs:9`): `dispatch` replaces `#[autoversion]`.
// ---------------------------------------------------------------------------

#[attune(dispatch, +v4x(cfg(avx512)))]
pub fn sum_squares(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum()
}

#[attune(dispatch(pub(crate)))]
fn sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x - y;
    }
}

pub fn sub_entry(a: &[f32], b: &[f32], out: &mut [f32]) {
    sub(a, b, out)
}

// ---------------------------------------------------------------------------
// Grouped outputs with a const generic and a fallible return
// (butteraugli `butteraugli/src/exact.rs:82`).
// ---------------------------------------------------------------------------

#[allow(clippy::manual_is_multiple_of)]
#[archmage::attune(make(dispatch, +v4(avx512)), define(f32x8))]
fn blur3_pass<const VERTICAL: bool>(
    token: Token,
    input: &[f32],
    width: usize,
    out: &mut [f32],
    w: [f32; 2],
) -> Result<(), &'static str> {
    if input.len() != out.len() || width == 0 || input.len() % width != 0 {
        return Err("shape");
    }
    let height = input.len() / width;
    let w0 = f32x8::splat_t(token, w[0]);
    let w1 = f32x8::splat_t(token, w[1]);
    let at = |x: isize, y: isize| -> f32 {
        let xc = x.clamp(0, width as isize - 1) as usize;
        let yc = y.clamp(0, height as isize - 1) as usize;
        input[yc * width + xc]
    };
    for y in 0..height {
        let mut x = 0;
        while x + 8 <= width {
            let lane = |d: isize| {
                f32x8::from_array_t(
                    token,
                    core::array::from_fn(|i| {
                        let (xi, yi) = ((x + i) as isize, y as isize);
                        if VERTICAL {
                            at(xi, yi + d)
                        } else {
                            at(xi + d, yi)
                        }
                    }),
                )
            };
            let v = lane(0) * w0 + (lane(-1) + lane(1)) * w1;
            v.store(
                (&mut out[y * width + x..y * width + x + 8])
                    .try_into()
                    .unwrap(),
            );
            x += 8;
        }
        for x in x..width {
            let (xi, yi) = (x as isize, y as isize);
            let (p, n) = if VERTICAL {
                (at(xi, yi - 1), at(xi, yi + 1))
            } else {
                (at(xi - 1, yi), at(xi + 1, yi))
            };
            out[y * width + x] = at(xi, yi) * w[0] + (p + n) * w[1];
        }
    }
    Ok(())
}

pub fn blur3(input: &[f32], width: usize, out: &mut [f32]) -> Result<(), &'static str> {
    let mut tmp = vec![0.0; input.len()];
    // The dispatcher keeps the token-first signature with `ScalarToken`.
    blur3_pass::<false>(ScalarToken, input, width, &mut tmp, [0.5, 0.25])?;
    blur3_pass::<true>(ScalarToken, &tmp, width, out, [0.5, 0.25])
}

// ---------------------------------------------------------------------------
// `_*` direct variants plus `_*_t` proof wrappers in one declaration
// (zenmetrics `gmsd/src/ms_gmsd.rs:81`); the direct `_v3` body is called
// from another v3 body without a boundary.
// ---------------------------------------------------------------------------

#[attune(_*, _*_t, define(f64x4))]
fn dot_f64(token: Token, a: &[f64], b: &[f64]) -> f64 {
    let (ac, at) = f64x4::partition_slice_t(token, a);
    let (bc, bt) = f64x4::partition_slice_t(token, b);
    let mut acc = f64x4::zero_t(token);
    for (x, y) in ac.iter().zip(bc) {
        acc = f64x4::load_t(token, x) * f64x4::load_t(token, y) + acc;
    }
    acc.reduce_add() + at.iter().zip(bt).map(|(x, y)| x * y).sum::<f64>()
}

pub fn dot_f64_dispatch(a: &[f64], b: &[f64]) -> f64 {
    archmage::attuned!(dot_f64(a, b), [_v3, _neon, _wasm128, _scalar])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{close, ramp};
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    #[test]
    fn matrix_family_matches_scalar_on_every_tier() {
        let m = [[0.3, 0.59, 0.11], [-0.1, 0.5, 0.2], [0.7, -0.2, 0.05]];
        let a0 = ramp(37, -1.0, 2.0);
        let b0 = ramp(37, 0.5, -0.5);
        let c0 = ramp(37, 3.0, 1.0);
        let mut want = (a0.clone(), b0.clone(), c0.clone());
        matrix3_planes_inner_scalar_t(ScalarToken, &m, &mut want.0, &mut want.1, &mut want.2);
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
            let (mut a, mut b, mut c) = (a0.clone(), b0.clone(), c0.clone());
            matrix3_planes(&m, &mut a, &mut b, &mut c);
            for (g, w) in a
                .iter()
                .chain(&b)
                .chain(&c)
                .zip(want.0.iter().chain(&want.1).chain(&want.2))
            {
                assert!(close(*g, *w, 1e-6), "{g} vs {w}");
            }
        });
    }

    #[test]
    fn width_split_and_dispatchers_match_scalar() {
        let x = ramp(53, -3.0, 3.0);
        let mut want = vec![1.0; 53];
        scale_add_scalar_t(ScalarToken, &x, 0.75, &mut want);
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
            let mut out = vec![1.0; 53];
            scale_add_dispatch(&x, 0.75, &mut out);
            assert_eq!(out, want);
            let s = sum_squares(&x);
            assert!(close(s, x.iter().map(|v| v * v).sum(), 1e-5));
            let mut d = vec![0.0; 53];
            sub_entry(&x, &want, &mut d);
            assert!(
                d.iter()
                    .zip(x.iter().zip(&want))
                    .all(|(o, (a, b))| *o == a - b)
            );
        });
    }

    #[test]
    fn grouped_dispatch_with_const_generic_and_result() {
        let width = 19;
        let input = ramp(width * 5, 0.0, 1.0);
        let mut out = vec![0.0; input.len()];
        assert_eq!(blur3(&input, width, &mut out[..3]), Err("shape"));
        blur3(&input, width, &mut out).unwrap();
        // A separable [0.25 0.5 0.25] blur of a ramp keeps interior values.
        assert!(close(out[2 * width + 9], input[2 * width + 9], 1e-5));
    }

    #[test]
    fn direct_and_proof_variants_agree() {
        let a: Vec<f64> = (0..23).map(|i| i as f64 * 0.5).collect();
        let b: Vec<f64> = (0..23).map(|i| 1.0 - i as f64 * 0.01).collect();
        let want: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let got = dot_f64_dispatch(&a, &b);
        assert!((got - want).abs() < 1e-9 * want.abs());
    }
}
