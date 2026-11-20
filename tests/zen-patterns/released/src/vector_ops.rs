//! Vector operations zen* kernels call most, exercised through one
//! `#[magetypes]` family so every tier runs them: flat transcendentals
//! (zensim `color.rs`, linear-srgb, zenanalyze, zenpixels), multiply-add
//! spellings, reductions, comparisons/masks, integer pairwise ops (zenav1-svt,
//! aom-rs) and f16 slice conversion (zenpixels-convert).

use archmage::{ScalarToken, incant};

/// Flat transcendentals in the forms zen* uses, one output plane each.
#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn transcendentals_tier(token: Token, x: &[f32; 8], out: &mut [[f32; 8]; 12]) {
    let v = f32x8::load_t(token, x);
    v.cbrt_midp().store(&mut out[0]); // 139 sites
    v.pow_midp(2.4).store(&mut out[1]); // 69 sites
    v.cbrt_lowp().store(&mut out[2]); // 23 sites
    v.log2_midp().store(&mut out[3]); // 22 sites
    v.pow_lowp_unchecked(1.0 / 2.4).store(&mut out[4]); // 14 sites
    v.exp_midp().store(&mut out[5]);
    (v - f32x8::splat_t(token, 1.0))
        .exp2_midp()
        .store(&mut out[6]);
    v.pow_midp_unchecked(0.5).store(&mut out[7]);
    v.log2_midp_precise().store(&mut out[8]);
    v.pow_midp_precise(2.2).store(&mut out[9]);
    v.ln_midp().store(&mut out[10]);
    v.log10_midp().store(&mut out[11]);
}

pub fn transcendentals(x: &[f32; 8], out: &mut [[f32; 8]; 12]) {
    incant!(
        transcendentals_tier(x, out),
        [v4(cfg(avx512)), v3, neon, wasm128, scalar]
    )
}

/// Multiply-add spellings, sqrt/reciprocals, reductions and masks.
#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn float_ops_tier(token: Token, a: &[f32; 8], b: &[f32; 8], out: &mut [[f32; 8]; 6]) -> [f32; 4] {
    let (x, y) = (f32x8::load_t(token, a), f32x8::load_t(token, b));
    let one = f32x8::splat_t(token, 1.0);
    #[allow(deprecated)] // 3,367 zen sites still call `mul_add`.
    x.mul_add(y, one).store(&mut out[0]);
    x.mul_add_portable(y, one).store(&mut out[1]);
    x.abs().sqrt().store(&mut out[2]);
    x.clamp(f32x8::zero_t(token), one).store(&mut out[3]);
    let m = x.simd_lt(y);
    f32x8::blend(m, x, y).store(&mut out[4]);
    x.max(y).min(one).round().store(&mut out[5]);
    // Integer masks carry `bitmask`/`any_true` (zenjpeg `encode/blocks.rs:690`).
    let mi = m.bitcast_to_i32();
    [
        x.reduce_add(),
        x.reduce_max(),
        mi.bitmask() as f32,
        if mi.any_true() { 1.0 } else { 0.0 },
    ]
}

pub fn float_ops(a: &[f32; 8], b: &[f32; 8], out: &mut [[f32; 8]; 6]) -> [f32; 4] {
    incant!(float_ops_tier(a, b, out), [v3, neon, wasm128, scalar])
}

/// Integer pairwise products, SAD, widening and saturating narrowing.
#[archmage::magetypes(define(i16x16, u8x32), v3, neon, wasm128, scalar)]
fn int_ops_tier(
    token: Token,
    a: &[i16; 16],
    b: &[i16; 16],
    p: &[u8; 32],
    q: &[u8; 32],
) -> (i32, u32, [i16; 16]) {
    let (x, y) = (i16x16::load_t(token, a), i16x16::load_t(token, b));
    let dot = x.madd_adjacent(y).reduce_add(); // 31 sites
    let sad = u8x32::load_t(token, p).sum_abs_diff(u8x32::load_t(token, q));
    let sum = x.saturating_add(y).to_array();
    (dot, sad, sum)
}

pub fn int_ops(a: &[i16; 16], b: &[i16; 16], p: &[u8; 32], q: &[u8; 32]) -> (i32, u32, [i16; 16]) {
    incant!(int_ops_tier(a, b, p, q), [v3, neon, wasm128, scalar])
}

/// f16 slice conversion through the token trait (zenpixels-convert
/// `f16_scalar.rs:233`).
pub fn f16_roundtrip(input: &[f32]) -> Vec<f32> {
    use magetypes::simd::generic::F16Convert;
    let mut half = vec![0u16; input.len()];
    let mut back = vec![0f32; input.len()];
    ScalarToken.f32_to_f16_slice(input, &mut half);
    ScalarToken.f16_to_f32_slice(&half, &mut back);
    back
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::close;
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    #[test]
    fn transcendentals_stay_within_documented_bounds_on_every_tier() {
        let x = [0.001, 0.05, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0];
        let mut scalar = [[0.0; 8]; 12];
        transcendentals_tier_scalar(ScalarToken, &x, &mut scalar);
        let refs: [fn(f32) -> f32; 12] = [
            f32::cbrt,
            |v| v.powf(2.4),
            f32::cbrt,
            f32::log2,
            |v| v.powf(1.0 / 2.4),
            f32::exp,
            |v| (v - 1.0).exp2(),
            |v| v.powf(0.5),
            f32::log2,
            |v| v.powf(2.2),
            f32::ln,
            f32::log10,
        ];
        // LOWP cbrt and pow carry percent-level budgets; MIDP is near 1e-5.
        let tol = [
            2e-5, 2e-4, 2e-2, 2e-5, 2e-2, 2e-5, 2e-5, 2e-5, 2e-5, 2e-4, 2e-5, 2e-5,
        ];
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
            let mut out = [[0.0; 8]; 12];
            transcendentals(&x, &mut out);
            for (k, plane) in out.iter().enumerate() {
                for (i, &got) in plane.iter().enumerate() {
                    let want = refs[k](x[i]);
                    // log2/ln/log10 of 1.0 must be exactly zero.
                    let tol = if want == 0.0 { 0.0 } else { tol[k] };
                    assert!(
                        close(got, want, tol),
                        "fn {k} x={} got {got} want {want}",
                        x[i]
                    );
                }
            }
        });
    }

    #[test]
    fn float_and_int_ops_match_scalar() {
        let a = [-2.0, -0.5, 0.0, 0.25, 0.5, 1.5, 3.0, -4.0];
        let b = [1.0, -1.0, 0.5, 0.25, 2.0, 1.0, -3.0, 4.0];
        let (mut want, mut got) = ([[0.0; 8]; 6], [[0.0; 8]; 6]);
        let want_r = float_ops_tier_scalar(ScalarToken, &a, &b, &mut want);
        let ia: [i16; 16] = core::array::from_fn(|i| i as i16 * 300 - 2000);
        let ib: [i16; 16] = core::array::from_fn(|i| 1000 - i as i16 * 170);
        let p: [u8; 32] = core::array::from_fn(|i| (i * 7) as u8);
        let q: [u8; 32] = core::array::from_fn(|i| 255 - (i * 3) as u8);
        let want_i = int_ops_tier_scalar(ScalarToken, &ia, &ib, &p, &q);
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
            let r = float_ops(&a, &b, &mut got);
            // Rows 0 (relaxed multiply-add) may fuse; every other row is exact.
            for (row, (g, w)) in got.iter().zip(&want).enumerate() {
                for (gv, wv) in g.iter().zip(w) {
                    assert!(
                        close(*gv, *wv, if row == 0 { 1e-6 } else { 0.0 }),
                        "row {row}"
                    );
                }
            }
            assert_eq!(r, want_r);
            assert_eq!(int_ops(&ia, &ib, &p, &q), want_i);
        });
    }

    #[test]
    fn f16_slices_round_trip_representable_values() {
        let v = [0.0, 1.0, -2.5, 65504.0, 0.000061035156];
        assert_eq!(f16_roundtrip(&v), v);
    }
}
