//! `exp2_lowp`, and `exp_lowp` through it, keep the lowp error up to the top of
//! the f32 range. The input used to be clamped to 126.0, so every x in [126, 128)
//! returned 2^126: 50% low at x = 127 and 75% low near 128.
#![forbid(unsafe_code)]
use archmage::{ScalarToken, incant, magetypes};

/// Measured maximum relative error of exp2_lowp/exp_lowp is 0.557%.
const TOL: f64 = 6e-3;

#[magetypes(v3, neon, wasm128, scalar)]
fn run(token: Token) {
    use magetypes::simd::generic::f32x4;
    // Every 97th f32 in [120, 127.99].
    let xs: Vec<f32> = (120f32.to_bits()..=127.99f32.to_bits())
        .step_by(97)
        .map(f32::from_bits)
        .collect();
    for chunk in xs.chunks(4) {
        let mut lanes = [120.0f32; 4];
        lanes[..chunk.len()].copy_from_slice(chunk);
        let got = f32x4::from_array_t(token, lanes).exp2_lowp().to_array();
        for (&y, &x) in got.iter().zip(chunk) {
            let want = (x as f64).exp2();
            assert!(y.is_finite(), "exp2_lowp({x}) = {y}");
            let rel = ((y as f64 - want) / want).abs();
            assert!(
                rel < TOL,
                "exp2_lowp({x}) = {y:e}, want {want:e} (rel {rel:.2e})"
            );
        }
    }
    // exp(88) = 2^126.96, which the old clamp cut to 2^126.
    let e = f32x4::splat_t(token, 88.0).exp_lowp().to_array()[0];
    let want = 88f64.exp();
    assert!(
        ((e as f64 - want) / want).abs() < TOL,
        "exp_lowp(88) = {e:e}, want {want:e}"
    );
}

#[test]
fn exp2_lowp_holds_its_error_up_to_f32_max() {
    run_scalar(ScalarToken);
    incant!(run(), [v3, neon, wasm128, scalar]);
}
