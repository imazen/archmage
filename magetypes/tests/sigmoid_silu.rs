//! `sigmoid_midp` / `silu_midp` accuracy and rail behavior on every backend.
//!
//! Regression coverage for <https://github.com/imazen/archmage/issues/64>:
//! the natural sigmoid spelling `((-v).exp_midp() + one).recip()` NaN'd 97%
//! of a real conv output tensor, because `exp_midp` correctly saturates to
//! `inf` for pre-activations ≲ -88 but the then-Newton-refined `recip()`
//! computed `inf * 0 = NaN` instead of the IEEE `1/inf = 0`. `recip()` is now
//! exact division on every backend and these helpers are built on it, so
//! ±100-range pre-activations — which real networks produce — give clean 0/1
//! lanes. Mid-range accuracy is checked against an f64 scalar reference at
//! the exp2_midp tier tolerance (1e-5 rel, transcendental_accuracy.rs).

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

const REL_TOL: f32 = 2e-5;

fn sigmoid_ref(x: f32) -> f32 {
    (1.0 / (1.0 + (-(x as f64)).exp())) as f32
}

fn silu_ref(x: f32) -> f32 {
    (x as f64 / (1.0 + (-(x as f64)).exp())) as f32
}

/// Pre-activation-shaped inputs: the saturation zones the issue hit (±100),
/// the exp_midp saturation boundary (~±88), and a mid-range sweep.
fn inputs() -> Vec<f32> {
    let mut v = vec![
        -100.0, -95.0, -90.0, -88.73, -88.0, -87.0, -50.0, -20.0, -10.0, -4.0, -1.0, -0.5, -0.1,
        -1e-4, 0.0, 1e-4, 0.1, 0.5, 1.0, 4.0, 10.0, 20.0, 50.0, 87.0, 88.0, 90.0, 100.0,
    ];
    for i in 0..256 {
        v.push((i as f32 - 128.0) / 8.0); // dense sweep of [-16, 16)
    }
    v
}

fn check_backend<T: F32x8Convert>(token: T, tier: &str) {
    let inputs = inputs();
    for chunk in inputs.chunks(8) {
        let mut arr = [0.0f32; 8];
        arr[..chunk.len()].copy_from_slice(chunk);
        let v = f32x8::<T>::from_array(token, arr);

        let sig = v.sigmoid_midp().to_array();
        let sil = v.silu_midp().to_array();
        for (i, &x) in arr.iter().enumerate() {
            let (gs, gl) = (sig[i], sil[i]);
            assert!(
                !gs.is_nan() && !gl.is_nan(),
                "{tier}: sigmoid/silu produced NaN at x = {x} (issue #64 regression)"
            );
            assert!(
                (0.0..=1.0).contains(&gs),
                "{tier}: sigmoid_midp({x}) = {gs} out of [0, 1]"
            );

            let (ws, wl) = (sigmoid_ref(x), silu_ref(x));
            // Three zones: relative check in the normal range; exactly 1 where
            // the f32 reference itself rounds to 1; and "tiny, not NaN" where
            // the true value is subnormal — the midp tier saturates `exp` to
            // inf near x ≈ -88.7, so lanes there land at 0 or a subnormal
            // (both acceptable), never NaN.
            if ws >= f32::MIN_POSITIVE && ws < 1.0 {
                let rel = ((gs - ws) / ws).abs();
                assert!(
                    rel <= REL_TOL,
                    "{tier}: sigmoid_midp({x}) = {gs}, want {ws} (rel {rel:e} > {REL_TOL:e})"
                );
            } else if ws >= 1.0 {
                assert!(
                    gs == 1.0,
                    "{tier}: sigmoid_midp({x}) = {gs}, want exactly 1.0 (saturated high)"
                );
            } else {
                assert!(
                    gs < f32::MIN_POSITIVE,
                    "{tier}: sigmoid_midp({x}) = {gs}, want a subnormal or 0 (true value {ws:e})"
                );
            }
            if wl.abs() >= f32::MIN_POSITIVE * 256.0 {
                let rel = ((gl - wl) / wl).abs();
                assert!(
                    rel <= REL_TOL,
                    "{tier}: silu_midp({x}) = {gl}, want {wl} (rel {rel:e} > {REL_TOL:e})"
                );
            } else {
                assert!(
                    gl.abs() < 1e-35,
                    "{tier}: silu_midp({x}) = {gl}, want ~±0 (true value {wl:e})"
                );
            }
        }
    }

    // The exact rails, spelled out.
    let v = f32x8::<T>::from_array(token, [-100.0, 100.0, 0.0, -88.73, 88.73, -1.0, 1.0, 0.0]);
    let sig = v.sigmoid_midp().to_array();
    assert_eq!(sig[0], 0.0, "{tier}: sigmoid_midp(-100) must be exactly 0");
    assert_eq!(sig[1], 1.0, "{tier}: sigmoid_midp(100) must be exactly 1");
    let sil = v.silu_midp().to_array();
    assert_eq!(sil[0], 0.0, "{tier}: silu_midp(-100) must be exactly ±0");
    assert_eq!(sil[1], 100.0, "{tier}: silu_midp(100) must be exactly x");

    // Monotonicity of sigmoid over the dense sweep (non-strict: saturated and
    // adjacent lanes may tie).
    let sweep: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) / 2.0).collect();
    let mut last = 0.0f32;
    for chunk in sweep.chunks(8) {
        let mut arr = [256.0f32; 8];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f32x8::<T>::from_array(token, arr).sigmoid_midp().to_array();
        for (i, g) in got.iter().enumerate().take(chunk.len()) {
            assert!(
                *g >= last,
                "{tier}: sigmoid_midp not monotonic at x = {} ({g} < {last})",
                arr[i]
            );
            last = *g;
        }
    }
}

#[test]
fn scalar_backend_sigmoid_silu() {
    check_backend(ScalarToken::summon().unwrap(), "scalar");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_backend_sigmoid_silu() {
    if let Some(t) = archmage::X64V3Token::summon() {
        check_backend(t, "x86_v3");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_backend_sigmoid_silu() {
    if let Some(t) = archmage::NeonToken::summon() {
        check_backend(t, "neon");
    }
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_backend_sigmoid_silu() {
    if let Some(t) = archmage::Wasm128Token::summon() {
        check_backend(t, "wasm128");
    }
}
