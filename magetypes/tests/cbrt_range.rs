//! `cbrt_midp_precise` covers the whole f32 range. `cbrt_midp` and `cbrt_lowp`
//! are fast and documented for magnitudes below `f32::MAX / 3`: their Halley step
//! forms `y³ + 2x`, which overflows above that and returns ±inf. The precise variant
//! used to inherit the overflow (every input from 1.1342859e38 up returned +inf).
#![forbid(unsafe_code)]
use archmage::{ScalarToken, incant, magetypes};

/// Every 4,099th f32 from `lo` up to `hi`, plus their negatives.
fn sweep(lo: f32, hi: f32) -> Vec<f32> {
    let mut xs: Vec<f32> = (lo.to_bits()..=hi.to_bits())
        .step_by(4099)
        .map(f32::from_bits)
        .collect();
    xs.push(hi);
    let negatives: Vec<f32> = xs.iter().map(|x| -x).collect();
    xs.extend(negatives);
    xs
}

fn check(name: &str, got: &[f32], xs: &[f32], tol: f64) {
    for (&y, &x) in got.iter().zip(xs) {
        let want = (x as f64).cbrt();
        assert!(y.is_finite(), "{name}({x:e}) = {y}");
        let rel = ((y as f64 - want) / want).abs();
        assert!(
            rel <= tol,
            "{name}({x:e}) = {y:e}, want {want:e} (rel {rel:.2e})"
        );
    }
}

// Measured over every positive normal f32: midp at most 3.2 ULP (relative error
// under 2.5e-7), lowp at most 3e-5 relative.
const MIDP_TOL: f64 = 4e-7;
const LOWP_TOL: f64 = 4e-5;

#[magetypes(v3, neon, wasm128, scalar)]
fn run(token: Token) {
    use magetypes::simd::generic::f32x4;
    let apply = |xs: &[f32], f: &dyn Fn(f32x4<Token>) -> f32x4<Token>| -> Vec<f32> {
        xs.chunks(4)
            .flat_map(|chunk| {
                let mut lanes = [1.0f32; 4];
                lanes[..chunk.len()].copy_from_slice(chunk);
                f(f32x4::from_array_t(token, lanes)).to_array()[..chunk.len()].to_vec()
            })
            .collect()
    };

    // The precise variant: the top of the range, normals near the bottom, denormals.
    let mut xs = sweep(f32::MAX / 4.0, f32::MAX);
    xs.extend(sweep(f32::MIN_POSITIVE, f32::MIN_POSITIVE * 64.0));
    xs.extend([
        f32::from_bits(1),
        f32::from_bits(0x0040_0000),
        f32::from_bits(0x007f_ffff),
    ]);
    xs.extend([1.0, 27.0, -8.0]);
    check(
        "cbrt_midp_precise",
        &apply(&xs, &|v| v.cbrt_midp_precise()),
        &xs,
        MIDP_TOL,
    );
    let specials = [f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
    let got = apply(&specials, &|v| v.cbrt_midp_precise());
    for (y, x) in got.iter().zip(specials) {
        assert_eq!(y.to_bits(), x.to_bits(), "cbrt_midp_precise({x})");
    }

    // The fast variants hold their accuracy up to the documented limit.
    let xs = sweep(f32::MAX / 16.0, f32::MAX / 3.0);
    check("cbrt_midp", &apply(&xs, &|v| v.cbrt_midp()), &xs, MIDP_TOL);
    check("cbrt_lowp", &apply(&xs, &|v| v.cbrt_lowp()), &xs, LOWP_TOL);
}

#[test]
fn cbrt_midp_precise_covers_the_whole_range() {
    run_scalar(ScalarToken);
    incant!(run(), [v3, neon, wasm128, scalar]);
}
