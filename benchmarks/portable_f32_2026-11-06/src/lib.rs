#![forbid(unsafe_code)]
use archmage::prelude::*;
pub const CANDIDATES: &[&str] = &[
    "std",
    "metallic",
    "core_scalar",
    "core_vector",
    "libm",
    "fpmath",
    "micromath",
    "magetypes_midp",
    "magetypes_lowp",
];
macro_rules! approx {
    ($name:ident,$exp:ident,$ln:ident,$log2:ident,$pow:ident) => {
        #[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
        fn $name(token: Token, op: &str, x: &mut [f32], y: &[f32]) {
            let n = x.len();
            for start in (0..n).step_by(8) {
                let len = (n - start).min(8);
                let mut a = [1.0; 8];
                a[..len].copy_from_slice(&x[start..start + len]);
                let v = f32x8::from_array_t(token, a);
                let out = match op {
                    "exp" => v.$exp(),
                    "ln" => v.$ln(),
                    "log2" => v.$log2(),
                    // The public approximation accepts a scalar exponent. Evaluate
                    // each lane's own exponent; do not substitute a fixed exponent.
                    "pow" => {
                        if y[start..start + len]
                            .iter()
                            .all(|b| b.to_bits() == y[start].to_bits())
                        {
                            v.$pow(y[start])
                        } else {
                            let r = core::array::from_fn(|i| {
                                if i < len {
                                    f32x8::splat_t(token, a[i]).$pow(y[start + i]).to_array()[0]
                                } else {
                                    1.0
                                }
                            });
                            f32x8::from_array_t(token, r)
                        }
                    }
                    _ => panic!("unknown op"),
                }
                .to_array();
                x[start..start + len].copy_from_slice(&out[..len]);
            }
        }
    };
}
approx!(mid, exp_midp, ln_midp, log2_midp, pow_midp);
approx!(low, exp_lowp, ln_lowp, log2_lowp, pow_lowp);
pub fn eval(candidate: &str, op: &str, x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    macro_rules! scalar {
        ($exp:expr,$ln:expr,$log:expr,$pow:expr) => {
            match op {
                "exp" => x.iter_mut().for_each(|a| *a = ($exp)(*a)),
                "ln" => x.iter_mut().for_each(|a| *a = ($ln)(*a)),
                "log2" => x.iter_mut().for_each(|a| *a = ($log)(*a)),
                "pow" => x.iter_mut().zip(y).for_each(|(a, b)| *a = ($pow)(*a, *b)),
                _ => panic!("unknown op"),
            }
        };
    }
    match candidate {
        "std" => scalar!(f32::exp, f32::ln, f32::log2, f32::powf),
        "core_scalar" => scalar!(
            zenmath::scalar::expf,
            zenmath::scalar::logf,
            zenmath::scalar::log2f,
            zenmath::scalar::powf
        ),
        "metallic" => scalar!(
            metallic::expf,
            metallic::logf,
            metallic::log2f,
            metallic::powf
        ),
        "libm" => scalar!(libm::expf, libm::logf, libm::log2f, libm::powf),
        "fpmath" => scalar!(fpmath::exp, fpmath::ln, fpmath::log2, fpmath::pow),
        "micromath" => scalar!(
            |a| micromath::F32(a).exp().0,
            |a| micromath::F32(a).ln().0,
            |a| micromath::F32(a).log2().0,
            |a, b| micromath::F32(a).powf(micromath::F32(b)).0
        ),
        "core_vector" => match op {
            "exp" => magemath::portable::slice::exp_f32(x),
            "ln" => magemath::portable::slice::ln_f32(x),
            "log2" => magemath::portable::slice::log2_f32(x),
            "pow" => magemath::portable::slice::pow_f32(x, y),
            _ => panic!("unknown op"),
        },
        "magetypes_midp" => incant!(mid(op, x, y), [v3, neon, wasm128, scalar]),
        "magetypes_lowp" => incant!(low(op, x, y), [v3, neon, wasm128, scalar]),
        _ => panic!("unknown candidate"),
    }
}
