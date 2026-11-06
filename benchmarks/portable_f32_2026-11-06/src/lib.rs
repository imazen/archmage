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
macro_rules! unary {
    ($name:ident, $method:ident) => {
        #[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
        fn $name(token: Token, x: &mut [f32], _y: &[f32]) {
            for chunk in x.chunks_mut(8) {
                let mut a = [1.0; 8];
                a[..chunk.len()].copy_from_slice(chunk);
                let out = f32x8::from_array_t(token, a).$method().to_array();
                chunk.copy_from_slice(&out[..chunk.len()]);
            }
        }
    };
}
unary!(exp_mid, exp_midp);
unary!(ln_mid, ln_midp);
unary!(log2_mid, log2_midp);
unary!(exp_low, exp_lowp);
unary!(ln_low, ln_lowp);
unary!(log2_low, log2_lowp);
macro_rules! power {
    ($name:ident,$method:ident) => {
        #[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
        fn $name(token: Token, x: &mut [f32], y: &[f32]) {
            if y.iter().all(|b| b.to_bits() == y[0].to_bits()) {
                let exponent = y[0];
                for chunk in x.chunks_mut(8) {
                    let mut a = [1.0; 8];
                    a[..chunk.len()].copy_from_slice(chunk);
                    let out = f32x8::from_array_t(token, a).$method(exponent).to_array();
                    chunk.copy_from_slice(&out[..chunk.len()]);
                }
            } else {
                // API adapter: public pow takes a scalar exponent, so this
                // general-exponent control uses one splat/pow/extract per lane.
                for (a, b) in x.iter_mut().zip(y) {
                    *a = f32x8::splat_t(token, *a).$method(*b).to_array()[0];
                }
            }
        }
    };
}
power!(pow_mid, pow_midp);
power!(pow_low, pow_lowp);
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
        "magetypes_midp" => match op {
            "exp" => incant!(exp_mid(x, y), [v3, neon, wasm128, scalar]),
            "ln" => incant!(ln_mid(x, y), [v3, neon, wasm128, scalar]),
            "log2" => incant!(log2_mid(x, y), [v3, neon, wasm128, scalar]),
            "pow" => incant!(pow_mid(x, y), [v3, neon, wasm128, scalar]),
            _ => panic!("unknown op"),
        },
        "magetypes_lowp" => match op {
            "exp" => incant!(exp_low(x, y), [v3, neon, wasm128, scalar]),
            "ln" => incant!(ln_low(x, y), [v3, neon, wasm128, scalar]),
            "log2" => incant!(log2_low(x, y), [v3, neon, wasm128, scalar]),
            "pow" => incant!(pow_low(x, y), [v3, neon, wasm128, scalar]),
            _ => panic!("unknown op"),
        },
        _ => panic!("unknown candidate"),
    }
}
