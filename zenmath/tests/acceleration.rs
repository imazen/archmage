//! Exact instruction, boundary and shared-kernel checks. No tolerances.
#![cfg(feature = "archmage")]
use archmage::prelude::*;
use zenmath::{
    scalar as oracle,
    simd::{array, scalar, slice},
};
#[path = "data/fma_vectors.rs"]
mod vectors;
fn eq32(got: f32, want: f32, label: &str, bits: u32) {
    if want.is_nan() {
        assert!(
            got.is_nan(),
            "{label} input {bits:08x}: got {:08x}, expected NaN",
            got.to_bits()
        );
    } else {
        assert_eq!(got.to_bits(), want.to_bits(), "{label} input {bits:08x}");
    }
}
fn eq64(got: f64, want: f64, label: &str, bits: u64) {
    if want.is_nan() {
        assert!(
            got.is_nan(),
            "{label} input {bits:016x}: got {:016x}, expected NaN",
            got.to_bits()
        );
    } else {
        assert_eq!(got.to_bits(), want.to_bits(), "{label} input {bits:016x}");
    }
}

#[test]
fn sqrtf_instruction_bits() {
    type Op = Box<dyn Fn(f32) -> f32>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::sqrtf)),
        (
            "scalar",
            Box::new(|x| scalar::sqrtf_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V1Token::summon() {
        forms.push(("v1", Box::new(move |x| scalar::sqrtf_v1_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::sqrtf_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push(("wasm128", Box::new(move |x| scalar::sqrtf_wasm128_t(t, x))));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f32> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
    ];
    for center in [0.5_f32, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f32::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f32::from_bits(rng as u32)
        };
        for (label, fun) in &forms {
            eq32(fun(x), x.sqrt(), label, x.to_bits());
        }
    }
}

#[test]
fn floorf_instruction_bits() {
    type Op = Box<dyn Fn(f32) -> f32>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::floorf)),
        (
            "scalar",
            Box::new(|x| scalar::floorf_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V2Token::summon() {
        forms.push(("v2", Box::new(move |x| scalar::floorf_v2_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::floorf_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push(("wasm128", Box::new(move |x| scalar::floorf_wasm128_t(t, x))));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f32> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
    ];
    for center in [0.5_f32, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f32::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f32::from_bits(rng as u32)
        };
        for (label, fun) in &forms {
            eq32(fun(x), x.floor(), label, x.to_bits());
        }
    }
}

#[test]
fn ceilf_instruction_bits() {
    type Op = Box<dyn Fn(f32) -> f32>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::ceilf)),
        (
            "scalar",
            Box::new(|x| scalar::ceilf_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V2Token::summon() {
        forms.push(("v2", Box::new(move |x| scalar::ceilf_v2_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::ceilf_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push(("wasm128", Box::new(move |x| scalar::ceilf_wasm128_t(t, x))));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f32> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
    ];
    for center in [0.5_f32, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f32::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f32::from_bits(rng as u32)
        };
        for (label, fun) in &forms {
            eq32(fun(x), x.ceil(), label, x.to_bits());
        }
    }
}

#[test]
fn truncf_instruction_bits() {
    type Op = Box<dyn Fn(f32) -> f32>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::truncf)),
        (
            "scalar",
            Box::new(|x| scalar::truncf_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V2Token::summon() {
        forms.push(("v2", Box::new(move |x| scalar::truncf_v2_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::truncf_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push(("wasm128", Box::new(move |x| scalar::truncf_wasm128_t(t, x))));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f32> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
    ];
    for center in [0.5_f32, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f32::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f32::from_bits(rng as u32)
        };
        for (label, fun) in &forms {
            eq32(fun(x), x.trunc(), label, x.to_bits());
        }
    }
}

#[test]
fn roundevenf_instruction_bits() {
    type Op = Box<dyn Fn(f32) -> f32>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::roundevenf)),
        (
            "scalar",
            Box::new(|x| scalar::roundevenf_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V2Token::summon() {
        forms.push(("v2", Box::new(move |x| scalar::roundevenf_v2_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::roundevenf_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push((
            "wasm128",
            Box::new(move |x| scalar::roundevenf_wasm128_t(t, x)),
        ));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f32> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
    ];
    for center in [0.5_f32, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f32::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f32::from_bits(rng as u32)
        };
        for (label, fun) in &forms {
            eq32(fun(x), x.round_ties_even(), label, x.to_bits());
        }
    }
}

#[test]
fn roundf_instruction_bits() {
    type Op = Box<dyn Fn(f32) -> f32>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::roundf)),
        (
            "scalar",
            Box::new(|x| scalar::roundf_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V2Token::summon() {
        forms.push(("v2", Box::new(move |x| scalar::roundf_v2_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::roundf_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push(("wasm128", Box::new(move |x| scalar::roundf_wasm128_t(t, x))));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f32> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
    ];
    for center in [0.5_f32, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f32::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f32::from_bits(rng as u32)
        };
        for (label, fun) in &forms {
            eq32(fun(x), x.round(), label, x.to_bits());
        }
    }
}

#[test]
fn fmaf_independent_exact_bits() {
    type Op = Box<dyn Fn(f32, f32, f32) -> f32>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::fmaf)),
        (
            "scalar",
            Box::new(|a, b, c| scalar::fmaf_scalar_t(ScalarToken, a, b, c)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        forms.push(("v3", Box::new(move |a, b, c| scalar::fmaf_v3_t(t, a, b, c))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push((
            "neon",
            Box::new(move |a, b, c| scalar::fmaf_neon_t(t, a, b, c)),
        ));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push((
            "wasm128",
            Box::new(move |a, b, c| scalar::fmaf_wasm128_t(t, a, b, c)),
        ));
    }
    for [a, b, c, want] in vectors::F32 {
        for (label, fun) in &forms {
            eq32(
                fun(f32::from_bits(a), f32::from_bits(b), f32::from_bits(c)),
                f32::from_bits(want),
                label,
                a,
            );
        }
    }
}

#[test]
fn sqrt_instruction_bits() {
    type Op = Box<dyn Fn(f64) -> f64>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::sqrt)),
        (
            "scalar",
            Box::new(|x| scalar::sqrt_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V1Token::summon() {
        forms.push(("v1", Box::new(move |x| scalar::sqrt_v1_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::sqrt_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push(("wasm128", Box::new(move |x| scalar::sqrt_wasm128_t(t, x))));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f64> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
    ];
    for center in [0.5_f64, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f64::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f64::from_bits(rng)
        };
        for (label, fun) in &forms {
            eq64(fun(x), x.sqrt(), label, x.to_bits());
        }
    }
}

#[test]
fn floor_instruction_bits() {
    type Op = Box<dyn Fn(f64) -> f64>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::floor)),
        (
            "scalar",
            Box::new(|x| scalar::floor_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V2Token::summon() {
        forms.push(("v2", Box::new(move |x| scalar::floor_v2_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::floor_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push(("wasm128", Box::new(move |x| scalar::floor_wasm128_t(t, x))));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f64> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
    ];
    for center in [0.5_f64, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f64::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f64::from_bits(rng)
        };
        for (label, fun) in &forms {
            eq64(fun(x), x.floor(), label, x.to_bits());
        }
    }
}

#[test]
fn ceil_instruction_bits() {
    type Op = Box<dyn Fn(f64) -> f64>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::ceil)),
        (
            "scalar",
            Box::new(|x| scalar::ceil_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V2Token::summon() {
        forms.push(("v2", Box::new(move |x| scalar::ceil_v2_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::ceil_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push(("wasm128", Box::new(move |x| scalar::ceil_wasm128_t(t, x))));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f64> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
    ];
    for center in [0.5_f64, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f64::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f64::from_bits(rng)
        };
        for (label, fun) in &forms {
            eq64(fun(x), x.ceil(), label, x.to_bits());
        }
    }
}

#[test]
fn trunc_instruction_bits() {
    type Op = Box<dyn Fn(f64) -> f64>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::trunc)),
        (
            "scalar",
            Box::new(|x| scalar::trunc_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V2Token::summon() {
        forms.push(("v2", Box::new(move |x| scalar::trunc_v2_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::trunc_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push(("wasm128", Box::new(move |x| scalar::trunc_wasm128_t(t, x))));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f64> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
    ];
    for center in [0.5_f64, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f64::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f64::from_bits(rng)
        };
        for (label, fun) in &forms {
            eq64(fun(x), x.trunc(), label, x.to_bits());
        }
    }
}

#[test]
fn roundeven_instruction_bits() {
    type Op = Box<dyn Fn(f64) -> f64>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::roundeven)),
        (
            "scalar",
            Box::new(|x| scalar::roundeven_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V2Token::summon() {
        forms.push(("v2", Box::new(move |x| scalar::roundeven_v2_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::roundeven_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push((
            "wasm128",
            Box::new(move |x| scalar::roundeven_wasm128_t(t, x)),
        ));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f64> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
    ];
    for center in [0.5_f64, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f64::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f64::from_bits(rng)
        };
        for (label, fun) in &forms {
            eq64(fun(x), x.round_ties_even(), label, x.to_bits());
        }
    }
}

#[test]
fn round_instruction_bits() {
    type Op = Box<dyn Fn(f64) -> f64>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::round)),
        (
            "scalar",
            Box::new(|x| scalar::round_scalar_t(ScalarToken, x)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V2Token::summon() {
        forms.push(("v2", Box::new(move |x| scalar::round_v2_t(t, x))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push(("neon", Box::new(move |x| scalar::round_neon_t(t, x))));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push(("wasm128", Box::new(move |x| scalar::round_wasm128_t(t, x))));
    }
    let mut rng = 0x1d573a42u64;
    let mut edges: Vec<f64> = vec![
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
    ];
    for center in [0.5_f64, -0.5, 1.5, -1.5] {
        let b = center.to_bits();
        edges.extend([b - 1, b + 1].map(f64::from_bits));
    }
    for i in 0..20000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = if i < edges.len() {
            edges[i]
        } else {
            f64::from_bits(rng)
        };
        for (label, fun) in &forms {
            eq64(fun(x), x.round(), label, x.to_bits());
        }
    }
}

#[test]
fn fma_independent_exact_bits() {
    type Op = Box<dyn Fn(f64, f64, f64) -> f64>;
    let mut forms: Vec<(&str, Op)> = vec![
        ("selected", Box::new(scalar::fma)),
        (
            "scalar",
            Box::new(|a, b, c| scalar::fma_scalar_t(ScalarToken, a, b, c)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        forms.push(("v3", Box::new(move |a, b, c| scalar::fma_v3_t(t, a, b, c))));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        forms.push((
            "neon",
            Box::new(move |a, b, c| scalar::fma_neon_t(t, a, b, c)),
        ));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        forms.push((
            "wasm128",
            Box::new(move |a, b, c| scalar::fma_wasm128_t(t, a, b, c)),
        ));
    }
    for [a, b, c, want] in vectors::F64 {
        for (label, fun) in &forms {
            eq64(
                fun(f64::from_bits(a), f64::from_bits(b), f64::from_bits(c)),
                f64::from_bits(want),
                label,
                a,
            );
        }
    }
}

#[test]
fn arrays_slices_and_tail_bits() {
    type A = fn([f32; 4]) -> [f32; 4];
    type S = fn(&mut [f32]);
    type O = fn(f32) -> f32;
    let ops: [(&str, A, S, O); 3] = [
        ("exp", array::exp_f32x4, slice::exp_f32, oracle::expf),
        ("ln", array::ln_f32x4, slice::ln_f32, oracle::logf),
        ("log2", array::log2_f32x4, slice::log2_f32, oracle::log2f),
    ];
    let bits = [
        0, 0x80000000, 1, 0x80000001, 0x7fffff, 0x800000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0400000, 0x7f800000, 0xff800000, 0x7fc00001, 0x3f800001, 0x00800001, 0x7f7fffff,
        0x3eaaaaab,
    ];
    for n in 0..=17 {
        let input: Vec<_> = bits[..n].iter().map(|&v| f32::from_bits(v)).collect();
        for (label, a, s, o) in ops {
            let mut data = input.clone();
            s(&mut data);
            for (&x, &got) in input.iter().zip(&data) {
                eq32(got, o(x), label, x.to_bits());
            }
            for chunk in input.chunks(4) {
                let mut pad = [1.; 4];
                pad[..chunk.len()].copy_from_slice(chunk);
                let got = a(pad);
                for i in 0..chunk.len() {
                    eq32(got[i], o(chunk[i]), label, chunk[i].to_bits());
                }
            }
        }
        let exp: Vec<_> = (0..n)
            .map(|i| [-3., 0., 0.5, 1., 2., 2.4, f32::NAN][i % 7])
            .collect();
        let mut data = input.clone();
        slice::pow_f32(&mut data, &exp);
        for i in 0..n {
            eq32(data[i], oracle::powf(input[i], exp[i]), "pow tail", bits[i]);
        }
    }
}
#[test]
fn power_length_failure_precedes_mutation() {
    let mut data = [2., 3., 4.];
    let saved = data;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        slice::pow_f32(&mut data, &[2., 3.])
    }));
    assert!(result.is_err());
    assert_eq!(data, saved);
}

#[test]
fn required_tier_inventory() {
    let mut tiers = vec![
        ("scalar", ScalarToken::summon().is_some()),
        ("v1", X64V1Token::summon().is_some()),
        ("v2", X64V2Token::summon().is_some()),
        ("v3", X64V3Token::summon().is_some()),
        ("neon", NeonToken::summon().is_some()),
        ("wasm128", Wasm128Token::summon().is_some()),
    ];
    #[cfg(feature = "avx512")]
    tiers.push(("v4x", X64V4xToken::summon().is_some()));
    for (name, ok) in &tiers {
        println!("TIER {name} supported={ok}");
    }
    if let Ok(required) = std::env::var("ZENMATH_REQUIRE_TIERS") {
        for name in required.split(',') {
            assert!(
                tiers.iter().any(|(n, ok)| *n == name && *ok),
                "required validation tier missing: {name}"
            );
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn eight_binary64_lane_contract() {
    if let Some(t) = X64V4xToken::summon() {
        let mut state = 0x3457b397u32;
        for _ in 0..8192 {
            let x: [f32; 8] = core::array::from_fn(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                f32::from_bits(state)
            });
            let y: [f32; 8] = core::array::from_fn(|i| [-3., -1., 0., 0.5, 1., 2., 2.4, 17.][i]);
            for (got, op, label) in [
                (
                    array::exp_f32x8_v4x_t(t, x),
                    oracle::expf as fn(f32) -> f32,
                    "wide exp",
                ),
                (array::ln_f32x8_v4x_t(t, x), oracle::logf, "wide ln"),
                (array::log2_f32x8_v4x_t(t, x), oracle::log2f, "wide log2"),
            ] {
                for i in 0..8 {
                    eq32(got[i], op(x[i]), label, x[i].to_bits());
                }
            }
            let got = array::pow_f32x8_v4x_t(t, x, y);
            for i in 0..8 {
                eq32(got[i], oracle::powf(x[i], y[i]), "wide pow", x[i].to_bits());
            }
        }
        for n in 0..=17 {
            let x: Vec<_> = (0..n)
                .map(|i| {
                    f32::from_bits(
                        [
                            0, 0x80000000, 1, 0x7fffff, 0x3f800000, 0xbf800000, 0x40000000,
                            0x7f800000, 0x7fc00001,
                        ][i % 9],
                    )
                })
                .collect();
            let y: Vec<_> = (0..n).map(|i| [0., 2.4, -3., 0.5][i % 4]).collect();
            type S = fn(X64V4xToken, &mut [f32]);
            for (op, fun) in [
                (
                    oracle::expf as fn(f32) -> f32,
                    slice::exp_f32_wide_v4x_t as S,
                ),
                (oracle::logf, slice::ln_f32_wide_v4x_t),
                (oracle::log2f, slice::log2_f32_wide_v4x_t),
            ] {
                let mut a = x.clone();
                fun(t, &mut a);
                for i in 0..n {
                    eq32(a[i], op(x[i]), "wide tail", x[i].to_bits());
                }
            }
            let mut a = x.clone();
            slice::pow_f32_wide_v4x_t(t, &mut a, &y);
            for i in 0..n {
                eq32(
                    a[i],
                    oracle::powf(x[i], y[i]),
                    "wide power tail",
                    x[i].to_bits(),
                );
            }
        }
        let mut x = [2., 3., 4.];
        let saved = x;
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| slice::pow_f32_wide_v4x_t(
                t,
                &mut x,
                &[2.]
            )))
            .is_err()
        );
        assert_eq!(x, saved);
    } else {
        println!(
            "UNAVAILABLE eight-binary64 AVX-512; required_tier_inventory enforces caller expectations"
        );
    }
}
