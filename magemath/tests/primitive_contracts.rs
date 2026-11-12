//! Semantic references are independent of the configured vector graphs.
#![forbid(unsafe_code)]
use archmage::ScalarToken;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
use archmage::SimdToken;
use magemath::{PrimitiveErrorKind, minmax::*, recip::*};
#[path = "data/reciprocal_vectors.rs"]
mod fixtures;
const NUMBER: u32 = MinMaxOptions::PORTABLE_NUMBER.bits();
const PROP: u32 = MinMaxOptions::PORTABLE_PROPAGATE.bits();
const NORMAL: u32 = MinMaxOptions::NORMAL.bits();
const HIGH: u32 = RecipOptions::HIGHP.bits();
const LOW: u32 = RecipOptions::PORTABLE_LOWP.bits();
const MID: u32 = RecipOptions::PORTABLE_MIDP.bits();
const NLOW: u32 = RecipOptions::BACKEND_LOWP.bits();
const NMID: u32 = RecipOptions::BACKEND_MIDP.bits();

fn same(a: f32, bits: u32) {
    if f32::from_bits(bits).is_nan() {
        assert!(a.is_nan());
    } else {
        assert_eq!(a.to_bits(), bits);
    }
}
fn reference(a: f32, b: f32, prop: bool, max: bool) -> f32 {
    if a.is_nan() || b.is_nan() {
        return if prop || a.is_nan() && b.is_nan() {
            f32::NAN
        } else if a.is_nan() {
            b
        } else {
            a
        };
    }
    if a == 0.0 && b == 0.0 {
        return f32::from_bits(if max {
            a.to_bits() & b.to_bits()
        } else {
            a.to_bits() | b.to_bits()
        });
    }
    if (a.total_cmp(&b).is_gt()) == max {
        a
    } else {
        b
    }
}
const INPUTS: &[u32] = &[
    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x01000000, 0x3f800000, 0xbf800000,
    0x40000000, 0xc0000000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00123,
    0x7f800001, 0xff800001,
];

macro_rules! minmax_width {
    ($name:ident,$v:ident,$bound:ident,$n:literal,$min:ident,$max:ident,$tail:ident) => {
        fn $name<T: magetypes::simd::backends::$bound>(t: T) {
            use magetypes::simd::generic::$v as V;
            for &a in INPUTS {
                for chunk in INPUTS.chunks($n) {
                    let aa = [f32::from_bits(a); $n];
                    let bb = core::array::from_fn(|i| f32::from_bits(chunk[i % chunk.len()]));
                    let x = V::from_array_t(t, aa);
                    let y = V::from_array_t(t, bb);
                    for (actual, prop, max) in [
                        ($min::<NUMBER, _>(t, x, y).unwrap().to_array(), false, false),
                        ($max::<NUMBER, _>(t, x, y).unwrap().to_array(), false, true),
                        ($min::<PROP, _>(t, x, y).unwrap().to_array(), true, false),
                        ($max::<PROP, _>(t, x, y).unwrap().to_array(), true, true),
                    ] {
                        for i in 0..$n {
                            same(actual[i], reference(aa[i], bb[i], prop, max).to_bits());
                        }
                    }
                    same(
                        $tail::<NUMBER, _>(t, aa[0], bb[0]).unwrap(),
                        reference(aa[0], bb[0], false, false).to_bits(),
                    );
                }
            }
            let a = V::from_array_t(
                t,
                core::array::from_fn(|i| if i % 2 == 0 { 0.0 } else { 1.0 }),
            );
            let b = V::splat_t(t, 1.0);
            let e = $min::<NORMAL, _>(t, a, b).unwrap_err();
            assert_eq!(e.kind, PrimitiveErrorKind::InvalidInput);
            assert_eq!(
                e.lane_mask,
                (0..$n).filter(|i| i % 2 == 0).fold(0, |m, i| m | (1 << i))
            );
        }
    };
}
minmax_width!(
    mm4,
    f32x4,
    F32x4Backend,
    4,
    try_min_f32x4_with,
    try_max_f32x4_with,
    try_min_f32x4_lane_with
);
minmax_width!(
    mm8,
    f32x8,
    F32x8Backend,
    8,
    try_min_f32x8_with,
    try_max_f32x8_with,
    try_min_f32x8_lane_with
);
#[cfg(feature = "w512")]
minmax_width!(
    mm16,
    f32x16,
    F32x16Backend,
    16,
    try_min_f32x16_with,
    try_max_f32x16_with,
    try_min_f32x16_lane_with
);

macro_rules! recip_width {
    ($name:ident,$v:ident,$bound:ident,$n:literal,$rc:ident,$rs:ident,$rct:ident,$rst:ident) => {
        fn $name<T: magetypes::simd::backends::$bound>(t: T) {
            use magetypes::simd::generic::$v as V;
            for chunk in fixtures::F32.chunks($n) {
                let cases: [(u32, u32, u32, u32); $n] =
                    core::array::from_fn(|i| chunk[i % chunk.len()]);
                let x = V::from_array_t(t, cases.map(|c| f32::from_bits(c.0)));
                let rc = $rc::<HIGH, _>(t, x).unwrap().to_array();
                let rs = $rs::<HIGH, _>(t, x).unwrap().to_array();
                for i in 0..$n {
                    same(rc[i], cases[i].1);
                    same(rs[i], cases[i].3);
                }
            }
            for e in [
                2, 3, 62, 63, 64, 125, 126, 127, 128, 190, 191, 192, 250, 251, 252,
            ] {
                for m in [
                    0, 1, 0x1fffff, 0x3fffff, 0x400000, 0x600001, 0x7ffffe, 0x7fffff,
                ] {
                    let value = f32::from_bits((e << 23) | m);
                    if value > f32::from_bits(252 << 23) {
                        continue;
                    }
                    let input: [f32; $n] =
                        core::array::from_fn(|i| value * (1.0 + (i as f32) * 0.001));
                    if input.iter().any(|&x| x > f32::from_bits(252 << 23)) {
                        continue;
                    }
                    let x = V::from_array_t(t, input);
                    macro_rules! profile {
                        ($o:ident,$tol:expr) => {{
                            let rc = $rc::<$o, _>(t, x).unwrap().to_array();
                            let rs = $rs::<$o, _>(t, x).unwrap().to_array();
                            for i in 0..$n {
                                if $o == LOW || $o == MID {
                                    let x = input[i];
                                    let mut r =
                                        f32::from_bits(0x7ef127ea_u32.wrapping_sub(x.to_bits()));
                                    let mut y = f32::from_bits(
                                        0x5f3759df_u32.wrapping_sub(x.to_bits() >> 1),
                                    );
                                    for _ in 0..if $o == LOW { 1 } else { 2 } {
                                        r = r * (-x).mul_add(r, 2.0);
                                        y = y * (-(0.5 * x * y)).mul_add(y, 1.5);
                                    }
                                    same(rc[i], r.to_bits());
                                    same(rs[i], y.to_bits());
                                }
                                let xr = input[i] as f64;
                                let reference = 1.0 / xr;
                                assert!(
                                    ((rc[i] as f64 - reference) / reference).abs() <= $tol,
                                    "recip flags={} x={:x}",
                                    $o,
                                    input[i].to_bits()
                                );
                                let reference = 1.0 / xr.sqrt();
                                assert!(
                                    ((rs[i] as f64 - reference) / reference).abs() <= $tol,
                                    "rsqrt flags={} x={:x}",
                                    $o,
                                    input[i].to_bits()
                                );
                                same($rct::<$o, _>(t, input[i]).unwrap(), rc[i].to_bits());
                                same($rst::<$o, _>(t, input[i]).unwrap(), rs[i].to_bits());
                            }
                        }};
                    }
                    profile!(LOW, 6e-3);
                    profile!(MID, 1e-4);
                    profile!(NLOW, 6e-3);
                    profile!(NMID, 1e-4);
                }
            }
            let x = V::from_array_t(
                t,
                core::array::from_fn(|i| if i % 2 == 0 { f32::NAN } else { 1.0 }),
            );
            let e = $rc::<MID, _>(t, x).unwrap_err();
            assert_eq!(
                e.lane_mask,
                (0..$n).filter(|i| i % 2 == 0).fold(0, |m, i| m | (1 << i))
            );
        }
    };
}
recip_width!(
    rc4,
    f32x4,
    F32x4Convert,
    4,
    try_recip_f32x4_with,
    try_rsqrt_f32x4_with,
    try_recip_f32x4_lane_with,
    try_rsqrt_f32x4_lane_with
);
recip_width!(
    rc8,
    f32x8,
    F32x8Convert,
    8,
    try_recip_f32x8_with,
    try_rsqrt_f32x8_with,
    try_recip_f32x8_lane_with,
    try_rsqrt_f32x8_lane_with
);
#[cfg(feature = "w512")]
recip_width!(
    rc16,
    f32x16,
    F32x16Convert,
    16,
    try_recip_f32x16_with,
    try_rsqrt_f32x16_with,
    try_recip_f32x16_lane_with,
    try_rsqrt_f32x16_lane_with
);

macro_rules! all {
    ($t:expr) => {{
        let t = $t;
        mm4(t);
        mm8(t);
        mm64x2(t);
        mm64x4(t);
        rc4(t);
        rc8(t);
        #[cfg(feature = "w512")]
        {
            mm16(t);
            mm64x8(t);
            rc16(t);
        }
    }};
}
#[test]
fn scalar_contracts() {
    all!(ScalarToken);
    assert_eq!(
        try_recip_array_with::<HIGH, 3>([1.; 3]).unwrap_err().kind,
        PrimitiveErrorKind::UnsupportedWidth
    );
    for &(x, rc, _, rs) in fixtures::F32 {
        same(try_recip_with::<HIGH>(f32::from_bits(x)).unwrap(), rc);
        same(try_rsqrt_with::<HIGH>(f32::from_bits(x)).unwrap(), rs);
    }
}
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_contracts() {
    if let Some(t) = archmage::X64V3Token::summon() {
        all!(t);
    }
    #[cfg(feature = "avx512")]
    {
        if let Some(t) = archmage::X64V4Token::summon() {
            all!(t);
        }
        if let Some(t) = archmage::X64V4xToken::summon() {
            all!(t);
        }
    }
}
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_contracts() {
    all!(archmage::NeonToken::summon().expect("native NEON"));
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn wasm_contracts() {
    all!(archmage::Wasm128Token::summon().expect("SIMD128 build"));
}

#[test]
fn f64_minmax_specials() {
    for a in [
        0.0_f64,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(0x7ff0000000000001),
    ] {
        for b in [
            0.0_f64,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let got = try_min_f64_with::<NUMBER>(a, b).unwrap();
            if a.is_nan() && b.is_nan() {
                assert!(got.is_nan());
            } else if a.is_nan() {
                assert_eq!(got.to_bits(), b.to_bits());
            } else if b.is_nan() {
                assert_eq!(got.to_bits(), a.to_bits());
            } else if a == 0.0 && b == 0.0 {
                assert_eq!(got.to_bits(), a.to_bits() | b.to_bits());
            } else {
                assert_eq!(got, a.min(b));
            }
        }
    }
}

macro_rules! f64_width {
    ($name:ident,$v:ident,$bound:ident,$n:literal,$min:ident,$max:ident) => {
        fn $name<T: magetypes::simd::backends::$bound>(t: T) {
            use magetypes::simd::generic::$v as V;
            let cases = [
                0.0,
                -0.0,
                1.0,
                -1.0,
                f64::MIN_POSITIVE,
                f64::from_bits(1),
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                f64::from_bits(0x7ff0000000000001),
            ];
            for &a in &cases {
                for &b in &cases {
                    let x = V::splat_t(t, a);
                    let y = V::splat_t(t, b);
                    for (out, max, prop) in [
                        ($min::<NUMBER, _>(t, x, y).unwrap().to_array(), false, false),
                        ($max::<NUMBER, _>(t, x, y).unwrap().to_array(), true, false),
                        ($min::<PROP, _>(t, x, y).unwrap().to_array(), false, true),
                        ($max::<PROP, _>(t, x, y).unwrap().to_array(), true, true),
                    ] {
                        let expected = if a.is_nan() || b.is_nan() {
                            if prop || a.is_nan() && b.is_nan() {
                                f64::NAN
                            } else if a.is_nan() {
                                b
                            } else {
                                a
                            }
                        } else if a == 0.0 && b == 0.0 {
                            f64::from_bits(if max {
                                a.to_bits() & b.to_bits()
                            } else {
                                a.to_bits() | b.to_bits()
                            })
                        } else if (a.total_cmp(&b).is_gt()) == max {
                            a
                        } else {
                            b
                        };
                        for got in out {
                            if expected.is_nan() {
                                assert!(got.is_nan());
                            } else {
                                assert_eq!(got.to_bits(), expected.to_bits());
                            }
                        }
                    }
                }
            }
        }
    };
}
f64_width!(
    mm64x2,
    f64x2,
    F64x2Backend,
    2,
    try_min_f64x2_with,
    try_max_f64x2_with
);
f64_width!(
    mm64x4,
    f64x4,
    F64x4Backend,
    4,
    try_min_f64x4_with,
    try_max_f64x4_with
);
#[cfg(feature = "w512")]
f64_width!(
    mm64x8,
    f64x8,
    F64x8Backend,
    8,
    try_min_f64x8_with,
    try_max_f64x8_with
);

#[test]
fn rsqrt_highp_is_the_declared_two_round_graph() {
    let witnesses = fixtures::F32
        .iter()
        .filter(|&&(x, _, mathematical, composed)| {
            f32::from_bits(x) > 0.0 && f32::from_bits(x).is_finite() && mathematical != composed
        })
        .count();
    assert!(
        witnesses > 0,
        "independent oracle must distinguish true rsqrt from composition"
    );
}

#[test]
fn checked_domains_and_canonical_options() {
    const FIN: u32 = MinMaxOptions::new()
        .with_domain(MinMaxDomain::Finite)
        .bits();
    let e =
        try_min_array_with::<FIN, 4>([f32::NAN, f32::INFINITY, -0.0, 1.0], [0.0; 4]).unwrap_err();
    assert_eq!(e.lane_mask, 3);
    assert_eq!(
        MinMaxOptions::NORMAL
            .with_nan(MinMaxNaN::Backend)
            .with_zero(MinMaxZero::Backend)
            .bits(),
        NORMAL
    );
    assert_eq!(
        RecipOptions::HIGHP
            .with_portability(RecipPortability::BackendDependent)
            .with_validation(RecipValidation::AssumeValid)
            .bits(),
        HIGH
    );
    assert_eq!(try_recip_with::<MID>(0.0).unwrap_err().lane_mask, 1);
    assert_eq!(try_rsqrt_with::<MID>(-1.0).unwrap_err().lane_mask, 1);
    for &bits in INPUTS {
        let a = f32::from_bits(bits);
        if a.is_normal()
            && a.abs() >= f32::from_bits(2 << 23)
            && a.abs() <= f32::from_bits(252 << 23)
        {
            for actual in [
                try_recip_with::<LOW>(a).unwrap(),
                try_recip_with::<MID>(a).unwrap(),
                try_recip_with::<NLOW>(a).unwrap(),
                try_recip_with::<NMID>(a).unwrap(),
            ] {
                assert_eq!(actual.is_sign_negative(), a.is_sign_negative());
            }
        }
    }
}

#[test]
fn portable_fusion_caller_matches_automatic_scalar() {
    const FLAGS: u32 = RecipOptions::PORTABLE_MIDP
        .with_execution(RecipExecution::Caller)
        .bits();
    for bits in [0x01000000, 0x3f800001, 0x3ff12999, 0x406eb88c, 0x7e000000] {
        let x = f32::from_bits(bits);
        same(
            try_recip_with::<FLAGS>(x).unwrap(),
            try_recip_with::<MID>(x).unwrap().to_bits(),
        );
        same(
            try_rsqrt_with::<FLAGS>(x).unwrap(),
            try_rsqrt_with::<MID>(x).unwrap().to_bits(),
        );
    }
}
