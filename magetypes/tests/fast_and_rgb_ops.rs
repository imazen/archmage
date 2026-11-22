//! Integration tests for 3-tier float `min` / `minimum` / `min_fast` (and
//! `max` / `maximum` / `max_fast`), matching scalar equivalents, `to_i32_fast`,
//! and 3-channel RGB block helpers (`deinterleave_3ch`, `interleave_3ch`,
//! `load_rgb_u8_t`, `load_rgb_u16_t`, `store_rgb_u8`).

#![forbid(unsafe_code)]

use archmage::{ScalarToken, SimdToken};
use magetypes::scalar::{
    self, F32ScalarBackend, F64ScalarBackend, maximum_f32, maximum_f64, minimum_f32, minimum_f64,
};
use magetypes::simd::backends::{
    F32x4Backend, F32x4Convert, F32x8Backend, F32x8Convert, F64x2Backend, F64x4Backend,
};
#[cfg(feature = "w512")]
use magetypes::simd::backends::{F32x16Backend, F32x16Convert, F64x8Backend};
use magetypes::simd::generic::{f32x4, f32x8, f64x2, f64x4};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{f32x16, f64x8};
use magetypes::simd::scalar::{f32x1, f64x1};

#[cfg(all(
    feature = "w512",
    any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
#[path = "support/f32x16_block.rs"]
mod f32x16_block;

#[cfg(feature = "w512")]
fn check_w512_rgb<T: F32x16Backend>(token: T, tier: &str) {
    use magetypes::numerics::IntConvert;

    let packed_bits: [u32; 48] = core::array::from_fn(|i| {
        (if i % 2 == 0 { 0x7f80_0000 } else { 0xffc0_0000 }) | (i + 1) as u32
    });
    let packed = packed_bits.map(f32::from_bits);
    let bits = |v: f32x16<T>| v.to_array().map(f32::to_bits);
    let aos = core::array::from_fn(|i| f32x16::load_t(token, &packed.as_chunks::<16>().0[i]));
    let planar_bits = |channel| core::array::from_fn(|lane| packed_bits[lane * 3 + channel]);
    let soa =
        core::array::from_fn(|c| f32x16::from_array_t(token, planar_bits(c).map(f32::from_bits)));
    assert_eq!(
        f32x16::deinterleave_3ch(aos).map(bits),
        core::array::from_fn(planar_bits),
        "{tier} deinterleave RGB bits",
    );
    assert_eq!(
        f32x16::interleave_3ch(soa).map(bits).as_flattened(),
        &packed_bits,
        "{tier} interleave RGB bits",
    );

    let bytes = core::array::from_fn(|i| (i * 37 + 255) as u8);
    let [r, g, b] = {
        let (r, g, b) = f32x16::load_rgb_u8_t(token, &bytes);
        [r, g, b]
    };
    for (channel, v) in [r, g, b].into_iter().enumerate() {
        let expected = core::array::from_fn(|i| f32::from(bytes[i * 3 + channel]));
        assert_eq!(v.to_array(), expected, "{tier} RGB u8 channel {channel}");
    }
    let mut out = [0; 48];
    f32x16::store_rgb_u8(r, g, b, &mut out);
    assert_eq!(out, bytes, "{tier} RGB u8 round trip");

    let words = core::array::from_fn(|i| (i * 2377 + 65535) as u16);
    let (r, g, b) = f32x16::load_rgb_u16_t(token, &words);
    for (channel, v) in [r, g, b].into_iter().enumerate() {
        let expected = core::array::from_fn(|i| f32::from(words[i * 3 + channel]));
        assert_eq!(v.to_array(), expected, "{tier} RGB u16 channel {channel}");
    }

    let rails = [
        f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::NEG_INFINITY,
        f32::INFINITY,
        -f32::MAX,
        f32::MAX,
        -1.5,
        -0.0,
        0.5,
        1.5,
        2.5,
        17.75,
        126.5,
        253.5,
        254.75,
        255.5,
    ];
    let plane = |channel: usize, lane: usize| rails[(lane + channel * 5) % 16];
    let [r, g, b] = core::array::from_fn(|c| {
        f32x16::from_array_t(token, core::array::from_fn(|i| plane(c, i)))
    });
    let packed = |i: usize| plane(i % 3, i / 3);
    let nearest = core::array::from_fn(|i| packed(i).round_ties_even() as u8);
    let trunc = core::array::from_fn(|i| packed(i) as u8);
    f32x16::store_rgb_u8(r, g, b, &mut out);
    assert_eq!(out, nearest, "{tier} RGB default saturation");
    f32x16::store_rgb_u8_with::<{ IntConvert::new().nearest_even().saturate_all().bits() }>(
        r, g, b, &mut out,
    );
    assert_eq!(out, nearest, "{tier} RGB nearest-even saturation");
    f32x16::store_rgb_u8_with::<{ IntConvert::new().trunc().saturate_all().bits() }>(
        r, g, b, &mut out,
    );
    assert_eq!(out, trunc, "{tier} RGB truncating saturation");
}

fn assert_f32_bits_or_nan(got: f32, exp: f32, ctx: &str) {
    if got.is_nan() && exp.is_nan() {
        return;
    }
    assert_eq!(
        got.to_bits(),
        exp.to_bits(),
        "{ctx}: got {got} ({:#010x}) != expected {exp} ({:#010x})",
        got.to_bits(),
        exp.to_bits()
    );
}

fn assert_f64_bits_or_nan(got: f64, exp: f64, ctx: &str) {
    if got.is_nan() && exp.is_nan() {
        return;
    }
    assert_eq!(
        got.to_bits(),
        exp.to_bits(),
        "{ctx}: got {got} ({:#018x}) != expected {exp} ({:#018x})",
        got.to_bits(),
        exp.to_bits()
    );
}

fn f32_edge_pairs() -> [(f32, f32); 16] {
    let qnan = f32::NAN;
    let snan = f32::from_bits(0x7f80_0001);
    let nnan = f32::from_bits(0xffc0_1234);
    [
        (-3.5, -1.0),
        (2.25, -4.5),
        (7.0, snan),
        (100.0, 100.0),
        (qnan, 5.0),
        (5.0, qnan),
        (snan, -7.0),
        (-7.0, nnan),
        (qnan, f32::NEG_INFINITY),
        (f32::INFINITY, qnan),
        (qnan, snan),
        (-0.0, 0.0),
        (0.0, -0.0),
        (-0.0, -0.0),
        (0.0, 0.0),
        (f32::NEG_INFINITY, f32::INFINITY),
    ]
}

fn f64_edge_pairs() -> [(f64, f64); 16] {
    let qnan = f64::NAN;
    let snan = f64::from_bits(0x7ff0_0000_0000_0001);
    let nnan = f64::from_bits(0xfff8_0000_0000_1234);
    [
        (-3.5, -1.0),
        (2.25, -4.5),
        (7.0, snan),
        (100.0, 100.0),
        (qnan, 5.0),
        (5.0, qnan),
        (snan, -7.0),
        (-7.0, nnan),
        (qnan, f64::NEG_INFINITY),
        (f64::INFINITY, qnan),
        (qnan, snan),
        (-0.0, 0.0),
        (0.0, -0.0),
        (-0.0, -0.0),
        (0.0, 0.0),
        (f64::NEG_INFINITY, f64::INFINITY),
    ]
}

fn check_f32_edge_pair_results(
    tier: &str,
    a: f32,
    b: f32,
    v_min: f32,
    v_max: f32,
    v_minimum: f32,
    v_maximum: f32,
) {
    // Tier 1: min / max (matches f32::min / f32::max; signed zero tie unspecified).
    if a == 0.0 && b == 0.0 {
        assert_eq!(v_min, 0.0, "{tier} min signed zero");
        assert_eq!(v_max, 0.0, "{tier} max signed zero");
    } else if a.is_nan() && b.is_nan() {
        assert!(v_min.is_nan(), "{tier} min(NaN, NaN)");
        assert!(v_max.is_nan(), "{tier} max(NaN, NaN)");
    } else if (a.is_nan() && a.to_bits() & 0x0040_0000 == 0)
        || (b.is_nan() && b.to_bits() & 0x0040_0000 == 0)
    {
        // Signalling NaN beside a number: unspecified, as for `f32::min`. Accept
        // the numeric operand (x86, WASM, scalar) or a quiet NaN (NEON).
        let numeric = if a.is_nan() { b } else { a };
        for (name, v) in [("min", v_min), ("max", v_max)] {
            assert!(
                v.to_bits() == numeric.to_bits() || (v.is_nan() && v.to_bits() & 0x0040_0000 != 0),
                "{tier} {name}({a}, {b}) = {v} ({:#x})",
                v.to_bits()
            );
        }
    } else {
        assert_eq!(v_min.to_bits(), a.min(b).to_bits(), "{tier} min({a}, {b})");
        assert_eq!(v_max.to_bits(), a.max(b).to_bits(), "{tier} max({a}, {b})");
    }

    // Tier 2: minimum / maximum (IEEE 754-2019: quiet NaN propagation, -0.0 < +0.0).
    let exp_minimum = minimum_f32(a, b);
    let exp_maximum = maximum_f32(a, b);
    if exp_minimum.is_nan() {
        assert!(v_minimum.is_nan(), "{tier} minimum({a}, {b})");
        assert_ne!(
            v_minimum.to_bits() & 0x0040_0000,
            0,
            "{tier} minimum must produce quiet NaN"
        );
    } else {
        assert_eq!(
            v_minimum.to_bits(),
            exp_minimum.to_bits(),
            "{tier} minimum({a}, {b})"
        );
    }
    if exp_maximum.is_nan() {
        assert!(v_maximum.is_nan(), "{tier} maximum({a}, {b})");
        assert_ne!(
            v_maximum.to_bits() & 0x0040_0000,
            0,
            "{tier} maximum must produce quiet NaN"
        );
    } else {
        assert_eq!(
            v_maximum.to_bits(),
            exp_maximum.to_bits(),
            "{tier} maximum({a}, {b})"
        );
    }
}

fn check_f64_edge_pair_results(
    tier: &str,
    a: f64,
    b: f64,
    v_min: f64,
    v_max: f64,
    v_minimum: f64,
    v_maximum: f64,
) {
    if a == 0.0 && b == 0.0 {
        assert_eq!(v_min, 0.0, "{tier} min signed zero");
        assert_eq!(v_max, 0.0, "{tier} max signed zero");
    } else if a.is_nan() && b.is_nan() {
        assert!(v_min.is_nan(), "{tier} min(NaN, NaN)");
        assert!(v_max.is_nan(), "{tier} max(NaN, NaN)");
    } else if (a.is_nan() && a.to_bits() & 0x0008_0000_0000_0000 == 0)
        || (b.is_nan() && b.to_bits() & 0x0008_0000_0000_0000 == 0)
    {
        // Signalling NaN beside a number: unspecified, as for `f64::min`. Accept
        // the numeric operand (x86, WASM, scalar) or a quiet NaN (NEON).
        let numeric = if a.is_nan() { b } else { a };
        for (name, v) in [("min", v_min), ("max", v_max)] {
            assert!(
                v.to_bits() == numeric.to_bits()
                    || (v.is_nan() && v.to_bits() & 0x0008_0000_0000_0000 != 0),
                "{tier} {name}({a}, {b}) = {v} ({:#x})",
                v.to_bits()
            );
        }
    } else {
        assert_eq!(v_min.to_bits(), a.min(b).to_bits(), "{tier} min({a}, {b})");
        assert_eq!(v_max.to_bits(), a.max(b).to_bits(), "{tier} max({a}, {b})");
    }

    let exp_minimum = minimum_f64(a, b);
    let exp_maximum = maximum_f64(a, b);
    if exp_minimum.is_nan() {
        assert!(v_minimum.is_nan(), "{tier} minimum({a}, {b})");
        assert_ne!(
            v_minimum.to_bits() & 0x0008_0000_0000_0000,
            0,
            "{tier} minimum must produce quiet NaN"
        );
    } else {
        assert_eq!(
            v_minimum.to_bits(),
            exp_minimum.to_bits(),
            "{tier} minimum({a}, {b})"
        );
    }
    if exp_maximum.is_nan() {
        assert!(v_maximum.is_nan(), "{tier} maximum({a}, {b})");
        assert_ne!(
            v_maximum.to_bits() & 0x0008_0000_0000_0000,
            0,
            "{tier} maximum must produce quiet NaN"
        );
    } else {
        assert_eq!(
            v_maximum.to_bits(),
            exp_maximum.to_bits(),
            "{tier} maximum({a}, {b})"
        );
    }
}

fn check_narrow_backend<T>(token: T, tier: &str)
where
    T: SimdToken + F32x4Backend + F32x4Convert + F32x8Backend + F32x8Convert,
{
    let pairs = f32_edge_pairs();

    // --- f32x4: 3-tier min/max + scalar matching ---
    for chunk in pairs.as_chunks::<4>().0 {
        let a = [chunk[0].0, chunk[1].0, chunk[2].0, chunk[3].0];
        let b = [chunk[0].1, chunk[1].1, chunk[2].1, chunk[3].1];
        let va = f32x4::from_array_t(token, a);
        let vb = f32x4::from_array_t(token, b);

        let r_min = va.min(vb).to_array();
        let r_max = va.max(vb).to_array();
        let r_minimum = va.minimum(vb).to_array();
        let r_maximum = va.maximum(vb).to_array();
        let r_min_fast = va.min_fast(vb).to_array();
        let r_max_fast = va.max_fast(vb).to_array();

        for i in 0..4 {
            check_f32_edge_pair_results(
                tier,
                a[i],
                b[i],
                r_min[i],
                r_max[i],
                r_minimum[i],
                r_maximum[i],
            );
            assert_f32_bits_or_nan(
                r_min_fast[i],
                scalar::min_fast_f32(token, a[i], b[i]),
                &format!("{tier} f32x4 min_fast_f32"),
            );
            assert_f32_bits_or_nan(
                r_max_fast[i],
                scalar::max_fast_f32(token, a[i], b[i]),
                &format!("{tier} f32x4 max_fast_f32"),
            );
        }
    }

    // --- f32x8: 3-tier min/max + scalar matching ---
    for chunk in pairs.as_chunks::<8>().0 {
        let a: [f32; 8] = core::array::from_fn(|i| chunk[i].0);
        let b: [f32; 8] = core::array::from_fn(|i| chunk[i].1);
        let va = f32x8::from_array_t(token, a);
        let vb = f32x8::from_array_t(token, b);

        let r_min = va.min(vb).to_array();
        let r_max = va.max(vb).to_array();
        let r_minimum = va.minimum(vb).to_array();
        let r_maximum = va.maximum(vb).to_array();
        let r_min_fast = va.min_fast(vb).to_array();
        let r_max_fast = va.max_fast(vb).to_array();

        for i in 0..8 {
            check_f32_edge_pair_results(
                tier,
                a[i],
                b[i],
                r_min[i],
                r_max[i],
                r_minimum[i],
                r_maximum[i],
            );
            assert_f32_bits_or_nan(
                r_min_fast[i],
                scalar::min_fast_f32(token, a[i], b[i]),
                &format!("{tier} f32x8 min_fast_f32"),
            );
            assert_f32_bits_or_nan(
                r_max_fast[i],
                scalar::max_fast_f32(token, a[i], b[i]),
                &format!("{tier} f32x8 max_fast_f32"),
            );
        }
    }

    // --- min_fast / max_fast finite & ±inf clamping ---
    let lo4 = f32x4::zero_t(token);
    let hi4 = f32x4::splat_t(token, 255.0);
    let clamp4_in = f32x4::from_array_t(token, [-0.5, f32::NEG_INFINITY, f32::INFINITY, 42.5]);
    assert_eq!(
        clamp4_in.max_fast(lo4).min_fast(hi4).to_array(),
        [0.0, 0.0, 255.0, 42.5],
        "{tier} f32x4 edge clamp"
    );

    let lo8 = f32x8::zero_t(token);
    let hi8 = f32x8::splat_t(token, 255.0);
    let clamp8_in = f32x8::from_array_t(
        token,
        [
            -1e20,
            1e20,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -0.5,
            0.0,
            128.25,
            255.5,
        ],
    );
    assert_eq!(
        clamp8_in.max_fast(lo8).min_fast(hi8).to_array(),
        [0.0, 255.0, 0.0, 255.0, 0.0, 0.0, 128.25, 255.0],
        "{tier} f32x8 edge clamp"
    );

    // --- to_i32_fast + scalar::to_i32_fast ---
    let fi4_src = [0.0, -1.875, 12345.75, -2_147_483_520.0];
    let fi4 = f32x4::from_array_t(token, fi4_src);
    assert_eq!(
        fi4.to_i32_fast().to_array(),
        [0, -1, 12345, -2_147_483_520],
        "{tier} f32x4 to_i32_fast"
    );
    for (got, &src) in fi4.to_i32_fast().to_array().iter().zip(&fi4_src) {
        assert_eq!(
            *got,
            scalar::to_i32_fast(token, src),
            "{tier} scalar::to_i32_fast (f32x4)"
        );
    }

    let fi8_src = [
        0.0,
        1.99,
        -1.99,
        255.5,
        -255.5,
        2_147_483_520.0,
        -2_147_483_520.0,
        -2_147_483_648.0,
    ];
    let fi8 = f32x8::from_array_t(token, fi8_src);
    assert_eq!(
        fi8.to_i32_fast().to_array(),
        [0, 1, -1, 255, -255, 2_147_483_520, -2_147_483_520, i32::MIN],
        "{tier} f32x8 to_i32_fast"
    );
    for (got, &src) in fi8.to_i32_fast().to_array().iter().zip(&fi8_src) {
        assert_eq!(
            *got,
            scalar::to_i32_fast(token, src),
            "{tier} scalar::to_i32_fast (f32x8)"
        );
    }

    // --- 3-channel deinterleave_3ch / interleave_3ch (numeric + bit-exact) ---
    let rgb4_aos = [
        f32x4::from_array_t(token, [1.0, 2.0, 3.0, 4.0]),
        f32x4::from_array_t(token, [5.0, 6.0, 7.0, 8.0]),
        f32x4::from_array_t(token, [9.0, 10.0, 11.0, 12.0]),
    ];
    let [r4, g4, b4] = f32x4::deinterleave_3ch(rgb4_aos);
    assert_eq!(r4.to_array(), [1.0, 4.0, 7.0, 10.0], "{tier}");
    assert_eq!(g4.to_array(), [2.0, 5.0, 8.0, 11.0], "{tier}");
    assert_eq!(b4.to_array(), [3.0, 6.0, 9.0, 12.0], "{tier}");
    for (got, exp) in f32x4::interleave_3ch([r4, g4, b4]).iter().zip(&rgb4_aos) {
        assert_eq!(got.to_array(), exp.to_array(), "{tier}");
    }

    let rgb8_aos = [
        f32x8::from_array_t(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        f32x8::from_array_t(token, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
        f32x8::from_array_t(token, [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]),
    ];
    let [r8, g8, b8] = f32x8::deinterleave_3ch(rgb8_aos);
    assert_eq!(
        r8.to_array(),
        [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0],
        "{tier}"
    );
    assert_eq!(
        g8.to_array(),
        [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0],
        "{tier}"
    );
    assert_eq!(
        b8.to_array(),
        [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0],
        "{tier}"
    );
    for (got, exp) in f32x8::interleave_3ch([r8, g8, b8]).iter().zip(&rgb8_aos) {
        assert_eq!(got.to_array(), exp.to_array(), "{tier}");
    }

    // Bit-exact preservation across deinterleave_3ch <-> interleave_3ch for
    // -0.0, subnormals, infinities, and distinct quiet NaN payloads.
    let bit_patterns_12: [u32; 12] = [
        0x8000_0000, // -0.0
        0x0000_0001, // smallest positive subnormal
        0x8000_0001, // smallest negative subnormal
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7fc0_0001, // qNaN payload 1
        0x7fc0_0002, // qNaN payload 2
        0x7fc0_1234, // qNaN payload 0x1234
        0xffc0_5678, // negative qNaN payload 0x5678
        0x3f80_0000, // 1.0
        0xbf80_0000, // -1.0
        0x437f_0000, // 255.0
    ];
    let bits_v4 = [
        f32x4::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_12[i])),
        ),
        f32x4::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_12[4 + i])),
        ),
        f32x4::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_12[8 + i])),
        ),
    ];
    let [br4, bg4, bb4] = f32x4::deinterleave_3ch(bits_v4);
    let br4_arr = br4.to_array();
    let bg4_arr = bg4.to_array();
    let bb4_arr = bb4.to_array();
    for p in 0..4 {
        assert_eq!(br4_arr[p].to_bits(), bit_patterns_12[p * 3], "{tier}");
        assert_eq!(bg4_arr[p].to_bits(), bit_patterns_12[p * 3 + 1], "{tier}");
        assert_eq!(bb4_arr[p].to_bits(), bit_patterns_12[p * 3 + 2], "{tier}");
    }
    let bits_re4 = f32x4::interleave_3ch([br4, bg4, bb4]);
    for (vec_idx, vec_item) in bits_re4.iter().enumerate() {
        let arr = vec_item.to_array();
        for (lane, val) in arr.iter().enumerate() {
            assert_eq!(val.to_bits(), bit_patterns_12[vec_idx * 4 + lane], "{tier}");
        }
    }

    let bit_patterns_24: [u32; 24] = core::array::from_fn(|i| {
        if i < 12 {
            bit_patterns_12[i]
        } else {
            (0x3f00_0000 + ((i as u32) << 13)) ^ (((i as u32) & 1) << 31)
        }
    });
    let bits_v8 = [
        f32x8::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_24[i])),
        ),
        f32x8::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_24[8 + i])),
        ),
        f32x8::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_24[16 + i])),
        ),
    ];
    let [br8, bg8, bb8] = f32x8::deinterleave_3ch(bits_v8);
    let br8_arr = br8.to_array();
    let bg8_arr = bg8.to_array();
    let bb8_arr = bb8.to_array();
    for p in 0..8 {
        assert_eq!(br8_arr[p].to_bits(), bit_patterns_24[p * 3], "{tier}");
        assert_eq!(bg8_arr[p].to_bits(), bit_patterns_24[p * 3 + 1], "{tier}");
        assert_eq!(bb8_arr[p].to_bits(), bit_patterns_24[p * 3 + 2], "{tier}");
    }
    let bits_re8 = f32x8::interleave_3ch([br8, bg8, bb8]);
    for (vec_idx, vec_item) in bits_re8.iter().enumerate() {
        let arr = vec_item.to_array();
        for (lane, val) in arr.iter().enumerate() {
            assert_eq!(val.to_bits(), bit_patterns_24[vec_idx * 8 + lane], "{tier}");
        }
    }

    // --- Full 0..=255 load_{4,8}_rgb_u8_t + store_{4,8}_rgb_u8 roundtrip & parity ---
    let mut full_rgb = [0u8; 256 * 3 + 7];
    for p in 0..256 {
        full_rgb[3 + p * 3] = p as u8;
        full_rgb[3 + p * 3 + 1] = (255 - p) as u8;
        full_rgb[3 + p * 3 + 2] = (p as u8).wrapping_mul(73).wrapping_add(19);
    }
    for base in (3..(3 + 256 * 3)).step_by(24) {
        let chunk24: &[u8; 24] = full_rgb[base..base + 24].try_into().unwrap();
        let (r, g, b) = f32x8::load_rgb_u8_t(token, chunk24);
        let exp_r: [f32; 8] = core::array::from_fn(|i| chunk24[i * 3] as f32);
        let exp_g: [f32; 8] = core::array::from_fn(|i| chunk24[i * 3 + 1] as f32);
        let exp_b: [f32; 8] = core::array::from_fn(|i| chunk24[i * 3 + 2] as f32);
        assert_eq!(r.to_array(), exp_r, "{tier}");
        assert_eq!(g.to_array(), exp_g, "{tier}");
        assert_eq!(b.to_array(), exp_b, "{tier}");
        let mut out = [0; 24];
        f32x8::store_rgb_u8(r, g, b, &mut out);
        assert_eq!(&out, chunk24, "{tier}");

        let chunk12_lo: &[u8; 12] = chunk24[..12].try_into().unwrap();
        let chunk12_hi: &[u8; 12] = chunk24[12..].try_into().unwrap();
        let (r4l, g4l, b4l) = f32x4::load_rgb_u8_t(token, chunk12_lo);
        let (r4h, g4h, b4h) = f32x4::load_rgb_u8_t(token, chunk12_hi);
        let mut out = [0; 12];
        f32x4::store_rgb_u8(r4l, g4l, b4l, &mut out);
        assert_eq!(&out, chunk12_lo, "{tier}");
        let mut out = [0; 12];
        f32x4::store_rgb_u8(r4h, g4h, b4h, &mut out);
        assert_eq!(&out, chunk12_hi, "{tier}");
    }

    // Halfway ties, NaN, infinities, negatives, and out-of-range floats match store_rgba_u8.
    let edge_vals: [f32; 24] = [
        0.0,
        -0.0,
        0.5,
        1.5,
        2.5,
        3.5,
        127.5,
        128.5,
        253.5,
        254.5,
        255.0,
        255.5,
        256.0,
        -0.5,
        -1.0,
        -1e30,
        1e30,
        1e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xffc0_0001),
        0.499_999_97,
        254.500_02,
    ];
    let er8 = f32x8::from_array_t(token, core::array::from_fn(|i| edge_vals[i]));
    let eg8 = f32x8::from_array_t(token, core::array::from_fn(|i| edge_vals[8 + i]));
    let eb8 = f32x8::from_array_t(token, core::array::from_fn(|i| edge_vals[16 + i]));
    let mut rgba8_ref = [0; 32];
    f32x8::store_rgba_u8(er8, eg8, eb8, f32x8::zero_t(token), &mut rgba8_ref);
    let rgb8_expected: [u8; 24] = core::array::from_fn(|i| rgba8_ref[(i / 3) * 4 + (i % 3)]);
    let mut packed = [0; 24];
    f32x8::store_rgb_u8(er8, eg8, eb8, &mut packed);
    assert_eq!(packed, rgb8_expected, "{tier}");

    const TRUNC: u32 = magetypes::numerics::IntConvert::new()
        .trunc()
        .saturate_all()
        .bits();
    let alpha = f32x8::splat_t(token, 127.9);
    let mut rgb_buf = [0xa5; 26];
    let mut rgba_buf = [0xa5; 34];
    f32x8::store_rgb_u8_with::<TRUNC>(er8, eg8, eb8, (&mut rgb_buf[1..25]).try_into().unwrap());
    f32x8::store_rgba_u8_with::<TRUNC>(
        er8,
        eg8,
        eb8,
        alpha,
        (&mut rgba_buf[1..33]).try_into().unwrap(),
    );
    for i in 0..8 {
        let expected = [
            edge_vals[i] as u8,
            edge_vals[8 + i] as u8,
            edge_vals[16 + i] as u8,
        ];
        assert_eq!(rgb_buf[1 + 3 * i..4 + 3 * i], expected, "{tier}");
        assert_eq!(rgba_buf[1 + 4 * i..4 + 4 * i], expected, "{tier}");
        assert_eq!(rgba_buf[4 + 4 * i], 127, "{tier}");
    }
    assert_eq!(
        [rgb_buf[0], rgb_buf[25], rgba_buf[0], rgba_buf[33]],
        [0xa5; 4]
    );

    for half in [0usize, 4usize] {
        let er4 = f32x4::from_array_t(token, core::array::from_fn(|i| edge_vals[half + i]));
        let eg4 = f32x4::from_array_t(token, core::array::from_fn(|i| edge_vals[8 + half + i]));
        let eb4 = f32x4::from_array_t(token, core::array::from_fn(|i| edge_vals[16 + half + i]));
        let mut rgba4_ref = [0; 16];
        f32x4::store_rgba_u8(er4, eg4, eb4, f32x4::zero_t(token), &mut rgba4_ref);
        let rgb4_expected: [u8; 12] = core::array::from_fn(|i| rgba4_ref[(i / 3) * 4 + (i % 3)]);
        let mut packed = [0; 12];
        f32x4::store_rgb_u8(er4, eg4, eb4, &mut packed);
        assert_eq!(packed, rgb4_expected, "{tier}");
    }
}

fn check_f64_narrow<T>(token: T, tier: &str)
where
    T: SimdToken + F64x2Backend + F64x4Backend,
{
    let pairs = f64_edge_pairs();

    for chunk in pairs.as_chunks::<2>().0 {
        let a = [chunk[0].0, chunk[1].0];
        let b = [chunk[0].1, chunk[1].1];
        let va = f64x2::from_array_t(token, a);
        let vb = f64x2::from_array_t(token, b);

        let r_min = va.min(vb).to_array();
        let r_max = va.max(vb).to_array();
        let r_minimum = va.minimum(vb).to_array();
        let r_maximum = va.maximum(vb).to_array();
        let r_min_fast = va.min_fast(vb).to_array();
        let r_max_fast = va.max_fast(vb).to_array();

        for i in 0..2 {
            check_f64_edge_pair_results(
                tier,
                a[i],
                b[i],
                r_min[i],
                r_max[i],
                r_minimum[i],
                r_maximum[i],
            );
            assert_f64_bits_or_nan(
                r_min_fast[i],
                scalar::min_fast_f64(token, a[i], b[i]),
                &format!("{tier} f64x2 min_fast_f64"),
            );
            assert_f64_bits_or_nan(
                r_max_fast[i],
                scalar::max_fast_f64(token, a[i], b[i]),
                &format!("{tier} f64x2 max_fast_f64"),
            );
        }
    }

    for chunk in pairs.as_chunks::<4>().0 {
        let a: [f64; 4] = core::array::from_fn(|i| chunk[i].0);
        let b: [f64; 4] = core::array::from_fn(|i| chunk[i].1);
        let va = f64x4::from_array_t(token, a);
        let vb = f64x4::from_array_t(token, b);

        let r_min = va.min(vb).to_array();
        let r_max = va.max(vb).to_array();
        let r_minimum = va.minimum(vb).to_array();
        let r_maximum = va.maximum(vb).to_array();
        let r_min_fast = va.min_fast(vb).to_array();
        let r_max_fast = va.max_fast(vb).to_array();

        for i in 0..4 {
            check_f64_edge_pair_results(
                tier,
                a[i],
                b[i],
                r_min[i],
                r_max[i],
                r_minimum[i],
                r_maximum[i],
            );
            assert_f64_bits_or_nan(
                r_min_fast[i],
                scalar::min_fast_f64(token, a[i], b[i]),
                &format!("{tier} f64x4 min_fast_f64"),
            );
            assert_f64_bits_or_nan(
                r_max_fast[i],
                scalar::max_fast_f64(token, a[i], b[i]),
                &format!("{tier} f64x4 max_fast_f64"),
            );
        }
    }

    let lo4 = f64x4::zero_t(token);
    let hi4 = f64x4::splat_t(token, 255.0);
    let clamp4 = f64x4::from_array_t(token, [-0.5, f64::NEG_INFINITY, f64::INFINITY, 42.5]);
    assert_eq!(
        clamp4.max_fast(lo4).min_fast(hi4).to_array(),
        [0.0, 0.0, 255.0, 42.5],
        "{tier} f64x4 edge clamp"
    );
}

#[cfg(feature = "w512")]
fn check_w512_backend<T>(token: T, tier: &str)
where
    T: SimdToken + F32x16Backend + F32x16Convert + F64x8Backend,
{
    check_w512_rgb(token, tier);
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    f32x16_block::check(token, tier);

    let pairs32 = f32_edge_pairs();
    let a16: [f32; 16] = core::array::from_fn(|i| pairs32[i].0);
    let b16: [f32; 16] = core::array::from_fn(|i| pairs32[i].1);
    let va16 = f32x16::from_array_t(token, a16);
    let vb16 = f32x16::from_array_t(token, b16);

    let r_min = va16.min(vb16).to_array();
    let r_max = va16.max(vb16).to_array();
    let r_minimum = va16.minimum(vb16).to_array();
    let r_maximum = va16.maximum(vb16).to_array();
    let r_min_fast = va16.min_fast(vb16).to_array();
    let r_max_fast = va16.max_fast(vb16).to_array();

    for i in 0..16 {
        check_f32_edge_pair_results(
            tier,
            a16[i],
            b16[i],
            r_min[i],
            r_max[i],
            r_minimum[i],
            r_maximum[i],
        );
        assert_f32_bits_or_nan(
            r_min_fast[i],
            scalar::min_fast_f32(token, a16[i], b16[i]),
            &format!("{tier} f32x16 min_fast_f32"),
        );
        assert_f32_bits_or_nan(
            r_max_fast[i],
            scalar::max_fast_f32(token, a16[i], b16[i]),
            &format!("{tier} f32x16 max_fast_f32"),
        );
    }

    let lo16 = f32x16::zero_t(token);
    let hi16 = f32x16::splat_t(token, 255.0);
    assert_eq!(
        f32x16::from_array_t(token, [f32::NEG_INFINITY; 16])
            .max_fast(lo16)
            .min_fast(hi16)
            .to_array(),
        [0.0; 16],
        "{tier} f32x16 -inf clamp"
    );
    assert_eq!(
        f32x16::from_array_t(token, [f32::INFINITY; 16])
            .max_fast(lo16)
            .min_fast(hi16)
            .to_array(),
        [255.0; 16],
        "{tier} f32x16 +inf clamp"
    );

    let fi16 = f32x16::from_array_t(
        token,
        core::array::from_fn(|i| match i {
            0 => -2_147_483_520.0,
            15 => 2_147_483_520.0,
            _ => ((i as f32) - 8.0) * 100.75,
        }),
    );
    let fi16_got = fi16.to_i32_fast().to_array();
    let fi16_src = fi16.to_array();
    for (i, (&got, &src)) in fi16_got.iter().zip(&fi16_src).enumerate() {
        assert_eq!(got, src as i32, "{tier} to_i32_fast lane {i}");
        assert_eq!(
            got,
            scalar::to_i32_fast(token, src),
            "{tier} scalar::to_i32_fast (f32x16) lane {i}"
        );
    }

    // f64x8 across both halves of f64_edge_pairs
    let pairs64 = f64_edge_pairs();
    for chunk in pairs64.as_chunks::<8>().0 {
        let a8: [f64; 8] = core::array::from_fn(|i| chunk[i].0);
        let b8: [f64; 8] = core::array::from_fn(|i| chunk[i].1);
        let va8 = f64x8::from_array_t(token, a8);
        let vb8 = f64x8::from_array_t(token, b8);

        let r_min = va8.min(vb8).to_array();
        let r_max = va8.max(vb8).to_array();
        let r_minimum = va8.minimum(vb8).to_array();
        let r_maximum = va8.maximum(vb8).to_array();
        let r_min_fast = va8.min_fast(vb8).to_array();
        let r_max_fast = va8.max_fast(vb8).to_array();

        for i in 0..8 {
            check_f64_edge_pair_results(
                tier,
                a8[i],
                b8[i],
                r_min[i],
                r_max[i],
                r_minimum[i],
                r_maximum[i],
            );
            assert_f64_bits_or_nan(
                r_min_fast[i],
                scalar::min_fast_f64(token, a8[i], b8[i]),
                &format!("{tier} f64x8 min_fast_f64"),
            );
            assert_f64_bits_or_nan(
                r_max_fast[i],
                scalar::max_fast_f64(token, a8[i], b8[i]),
                &format!("{tier} f64x8 max_fast_f64"),
            );
        }
    }
}

fn check_scalar1() {
    for &(a, b) in &f32_edge_pairs() {
        let va = f32x1(a);
        let vb = f32x1(b);
        check_f32_edge_pair_results(
            "f32x1",
            a,
            b,
            va.min(vb).0,
            va.max(vb).0,
            va.minimum(vb).0,
            va.maximum(vb).0,
        );
        assert_f32_bits_or_nan(
            va.min_fast(vb).0,
            scalar::min_fast_f32(ScalarToken, a, b),
            "f32x1 min_fast",
        );
        assert_f32_bits_or_nan(
            va.max_fast(vb).0,
            scalar::max_fast_f32(ScalarToken, a, b),
            "f32x1 max_fast",
        );
    }
    assert_eq!(f32x1(-12.75).to_i32_fast().0, -12);

    for &(a, b) in &f64_edge_pairs() {
        let va = f64x1(a);
        let vb = f64x1(b);
        check_f64_edge_pair_results(
            "f64x1",
            a,
            b,
            va.min(vb).0,
            va.max(vb).0,
            va.minimum(vb).0,
            va.maximum(vb).0,
        );
        assert_f64_bits_or_nan(
            va.min_fast(vb).0,
            scalar::min_fast_f64(ScalarToken, a, b),
            "f64x1 min_fast",
        );
        assert_f64_bits_or_nan(
            va.max_fast(vb).0,
            scalar::max_fast_f64(ScalarToken, a, b),
            "f64x1 max_fast",
        );
    }
    for (x, lo, hi) in [
        (-10.0, -5.0, 5.0),
        (0.5, -5.0, 5.0),
        (10.0, -5.0, 5.0),
        (f64::NAN, -5.0, 5.0),
        (f64::NEG_INFINITY, -5.0, 5.0),
        (f64::INFINITY, -5.0, 5.0),
        (1.0, 10.0, -10.0),
        (1.0, f64::NAN, 5.0),
    ] {
        assert_f64_bits_or_nan(
            f64x1(x).clamp(f64x1(lo), f64x1(hi)).0,
            scalar::clamp_f64(ScalarToken, x, lo, hi),
            "f64x1 clamp",
        );
    }
}

/// Exhaustive parity check between `magetypes::scalar` helpers and `f32x4` /
/// `f64x2` lane 0 across edge inputs for any active token.
fn check_scalar_module_parity<T>(token: T, tier: &str)
where
    T: F32x4Backend + F32x4Convert + F64x2Backend + F32ScalarBackend + F64ScalarBackend,
{
    let f32_vals = [
        -0.0f32,
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.5,
        3.5,
        -2.5,
        -3.5,
        4.0,
        9.0,
        1e-20,
        1e20,
        1.0 + f32::EPSILON,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7f80_0001),
        3_000_000_000.0,
        -3_000_000_000.0,
    ];
    for &x in &f32_vals {
        let vx = f32x4::splat_t(token, x);
        assert_f32_bits_or_nan(
            vx.round().to_array()[0],
            scalar::round_f32(x),
            &format!("{tier} round_f32({x})"),
        );
        assert_f32_bits_or_nan(
            vx.floor().to_array()[0],
            scalar::floor_f32(x),
            &format!("{tier} floor_f32({x})"),
        );
        assert_f32_bits_or_nan(
            vx.ceil().to_array()[0],
            scalar::ceil_f32(x),
            &format!("{tier} ceil_f32({x})"),
        );
        assert_f32_bits_or_nan(
            vx.sqrt().to_array()[0],
            scalar::sqrt_f32(x),
            &format!("{tier} sqrt_f32({x})"),
        );
        assert_eq!(
            vx.to_i32().to_array()[0],
            scalar::to_i32(token, x),
            "{tier} to_i32({x})"
        );
        if x.is_finite() && (-2_147_483_520.0..=2_147_483_520.0).contains(&x) {
            assert_eq!(
                vx.to_i32_fast().to_array()[0],
                scalar::to_i32_fast(token, x),
                "{tier} to_i32_fast({x})"
            );
        }
        for (lo, hi) in [
            (-5.0f32, 5.0f32),
            (0.0, 255.0),
            (10.0, -10.0),
            (f32::NAN, 5.0),
            (-5.0, f32::NAN),
        ] {
            let got = vx
                .clamp(f32x4::splat_t(token, lo), f32x4::splat_t(token, hi))
                .to_array()[0];
            assert_f32_bits_or_nan(
                got,
                scalar::clamp_f32(token, x, lo, hi),
                &format!("{tier} clamp_f32({x}, {lo}, {hi})"),
            );
            #[cfg(not(all(target_arch = "wasm32", target_feature = "relaxed-simd")))]
            if x.is_nan() && x.to_bits() & 0x0040_0000 != 0 && !lo.is_nan() && !hi.is_nan() {
                assert_eq!(
                    got.to_bits(),
                    lo.min(hi).to_bits(),
                    "{tier} quiet NaN clamp({x}, {lo}, {hi}) must map to lo (clamped by hi)"
                );
            }
        }
    }
    let f32_triples = [
        (1.0 + f32::EPSILON, 1.0 - f32::EPSILON, -1.0f32),
        (1.0 + 1.0 / 4096.0, 1.0 + 1.0 / 4096.0, 1.0 + 1.0 / 2048.0),
        (-3.25, 4.5, 0.125),
        (1e10, 1e10, -1e20),
        (0.0, 5.0, -0.0),
    ];
    for &(a, b, c) in &f32_triples {
        let va = f32x4::splat_t(token, a);
        let vb = f32x4::splat_t(token, b);
        let vc = f32x4::splat_t(token, c);
        assert_f32_bits_or_nan(
            va.mul_add_fused(vb, vc).to_array()[0],
            scalar::mul_add_fused_f32(a, b, c),
            &format!("{tier} mul_add_fused_f32"),
        );
        assert_f32_bits_or_nan(
            va.mul_sub_fused(vb, vc).to_array()[0],
            scalar::mul_sub_fused_f32(a, b, c),
            &format!("{tier} mul_sub_fused_f32"),
        );
        assert_f32_bits_or_nan(
            va.mul_add_relaxed(vb, vc).to_array()[0],
            scalar::mul_add_relaxed_f32(token, a, b, c),
            &format!("{tier} mul_add_relaxed_f32"),
        );
        assert_f32_bits_or_nan(
            va.mul_sub_relaxed(vb, vc).to_array()[0],
            scalar::mul_sub_relaxed_f32(token, a, b, c),
            &format!("{tier} mul_sub_relaxed_f32"),
        );
    }

    let f64_vals = [
        -0.0f64,
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.5,
        3.5,
        -2.5,
        -3.5,
        4.0,
        9.0,
        1e-40,
        1e40,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    for &x in &f64_vals {
        let vx = f64x2::splat_t(token, x);
        assert_f64_bits_or_nan(
            vx.round().to_array()[0],
            scalar::round_f64(x),
            &format!("{tier} round_f64({x})"),
        );
        assert_f64_bits_or_nan(
            vx.floor().to_array()[0],
            scalar::floor_f64(x),
            &format!("{tier} floor_f64({x})"),
        );
        assert_f64_bits_or_nan(
            vx.ceil().to_array()[0],
            scalar::ceil_f64(x),
            &format!("{tier} ceil_f64({x})"),
        );
        assert_f64_bits_or_nan(
            vx.sqrt().to_array()[0],
            scalar::sqrt_f64(x),
            &format!("{tier} sqrt_f64({x})"),
        );
        for (lo, hi) in [
            (-5.0f64, 5.0f64),
            (0.0, 255.0),
            (10.0, -10.0),
            (f64::NAN, 5.0),
            (-5.0, f64::NAN),
        ] {
            let got = vx
                .clamp(f64x2::splat_t(token, lo), f64x2::splat_t(token, hi))
                .to_array()[0];
            assert_f64_bits_or_nan(
                got,
                scalar::clamp_f64(token, x, lo, hi),
                &format!("{tier} clamp_f64({x}, {lo}, {hi})"),
            );
            #[cfg(not(all(target_arch = "wasm32", target_feature = "relaxed-simd")))]
            if x.is_nan() && !lo.is_nan() && !hi.is_nan() {
                assert_eq!(
                    got.to_bits(),
                    lo.min(hi).to_bits(),
                    "{tier} quiet NaN f64 clamp({x}, {lo}, {hi}) must map to lo"
                );
            }
        }
    }
    let e64 = 1.0 / 134_217_728.0f64;
    let f64_triples = [
        (1.0 + e64, 1.0 - e64, -1.0f64),
        (1.0 + e64, 1.0 + e64, 1.0 + 2.0 * e64),
        (-3.25, 4.5, 0.125),
        (1e20, 1e20, -1e40),
    ];
    for &(a, b, c) in &f64_triples {
        let va = f64x2::splat_t(token, a);
        let vb = f64x2::splat_t(token, b);
        let vc = f64x2::splat_t(token, c);
        assert_f64_bits_or_nan(
            va.mul_add_fused(vb, vc).to_array()[0],
            scalar::mul_add_fused_f64(a, b, c),
            &format!("{tier} mul_add_fused_f64"),
        );
        assert_f64_bits_or_nan(
            va.mul_sub_fused(vb, vc).to_array()[0],
            scalar::mul_sub_fused_f64(a, b, c),
            &format!("{tier} mul_sub_fused_f64"),
        );
        assert_f64_bits_or_nan(
            va.mul_add_relaxed(vb, vc).to_array()[0],
            scalar::mul_add_relaxed_f64(token, a, b, c),
            &format!("{tier} mul_add_relaxed_f64"),
        );
        assert_f64_bits_or_nan(
            va.mul_sub_relaxed(vb, vc).to_array()[0],
            scalar::mul_sub_relaxed_f64(token, a, b, c),
            &format!("{tier} mul_sub_relaxed_f64"),
        );
    }
}

#[test]
fn fast_and_rgb_ops_all_backends() {
    check_scalar1();
    check_scalar_module_parity(ScalarToken, "scalar");
    check_narrow_backend(ScalarToken, "scalar");
    check_f64_narrow(ScalarToken, "scalar");
    #[cfg(feature = "w512")]
    check_w512_backend(ScalarToken, "scalar");

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = archmage::X64V3Token::summon() {
            check_scalar_module_parity(t, "x86-v3");
            check_narrow_backend(t, "x86-v3");
            check_f64_narrow(t, "x86-v3");
            #[cfg(feature = "w512")]
            check_w512_backend(t, "x86-v3");
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = archmage::X64V4Token::summon() {
                check_scalar_module_parity(t, "x86-v4");
                check_narrow_backend(t, "x86-v4");
                check_f64_narrow(t, "x86-v4");
                #[cfg(feature = "w512")]
                check_w512_backend(t, "x86-v4");
            }
            if let Some(t) = archmage::X64V4xToken::summon() {
                check_scalar_module_parity(t, "x86-v4x");
                check_narrow_backend(t, "x86-v4x");
                check_f64_narrow(t, "x86-v4x");
                #[cfg(feature = "w512")]
                check_w512_backend(t, "x86-v4x");
            }
            if let Some(t) = archmage::Avx512Fp16Token::summon() {
                check_scalar_module_parity(t, "x86-fp16");
                check_narrow_backend(t, "x86-fp16");
                check_f64_narrow(t, "x86-fp16");
                #[cfg(feature = "w512")]
                check_w512_backend(t.v4(), "x86-fp16-v4");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let Some(t) = archmage::NeonToken::summon() {
            check_scalar_module_parity(t, "neon");
            check_narrow_backend(t, "neon");
            check_f64_narrow(t, "neon");
            #[cfg(feature = "w512")]
            check_w512_backend(t, "neon");
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if let Some(t) = archmage::Wasm128Token::summon() {
            check_scalar_module_parity(t, "wasm128");
            check_narrow_backend(t, "wasm128");
            check_f64_narrow(t, "wasm128");
            #[cfg(feature = "w512")]
            check_w512_backend(t, "wasm128");
        }
    }
}
