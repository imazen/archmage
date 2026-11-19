use archmage::{ScalarToken, SimdToken};
use magetypes::simd::backends::{F32x4Backend, F32x8Backend};
use magetypes::simd::generic::{f32x4, f32x8};

const FLOATS: [f32; 20] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.999,
    1.9,
    32767.0,
    32767.99,
    32768.0,
    49153.0,
    65534.99,
    65535.0,
    65536.0,
    1e20,
    -1e20,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::from_bits(0x7f80_0042),
    f32::from_bits(0xffc0_0123),
];
const BITS: [u32; 16] = [
    0,
    0x8000_0000,
    0x3f80_0000,
    0x7f80_0042,
    0xff80_0000,
    0x8000_0001,
    0x7f80_0000,
    0xffc0_0123,
    0x7f80_0001,
    0x7fc0_0002,
    0x0000_0001,
    0x8000_0000,
    0x7fc0_0003,
    0xff80_0001,
    0x4000_0000,
    0,
];

macro_rules! check_lanes {
    ($token:expr, $vec:ident, $n:literal) => {{
        // Cover every unsigned code in consecutive groups.
        for base in (0..=u16::MAX).step_by($n) {
            let values = core::array::from_fn(|i| base + u16::try_from(i).unwrap());
            let loaded = $vec::from_u16_t($token, &values);
            assert_eq!(loaded.to_array(), values.map(f32::from));
            assert_eq!(loaded.to_u16_trunc_saturating(), values);
        }
        // Rotate rails and halfway values through all lane positions.
        for start in 0..FLOATS.len() {
            let values = core::array::from_fn(|i| FLOATS[(start + i) % FLOATS.len()]);
            assert_eq!(
                $vec::from_array_t($token, values).to_u16_trunc_saturating(),
                values.map(|x| x as u16)
            );
        }

        let a = core::array::from_fn(|i| BITS[i]);
        let b = core::array::from_fn(|i| BITS[(i + 5) % BITS.len()]);
        let lhs = $vec::from_array_t($token, a.map(f32::from_bits));
        let rhs = $vec::from_array_t($token, b.map(f32::from_bits));
        assert_eq!(
            lhs.broadcast_rgba_alpha().to_array().map(f32::to_bits),
            core::array::from_fn(|i| a[i | 3]),
        );
        assert_eq!(
            lhs.with_rgba_alpha(rhs).to_array().map(f32::to_bits),
            core::array::from_fn(|i| if i % 4 == 3 { b[i] } else { a[i] }),
        );
    }};
}

macro_rules! check_rgb {
    ($token:expr, $vec:ident, $load:ident, $n:literal) => {{
        let words = core::array::from_fn(|i| u16::MAX - 1801 * u16::try_from(i).unwrap());
        let (r, g, b) = $vec::$load($token, &words);
        let actual = [r, g, b].map(|v| v.to_array());
        for c in 0..3 {
            let expected: [f32; $n] = core::array::from_fn(|i| f32::from(words[i * 3 + c]));
            assert_eq!(actual[c], expected);
        }
    }};
}

fn check_narrow<T: F32x4Backend + F32x8Backend>(token: T) {
    check_lanes!(token, f32x4, 4);
    check_lanes!(token, f32x8, 8);
    check_rgb!(token, f32x4, load_4_rgb_u16_t, 4);
    check_rgb!(token, f32x8, load_8_rgb_u16_t, 8);

    let rows = core::array::from_fn(|i| {
        f32x4::from_array_t(
            token,
            core::array::from_fn(|j| f32::from_bits(BITS[i * 4 + j])),
        )
    });
    let columns = core::array::from_fn(|i| core::array::from_fn(|j| BITS[j * 4 + i]));
    let bits = |v: f32x4<T>| v.to_array().map(f32::to_bits);
    assert_eq!(f32x4::transpose_4x4_copy(rows).map(bits), columns);
    let mut transposed = rows;
    f32x4::transpose_4x4(&mut transposed);
    assert_eq!(transposed.map(bits), columns);
    let planes = f32x4::deinterleave_4ch(rows);
    assert_eq!(planes.map(bits), columns);
    assert_eq!(f32x4::interleave_4ch(planes).map(bits), rows.map(bits));

    let rows = core::array::from_fn(|i| {
        f32x8::from_array_t(
            token,
            core::array::from_fn(|j| f32::from_bits(BITS[(i * 8 + j) % 16])),
        )
    });
    let bits = |v: f32x8<T>| v.to_array().map(f32::to_bits);
    let planes = f32x8::deinterleave_3ch(rows);
    assert_eq!(
        planes.map(bits),
        core::array::from_fn(|c| core::array::from_fn(|i| BITS[(i * 3 + c) % 16]))
    );
    assert_eq!(f32x8::interleave_3ch(planes).map(bits), rows.map(bits));
}

#[cfg(feature = "w512")]
fn check_wide<T: magetypes::simd::backends::F32x16Backend>(token: T) {
    use magetypes::simd::generic::f32x16;
    check_lanes!(token, f32x16, 16);

    let rgba = core::array::from_fn(|i| u8::try_from(i * 4).unwrap());
    let (r, g, b, a) = f32x16::load_16_rgba_u8_t(token, &rgba);
    let channels = [r, g, b, a].map(|v| v.to_array());
    assert_eq!(
        channels,
        core::array::from_fn(|c| core::array::from_fn(|i| f32::from(rgba[i * 4 + c])))
    );
    assert_eq!(f32x16::store_16_rgba_u8(r, g, b, a), rgba);

    let rows = core::array::from_fn(|i| {
        f32x16::from_array_t(
            token,
            core::array::from_fn(|j| f32::from_bits(BITS[(i + j) % 16])),
        )
    });
    let bits = |v: f32x16<T>| v.to_array().map(f32::to_bits);
    let planes = f32x16::deinterleave_4ch(rows);
    assert_eq!(
        planes.map(bits),
        core::array::from_fn(|c| core::array::from_fn(|i| BITS[(i / 4 + 4 * (i % 4) + c) % 16]))
    );
    assert_eq!(f32x16::interleave_4ch(planes).map(bits), rows.map(bits));

    let bytes = [
        0.0,
        -0.0,
        0.5,
        1.5,
        2.5,
        3.5,
        255.0,
        256.0,
        254.5,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e20,
        3.0,
        255.5,
    ];
    assert_eq!(
        f32x16::from_array_t(token, bytes).to_u8(),
        [0, 0, 0, 2, 2, 4, 255, 255, 254, 0, 255, 0, 0, 255, 3, 255],
    );
}

#[test]
fn scalar_layout() {
    check_narrow(ScalarToken::summon().unwrap());
    #[cfg(feature = "w512")]
    check_wide(ScalarToken::summon().unwrap());
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
macro_rules! test_tier {
    ($name:ident, $token:ident) => {
        #[test]
        fn $name() {
            if let Some(token) = archmage::$token::summon() {
                check_narrow(token);
                #[cfg(feature = "w512")]
                check_wide(token);
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
test_tier!(v3_layout, X64V3Token);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
test_tier!(v4_layout, X64V4Token);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
test_tier!(v4x_layout, X64V4xToken);
#[cfg(target_arch = "aarch64")]
test_tier!(neon_layout, NeonToken);
#[cfg(target_arch = "wasm32")]
test_tier!(wasm_layout, Wasm128Token);
