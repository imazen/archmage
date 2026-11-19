use archmage::{ScalarToken, SimdToken};
use magetypes::numerics::{ExtremaOptions, HalfConvert, IntConvert, RecipOptions};
use magetypes::scalar;
use magetypes::simd::backends::{F32x4Convert, F32x8Convert, U32x4Backend, U32x8Backend};
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
            assert_eq!(
                loaded.to_u16_with::<{
                    magetypes::numerics::IntConvert::new()
                        .trunc()
                        .saturate_all()
                        .bits()
                }>(),
                values
            );
        }
        // Rotate rails and halfway values through all lane positions.
        for start in 0..FLOATS.len() {
            let values = core::array::from_fn(|i| FLOATS[(start + i) % FLOATS.len()]);
            assert_eq!(
                $vec::from_array_t($token, values).to_u16_with::<{
                    magetypes::numerics::IntConvert::new()
                        .trunc()
                        .saturate_all()
                        .bits()
                }>(),
                values.map(|x| x as u16)
            );
        }

        const SAT: u32 = IntConvert::new().saturate_all().bits();
        const NONNEGATIVE: u32 = IntConvert::new().non_negative_finite().bits();
        const FULL: u32 = HalfConvert::new().all_floats().normalize().bits();
        const FLUSH: u32 = HalfConvert::new().all_floats().bits();
        const TRUNC_HALF: u32 = HalfConvert::new().all_floats().normalize().trunc().bits();
        const EXACT: u32 = RecipOptions::new().exact().bits();
        const PROPAGATE: u32 = ExtremaOptions::new().propagate_nan().bits();
        const DROP_NAN: u32 = ExtremaOptions::new().drop_nan().bits();
        const NON_NAN_RHS: u32 = ExtremaOptions::new().non_nan_rhs().bits();
        for start in 0..FLOATS.len() {
            let x = core::array::from_fn(|i| FLOATS[(start + i) % FLOATS.len()]);
            let v = $vec::from_array_t($token, x);
            assert_eq!(
                v.to_i32_with::<SAT>().to_array(),
                x.map(scalar::to_i32_with::<SAT>)
            );
            assert_eq!(
                v.to_u32_with::<SAT>().to_array(),
                x.map(scalar::to_u32_with::<SAT>)
            );
            assert_eq!(v.to_u8_with::<SAT>(), x.map(scalar::to_u8_with::<SAT>));
            assert_eq!(v.to_u16_with::<SAT>(), x.map(scalar::to_u16_with::<SAT>));
            let y = v.trunc().to_array();
            let reciprocal = v.recip_with::<EXACT>().to_array();
            let half = v.to_f16_bits_with::<FULL>();
            let trunc_half = v.to_f16_bits_with::<TRUNC_HALF>();
            for lane in 0..$n {
                if x[lane].is_nan() {
                    assert!(y[lane].is_nan());
                    assert!(reciprocal[lane].is_nan());
                    assert!(half[lane] & 0x7fff > 0x7c00);
                    assert!(trunc_half[lane] & 0x7fff > 0x7c00);
                } else {
                    assert_eq!(y[lane].to_bits(), x[lane].trunc().to_bits());
                    assert_eq!(reciprocal[lane].to_bits(), (1.0 / x[lane]).to_bits());
                    assert_eq!(half[lane], scalar::to_f16_bits_with::<FULL>(x[lane]));
                    assert_eq!(
                        trunc_half[lane],
                        scalar::to_f16_bits_with::<TRUNC_HALF>(x[lane])
                    );
                }
            }
        }
        let nonnegative = [0.0, 0.5, 1.5, 254.5, 255.0, 255.5, 256.0, 16777216.0];
        let lanes = core::array::from_fn(|i| nonnegative[i % nonnegative.len()]);
        let v = $vec::from_array_t($token, lanes);
        assert_eq!(
            v.to_u8_with::<NONNEGATIVE>(),
            lanes.map(scalar::to_u8_with::<NONNEGATIVE>)
        );
        // Carry into the first half normal when rounded to nearest.
        let just_normal = f32::from_bits(0x387f_e000);
        let halves = [
            just_normal,
            just_normal.next_down(),
            -just_normal,
            -just_normal.next_down(),
        ];
        let v = $vec::from_array_t($token, core::array::from_fn(|i| halves[i % 4]));
        assert_eq!(
            v.to_f16_bits_with::<FLUSH>(),
            core::array::from_fn(|i| [0x0400, 0, 0x8400, 0x8000][i % 4])
        );
        let a = $vec::splat_t($token, 1.0_f32.next_up());
        let b = $vec::splat_t($token, 1.0 - f32::EPSILON);
        let c = $vec::splat_t($token, 1.0);
        assert_eq!(
            a.mul_neg_add_fused(b, c).to_array(),
            [f32::from_bits(81 << 23); $n]
        );
        let nan = $vec::splat_t($token, f32::NAN);
        assert_eq!(nan.min_with::<NON_NAN_RHS>(c).to_array(), [1.0; $n]);
        assert_eq!(nan.max_with::<NON_NAN_RHS>(c).to_array(), [1.0; $n]);
        assert!(nan.reduce_min_with::<DROP_NAN>().is_nan());
        assert!(nan.reduce_max_with::<DROP_NAN>().is_nan());
        let mixed = $vec::from_array_t(
            $token,
            core::array::from_fn(|i| [f32::NAN, 2.0, -3.0, f32::NAN][i % 4]),
        );
        assert_eq!(mixed.reduce_min_with::<DROP_NAN>(), -3.0);
        assert_eq!(mixed.reduce_max_with::<DROP_NAN>(), 2.0);
        assert!(
            a.min_with::<PROPAGATE>(nan)
                .reduce_max_with::<PROPAGATE>()
                .is_nan()
        );

        // Widen every half encoding outside Miri. Retain boundary groups in Miri.
        let step = if cfg!(miri) { $n * 257 } else { $n };
        for base in (0..=u16::MAX).step_by(step) {
            let bits = core::array::from_fn(|i| base + i as u16);
            let normalized = $vec::from_f16_with_t::<FULL>($token, &bits).to_array();
            let flushed = $vec::from_f16_with_t::<FLUSH>($token, &bits).to_array();
            let bf16 = $vec::from_bf16_t($token, &bits);
            assert_eq!(
                bf16.to_array().map(f32::to_bits),
                bits.map(|b| u32::from(b) << 16)
            );
            let bf16_roundtrip = bf16.to_bf16_with::<FULL>();
            for lane in 0..$n {
                let want = scalar::from_f16_with::<FULL>(bits[lane]);
                if want.is_nan() {
                    assert!(normalized[lane].is_nan());
                    assert!(flushed[lane].is_nan());
                } else {
                    assert_eq!(normalized[lane].to_bits(), want.to_bits());
                    assert_eq!(
                        flushed[lane].to_bits(),
                        scalar::from_f16_with::<FLUSH>(bits[lane]).to_bits()
                    );
                }
            }
            assert_eq!(
                bf16_roundtrip,
                bits.map(|b| scalar::to_bf16_with::<FULL>(f32::from_bits(u32::from(b) << 16)))
            );
        }

        let unsigned = $vec::from_array_t(
            $token,
            core::array::from_fn(|i| [0.0, 2147483648.0, 2147483904.0, 4294967040.0][i % 4]),
        );
        assert_eq!(
            $vec::from_u32_t(
                $token,
                unsigned.to_u32_with::<{ IntConvert::new().in_range().bits() }>()
            )
            .to_array(),
            unsigned.to_array()
        );

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

fn check_narrow<T: F32x4Convert + F32x8Convert + U32x4Backend + U32x8Backend>(token: T) {
    check_lanes!(token, f32x4, 4);
    check_lanes!(token, f32x8, 8);
    check_rgb!(token, f32x4, load_rgb_u16_t, 4);
    check_rgb!(token, f32x8, load_rgb_u16_t, 8);

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
fn check_wide<
    T: magetypes::simd::backends::F32x16Convert + magetypes::simd::backends::U32x16Backend,
>(
    token: T,
) {
    use magetypes::simd::generic::f32x16;
    check_lanes!(token, f32x16, 16);

    let rgba = core::array::from_fn(|i| u8::try_from(i * 4).unwrap());
    let (r, g, b, a) = f32x16::load_rgba_u8_t(token, &rgba);
    let channels = [r, g, b, a].map(|v| v.to_array());
    assert_eq!(
        channels,
        core::array::from_fn(|c| core::array::from_fn(|i| f32::from(rgba[i * 4 + c])))
    );
    let mut out = [0; 64];
    f32x16::store_rgba_u8(r, g, b, a, &mut out);
    assert_eq!(out, rgba);

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
