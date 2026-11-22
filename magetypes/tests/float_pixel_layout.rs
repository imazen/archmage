use archmage::{ScalarToken, SimdToken};
use magetypes::numerics::{
    ExtremaOptions, HalfConvert, IntConvert, NumericsError, NumericsErrorKind, RecipOptions,
};
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
        // Carry into the first half normal when rounded to nearest, under both
        // the default finite policy (`to_f16_bits()`) and `AllFloats` (`FLUSH`).
        let just_normal = f32::from_bits(0x387f_e000);
        let halves = [
            just_normal,
            just_normal.next_down(),
            -just_normal,
            f32::from_bits(0x387f_efff),
        ];
        let v = $vec::from_array_t($token, core::array::from_fn(|i| halves[i % 4]));
        let want_halves = core::array::from_fn(|i| [0x0400, 0, 0x8400, 0x0400][i % 4]);
        assert_eq!(v.to_f16_bits(), want_halves);
        assert_eq!(v.to_f16_bits_with::<FLUSH>(), want_halves);

        // `to_bf16()` preserves bf16 subnormals while `HalfConvert::new()` flushes them.
        const BF16_FLUSH_FINITE: u32 = HalfConvert::new().bits();
        let bf16_sub = [
            f32::from_bits(0x0040_0000),
            f32::from_bits(0x8040_0000),
            1.0,
            -2.5,
        ];
        let v_bf16 = $vec::from_array_t($token, core::array::from_fn(|i| bf16_sub[i % 4]));
        assert_eq!(
            v_bf16.to_bf16(),
            core::array::from_fn(|i| [0x0040, 0x8040, 0x3f80, 0xc020][i % 4])
        );
        assert_eq!(
            v_bf16.to_bf16_with::<BF16_FLUSH_FINITE>(),
            core::array::from_fn(|i| [0x0000, 0x8000, 0x3f80, 0xc020][i % 4])
        );

        // `Nr1Ceil` must upper-bound mathematical 1/x, including V3's 2-ULP downward witness.
        const NR1_CEIL: u32 = RecipOptions::new().nr1_ceil().bits();
        const NR1_CEIL_ALL: u32 = RecipOptions::new().nr1_ceil().all_floats().bits();
        let ceil_inputs = [
            f32::from_bits(0x3f80_8020), // 1.0039101 (V3 rcpps+NR is 2 ULPs below ceil)
            -f32::from_bits(0x3f80_8020),
            3.0,
            -7.0,
        ];
        let v_ceil = $vec::from_array_t($token, core::array::from_fn(|i| ceil_inputs[i % 4]));
        let got_ceil = v_ceil.recip_with::<NR1_CEIL>().to_array();
        assert_eq!(
            v_ceil
                .recip_with::<NR1_CEIL_ALL>()
                .to_array()
                .map(f32::to_bits),
            got_ceil.map(f32::to_bits)
        );
        for lane in 0..$n {
            let exact = 1.0_f64 / f64::from(ceil_inputs[lane % 4]);
            assert!(
                f64::from(got_ceil[lane]) >= exact,
                "Nr1Ceil failed upper bound at {}: {} < {}",
                ceil_inputs[lane % 4],
                got_ceil[lane],
                exact
            );
        }
        let ceil_rails = [0.0_f32, -0.0, f32::INFINITY, f32::NEG_INFINITY];
        let v_rails = $vec::from_array_t($token, core::array::from_fn(|i| ceil_rails[i % 4]));
        let got_rails = v_rails.recip_with::<NR1_CEIL_ALL>().to_array();
        assert_eq!(
            got_rails.map(f32::to_bits),
            core::array::from_fn(|i| {
                [
                    f32::INFINITY.to_bits(),
                    f32::NEG_INFINITY.to_bits(),
                    0.0_f32.to_bits(),
                    (-0.0_f32).to_bits(),
                ][i % 4]
            })
        );

        // Quiet NaNs must propagate through exp2_midp, exp2_lowp, exp_midp, pow_midp, sigmoid_midp.
        let qnan = $vec::splat_t($token, f32::NAN);
        let neg_base = $vec::splat_t($token, -1.0);
        for (label, out) in [
            ("exp2_midp(NaN)", qnan.exp2_midp().to_array()),
            ("exp2_lowp(NaN)", qnan.exp2_lowp().to_array()),
            ("exp_midp(NaN)", qnan.exp_midp().to_array()),
            ("sigmoid_midp(NaN)", qnan.sigmoid_midp().to_array()),
            ("pow_midp(-1, 0.5)", neg_base.pow_midp(0.5).to_array()),
        ] {
            for lane in 0..$n {
                assert!(out[lane].is_nan(), "{label}[{lane}] = {}", out[lane]);
            }
        }
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

// Check the expected mask independently, then verify each scalar tail and
// compare accepted vector results to the pre-existing lane conversions.
macro_rules! check_checked_conversions {
    ($token:ident, $vec:ident, $lanes:literal) => {{
        const FINITE: u32 = IntConvert::new().bits();
        const RANGE: u32 = IntConvert::new().trunc().in_range().bits();
        const NONNEG: u32 = IntConvert::new().non_negative_finite().bits();
        const ALL: u32 = IntConvert::new().saturate_all().bits();
        const HALF_FINITE: u32 = HalfConvert::new().normalize().bits();
        const HALF_ALL: u32 = HalfConvert::new().normalize().all_floats().bits();
        let pattern_mask =
            |pattern: u16| (0..$lanes).fold(0, |mask, i| mask | (((pattern >> (i % 4)) & 1) << i));
        let vector = |xs: [f32; 4]| $vec::from_array_t($token, core::array::from_fn(|i| xs[i % 4]));

        macro_rules! check_int {
            ($name:ident, $values:expr, $cfg:ident, $bad:literal) => {{
                let lanes: [f32; 4] = $values;
                let want = NumericsError {
                    kind: NumericsErrorKind::IntegerBounds,
                    lane_mask: pattern_mask($bad),
                };
                assert_eq!(vector(lanes).$name::<$cfg>().unwrap_err(), want);
                for (i, x) in lanes.into_iter().enumerate() {
                    let scalar = scalar::$name::<$cfg>(x);
                    if $bad & (1 << i) != 0 {
                        assert_eq!(
                            scalar.unwrap_err(),
                            NumericsError {
                                lane_mask: 1,
                                ..want
                            }
                        );
                    } else {
                        let actual = $vec::splat_t($token, x).$name::<$cfg>().unwrap();
                        assert_eq!(actual[0], scalar.unwrap());
                    }
                }
            }};
        }

        check_int!(
            try_to_i32_with,
            [f32::NAN, -2_147_483_648.0, 2_147_483_648.0, f32::INFINITY],
            FINITE,
            0b1101
        );
        check_int!(
            try_to_u32_with,
            [-0.25, 4_294_967_040.0, 4_294_967_296.0, f32::NEG_INFINITY],
            RANGE,
            0b1101
        );
        check_int!(
            try_to_u16_with,
            [-1.0, 65_535.0, 65_535.5, f32::NAN],
            RANGE,
            0b1101
        );
        check_int!(try_to_u8_with, [-1.0, -0.0, 256.0, 255.0], RANGE, 0b0101);
        check_int!(
            try_to_u8_with,
            [-1.0, 257.0, 2_147_483_648.0, f32::NEG_INFINITY],
            NONNEG,
            0b1101
        );
        check_int!(
            try_to_u16_with,
            [-1.0, 65_536.0, -2_147_483_904.0, 2_147_483_520.0],
            FINITE,
            0b0100
        );
        check_int!(
            try_to_u32_with,
            [-2_147_483_648.0, 2_147_483_520.0, f32::NAN, 2_147_483_648.0],
            FINITE,
            0b1100
        );

        let accepted = vector([0.5, 1.5, 2.5, 254.5]);
        assert_eq!(
            accepted.try_to_i32_with::<FINITE>().unwrap().to_array(),
            core::array::from_fn(|i| [0, 2, 2, 254][i % 4])
        );
        assert_eq!(
            accepted.try_to_u32_with::<FINITE>().unwrap().to_array(),
            core::array::from_fn(|i| [0, 2, 2, 254][i % 4])
        );
        assert_eq!(
            accepted.try_to_u16_with::<FINITE>().unwrap(),
            core::array::from_fn(|i| [0, 2, 2, 254][i % 4])
        );
        assert_eq!(
            accepted.try_to_u8_with::<FINITE>().unwrap(),
            core::array::from_fn(|i| [0, 2, 2, 254][i % 4])
        );

        let total = vector([f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -1.0]);
        let input = total.to_array();
        assert_eq!(
            total.try_to_i32_with::<ALL>().unwrap().to_array(),
            input.map(|x| scalar::try_to_i32_with::<ALL>(x).unwrap())
        );
        assert_eq!(
            total.try_to_u32_with::<ALL>().unwrap().to_array(),
            input.map(|x| scalar::try_to_u32_with::<ALL>(x).unwrap())
        );
        assert_eq!(
            total.try_to_u16_with::<ALL>().unwrap(),
            input.map(|x| scalar::try_to_u16_with::<ALL>(x).unwrap())
        );
        assert_eq!(
            total.try_to_u8_with::<ALL>().unwrap(),
            input.map(|x| scalar::try_to_u8_with::<ALL>(x).unwrap())
        );

        let nonfinite = NumericsError {
            kind: NumericsErrorKind::NonFinite,
            lane_mask: pattern_mask(0b0111),
        };
        assert_eq!(
            total.try_to_f16_bits_with::<HALF_FINITE>().unwrap_err(),
            nonfinite
        );
        assert_eq!(
            total.try_to_bf16_with::<HALF_FINITE>().unwrap_err(),
            nonfinite
        );
        let bits = core::array::from_fn(|i| [0x7c01, 0xfc00, 0x7c00, 0x8001][i % 4]);
        assert_eq!(
            $vec::try_from_f16_with_t::<HALF_FINITE>($token, &bits).unwrap_err(),
            nonfinite
        );
        let scalar_error = NumericsError {
            lane_mask: 1,
            ..nonfinite
        };
        for (i, (&x, &b)) in input.iter().zip(&bits).enumerate() {
            if i % 4 != 3 {
                assert_eq!(
                    scalar::try_to_f16_bits_with::<HALF_FINITE>(x).unwrap_err(),
                    scalar_error
                );
                assert_eq!(
                    scalar::try_to_bf16_with::<HALF_FINITE>(x).unwrap_err(),
                    scalar_error
                );
                assert_eq!(
                    scalar::try_from_f16_with::<HALF_FINITE>(b).unwrap_err(),
                    scalar_error
                );
            }
        }
        let binary16 = total.try_to_f16_bits_with::<HALF_ALL>().unwrap();
        let bf16 = total.try_to_bf16_with::<HALF_ALL>().unwrap();
        let decoded = $vec::try_from_f16_with_t::<HALF_ALL>($token, &bits)
            .unwrap()
            .to_array();
        for i in 0..$lanes {
            let x = input[i];
            if x.is_nan() {
                assert!(binary16[i] & 0x7fff > 0x7c00);
                assert!(bf16[i] & 0x7fff > 0x7f80);
                assert!(decoded[i].is_nan());
            } else {
                assert_eq!(
                    binary16[i],
                    scalar::try_to_f16_bits_with::<HALF_ALL>(x).unwrap()
                );
                assert_eq!(bf16[i], scalar::try_to_bf16_with::<HALF_ALL>(x).unwrap());
                assert_eq!(
                    decoded[i].to_bits(),
                    scalar::try_from_f16_with::<HALF_ALL>(bits[i])
                        .unwrap()
                        .to_bits()
                );
            }
        }

        // HalfDomain::Finite validates the source. Finite overflow and
        // subnormal handling are part of conversion rather than rejection.
        let finite = vector([f32::MAX, -0.0, f32::from_bits(1), 1.0]);
        let input = finite.to_array();
        assert_eq!(
            finite.try_to_f16_bits_with::<HALF_FINITE>().unwrap(),
            input.map(|x| scalar::try_to_f16_bits_with::<HALF_FINITE>(x).unwrap())
        );
        assert_eq!(
            finite.try_to_bf16_with::<HALF_FINITE>().unwrap(),
            input.map(|x| scalar::try_to_bf16_with::<HALF_FINITE>(x).unwrap())
        );
        let bytes = core::array::from_fn(|i| [0x0001, 0x8000, 0x8001, 0x3c00][i % 4]);
        let out = $vec::try_from_f16_with_t::<HALF_FINITE>($token, &bytes)
            .unwrap()
            .to_array();
        assert_eq!(
            out.map(f32::to_bits),
            bytes.map(|b| scalar::try_from_f16_with::<HALF_FINITE>(b)
                .unwrap()
                .to_bits())
        );
    }};
}

// Near MAX, division returns subnormal reciprocals. Check the signed upper
// bound independently of the scalar helper, then require exact tail parity.
const CEIL_ALL: u32 = RecipOptions::new().nr1_ceil().all_floats().bits();

macro_rules! check_ceil_near_max {
    ($token:ident, $vec:ident, $float:ident, $scalar:ident, $lanes:literal, $bound:ident) => {{
        let cases = [
            <$float>::MAX,
            <$float>::MAX.next_down(),
            <$float>::MAX.next_down().next_down(),
            <$float>::MAX.next_down().next_down().next_down(),
            <$float>::MAX
                .next_down()
                .next_down()
                .next_down()
                .next_down(),
            0.0,
            <$float>::INFINITY,
            <$float>::from_bits(1),
            1.0 / <$float>::MIN_POSITIVE,
            <$float>::NAN,
        ];
        for (i, _) in cases.iter().enumerate() {
            let input = core::array::from_fn::<_, $lanes, _>(|j| {
                let magnitude = cases[(i + j / 2) % cases.len()];
                if j % 2 == 0 { magnitude } else { -magnitude }
            });
            let actual = $vec::from_array_t($token, input)
                .recip_with::<CEIL_ALL>()
                .to_array();
            for (x, y) in input.into_iter().zip(actual) {
                if x.is_finite() && x.abs() >= cases[4] {
                    $bound(x, y);
                }
                let expected = scalar::$scalar::<CEIL_ALL>(x);
                if expected.is_nan() {
                    assert!(y.is_nan());
                } else {
                    assert_eq!(y.to_bits(), expected.to_bits(), "x={x:e}");
                }
            }
        }
    }};
}

fn exact_ceil_f32(x: f32, y: f32) {
    let product = f64::from(x) * f64::from(y);
    assert!(
        if x > 0.0 {
            product >= 1.0
        } else {
            product <= 1.0
        },
        "f32 x={:#010x} y={:#010x}",
        x.to_bits(),
        y.to_bits()
    );
    if x == f32::MAX {
        assert!(y.to_bits() >= 0x0020_0001);
    }
}

fn exact_ceil_f64(x: f64, y: f64) {
    assert_eq!(x.is_sign_negative(), y.is_sign_negative());
    assert!(y.is_subnormal());
    // x is normal with unbiased exponent 1023, y is subnormal.
    // Their unsigned significands give the exact product scaled by 2^103.
    let significand = (1 << 52) | (x.to_bits() & ((1 << 52) - 1));
    let product = u128::from(significand) * u128::from(y.abs().to_bits());
    assert!(
        if x > 0.0 {
            product >= 1 << 103
        } else {
            product <= 1 << 103
        },
        "f64 x={:#018x} y={:#018x}",
        x.to_bits(),
        y.to_bits()
    );
    if x == f64::MAX {
        assert!(y.to_bits() >= 0x0004_0000_0000_0001);
    }
}

#[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn reciprocal_ceil_rails(token: Token) {
    use magetypes::simd::generic::{f64x2, f64x4};
    check_ceil_near_max!(token, f32x4, f32, recip_f32_with, 4, exact_ceil_f32);
    check_ceil_near_max!(token, f32x8, f32, recip_f32_with, 8, exact_ceil_f32);
    check_ceil_near_max!(token, f64x2, f64, recip_f64_with, 2, exact_ceil_f64);
    check_ceil_near_max!(token, f64x4, f64, recip_f64_with, 4, exact_ceil_f64);
    #[cfg(feature = "w512")]
    {
        use magetypes::simd::generic::{f32x16, f64x8};
        check_ceil_near_max!(token, f32x16, f32, recip_f32_with, 16, exact_ceil_f32);
        check_ceil_near_max!(token, f64x8, f64, recip_f64_with, 8, exact_ceil_f64);
    }
}

#[test]
fn reciprocal_ceil_division_fallback_matches_scalar_bits() {
    reciprocal_ceil_rails_scalar_t(ScalarToken);
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        if let Some(t) = archmage::X64V3Token::summon() {
            reciprocal_ceil_rails_v3_t(t);
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = archmage::X64V4Token::summon() {
                reciprocal_ceil_rails_v4_t(t);
            }
            if let Some(t) = archmage::X64V4xToken::summon() {
                reciprocal_ceil_rails_v4x_t(t);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        reciprocal_ceil_rails_neon_t(t);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        reciprocal_ceil_rails_wasm128_t(t);
    }
}

#[test]
fn single_lane_clamp_matches_scalar_vectors_with_reversed_or_nan_bounds() {
    use magetypes::simd::scalar::f32x1;
    for (x, lo, hi) in [
        (8.0, 1.0, -1.0),
        (2.0, f32::NAN, 1.0),
        (2.0, 1.0, f32::NAN),
        (f32::NAN, 1.0, 3.0),
        (-0.0, 0.0, 1.0),
        (-2.0, -1.0, 1.0),
        (0.5, -1.0, 1.0),
    ] {
        let splat = |v| f32x4::splat_t(ScalarToken, v);
        let expected = splat(x).clamp(splat(lo), splat(hi)).to_array()[0];
        let actual = f32x1(x).clamp(f32x1(lo), f32x1(hi)).0;
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "x={x} lo={lo} hi={hi}"
            );
        }
    }
}

fn check_narrow<T: F32x4Convert + F32x8Convert + U32x4Backend + U32x8Backend>(token: T) {
    check_lanes!(token, f32x4, 4);
    check_checked_conversions!(token, f32x4, 4);
    check_lanes!(token, f32x8, 8);
    check_checked_conversions!(token, f32x8, 8);
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
    check_checked_conversions!(token, f32x16, 16);

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
fn policy_families_reject_foreign_bits() {
    let i = IntConvert::new().bits();
    let h = HalfConvert::new().bits();
    let r = RecipOptions::new().bits();
    let e = ExtremaOptions::new().bits();
    assert!(IntConvert::from_bits(i).is_some());
    assert!(IntConvert::from_bits(h).is_none());
    assert!(IntConvert::from_bits(r).is_none());
    assert!(IntConvert::from_bits(e).is_none());
    assert!(HalfConvert::from_bits(i).is_none());
    assert!(RecipOptions::from_bits(h).is_none());
    assert!(ExtremaOptions::from_bits(r).is_none());
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn narrow_v4_f64_approx_preserves_exact_division_and_wide_nr1_refines() {
    use magetypes::simd::generic::{f64x2, f64x4, f64x8};
    const NR1: u32 = RecipOptions::new().bits();
    const NR1_CEIL: u32 = RecipOptions::new().nr1_ceil().bits();
    if let Some(token) = archmage::X64V4Token::summon() {
        let v2 = f64x2::splat_t(token, 3.0);
        let v4 = f64x4::splat_t(token, 3.0);
        let exact_rcp = (1.0_f64 / 3.0).to_bits();
        let exact_rsqrt = (1.0_f64 / 3.0_f64.sqrt()).to_bits();
        assert_eq!(v2.rcp_approx().to_array().map(f64::to_bits), [exact_rcp; 2]);
        assert_eq!(
            v2.rsqrt_approx().to_array().map(f64::to_bits),
            [exact_rsqrt; 2]
        );
        assert_eq!(v4.rcp_approx().to_array().map(f64::to_bits), [exact_rcp; 4]);
        assert_eq!(
            v4.rsqrt_approx().to_array().map(f64::to_bits),
            [exact_rsqrt; 4]
        );

        let v8 = f64x8::splat_t(token, 3.0);
        let nr = v8.recip_with::<NR1>().to_array()[0];
        assert!(
            nr.to_bits().abs_diff(exact_rcp) <= 2,
            "f64x8 V4 Nr1 ULP error too high: {:#x} vs {:#x}",
            nr.to_bits(),
            exact_rcp
        );
        let ceil = v8.recip_with::<NR1_CEIL>().to_array()[0];
        assert!(ceil >= 1.0_f64 / 3.0);
    }
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
