use archmage::{IntoConcreteToken, ScalarToken, attune};
use magetypes::numerics::{NumericsError, NumericsErrorKind, RecipOptions};
use magetypes::scalar;
use magetypes::simd::generic::{f32x4, f32x8, f64x2, f64x4};

fn expected_error(mask: u16) -> NumericsError {
    NumericsError {
        kind: NumericsErrorKind::RecipDomain,
        lane_mask: mask,
    }
}

// Keep each policy and width in a separate debug-test stack frame.
#[inline(never)]
fn run_case(check: impl FnOnce()) {
    check();
}

macro_rules! check_operation {
    (
        $token:ident, $vec:ident, $float:ident, $try_vec:ident, $bare_vec:ident,
        $try_scalar:ident, $bare_scalar:ident, $lanes:literal, $root:literal, $cfg:expr
    ) => {
        run_case(|| {
            const CFG: u32 = $cfg.bits();
            let upper = 1.0 / <$float>::MIN_POSITIVE;
            // Expected acceptance by each restricted-domain operation.
            let cases = [
                (2.0, true, true),
                (-2.0, true, false),
                (<$float>::MIN_POSITIVE, true, true),
                (-<$float>::MIN_POSITIVE, true, false),
                (upper.next_down(), true, true),
                (-upper.next_down(), true, false),
                (upper, false, true),
                (-upper, false, false),
                (<$float>::MAX, false, true),
                (-<$float>::MAX, false, false),
                (0.0, false, false),
                (-0.0, false, false),
                (<$float>::from_bits(1), false, false),
                (-<$float>::from_bits(1), false, false),
                (<$float>::INFINITY, false, false),
                (<$float>::NEG_INFINITY, false, false),
                (<$float>::NAN, false, false),
            ];
            let total =
                CFG == RecipOptions::new().exact().bits() || (CFG == ($cfg).all_floats().bits());
            let same_bits = |got: $float, expected: $float| {
                if expected.is_nan() {
                    assert!(got.is_nan(), "expected NaN, got {got:?}");
                } else {
                    assert_eq!(got.to_bits(), expected.to_bits());
                }
            };
            for (start, (x, recip_ok, rsqrt_ok)) in cases.into_iter().enumerate() {
                let accepts = |recip, rsqrt| total || if $root { rsqrt } else { recip };
                let scalar = scalar::$try_scalar::<CFG>(x);
                let repeated = $vec::splat_t($token, x);
                if accepts(recip_ok, rsqrt_ok) {
                    let scalar = scalar.unwrap();
                    same_bits(scalar, scalar::$bare_scalar::<CFG>(x));
                    let vector = repeated.$try_vec::<CFG>().unwrap().to_array()[0];
                    same_bits(vector, repeated.$bare_vec::<CFG>().to_array()[0]);
                    let division_seed = $token.as_scalar().is_some();
                    #[cfg(target_arch = "wasm32")]
                    let division_seed = division_seed || $token.as_wasm128().is_some();
                    if CFG == RecipOptions::new().exact().bits() || (!$root && division_seed) {
                        same_bits(vector, scalar);
                    }
                } else {
                    assert_eq!(scalar.unwrap_err(), expected_error(1));
                    let mask = u16::MAX >> (16 - $lanes);
                    assert_eq!(
                        repeated.$try_vec::<CFG>().unwrap_err(),
                        expected_error(mask)
                    );
                }
                let group =
                    core::array::from_fn::<_, $lanes, _>(|i| cases[(i + start) % cases.len()]);
                let mask = group
                    .iter()
                    .enumerate()
                    .fold(0, |mask, (i, &(_, recip, rsqrt))| {
                        mask | (u16::from(!accepts(recip, rsqrt)) << i)
                    });
                let vector = $vec::from_array_t($token, group.map(|(value, _, _)| value));
                if mask == 0 {
                    let checked = vector.$try_vec::<CFG>().unwrap().to_array();
                    let unchecked = vector.$bare_vec::<CFG>().to_array();
                    for (got, expected) in checked.into_iter().zip(unchecked) {
                        same_bits(got, expected);
                    }
                } else {
                    assert_eq!(vector.$try_vec::<CFG>().unwrap_err(), expected_error(mask));
                }
            }
        })
    };
}

macro_rules! check_width {
    ($token:ident, $vec:ident, $float:ident, $lanes:literal, $cr:ident, $r:ident, $cs:ident, $s:ident) => {{
        check_operation!(
            $token,
            $vec,
            $float,
            try_recip_with,
            recip_with,
            $cr,
            $r,
            $lanes,
            false,
            RecipOptions::new()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_recip_with,
            recip_with,
            $cr,
            $r,
            $lanes,
            false,
            RecipOptions::new().approx()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_recip_with,
            recip_with,
            $cr,
            $r,
            $lanes,
            false,
            RecipOptions::new().nr1_ceil()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_recip_with,
            recip_with,
            $cr,
            $r,
            $lanes,
            false,
            RecipOptions::new().exact()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_recip_with,
            recip_with,
            $cr,
            $r,
            $lanes,
            false,
            RecipOptions::new().nr1_ceil().all_floats()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_recip_with,
            recip_with,
            $cr,
            $r,
            $lanes,
            false,
            RecipOptions::new().approx().all_floats()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_recip_with,
            recip_with,
            $cr,
            $r,
            $lanes,
            false,
            RecipOptions::new().nr1().all_floats()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_rsqrt_with,
            rsqrt_with,
            $cs,
            $s,
            $lanes,
            true,
            RecipOptions::new()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_rsqrt_with,
            rsqrt_with,
            $cs,
            $s,
            $lanes,
            true,
            RecipOptions::new().approx()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_rsqrt_with,
            rsqrt_with,
            $cs,
            $s,
            $lanes,
            true,
            RecipOptions::new().nr1_ceil()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_rsqrt_with,
            rsqrt_with,
            $cs,
            $s,
            $lanes,
            true,
            RecipOptions::new().exact()
        );
        check_operation!(
            $token,
            $vec,
            $float,
            try_rsqrt_with,
            rsqrt_with,
            $cs,
            $s,
            $lanes,
            true,
            RecipOptions::new().nr1_ceil().all_floats()
        );
    }};
}

#[attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn reciprocal_domains(token: Token) {
    macro_rules! f32_width {
        ($vec:ident, $lanes:literal) => {
            check_width!(
                token,
                $vec,
                f32,
                $lanes,
                try_recip_f32_with,
                recip_f32_with,
                try_rsqrt_f32_with,
                rsqrt_f32_with
            )
        };
    }
    macro_rules! f64_width {
        ($vec:ident, $lanes:literal) => {
            check_width!(
                token,
                $vec,
                f64,
                $lanes,
                try_recip_f64_with,
                recip_f64_with,
                try_rsqrt_f64_with,
                rsqrt_f64_with
            )
        };
    }
    f32_width!(f32x4, 4);
    f32_width!(f32x8, 8);
    f64_width!(f64x2, 2);
    f64_width!(f64x4, 4);
    #[cfg(feature = "w512")]
    {
        use magetypes::simd::generic::{f32x16, f64x8};
        f32_width!(f32x16, 16);
        f64_width!(f64x8, 8);
    }
}

#[test]
fn scalar_domains() {
    reciprocal_domains_scalar_t(ScalarToken);
}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
macro_rules! test_tier {
    ($name:ident, $token:ident, $check:ident) => {
        #[test]
        fn $name() {
            use archmage::SimdToken;
            if let Some(token) = archmage::$token::summon() {
                $check(token);
            }
        }
    };
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
test_tier!(v3_domains, X64V3Token, reciprocal_domains_v3_t);
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
test_tier!(v4_domains, X64V4Token, reciprocal_domains_v4_t);
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
test_tier!(v4x_domains, X64V4xToken, reciprocal_domains_v4x_t);
#[cfg(target_arch = "aarch64")]
test_tier!(neon_domains, NeonToken, reciprocal_domains_neon_t);
#[cfg(target_arch = "wasm32")]
test_tier!(wasm_domains, Wasm128Token, reciprocal_domains_wasm128_t);
