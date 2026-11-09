use super::*;
extern crate std;
#[path = "test_vectors.rs"]
mod fixtures;

fn options(domain: u32) -> PowfOptions {
    PowfOptions {
        domain: match domain {
            0 => PowfDomain::PositiveNormal,
            1 => PowfDomain::NonNegativeNormal,
            2 => PowfDomain::Gamma,
            _ => panic!("unknown test domain"),
        },
        execution: PowfExecution::Caller,
        ..PowfOptions::new()
    }
}

fn expected_kind(code: u32) -> PowfErrorKind {
    match code {
        1 => PowfErrorKind::InvalidExponent,
        2 => PowfErrorKind::InvalidBase,
        3 => PowfErrorKind::OutsideApproximationRange,
        _ => panic!("unexpected oracle rejection stage"),
    }
}

// Runtime dispatch exists only in the test harness. Every arm instantiates
// one literal options integer all the way through the public API.
macro_rules! const_options {
    ($options:expr, |$flags:ident| $body:expr) => {{
        match ($options.domain, $options.execution) {
            (PowfDomain::PositiveNormal, PowfExecution::Automatic) => {
                const $flags: u32 = 0;
                $body
            }
            (PowfDomain::NonNegativeNormal, PowfExecution::Automatic) => {
                const $flags: u32 = 1;
                $body
            }
            (PowfDomain::Gamma, PowfExecution::Automatic) => {
                const $flags: u32 = 2;
                $body
            }
            (PowfDomain::PositiveNormal, PowfExecution::Caller) => {
                const $flags: u32 = 4;
                $body
            }
            (PowfDomain::NonNegativeNormal, PowfExecution::Caller) => {
                const $flags: u32 = 5;
                $body
            }
            (PowfDomain::Gamma, PowfExecution::Caller) => {
                const $flags: u32 = 6;
                $body
            }
            _ => panic!("legacy test uses restricted domains"),
        }
    }};
}

macro_rules! check {
    ($token:expr) => {{
        let token = $token;
        for domain in [1, 2] {
            let input = [0.0, 1.0, 4.0, 16.0];
            assert_eq!(
                crate::simd::generic::f32x4::from_array_t(token, input)
                    .try_powf_runtime(0.5, options(domain))
                    .unwrap()
                    .to_array()
                    .map(f32::to_bits),
                [0.0_f32, 1.0, 2.0, 4.0].map(f32::to_bits),
            );
        }
        // Distinct accepted lanes catch packing/shuffle errors that splats miss.
        let bases = [
            0x3780_0000,
            0x3780_0001,
            0x3f7f_ffff,
            0x3f80_0000,
            0x3f80_0001,
            0x3f35_04f3,
            0x3fb5_04f3,
            0x4000_0000,
            0x4080_0000,
            0x477f_ffff,
            0x4780_0000,
        ];
        for domain in 0..3 {
            for exponent in [0.5_f32, 2.4, 4.0] {
                for execution in [PowfExecution::Caller, PowfExecution::Automatic] {
                    let exponent = std::hint::black_box(exponent);
                    let opts = PowfOptions {
                        execution,
                        ..options(domain)
                    };
                    macro_rules! mixed_width {
                        ($ty:ident, $n:literal) => {{
                            let input: [f32; $n] =
                                std::hint::black_box(core::array::from_fn(|i| {
                                    f32::from_bits(if i == 0 && domain != 0 {
                                        0
                                    } else {
                                        bases[i % bases.len()]
                                    })
                                }));
                            let expected: [u32; $n] = core::array::from_fn(|i| {
                                let row = fixtures::POWF
                                    .iter()
                                    .find(|r| {
                                        r[0] == input[i].to_bits()
                                            && r[1] == exponent.to_bits()
                                            && r[4] == domain
                                    })
                                    .expect("independent mixed-lane fixture");
                                assert_eq!(row[3], 0);
                                row[2]
                            });
                            assert_eq!(
                                const_options!(opts, |FLAGS| {
                                    crate::simd::generic::$ty::from_array_t(token, input)
                                        .try_powf_with::<FLAGS>(exponent)
                                })
                                .unwrap()
                                .to_array()
                                .map(f32::to_bits),
                                expected
                            );
                            assert_eq!(
                                crate::simd::generic::$ty::from_array_t(token, input)
                                    .try_powf_runtime(exponent, opts)
                                    .unwrap()
                                    .to_array()
                                    .map(f32::to_bits),
                                expected
                            );
                        }};
                    }
                    mixed_width!(f32x4, 4);
                    mixed_width!(f32x8, 8);
                    #[cfg(feature = "w512")]
                    mixed_width!(f32x16, 16);
                }
            }
        }
        for &[base, exponent, expected, error, allow_zero] in &fixtures::POWF {
            let b = f32::from_bits(std::hint::black_box(base));
            let n = f32::from_bits(std::hint::black_box(exponent));
            let options = options(allow_zero);
            macro_rules! width {
                ($ty:ident, $width:literal) => {{
                    let result = crate::simd::generic::$ty::from_array_t(token, [b; $width])
                        .try_powf_runtime(n, options);
                    for execution in [PowfExecution::Caller, PowfExecution::Automatic] {
                        let opts = PowfOptions {
                            execution,
                            ..options
                        };
                        let constant =
                            const_options!(opts, |FLAGS| crate::simd::generic::$ty::from_array_t(
                                token,
                                [b; $width]
                            )
                            .try_powf_with::<FLAGS>(n));
                        assert_eq!(
                            constant.map(|v| v.to_array().map(f32::to_bits)),
                            result.map(|v| v.to_array().map(f32::to_bits)),
                            "const flags base={base:08x} exponent={exponent:08x}"
                        );
                    }
                    if error == 0 {
                        assert_eq!(
                            result.unwrap().to_array().map(f32::to_bits),
                            [expected; $width],
                            "base={base:08x} exponent={exponent:08x}"
                        );
                    } else {
                        assert_eq!(
                            result.unwrap_err(),
                            PowfError {
                                kind: expected_kind(error),
                                lane_mask: ((1_u32 << $width) - 1) as u16
                            }
                        );
                    }
                }};
            }
            width!(f32x4, 4);
            width!(f32x8, 8);
            #[cfg(feature = "w512")]
            width!(f32x16, 16);
        }
    }};
}

#[test]
fn scalar_array_and_auto_have_identical_graph() {
    for &[base, exponent, expected, error, allow_zero] in &fixtures::POWF {
        let b = f32::from_bits(std::hint::black_box(base));
        let n = f32::from_bits(std::hint::black_box(exponent));
        for execution in [PowfExecution::Caller, PowfExecution::Automatic] {
            let options = PowfOptions {
                execution,
                ..options(allow_zero)
            };
            let result = try_powf_runtime(b, n, options);
            let constant = const_options!(options, |FLAGS| try_powf_with::<FLAGS>(b, n));
            assert_eq!(constant.map(f32::to_bits), result.map(f32::to_bits));
            assert_eq!(
                const_options!(options, |FLAGS| crate::simd::scalar::f32x1(b)
                    .try_powf_with::<FLAGS>(n))
                .map(|v| v.0.to_bits()),
                result.map(f32::to_bits)
            );
            assert_eq!(
                const_options!(options, |FLAGS| try_powf_array_with::<FLAGS, 1>([b], n))
                    .map(|v| v.map(f32::to_bits)),
                result.map(|v| [v.to_bits()])
            );
            assert_eq!(
                crate::simd::scalar::f32x1(b)
                    .try_powf_runtime(n, options)
                    .map(|v| v.0.to_bits()),
                result.map(f32::to_bits),
            );
            if error == 0 {
                assert_eq!(
                    result.unwrap().to_bits(),
                    expected,
                    "base={base:08x}, exponent={exponent:08x}"
                );
                assert_eq!(
                    try_powf_array_runtime([b; 4], n, options)
                        .unwrap()
                        .map(f32::to_bits),
                    [expected; 4]
                );
            } else {
                assert_eq!(
                    result.unwrap_err(),
                    PowfError {
                        kind: expected_kind(error),
                        lane_mask: 1
                    }
                );
            }
        }
    }
}

#[test]
fn mixed_lanes_reject_atomically_with_exact_mask() {
    let options = options(0);
    let input = [2.0, f32::NAN, f32::from_bits(1), -1.0];
    assert_eq!(
        try_powf_array_runtime(input, 2.4, options).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::InvalidBase,
            lane_mask: 0b1110
        }
    );
    assert_eq!(
        try_powf_array_runtime([2.0, 4.0, f32::MAX, f32::MIN_POSITIVE], 2.0, options).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::OutsideApproximationRange,
            lane_mask: 0b1100
        }
    );
    assert_eq!(
        try_powf_array_runtime([2.0; 4], f32::NAN, options).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::InvalidExponent,
            lane_mask: 0b1111
        }
    );
    assert_eq!(
        try_powf_array_runtime([1.0; 3], 1.0, options)
            .unwrap_err()
            .kind,
        PowfErrorKind::UnsupportedWidth
    );
    assert_eq!(
        try_powf_array_runtime([], 1.0, options).unwrap_err().kind,
        PowfErrorKind::UnsupportedWidth
    );
    assert_eq!(
        try_powf_array_runtime([1.0; 32], 1.0, options).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
}

#[test]
fn caller_scalar_never_upgrades() {
    assert!(matches!(
        dispatch::select(archmage::ScalarToken, PowfExecution::Caller),
        dispatch::Plan::Software
    ));
}

#[test]
fn automatic_plan_matches_requested_runtime_context() {
    let plan = dispatch::select(archmage::ScalarToken, PowfExecution::Automatic);
    std::eprintln!("powf automatic plan: {plan:?}");
    if let Ok(expected) = std::env::var("ARCHMAGE_POWF_EXPECT_PLAN") {
        let matches = match expected.as_str() {
            "software" => matches!(plan, dispatch::Plan::Software),
            #[cfg(all(feature = "std", not(miri)))]
            "std" => matches!(plan, dispatch::Plan::Std),
            #[cfg(target_arch = "x86_64")]
            "v3" => matches!(plan, dispatch::Plan::V3(_)),
            #[cfg(target_arch = "aarch64")]
            "neon" => matches!(plan, dispatch::Plan::Neon(_)),
            #[cfg(target_arch = "wasm32")]
            "wasm-fused" => matches!(plan, dispatch::Plan::Wasm(_, true)),
            #[cfg(target_arch = "wasm32")]
            "wasm-software" => matches!(plan, dispatch::Plan::Wasm(_, false)),
            name => panic!("unknown requested plan {name}"),
        };
        assert!(matches, "expected {expected}, got {plan:?}");
    }
}

macro_rules! payloads {
    ($token:expr, $stride:expr) => {{
        let token = $token;
        for sign in [0, 0x8000_0000] {
            for payload in (1..=0x007f_ffff_u32).step_by($stride) {
                for rejected in [sign | 0x7f80_0000 | payload, sign | payload] {
                    let nan = f32::from_bits(std::hint::black_box(rejected));
                    macro_rules! width {
                        ($ty:ident, $n:literal, $domain:expr) => {{
                            let lane = payload as usize % $n;
                            let mut input = [1.0; $n];
                            input[lane] = nan;
                            let opts = options($domain);
                            let constant = const_options!(opts, |FLAGS| {
                                crate::simd::generic::$ty::from_array_t(token, input)
                                    .try_powf_with::<FLAGS>(2.4)
                            });
                            assert_eq!(
                                constant.unwrap_err(),
                                PowfError {
                                    kind: PowfErrorKind::InvalidBase,
                                    lane_mask: 1 << lane
                                }
                            );
                            let constant = const_options!(opts, |FLAGS| {
                                crate::simd::generic::$ty::from_array_t(token, [1.0; $n])
                                    .try_powf_with::<FLAGS>(nan)
                            });
                            assert_eq!(
                                constant.unwrap_err(),
                                PowfError {
                                    kind: PowfErrorKind::InvalidExponent,
                                    lane_mask: ((1_u32 << $n) - 1) as u16
                                }
                            );
                            assert_eq!(
                                crate::simd::generic::$ty::from_array_t(token, input)
                                    .try_powf_runtime(2.4, options($domain))
                                    .unwrap_err(),
                                PowfError {
                                    kind: PowfErrorKind::InvalidBase,
                                    lane_mask: 1 << lane
                                },
                                "base_bits={rejected:08x}, width={}, domain={}",
                                $n,
                                $domain
                            );
                            assert_eq!(
                                crate::simd::generic::$ty::from_array_t(token, [1.0; $n])
                                    .try_powf_runtime(nan, options($domain))
                                    .unwrap_err(),
                                PowfError {
                                    kind: PowfErrorKind::InvalidExponent,
                                    lane_mask: ((1_u32 << $n) - 1) as u16,
                                },
                                "exponent_bits={rejected:08x}, width={}, domain={}",
                                $n,
                                $domain
                            );
                        }};
                    }
                    for domain in 0..3 {
                        width!(f32x4, 4, domain);
                        width!(f32x8, 8, domain);
                        #[cfg(feature = "w512")]
                        width!(f32x16, 16, domain);
                    }
                }
            }
        }
    }};
}

#[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn run_rational(token: Token, expected: &str) {
    assert_eq!(
        <Token as archmage::SimdToken>::NAME,
        expected,
        "requested backend did not execute"
    );
    check!(token);
}

#[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn run_payloads(token: Token, expected: &str, stride: usize) {
    assert_eq!(
        <Token as archmage::SimdToken>::NAME,
        expected,
        "requested backend did not execute"
    );
    payloads!(token, stride);
}

macro_rules! require_tier {
    ($family:ident $(, $argument:expr)*) => {{
        use archmage::SimdToken;
        let tier = std::env::var("ARCHMAGE_POWF_TIER").unwrap_or_else(|_| "scalar".into());
        match tier.as_str() {
            "scalar" => archmage::attuned!($family(archmage::ScalarToken::NAME $(, $argument)*), [_scalar_t]),
            #[cfg(target_arch = "x86_64")]
            "v3" => archmage::attuned!($family(archmage::X64V3Token::NAME $(, $argument)*), [_v3_t, _scalar_t]),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            "v4" => archmage::attuned!($family(archmage::X64V4Token::NAME $(, $argument)*), [_v4_t, _scalar_t]),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            "v4x" => archmage::attuned!($family(archmage::X64V4xToken::NAME $(, $argument)*), [_v4x_t, _scalar_t]),
            #[cfg(target_arch = "aarch64")]
            "neon" => archmage::attuned!($family(archmage::NeonToken::NAME $(, $argument)*), [_neon_t, _scalar_t]),
            #[cfg(target_arch = "wasm32")]
            "wasm128" => archmage::attuned!($family(archmage::Wasm128Token::NAME $(, $argument)*), [_wasm128_t, _scalar_t]),
            other => panic!("unsupported requested powf backend {other}"),
        }
    }};
}

#[test]
fn rational_graph_and_selected_backend() {
    require_tier!(run_rational);
}

#[test]
fn nan_sign_payload_and_subnormal_families_reject_with_exact_masks() {
    let stride: usize = std::env::var("ARCHMAGE_NAN_STRIDE")
        .unwrap_or_else(|_| "65537".into())
        .parse()
        .unwrap();
    assert!(stride > 0);
    require_tier!(run_payloads, stride);
}

#[test]
fn const_array_width_and_rejection_precedence() {
    const FLAGS: u32 = PowfOptions::GAMMA | PowfOptions::CALLER;
    assert_eq!(
        try_powf_array_with::<FLAGS, 0>([], f32::NAN).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(
        try_powf_array_with::<FLAGS, 3>([1.0; 3], f32::NAN).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(
        try_powf_array_with::<FLAGS, 32>([1.0; 32], f32::NAN).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(
        try_powf_array_with::<FLAGS, 4>([f32::NAN; 4], f32::NAN).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::InvalidExponent,
            lane_mask: 15
        }
    );
    assert_eq!(
        try_powf_array_with::<{ PowfOptions::CALLER }, 4>(
            [2.0, f32::NAN, f32::from_bits(1), -1.0],
            2.4
        )
        .unwrap_err(),
        PowfError {
            kind: PowfErrorKind::InvalidBase,
            lane_mask: 14
        }
    );
    assert_eq!(
        try_powf_array_with::<{ PowfOptions::CALLER }, 4>(
            [2.0, 4.0, f32::MAX, f32::MIN_POSITIVE],
            2.0
        )
        .unwrap_err(),
        PowfError {
            kind: PowfErrorKind::OutsideApproximationRange,
            lane_mask: 12
        }
    );
}

#[test]
fn const_builder_encodes_each_supported_configuration() {
    for domain in [
        PowfDomain::PositiveNormal,
        PowfDomain::NonNegativeNormal,
        PowfDomain::Gamma,
    ] {
        for execution in [PowfExecution::Automatic, PowfExecution::Caller] {
            let builder = PowfOptions::new()
                .with_domain(domain)
                .with_execution(execution);
            assert_eq!(
                builder,
                PowfOptions {
                    domain,
                    execution,
                    ..PowfOptions::new()
                }
            );
            let expected = match domain {
                PowfDomain::PositiveNormal => 0,
                PowfDomain::NonNegativeNormal => 1,
                PowfDomain::Gamma => 2,
                PowfDomain::Full => unreachable!("legacy domain test"),
            } | if execution == PowfExecution::Caller {
                4
            } else {
                0
            };
            assert_eq!(builder.bits(), expected);
        }
    }
}

#[test]
fn new_policy_presets_and_invalid_runtime_combinations() {
    for preset in [
        PowfOptions::PORTABLE_GAMMA_MIDP,
        PowfOptions::PORTABLE_GAMMA_LOWP,
        PowfOptions::BACKEND_GAMMA_MIDP,
        PowfOptions::BACKEND_GAMMA_LOWP,
    ] {
        assert_eq!(preset.domain, PowfDomain::Gamma);
        assert_eq!(preset.validation, PowfValidation::Checked);
        assert_eq!(
            try_powf_array_runtime([0., 1., 4., 16.], 0., preset).unwrap()[0].to_bits(),
            1f32.to_bits()
        );
        let error = try_powf_array_runtime([0., f32::NAN, -0., f32::from_bits(1)], 2.4, preset)
            .unwrap_err();
        assert_eq!(
            error,
            PowfError {
                kind: PowfErrorKind::InvalidBase,
                lane_mask: 14
            }
        );
    }
    for opts in [
        PowfOptions::new().with_domain(PowfDomain::Full),
        PowfOptions::new()
            .with_precision(PowfPrecision::Lowp)
            .with_domain(PowfDomain::Full),
    ] {
        assert_eq!(
            try_powf_runtime(1., 2.4, opts).unwrap_err(),
            PowfError {
                kind: PowfErrorKind::UnsupportedOptions,
                lane_mask: 0
            }
        );
    }
    #[cfg(not(feature = "std"))]
    assert_eq!(
        try_powf_runtime(1., 2.4, PowfOptions::STD_HIGHP)
            .unwrap_err()
            .kind,
        PowfErrorKind::UnsupportedOptions
    );
    assert!(!PowfOptions::valid(128));
    assert!(!PowfOptions::valid(
        PowfOptions::LOWP | PowfOptions::HIGHP | PowfOptions::BACKEND
    ));
}

#[test]
fn lowp_admitted_zero_survives_huge_exponent_in_both_const_modes() {
    const PORTABLE: u32 = PowfOptions::PORTABLE_GAMMA_LOWP
        .with_domain(PowfDomain::NonNegativeNormal)
        .with_execution(PowfExecution::Caller)
        .bits();
    const BACKEND: u32 = PowfOptions::BACKEND_GAMMA_LOWP
        .with_domain(PowfDomain::NonNegativeNormal)
        .with_execution(PowfExecution::Caller)
        .bits();
    for n in [1_073_741_824., f32::MAX] {
        assert_eq!(try_powf_with::<PORTABLE>(0., n).unwrap().to_bits(), 0);
        assert_eq!(try_powf_with::<BACKEND>(0., n).unwrap().to_bits(), 0);
    }
}

#[test]
#[cfg(feature = "std")]
fn std_highp_full_domain_and_restricted_masks() {
    const STD: u32 = PowfOptions::STD_HIGHP.bits();
    let result = try_powf_array_with::<STD, 4>([-2., -0., f32::INFINITY, f32::NAN], 3.).unwrap();
    // Std power has unspecified precision, including under Miri. Only its
    // admitted domain and sign/classification are the HIGHP contract here.
    assert!(result[0].is_normal() && result[0].is_sign_negative());
    assert_eq!(result[1].to_bits(), (-0.0f32).to_bits());
    assert_eq!(result[2], f32::INFINITY);
    assert!(result[3].is_nan());
    assert_eq!(
        try_powf_array_with::<STD, 4>([f32::NAN, 0., -0., f32::INFINITY], 0.).unwrap(),
        [1.; 4]
    );
    assert_eq!(try_powf_with::<STD>(0., -1.).unwrap(), f32::INFINITY);
    let tiny = try_powf_with::<STD>(f32::from_bits(1), 1.).unwrap();
    assert!(tiny.is_subnormal() || tiny == 0.);
    assert_eq!(
        try_powf_array_with::<STD, 3>([1.; 3], 2.).unwrap_err().kind,
        PowfErrorKind::UnsupportedWidth
    );
    let opts = PowfOptions::STD_HIGHP.with_domain(PowfDomain::PositiveNormal);
    assert_eq!(
        try_powf_array_runtime([2., -1., f32::from_bits(1), f32::NAN], 2., opts).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::InvalidBase,
            lane_mask: 14
        }
    );
    assert_eq!(
        try_powf_array_runtime([2., 4., f32::MAX, f32::MIN_POSITIVE], 2., opts).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::OutsideApproximationRange,
            lane_mask: 12
        }
    );
}

// A feature-context tail must use the same graph as the packed body. In
// particular BackendDependent Caller must retain its token rather than switch
// to ScalarToken (which intentionally uses separate arithmetic).
#[archmage::magetypes(v4x(cfg(avx512)), v3, neon, wasm128, scalar)]
fn bounded_widths_and_tails(token: Token) {
    use crate::simd::generic::{f32x4, f32x8};
    fn check<const FLAGS: u32>(token: Token) {
        let mut state = 0x9e37_79b9_u32;
        for n in [0.0_f32, -0.0, 0.5, 2.4, 4.0] {
            let input: [f32; 513] = core::array::from_fn(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if FLAGS & 3 != 0 && i % 17 == 0 {
                    0.0
                } else {
                    f32::from_bits(0x3780_0000 + state % 0x1000_0001)
                }
            });
            let single = input.map(|x| {
                let result = f32x4::splat_t(token, x)
                    .try_powf_with::<FLAGS>(n)
                    .unwrap()
                    .to_array()[0];
                if FLAGS & PowfOptions::CALLER == 0 || FLAGS & PowfOptions::BACKEND == 0 {
                    assert_eq!(try_powf_with::<FLAGS>(x, n).unwrap().to_bits(), result.to_bits());
                }
                result.to_bits()
            });
            macro_rules! width {
                ($ty:ident, $lanes:literal) => {
                    for length in (0..=33).chain([64, 65, 127, 255, 511, 512, 513]) {
                        let mut chunks = input[..length].chunks_exact($lanes);
                        let mut offset = 0;
                        for chunk in chunks.by_ref() {
                            let result = $ty::from_array_t(token, chunk.try_into().unwrap())
                                .try_powf_with::<FLAGS>(n)
                                .unwrap()
                                .to_array()
                                .map(f32::to_bits);
                            assert_eq!(&result, &single[offset..offset + $lanes]);
                            offset += $lanes;
                        }
                        for &x in chunks.remainder() {
                            let result = f32x4::splat_t(token, x)
                                .try_powf_with::<FLAGS>(n)
                                .unwrap()
                                .to_array()[0];
                            assert_eq!(result.to_bits(), single[offset]);
                            offset += 1;
                        }
                        assert_eq!(offset, length);
                    }
                };
            }
            width!(f32x4, 4);
            width!(f32x8, 8);
            #[cfg(feature = "w512")]
            {
                use crate::simd::generic::f32x16;
                width!(f32x16, 16);
            }
        }
    }
    macro_rules! flags {
        ($($f:literal),* $(,)?) => { $(check::<$f>(token);)* };
    }
    flags!(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24,
        25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46,
        48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62,
    );
}

#[test]
fn all_bounded_policies_match_single_vector_and_tail_forms() {
    bounded_widths_and_tails_scalar(archmage::ScalarToken);
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            bounded_widths_and_tails_v3(token);
        }
        #[cfg(feature = "avx512")]
        if let Some(token) = archmage::X64V4xToken::summon() {
            bounded_widths_and_tails_v4x(token);
        }
    }
    #[cfg(target_arch = "aarch64")]
    bounded_widths_and_tails_neon(archmage::NeonToken::summon().unwrap());
    #[cfg(target_arch = "wasm32")]
    bounded_widths_and_tails_wasm128(archmage::Wasm128Token::summon().unwrap());
}
