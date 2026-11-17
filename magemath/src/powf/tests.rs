use super::*;
use crate::{Execution, Portability, SimdMath, Validation};
extern crate std;
#[path = "../../tests/data/powf.rs"]
mod fixtures;

// Keep full frozen fixtures available without making every ordinary test/Miri
// invocation replay the full cross-product. Both tiers use the same oracle and
// assertions; the explicit _exhaustive-tests feature restores every row.
fn oracle_rows() -> impl Iterator<Item = &'static [u32; 5]> {
    fixtures::POWF
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            let [base, exponent, _, _, _] = *row;
            let directed =
                exponent == 0x4019_999a || exponent == 0x3f00_0000 || base == 0x3f80_0000;
            (cfg!(feature = "_exhaustive-tests") || directed || index % 97 == 0).then_some(row)
        })
}

fn options(domain: u32) -> PowfOptions {
    let base = match domain {
        0 => PowfOptions::new(),
        1 => PowfOptions::new().admit_zero(),
        2 => PowfOptions::new().gamma(),
        _ => panic!("unknown test domain"),
    };
    base.caller()
}

fn expected_kind(code: u32) -> PowfErrorKind {
    match code {
        1 => PowfErrorKind::InvalidExponent,
        2 => PowfErrorKind::InvalidBase,
        3 => PowfErrorKind::OutsideApproximationRange,
        _ => panic!("unexpected oracle rejection stage"),
    }
}

const POS_UPGRADE: u128 = PowfOptions::new().upgrade().bits();
const ZERO_UPGRADE: u128 = PowfOptions::new().admit_zero().upgrade().bits();
const GAMMA_UPGRADE: u128 = PowfOptions::new().gamma().upgrade().bits();
const POS_CALLER: u128 = PowfOptions::new().caller().bits();
const ZERO_CALLER: u128 = PowfOptions::new().admit_zero().caller().bits();
const GAMMA_CALLER: u128 = PowfOptions::new().gamma().caller().bits();

// Runtime dispatch exists only in the test harness. Every arm instantiates
// one literal options integer all the way through the public API.
macro_rules! const_options {
    ($domain:expr, $execution:expr, |$cfg:ident| $body:expr) => {{
        match ($domain, $execution) {
            (0, Execution::Upgrade) => {
                const $cfg: u128 = POS_UPGRADE;
                $body
            }
            (1, Execution::Upgrade) => {
                const $cfg: u128 = ZERO_UPGRADE;
                $body
            }
            (2, Execution::Upgrade) => {
                const $cfg: u128 = GAMMA_UPGRADE;
                $body
            }
            (0, Execution::Caller) => {
                const $cfg: u128 = POS_CALLER;
                $body
            }
            (1, Execution::Caller) => {
                const $cfg: u128 = ZERO_CALLER;
                $body
            }
            (2, Execution::Caller) => {
                const $cfg: u128 = GAMMA_CALLER;
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
                options(domain)
                    .runtime()
                    .simd(
                        magetypes::simd::generic::f32x4::from_array_t(token, input),
                        0.5
                    )
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
                for execution in [Execution::Caller, Execution::Upgrade] {
                    let exponent = std::hint::black_box(exponent);
                    let opts = options(domain).execution(execution);
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
                                const_options!(domain, execution, |CFG| {
                                    magetypes::simd::generic::$ty::from_array_t(token, input)
                                        .try_powf_with::<CFG>(exponent)
                                })
                                .unwrap()
                                .to_array()
                                .map(f32::to_bits),
                                expected
                            );
                            assert_eq!(
                                opts.runtime()
                                    .simd(
                                        magetypes::simd::generic::$ty::from_array_t(token, input),
                                        exponent,
                                    )
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
        for &[base, exponent, expected, error, allow_zero] in oracle_rows() {
            let b = f32::from_bits(std::hint::black_box(base));
            let n = f32::from_bits(std::hint::black_box(exponent));
            let base_opts = options(allow_zero);
            macro_rules! width {
                ($ty:ident, $width:literal) => {{
                    let result = base_opts.runtime().simd(
                        magetypes::simd::generic::$ty::from_array_t(token, [b; $width]),
                        n,
                    );
                    for execution in [Execution::Caller, Execution::Upgrade] {
                        let constant = const_options!(allow_zero, execution, |CFG| {
                            magetypes::simd::generic::$ty::from_array_t(token, [b; $width])
                                .try_powf_with::<CFG>(n)
                        });
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
    for &[base, exponent, expected, error, allow_zero] in oracle_rows() {
        let b = f32::from_bits(std::hint::black_box(base));
        let n = f32::from_bits(std::hint::black_box(exponent));
        for execution in [Execution::Caller, Execution::Upgrade] {
            let opts = options(allow_zero).execution(execution);
            let rt = opts.runtime();
            let result = rt.scalar(b, n);
            let constant = const_options!(allow_zero, execution, |CFG| Powf::<CFG>::scalar(b, n));
            assert_eq!(constant.map(f32::to_bits), result.map(f32::to_bits));
            assert_eq!(
                const_options!(allow_zero, execution, |CFG| {
                    Powf::<CFG>::scalar_with_token(archmage::ScalarToken, b, n)
                })
                .map(f32::to_bits),
                result.map(f32::to_bits)
            );
            assert_eq!(
                const_options!(allow_zero, execution, |CFG| Powf::<CFG>::array::<1>([b], n))
                    .map(|v| v.map(f32::to_bits)),
                result.map(|v| [v.to_bits()])
            );
            assert_eq!(
                rt.scalar_with_token(archmage::ScalarToken, b, n)
                    .map(f32::to_bits),
                result.map(f32::to_bits),
            );
            if error == 0 {
                assert_eq!(
                    result.unwrap().to_bits(),
                    expected,
                    "base={base:08x}, exponent={exponent:08x}"
                );
                assert_eq!(
                    rt.array([b; 4], n).unwrap().map(f32::to_bits),
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
    let rt = options(0).runtime();
    let input = [2.0, f32::NAN, f32::from_bits(1), -1.0];
    assert_eq!(
        rt.array(input, 2.4).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::InvalidBase,
            lane_mask: 0b1110
        }
    );
    assert_eq!(
        rt.array([2.0, 4.0, f32::MAX, f32::MIN_POSITIVE], 2.0)
            .unwrap_err(),
        PowfError {
            kind: PowfErrorKind::OutsideApproximationRange,
            lane_mask: 0b1100
        }
    );
    assert_eq!(
        rt.array([2.0; 4], f32::NAN).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::InvalidExponent,
            lane_mask: 0b1111
        }
    );
    assert_eq!(
        rt.array([1.0; 3], 1.0).unwrap_err().kind,
        PowfErrorKind::UnsupportedWidth
    );
    assert_eq!(
        rt.array([], 1.0).unwrap_err().kind,
        PowfErrorKind::UnsupportedWidth
    );
    assert_eq!(
        rt.array([1.0; 32], 1.0).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
}

#[test]
fn caller_scalar_never_upgrades() {
    assert!(matches!(
        dispatch::select(archmage::ScalarToken, Execution::Caller, false),
        dispatch::Plan::Software
    ));
}

#[test]
fn automatic_plan_matches_requested_runtime_context() {
    let plan = dispatch::select(archmage::ScalarToken, Execution::Upgrade, false);
    std::eprintln!("powf upgrade plan: {plan:?}");
    if let Ok(expected) = std::env::var("ARCHMAGE_POWF_EXPECT_PLAN") {
        let matches = match expected.as_str() {
            "software" => matches!(plan, dispatch::Plan::Software),
            #[cfg(all(feature = "std", not(miri)))]
            "std" => matches!(plan, dispatch::Plan::Std),
            #[cfg(target_arch = "x86_64")]
            "v3" => matches!(plan, dispatch::Plan::V3(_)),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            "v4" => matches!(plan, dispatch::Plan::V4(_)),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            "v4x" => matches!(plan, dispatch::Plan::V4x(_)),
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
                            let rt = options($domain).runtime();
                            let constant = const_options!($domain, Execution::Caller, |CFG| {
                                magetypes::simd::generic::$ty::from_array_t(token, input)
                                    .try_powf_with::<CFG>(2.4)
                            });
                            assert_eq!(
                                constant.unwrap_err(),
                                PowfError {
                                    kind: PowfErrorKind::InvalidBase,
                                    lane_mask: 1 << lane
                                }
                            );
                            let constant = const_options!($domain, Execution::Caller, |CFG| {
                                magetypes::simd::generic::$ty::from_array_t(token, [1.0; $n])
                                    .try_powf_with::<CFG>(nan)
                            });
                            assert_eq!(
                                constant.unwrap_err(),
                                PowfError {
                                    kind: PowfErrorKind::InvalidExponent,
                                    lane_mask: ((1_u32 << $n) - 1) as u16
                                }
                            );
                            assert_eq!(
                                rt.simd(
                                    magetypes::simd::generic::$ty::from_array_t(token, input),
                                    2.4,
                                )
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
                                rt.simd(
                                    magetypes::simd::generic::$ty::from_array_t(token, [1.0; $n]),
                                    nan,
                                )
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
    const GAMMA_C: u128 = PowfOptions::new().gamma().caller().bits();
    const CALLER_DEFAULT: u128 = PowfOptions::new().caller().bits();
    assert_eq!(
        Powf::<GAMMA_C>::array::<0>([], f32::NAN).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(
        Powf::<GAMMA_C>::array::<3>([1.0; 3], f32::NAN).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(
        Powf::<GAMMA_C>::array::<32>([1.0; 32], f32::NAN).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(
        Powf::<GAMMA_C>::array::<4>([f32::NAN; 4], f32::NAN).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::InvalidExponent,
            lane_mask: 15
        }
    );
    assert_eq!(
        Powf::<CALLER_DEFAULT>::array::<4>([2.0, f32::NAN, f32::from_bits(1), -1.0], 2.4)
            .unwrap_err(),
        PowfError {
            kind: PowfErrorKind::InvalidBase,
            lane_mask: 14
        }
    );
    assert_eq!(
        Powf::<CALLER_DEFAULT>::array::<4>([2.0, 4.0, f32::MAX, f32::MIN_POSITIVE], 2.0)
            .unwrap_err(),
        PowfError {
            kind: PowfErrorKind::OutsideApproximationRange,
            lane_mask: 12
        }
    );
}

#[test]
fn const_builder_encodes_each_supported_configuration() {
    for base in [
        PowfOptions::new(),
        PowfOptions::new().admit_zero(),
        PowfOptions::new().gamma(),
        PowfOptions::new().positive_gamma(),
        PowfOptions::new()
            .base_range(1.0e-4, 65536.0)
            .exponent_range(-4.0, 4.0)
            .clamp_output(),
        PowfOptions::new().full_domain().highp(),
    ] {
        for execution in [Execution::Upgrade, Execution::Caller] {
            for portability in [Portability::BitExact, Portability::BackendDependent] {
                for validation in [Validation::Checked, Validation::AssumeValid] {
                    let opts = base
                        .execution(execution)
                        .portability(portability)
                        .validation(validation);
                    if !opts.is_valid() {
                        continue;
                    }
                    let bits = opts.bits();
                    assert_eq!(PowfOptions::from_bits(bits), Some(opts));
                }
            }
        }
    }
    assert_eq!(PowfOptions::from_bits(0), None);
    assert_eq!(
        PowfOptions::from_bits(crate::expf::ExpfOptions::new().bits()),
        None
    );
    assert_eq!(
        PowfOptions::from_bits(crate::logf::LogfOptions::new().bits()),
        None
    );
    assert_eq!(
        PowfOptions::from_bits(crate::cbrtf::CbrtfOptions::new().bits()),
        None
    );
}

#[test]
fn new_policy_presets_and_invalid_runtime_combinations() {
    for opts in [
        PowfOptions::new().gamma().midp().bit_exact(),
        PowfOptions::new().gamma().lowp().bit_exact(),
        PowfOptions::new().gamma().midp().backend_dependent(),
        PowfOptions::new().gamma().lowp().backend_dependent(),
    ] {
        let rt = opts.runtime();
        assert_eq!(
            rt.array([0., 1., 4., 16.], 0.).unwrap()[0].to_bits(),
            1f32.to_bits()
        );
        let error = rt
            .array([0., f32::NAN, -0., f32::from_bits(1)], 2.4)
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
        PowfOptions::new().full_domain(),
        PowfOptions::new().lowp().full_domain(),
    ] {
        assert_eq!(
            opts.runtime().scalar(1., 2.4).unwrap_err(),
            PowfError {
                kind: PowfErrorKind::UnsupportedOptions,
                lane_mask: 0
            }
        );
    }
    #[cfg(not(feature = "std"))]
    assert_eq!(
        PowfOptions::new()
            .full_domain()
            .highp()
            .backend_dependent()
            .runtime()
            .scalar(1., 2.4)
            .unwrap_err()
            .kind,
        PowfErrorKind::UnsupportedOptions
    );
}

#[test]
fn lowp_admitted_zero_survives_huge_exponent_in_both_const_modes() {
    const PORTABLE: u128 = PowfOptions::new().admit_zero().lowp().caller().bits();
    const BACKEND: u128 = PowfOptions::new()
        .admit_zero()
        .lowp()
        .backend_dependent()
        .caller()
        .bits();
    for n in [1_073_741_824., f32::MAX] {
        assert_eq!(Powf::<PORTABLE>::scalar(0., n).unwrap().to_bits(), 0);
        assert_eq!(Powf::<BACKEND>::scalar(0., n).unwrap().to_bits(), 0);
    }
}

#[test]
#[cfg(feature = "std")]
fn std_highp_full_domain_and_restricted_masks() {
    const STD: u128 = PowfOptions::new()
        .full_domain()
        .highp()
        .backend_dependent()
        .bits();
    let result = Powf::<STD>::array::<4>([-2., -0., f32::INFINITY, f32::NAN], 3.).unwrap();
    // Std power has unspecified precision, including under Miri. Only its
    // admitted domain and sign/classification are the HIGHP contract here.
    assert!(result[0].is_normal() && result[0].is_sign_negative());
    assert_eq!(result[1].to_bits(), (-0.0f32).to_bits());
    assert_eq!(result[2], f32::INFINITY);
    assert!(result[3].is_nan());
    assert_eq!(
        Powf::<STD>::array::<4>([f32::NAN, 0., -0., f32::INFINITY], 0.).unwrap(),
        [1.; 4]
    );
    assert_eq!(Powf::<STD>::scalar(0., -1.).unwrap(), f32::INFINITY);
    let tiny = Powf::<STD>::scalar(f32::from_bits(1), 1.).unwrap();
    assert!(tiny.is_subnormal() || tiny == 0.);
    assert_eq!(
        Powf::<STD>::array::<3>([1.; 3], 2.).unwrap_err().kind,
        PowfErrorKind::UnsupportedWidth
    );
    let rt = PowfOptions::new().highp().backend_dependent().runtime();
    assert_eq!(
        rt.array([2., -1., f32::from_bits(1), f32::NAN], 2.)
            .unwrap_err(),
        PowfError {
            kind: PowfErrorKind::InvalidBase,
            lane_mask: 14
        }
    );
    assert_eq!(
        rt.array([2., 4., f32::MAX, f32::MIN_POSITIVE], 2.)
            .unwrap_err(),
        PowfError {
            kind: PowfErrorKind::OutsideApproximationRange,
            lane_mask: 12
        }
    );
}

#[test]
fn positive_gamma_preserves_nonzero_graph_and_rejects_zero() {
    fn check<const PG: u128, const G: u128>(options: PowfOptions) {
        assert_eq!(options.bits(), PG);
        let is_std_highp = options.is_highp() && options.is_backend();
        let is_checked = !options.assume_valid();
        let rt = options.runtime();
        let compare = |left: [f32; 4], right: [f32; 4]| {
            if is_std_highp {
                // This provider delegates to std::powf, whose precision is
                // unspecified and can vary between calls under Miri. This
                // fixture checks ordinary results, not a portable std graph.
                for (a, b) in left.into_iter().zip(right) {
                    assert!(a.is_normal() && b.is_normal());
                    assert!((a - b).abs() <= a.abs() * 0.000_01);
                }
            } else {
                assert_eq!(
                    left.map(f32::to_bits),
                    right.map(f32::to_bits),
                    "options={options:?}"
                );
            }
        };
        for n in [0., -0., 0.5, 2.4, 4.] {
            let input = [1.0 / 65_536.0, 0.75, 1., 65_536.0];
            let got = Powf::<PG>::array::<4>(input, n).unwrap();
            let gamma = Powf::<G>::array::<4>(input, n).unwrap();
            compare(got, gamma);
            for (x, y) in input.into_iter().zip(got) {
                compare(
                    [Powf::<PG>::scalar_with_token(archmage::ScalarToken, x, n).unwrap(); 4],
                    [y; 4],
                );
            }
            compare(got, rt.array(input, n).unwrap());
            if is_checked {
                for x in [0., -0., f32::from_bits(1), f32::NAN, f32::INFINITY] {
                    assert_eq!(
                        Powf::<PG>::scalar_with_token(archmage::ScalarToken, x, n).unwrap_err(),
                        PowfError {
                            kind: PowfErrorKind::InvalidBase,
                            lane_mask: 1
                        }
                    );
                    assert_eq!(
                        Powf::<PG>::array::<4>([1., x, 1., x], n).unwrap_err(),
                        PowfError {
                            kind: PowfErrorKind::InvalidBase,
                            lane_mask: 10
                        }
                    );
                }
            }
        }
    }

    macro_rules! pair {
        ($prec:ident, $exec:ident, $port:ident, $val:ident) => {{
            const PG_OPTS: PowfOptions = PowfOptions::new()
                .positive_gamma()
                .$prec()
                .$exec()
                .$port()
                .$val();
            const G_OPTS: PowfOptions = PowfOptions::new().gamma().$prec().$exec().$port().$val();
            check::<{ PG_OPTS.bits() }, { G_OPTS.bits() }>(PG_OPTS);
        }};
    }

    macro_rules! sweep_exec_val {
        ($prec:ident, $port:ident) => {{
            pair!($prec, upgrade, $port, checked);
            pair!($prec, caller, $port, checked);
            pair!($prec, upgrade, $port, unchecked);
            pair!($prec, caller, $port, unchecked);
        }};
    }

    sweep_exec_val!(midp, bit_exact);
    sweep_exec_val!(midp, backend_dependent);
    sweep_exec_val!(lowp, bit_exact);
    sweep_exec_val!(lowp, backend_dependent);
    sweep_exec_val!(highp, bit_exact);
    #[cfg(feature = "std")]
    sweep_exec_val!(highp, backend_dependent);
}
