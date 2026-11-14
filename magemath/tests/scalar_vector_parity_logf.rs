//! Exhaustive const-generic scalar vs vector parity for `magemath::logf`
//! (`log2`, `ln`, `log10`, `log1p`) across all 32 valid option encodings and all tokens.
#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/scalar_vector_parity.rs"]
mod parity;

use magemath::logf::{
    self as l, LogfDomain, LogfError, LogfErrorKind, LogfExecution, LogfOptions, LogfPortability,
    LogfPrecision, LogfValidation,
};
use parity::{assert_array_res_eq, assert_f32_eq, assert_scalar_res_eq};

const LOG_POSITIVE_VALID: [f32; 16] = [
    1.175_494_4e-38,
    1.0e-20,
    1.0 / 65_536.0,
    0.125,
    0.5,
    0.707_106_77,
    0.999_999_9,
    1.0,
    1.000_000_1,
    1.5,
    2.0,
    core::f32::consts::E,
    10.0,
    256.0,
    1.0e20,
    f32::MAX,
];

const LOG_SPECIALS_EXTRA: [f32; 8] = [
    0.0,
    -0.0,
    -1.0,
    -100.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::from_bits(1),
];

const LOG1P_POSITIVE_VALID: [f32; 16] = [
    -0.999_999,
    -0.75,
    -0.5,
    -0.125,
    -1.0e-7,
    f32::from_bits(0x8000_0001),
    -0.0,
    0.0,
    f32::from_bits(1),
    1.0e-7,
    0.125,
    0.5,
    1.0,
    3.0,
    100.0,
    1.0e38,
];

const LOG1P_SPECIALS_EXTRA: [f32; 8] = [
    -1.0,
    -2.0,
    -100.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::from_bits(0x7fc0_5678),
    -1.000_001,
];

const fn logf_options_from_bits(f: u32) -> LogfOptions {
    LogfOptions::new()
        .with_domain(if f & 1 == 0 {
            LogfDomain::Positive
        } else {
            LogfDomain::Specials
        })
        .with_execution(if f & 4 == 0 {
            LogfExecution::Automatic
        } else {
            LogfExecution::Caller
        })
        .with_precision(if f & 8 == 0 {
            LogfPrecision::Midp
        } else {
            LogfPrecision::Lowp
        })
        .with_portability(if f & 16 == 0 {
            LogfPortability::BitExact
        } else {
            LogfPortability::BackendDependent
        })
        .with_validation(if f & 32 == 0 {
            LogfValidation::Checked
        } else {
            LogfValidation::AssumeValid
        })
}

macro_rules! all_32_flags {
    ($m:ident $(, $arg:expr)*) => {
        $m::<0>($($arg),*);
        $m::<1>($($arg),*);
        $m::<4>($($arg),*);
        $m::<5>($($arg),*);
        $m::<8>($($arg),*);
        $m::<9>($($arg),*);
        $m::<12>($($arg),*);
        $m::<13>($($arg),*);
        $m::<16>($($arg),*);
        $m::<17>($($arg),*);
        $m::<20>($($arg),*);
        $m::<21>($($arg),*);
        $m::<24>($($arg),*);
        $m::<25>($($arg),*);
        $m::<28>($($arg),*);
        $m::<29>($($arg),*);
        $m::<32>($($arg),*);
        $m::<33>($($arg),*);
        $m::<36>($($arg),*);
        $m::<37>($($arg),*);
        $m::<40>($($arg),*);
        $m::<41>($($arg),*);
        $m::<44>($($arg),*);
        $m::<45>($($arg),*);
        $m::<48>($($arg),*);
        $m::<49>($($arg),*);
        $m::<52>($($arg),*);
        $m::<53>($($arg),*);
        $m::<56>($($arg),*);
        $m::<57>($($arg),*);
        $m::<60>($($arg),*);
        $m::<61>($($arg),*);
    };
}

macro_rules! check_standalone_log_op {
    (
        $fn_name:ident,
        $try_default:path,
        $try_with:ident,
        $try_array_with:ident,
        $try_runtime:ident,
        $try_array_runtime:ident,
        $pos_valid:ident,
        $specials_extra:ident,
        $rejected:expr
    ) => {
        mod $fn_name {
            use super::*;

            #[inline(never)]
            fn rt_scalar(x: f32, opts: LogfOptions) -> Result<f32, LogfError> {
                l::$try_runtime(x, opts)
            }
            #[inline(never)]
            fn rt_arr<const N: usize>(
                xs: [f32; N],
                opts: LogfOptions,
            ) -> Result<[f32; N], LogfError> {
                l::$try_array_runtime::<N>(xs, opts)
            }
            #[inline(never)]
            fn c_scalar<const F: u32>(x: f32) -> Result<f32, LogfError> {
                l::$try_with::<F>(x)
            }
            #[inline(never)]
            fn c_arr<const F: u32, const N: usize>(xs: [f32; N]) -> Result<[f32; N], LogfError> {
                l::$try_array_with::<F, N>(xs)
            }
            #[inline(never)]
            fn def_scalar(x: f32) -> Result<f32, LogfError> {
                $try_default(x)
            }

            #[inline(never)]
            pub fn run<const F: u32>() {
                let opts = logf_options_from_bits(F);
                assert_eq!(opts.bits(), F);

                for &x in &$pos_valid {
                    let s = c_scalar::<F>(x);
                    assert!(s.is_ok(), "expected Ok on Positive valid input {x:?}");
                    if F == 0 {
                        assert_scalar_res_eq(def_scalar(x), s, "default scalar vs _with::<0>");
                    }
                    assert_scalar_res_eq(
                        c_arr::<F, 1>([x]).map(|a| a[0]),
                        s,
                        "array_with::<F, 1> vs scalar _with",
                    );
                    assert_scalar_res_eq(rt_scalar(x, opts), s, "runtime scalar vs const _with");
                    assert_scalar_res_eq(
                        rt_arr::<1>([x], opts).map(|a| a[0]),
                        s,
                        "array_runtime::<1> vs const _with",
                    );
                }

                if F & 1 != 0 {
                    for &x in &$specials_extra {
                        let s = c_scalar::<F>(x);
                        assert!(s.is_ok(), "expected Ok on Specials extra input {x:?}");
                        assert_scalar_res_eq(
                            c_arr::<F, 1>([x]).map(|a| a[0]),
                            s,
                            "Specials array_with::<F, 1> vs _with",
                        );
                        assert_scalar_res_eq(
                            rt_scalar(x, opts),
                            s,
                            "Specials runtime scalar vs _with",
                        );
                        assert_scalar_res_eq(
                            rt_arr::<1>([x], opts).map(|a| a[0]),
                            s,
                            "Specials array_runtime::<1> vs _with",
                        );
                    }
                }

                for &pack in $pos_valid.as_chunks::<4>().0 {
                    let a_with = c_arr::<F, 4>(pack);
                    let a_rt = rt_arr::<4>(pack, opts);
                    assert_array_res_eq(a_rt, a_with, "array_runtime::<4> vs array_with::<F, 4>");
                    let arr = a_with.unwrap();
                    for i in 0..4 {
                        assert_f32_eq(
                            arr[i],
                            c_scalar::<F>(pack[i]).unwrap(),
                            "array_with::<F, 4> lane vs scalar _with",
                        );
                    }
                }
                for &pack in $pos_valid.as_chunks::<8>().0 {
                    let a_with = c_arr::<F, 8>(pack);
                    let a_rt = rt_arr::<8>(pack, opts);
                    assert_array_res_eq(a_rt, a_with, "array_runtime::<8> vs array_with::<F, 8>");
                    let arr = a_with.unwrap();
                    for i in 0..8 {
                        assert_f32_eq(
                            arr[i],
                            c_scalar::<F>(pack[i]).unwrap(),
                            "array_with::<F, 8> lane vs scalar _with",
                        );
                    }
                }
                #[cfg(feature = "w512")]
                {
                    let a_with = c_arr::<F, 16>($pos_valid);
                    let a_rt = rt_arr::<16>($pos_valid, opts);
                    assert_array_res_eq(a_rt, a_with, "array_runtime::<16> vs array_with::<F, 16>");
                    let arr = a_with.unwrap();
                    for i in 0..16 {
                        assert_f32_eq(
                            arr[i],
                            c_scalar::<F>($pos_valid[i]).unwrap(),
                            "array_with::<F, 16> lane vs scalar _with",
                        );
                    }
                }

                let bad_w_const = c_arr::<F, 3>([1.0; 3]).unwrap_err();
                assert_eq!(bad_w_const.kind, LogfErrorKind::UnsupportedWidth);
                assert_eq!(bad_w_const.lane_mask, 0);
                let bad_w_rt = rt_arr::<3>([1.0; 3], opts).unwrap_err();
                assert_eq!(bad_w_rt, bad_w_const);

                if F & 32 == 0 && F & 1 == 0 {
                    let bads: &[f32] = &$rejected;
                    for &bad in bads {
                        let err = c_scalar::<F>(bad).unwrap_err();
                        assert_eq!(err.kind, LogfErrorKind::InvalidInput);
                        assert_eq!(err.lane_mask, 1);
                        if F == 0 {
                            assert_eq!(def_scalar(bad).unwrap_err(), err);
                        }
                        assert_eq!(c_arr::<F, 1>([bad]).unwrap_err(), err);
                        assert_eq!(rt_scalar(bad, opts).unwrap_err(), err);
                        assert_eq!(rt_arr::<1>([bad], opts).unwrap_err(), err);
                    }
                    let mixed4 = [1.0, bads[0], 0.5, bads[1]];
                    let err4 = c_arr::<F, 4>(mixed4).unwrap_err();
                    assert_eq!(err4.kind, LogfErrorKind::InvalidInput);
                    assert_eq!(err4.lane_mask, 0b1010);
                    assert_eq!(rt_arr::<4>(mixed4, opts).unwrap_err(), err4);
                }
            }
        }
    };
}

check_standalone_log_op!(
    standalone_log2,
    l::try_log2,
    try_log2_with,
    try_log2_array_with,
    try_log2_runtime,
    try_log2_array_runtime,
    LOG_POSITIVE_VALID,
    LOG_SPECIALS_EXTRA,
    [
        0.0,
        -0.0,
        -1.0,
        f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN
    ]
);

check_standalone_log_op!(
    standalone_ln,
    l::try_ln,
    try_ln_with,
    try_ln_array_with,
    try_ln_runtime,
    try_ln_array_runtime,
    LOG_POSITIVE_VALID,
    LOG_SPECIALS_EXTRA,
    [
        0.0,
        -0.0,
        -1.0,
        f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN
    ]
);

check_standalone_log_op!(
    standalone_log10,
    l::try_log10,
    try_log10_with,
    try_log10_array_with,
    try_log10_runtime,
    try_log10_array_runtime,
    LOG_POSITIVE_VALID,
    LOG_SPECIALS_EXTRA,
    [
        0.0,
        -0.0,
        -1.0,
        f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN
    ]
);

check_standalone_log_op!(
    standalone_log1p,
    l::try_log1p,
    try_log1p_with,
    try_log1p_array_with,
    try_log1p_runtime,
    try_log1p_array_runtime,
    LOG1P_POSITIVE_VALID,
    LOG1P_SPECIALS_EXTRA,
    [
        -1.0,
        -2.0,
        -100.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN
    ]
);

#[test]
fn standalone_logf_scalar_array_and_runtime_match_across_all_32_configs() {
    use standalone_ln::run as run_ln;
    use standalone_log1p::run as run_log1p;
    use standalone_log2::run as run_log2;
    use standalone_log10::run as run_log10;
    all_32_flags!(run_log2);
    all_32_flags!(run_ln);
    all_32_flags!(run_log10);
    all_32_flags!(run_log1p);
}

macro_rules! define_log_token_op {
    (
        $fn_name:ident,
        $tok_ty:ty,
        $vec4:ident,
        $vec8:ident,
        $vec16:ident,
        $tail_fn:ident,
        $with_fn:ident,
        $pos_valid:ident,
        $specials_extra:ident,
        $bad0:expr,
        $bad1:expr,
        $has_x16:tt
    ) => {
        #[inline(never)]
        fn $fn_name<const F: u32>(token: $tok_ty) {
            #[inline(never)]
            fn eval4<const FLAGS: u32>(t: $tok_ty, xs: [f32; 4]) -> Result<[f32; 4], LogfError> {
                use magetypes::simd::generic::f32x4;
                l::$vec4::<FLAGS, _>(t, f32x4::from_array_t(t, xs)).map(|v| v.to_array())
            }
            #[inline(never)]
            fn eval8<const FLAGS: u32>(t: $tok_ty, xs: [f32; 8]) -> Result<[f32; 8], LogfError> {
                use magetypes::simd::generic::f32x8;
                l::$vec8::<FLAGS, _>(t, f32x8::from_array_t(t, xs)).map(|v| v.to_array())
            }
            #[inline(never)]
            fn eval_tail<const FLAGS: u32>(t: $tok_ty, x: f32) -> Result<f32, LogfError> {
                l::$tail_fn::<FLAGS, _>(t, x)
            }
            #[inline(never)]
            fn eval_with<const FLAGS: u32>(x: f32) -> Result<f32, LogfError> {
                l::$with_fn::<FLAGS>(x)
            }

            for &xs in $pos_valid.as_chunks::<4>().0 {
                let v4 = eval4::<F>(token, xs).unwrap();
                for i in 0..4 {
                    let tail = eval_tail::<F>(token, xs[i]).unwrap();
                    assert_f32_eq(v4[i], tail, "logf x4 vs _with_token");
                    if F & 16 == 0 {
                        assert_f32_eq(
                            tail,
                            eval_with::<F>(xs[i]).unwrap(),
                            "logf BitExact _with_token vs _with",
                        );
                    }
                }
            }
            for &xs in $pos_valid.as_chunks::<8>().0 {
                let v8 = eval8::<F>(token, xs).unwrap();
                for i in 0..8 {
                    let tail = eval_tail::<F>(token, xs[i]).unwrap();
                    assert_f32_eq(v8[i], tail, "logf x8 vs _with_token");
                }
            }
            define_log_token_op!(@x16 $has_x16, $tok_ty, token, F, $pos_valid, $vec16, eval_tail);

            if F & 1 != 0 {
                let v8 = eval8::<F>(token, $specials_extra).unwrap();
                for i in 0..8 {
                    let tail = eval_tail::<F>(token, $specials_extra[i]).unwrap();
                    assert_f32_eq(v8[i], tail, "logf Specials x8 vs _with_token");
                    if F & 16 == 0 {
                        assert_f32_eq(
                            tail,
                            eval_with::<F>($specials_extra[i]).unwrap(),
                            "logf Specials BitExact _with_token vs _with",
                        );
                    }
                }
            } else if F & 32 == 0 {
                let err_s = eval_tail::<F>(token, $bad0).unwrap_err();
                assert_eq!(err_s.kind, LogfErrorKind::InvalidInput);
                assert_eq!(err_s.lane_mask, 1);
                let mixed = [0.5, $bad0, 1.0, $bad1];
                let err_v = eval4::<F>(token, mixed).unwrap_err();
                assert_eq!(err_v.kind, LogfErrorKind::InvalidInput);
                assert_eq!(err_v.lane_mask, 0b1010);
            }
        }
    };
    (@x16 yes, $tok_ty:ty, $token:ident, $f:ident, $arr:ident, $vec16:ident, $eval_tail:ident) => {
        #[cfg(feature = "w512")]
        {
            #[inline(never)]
            fn eval16<const FLAGS: u32>(t: $tok_ty, xs: [f32; 16]) -> Result<[f32; 16], LogfError> {
                use magetypes::simd::generic::f32x16;
                l::$vec16::<FLAGS, _>(t, f32x16::from_array_t(t, xs)).map(|v| v.to_array())
            }
            let v16 = eval16::<$f>($token, $arr).unwrap();
            for i in 0..16 {
                let tail = $eval_tail::<$f>($token, $arr[i]).unwrap();
                assert_f32_eq(v16[i], tail, "logf x16 vs _with_token");
            }
        }
    };
    (@x16 no, $tok_ty:ty, $token:ident, $f:ident, $arr:ident, $vec16:ident, $eval_tail:ident) => {};
}

#[archmage::magetypes(
    v4x(cfg(avx512)),
    v4(cfg(avx512)),
    v3,
    neon,
    wasm128_relaxed_fma,
    wasm128,
    scalar
)]
fn check_token(token: Token) {
    define_log_token_op!(
        tok_log2,
        Token,
        try_log2_f32x4_with,
        try_log2_f32x8_with,
        try_log2_f32x16_with,
        try_log2_with_token,
        try_log2_with,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        0.0,
        f32::NAN,
        yes
    );
    define_log_token_op!(
        tok_ln,
        Token,
        try_ln_f32x4_with,
        try_ln_f32x8_with,
        try_ln_f32x16_with,
        try_ln_with_token,
        try_ln_with,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        -0.0,
        -1.0,
        yes
    );
    define_log_token_op!(
        tok_log10,
        Token,
        try_log10_f32x4_with,
        try_log10_f32x8_with,
        try_log10_f32x16_with,
        try_log10_with_token,
        try_log10_with,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        f32::from_bits(1),
        f32::INFINITY,
        yes
    );
    define_log_token_op!(
        tok_log1p,
        Token,
        try_log1p_f32x4_with,
        try_log1p_f32x8_with,
        try_log1p_f32x16_with,
        try_log1p_with_token,
        try_log1p_with,
        LOG1P_POSITIVE_VALID,
        LOG1P_SPECIALS_EXTRA,
        -1.0,
        -2.0,
        yes
    );

    all_32_flags!(tok_log2, token);
    all_32_flags!(tok_ln, token);
    all_32_flags!(tok_log10, token);
    all_32_flags!(tok_log1p, token);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn check_token_fp16(token: archmage::Avx512Fp16Token) {
    define_log_token_op!(
        fp16_log2,
        archmage::Avx512Fp16Token,
        try_log2_f32x4_with,
        try_log2_f32x8_with,
        try_log2_f32x16_with,
        try_log2_with_token,
        try_log2_with,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        0.0,
        f32::NAN,
        no
    );
    define_log_token_op!(
        fp16_ln,
        archmage::Avx512Fp16Token,
        try_ln_f32x4_with,
        try_ln_f32x8_with,
        try_ln_f32x16_with,
        try_ln_with_token,
        try_ln_with,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        -0.0,
        -1.0,
        no
    );
    define_log_token_op!(
        fp16_log10,
        archmage::Avx512Fp16Token,
        try_log10_f32x4_with,
        try_log10_f32x8_with,
        try_log10_f32x16_with,
        try_log10_with_token,
        try_log10_with,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        f32::from_bits(1),
        f32::INFINITY,
        no
    );
    define_log_token_op!(
        fp16_log1p,
        archmage::Avx512Fp16Token,
        try_log1p_f32x4_with,
        try_log1p_f32x8_with,
        try_log1p_f32x16_with,
        try_log1p_with_token,
        try_log1p_with,
        LOG1P_POSITIVE_VALID,
        LOG1P_SPECIALS_EXTRA,
        -1.0,
        -2.0,
        no
    );

    all_32_flags!(fp16_log2, token);
    all_32_flags!(fp16_ln, token);
    all_32_flags!(fp16_log10, token);
    all_32_flags!(fp16_log1p, token);
}

#[test]
fn token_logf_scalar_and_vector_widths_match_across_all_32_configs() {
    run_all_summonable_tokens!();

    for &x in &LOG1P_POSITIVE_VALID {
        assert_f32_eq(
            l::try_log1p_with::<20>(x).unwrap(),
            l::try_log1p_with_token::<20, _>(archmage::ScalarToken, x).unwrap(),
            "log1p BackendDependent Caller standalone vs ScalarToken",
        );
        assert_f32_eq(
            l::try_log1p_with::<28>(x).unwrap(),
            l::try_log1p_with_token::<28, _>(archmage::ScalarToken, x).unwrap(),
            "log1p BackendDependent Lowp Caller standalone vs ScalarToken",
        );
    }
}
