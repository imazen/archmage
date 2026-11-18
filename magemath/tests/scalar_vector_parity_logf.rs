//! Exhaustive const-generic scalar vs vector parity for `magemath::logf`
//! (`log2`, `ln`, `log10`) and `magemath::log1pf` (`log1p`) across all 32
//! valid option encodings and all tokens.
#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/scalar_vector_parity.rs"]
mod parity;

use magemath::{
    Execution, Portability, SimdMath, Validation,
    log1pf::{self as l1p, Log1pf, Log1pfError, Log1pfErrorKind, Log1pfOptions, Log1pfPrecision},
    logf::{self as l, Logf, LogfError, LogfErrorKind, LogfOptions, LogfPrecision},
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
    -f32::MIN_POSITIVE,
    -0.0,
    0.0,
    f32::MIN_POSITIVE,
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
    f32::from_bits(1),
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::from_bits(0x8000_0001),
    -1.000_001,
];

const fn logf_options_from_bits(f: u32) -> LogfOptions {
    let base = if f & 1 == 0 {
        LogfOptions::new()
    } else {
        LogfOptions::new().full_domain()
    };
    base.execution(if f & 4 == 0 {
        Execution::Upgrade
    } else {
        Execution::Caller
    })
    .precision(if f & 8 == 0 {
        LogfPrecision::Midp
    } else {
        LogfPrecision::Lowp
    })
    .portability(if f & 16 == 0 {
        Portability::BitExact
    } else {
        Portability::BackendDependent
    })
    .validation(if f & 32 == 0 {
        Validation::Checked
    } else {
        Validation::AssumeValid
    })
}

const fn log1pf_options_from_bits(f: u32) -> Log1pfOptions {
    let base = if f & 1 == 0 {
        Log1pfOptions::new()
    } else {
        Log1pfOptions::new().full_domain()
    };
    base.execution(if f & 4 == 0 {
        Execution::Upgrade
    } else {
        Execution::Caller
    })
    .precision(if f & 8 == 0 {
        Log1pfPrecision::Midp
    } else {
        Log1pfPrecision::Lowp
    })
    .portability(if f & 16 == 0 {
        Portability::BitExact
    } else {
        Portability::BackendDependent
    })
    .validation(if f & 32 == 0 {
        Validation::Checked
    } else {
        Validation::AssumeValid
    })
}

macro_rules! all_32_flags {
    ($m:ident, $builder:ident $(, $arg:expr)*) => {
        macro_rules! call_one {
            ($f:literal) => {
                $m::<$f, { $builder($f).bits() }>($($arg),*);
            };
        }
        call_one!(0);
        call_one!(1);
        call_one!(4);
        call_one!(5);
        call_one!(8);
        call_one!(9);
        call_one!(12);
        call_one!(13);
        call_one!(16);
        call_one!(17);
        call_one!(20);
        call_one!(21);
        call_one!(24);
        call_one!(25);
        call_one!(28);
        call_one!(29);
        call_one!(32);
        call_one!(33);
        call_one!(36);
        call_one!(37);
        call_one!(40);
        call_one!(41);
        call_one!(44);
        call_one!(45);
        call_one!(48);
        call_one!(49);
        call_one!(52);
        call_one!(53);
        call_one!(56);
        call_one!(57);
        call_one!(60);
        call_one!(61);
    };
}

macro_rules! check_standalone_log_op {
    (
        $fn_name:ident,
        $op_ty:ident,
        $opts_ty:ty,
        $err_ty:ty,
        $err_kind:ident,
        $builder:ident,
        $try_default:path,
        $scalar_m:ident,
        $array_m:ident,
        $pos_valid:ident,
        $specials_extra:ident,
        $rejected:expr
    ) => {
        mod $fn_name {
            use super::*;

            #[inline(never)]
            fn rt_scalar(x: f32, opts: $opts_ty) -> Result<f32, $err_ty> {
                opts.runtime().$scalar_m(x)
            }
            #[inline(never)]
            fn rt_arr<const N: usize>(xs: [f32; N], opts: $opts_ty) -> Result<[f32; N], $err_ty> {
                opts.runtime().$array_m::<N>(xs)
            }
            #[inline(never)]
            fn c_scalar<const CFG: u128>(x: f32) -> Result<f32, $err_ty> {
                $op_ty::<CFG>::$scalar_m(x)
            }
            #[inline(never)]
            fn c_arr<const CFG: u128, const N: usize>(xs: [f32; N]) -> Result<[f32; N], $err_ty> {
                $op_ty::<CFG>::$array_m::<N>(xs)
            }
            #[inline(never)]
            fn def_scalar(x: f32) -> Result<f32, $err_ty> {
                $try_default(x)
            }

            #[inline(never)]
            pub fn run<const F: u32, const CFG: u128>() {
                let opts = $builder(F);
                assert_eq!(<$opts_ty>::from_bits(CFG), Some(opts));

                for &x in &$pos_valid {
                    let s = c_scalar::<CFG>(x);
                    assert!(s.is_ok(), "expected Ok on Positive valid input {x:?}");
                    if F == 0 {
                        assert_scalar_res_eq(def_scalar(x), s, "default scalar vs DEFAULT");
                    }
                    assert_scalar_res_eq(
                        c_arr::<CFG, 1>([x]).map(|a| a[0]),
                        s,
                        "array::<1> vs scalar",
                    );
                    assert_scalar_res_eq(rt_scalar(x, opts), s, "runtime scalar vs const");
                    assert_scalar_res_eq(
                        rt_arr::<1>([x], opts).map(|a| a[0]),
                        s,
                        "runtime array::<1> vs const",
                    );
                }

                if F & 1 != 0 {
                    for &x in &$specials_extra {
                        let s = c_scalar::<CFG>(x);
                        assert!(s.is_ok(), "expected Ok on Specials extra input {x:?}");
                        assert_scalar_res_eq(
                            c_arr::<CFG, 1>([x]).map(|a| a[0]),
                            s,
                            "Specials array::<1> vs scalar",
                        );
                        assert_scalar_res_eq(
                            rt_scalar(x, opts),
                            s,
                            "Specials runtime scalar vs const",
                        );
                        assert_scalar_res_eq(
                            rt_arr::<1>([x], opts).map(|a| a[0]),
                            s,
                            "Specials runtime array::<1> vs const",
                        );
                    }
                }

                for &pack in $pos_valid.as_chunks::<4>().0 {
                    let a_with = c_arr::<CFG, 4>(pack);
                    let a_rt = rt_arr::<4>(pack, opts);
                    assert_array_res_eq(a_rt, a_with, "runtime array::<4> vs const array::<4>");
                    let arr = a_with.unwrap();
                    for i in 0..4 {
                        assert_f32_eq(
                            arr[i],
                            c_scalar::<CFG>(pack[i]).unwrap(),
                            "array::<4> lane vs scalar",
                        );
                    }
                }
                for &pack in $pos_valid.as_chunks::<8>().0 {
                    let a_with = c_arr::<CFG, 8>(pack);
                    let a_rt = rt_arr::<8>(pack, opts);
                    assert_array_res_eq(a_rt, a_with, "runtime array::<8> vs const array::<8>");
                    let arr = a_with.unwrap();
                    for i in 0..8 {
                        assert_f32_eq(
                            arr[i],
                            c_scalar::<CFG>(pack[i]).unwrap(),
                            "array::<8> lane vs scalar",
                        );
                    }
                }
                #[cfg(feature = "w512")]
                {
                    let a_with = c_arr::<CFG, 16>($pos_valid);
                    let a_rt = rt_arr::<16>($pos_valid, opts);
                    assert_array_res_eq(a_rt, a_with, "runtime array::<16> vs const array::<16>");
                    let arr = a_with.unwrap();
                    for i in 0..16 {
                        assert_f32_eq(
                            arr[i],
                            c_scalar::<CFG>($pos_valid[i]).unwrap(),
                            "array::<16> lane vs scalar",
                        );
                    }
                }

                let bad_w_const = c_arr::<CFG, 3>([1.0; 3]).unwrap_err();
                assert_eq!(bad_w_const.kind, $err_kind::UnsupportedWidth);
                assert_eq!(bad_w_const.lane_mask, 0);
                let bad_w_rt = rt_arr::<3>([1.0; 3], opts).unwrap_err();
                assert_eq!(bad_w_rt, bad_w_const);

                if F & 32 == 0 && F & 1 == 0 {
                    let bads: &[f32] = &$rejected;
                    for &bad in bads {
                        let err = c_scalar::<CFG>(bad).unwrap_err();
                        assert_eq!(err.kind, $err_kind::InvalidInput);
                        assert_eq!(err.lane_mask, 1);
                        if F == 0 {
                            assert_eq!(def_scalar(bad).unwrap_err(), err);
                        }
                        assert_eq!(c_arr::<CFG, 1>([bad]).unwrap_err(), err);
                        assert_eq!(rt_scalar(bad, opts).unwrap_err(), err);
                        assert_eq!(rt_arr::<1>([bad], opts).unwrap_err(), err);
                    }
                    let mixed4 = [1.0, bads[0], 0.5, bads[1]];
                    let err4 = c_arr::<CFG, 4>(mixed4).unwrap_err();
                    assert_eq!(err4.kind, $err_kind::InvalidInput);
                    assert_eq!(err4.lane_mask, 0b1010);
                    assert_eq!(rt_arr::<4>(mixed4, opts).unwrap_err(), err4);
                }
            }
        }
    };
}

check_standalone_log_op!(
    standalone_log2,
    Logf,
    LogfOptions,
    LogfError,
    LogfErrorKind,
    logf_options_from_bits,
    l::try_log2,
    log2,
    log2_array,
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
    Logf,
    LogfOptions,
    LogfError,
    LogfErrorKind,
    logf_options_from_bits,
    l::try_ln,
    ln,
    ln_array,
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
    Logf,
    LogfOptions,
    LogfError,
    LogfErrorKind,
    logf_options_from_bits,
    l::try_log10,
    log10,
    log10_array,
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
    Log1pf,
    Log1pfOptions,
    Log1pfError,
    Log1pfErrorKind,
    log1pf_options_from_bits,
    l1p::try_log1p,
    scalar,
    array,
    LOG1P_POSITIVE_VALID,
    LOG1P_SPECIALS_EXTRA,
    [
        -1.0,
        -2.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
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
    all_32_flags!(run_log2, logf_options_from_bits);
    all_32_flags!(run_ln, logf_options_from_bits);
    all_32_flags!(run_log10, logf_options_from_bits);
    all_32_flags!(run_log1p, log1pf_options_from_bits);
}

macro_rules! define_log_token_op {
    (
        $fn_name:ident,
        $op_ty:ident,
        $err_ty:ty,
        $err_kind:ident,
        $tok_ty:ty,
        $simd_m:ident,
        $tail_m:ident,
        $scalar_m:ident,
        $pos_valid:ident,
        $specials_extra:ident,
        $bad0:expr,
        $bad1:expr,
        $has_x16:tt
    ) => {
        #[inline(never)]
        fn $fn_name<const F: u32, const CFG: u128>(token: $tok_ty) {
            #[inline(never)]
            fn eval4<const C: u128>(t: $tok_ty, xs: [f32; 4]) -> Result<[f32; 4], $err_ty> {
                use magetypes::simd::generic::f32x4;
                f32x4::from_array_t(t, xs)
                    .$simd_m::<C>()
                    .map(|v| v.to_array())
            }
            #[inline(never)]
            fn eval8<const C: u128>(t: $tok_ty, xs: [f32; 8]) -> Result<[f32; 8], $err_ty> {
                use magetypes::simd::generic::f32x8;
                f32x8::from_array_t(t, xs)
                    .$simd_m::<C>()
                    .map(|v| v.to_array())
            }
            #[inline(never)]
            fn eval_tail<const C: u128>(t: $tok_ty, x: f32) -> Result<f32, $err_ty> {
                $op_ty::<C>::$tail_m(t, x)
            }
            #[inline(never)]
            fn eval_with<const C: u128>(x: f32) -> Result<f32, $err_ty> {
                $op_ty::<C>::$scalar_m(x)
            }

            for &xs in $pos_valid.as_chunks::<4>().0 {
                let v4 = eval4::<CFG>(token, xs).unwrap();
                for i in 0..4 {
                    let tail = eval_tail::<CFG>(token, xs[i]).unwrap();
                    assert_f32_eq(v4[i], tail, "logf x4 vs _with_token");
                    if F & 16 == 0 {
                        assert_f32_eq(
                            tail,
                            eval_with::<CFG>(xs[i]).unwrap(),
                            "logf BitExact _with_token vs scalar",
                        );
                    }
                }
            }
            for &xs in $pos_valid.as_chunks::<8>().0 {
                let v8 = eval8::<CFG>(token, xs).unwrap();
                for i in 0..8 {
                    let tail = eval_tail::<CFG>(token, xs[i]).unwrap();
                    assert_f32_eq(v8[i], tail, "logf x8 vs _with_token");
                }
            }
            define_log_token_op!(
                @x16 $has_x16,
                $err_ty,
                $tok_ty,
                token,
                CFG,
                $pos_valid,
                $simd_m,
                eval_tail
            );

            if F & 1 != 0 {
                let v8 = eval8::<CFG>(token, $specials_extra).unwrap();
                for i in 0..8 {
                    let tail = eval_tail::<CFG>(token, $specials_extra[i]).unwrap();
                    assert_f32_eq(v8[i], tail, "logf Specials x8 vs _with_token");
                    if F & 16 == 0 {
                        assert_f32_eq(
                            tail,
                            eval_with::<CFG>($specials_extra[i]).unwrap(),
                            "logf Specials BitExact _with_token vs scalar",
                        );
                    }
                }
            } else if F & 32 == 0 {
                let err_s = eval_tail::<CFG>(token, $bad0).unwrap_err();
                assert_eq!(err_s.kind, $err_kind::InvalidInput);
                assert_eq!(err_s.lane_mask, 1);
                let mixed = [0.5, $bad0, 1.0, $bad1];
                let err_v = eval4::<CFG>(token, mixed).unwrap_err();
                assert_eq!(err_v.kind, $err_kind::InvalidInput);
                assert_eq!(err_v.lane_mask, 0b1010);
            }
        }
    };
    (@x16 yes, $err_ty:ty, $tok_ty:ty, $token:ident, $cfg:ident, $arr:ident, $simd_m:ident, $eval_tail:ident) => {
        #[cfg(feature = "w512")]
        {
            #[inline(never)]
            fn eval16<const C: u128>(t: $tok_ty, xs: [f32; 16]) -> Result<[f32; 16], $err_ty> {
                use magetypes::simd::generic::f32x16;
                f32x16::from_array_t(t, xs)
                    .$simd_m::<C>()
                    .map(|v| v.to_array())
            }
            let v16 = eval16::<$cfg>($token, $arr).unwrap();
            for i in 0..16 {
                let tail = $eval_tail::<$cfg>($token, $arr[i]).unwrap();
                assert_f32_eq(v16[i], tail, "logf x16 vs _with_token");
            }
        }
    };
    (@x16 no, $err_ty:ty, $tok_ty:ty, $token:ident, $cfg:ident, $arr:ident, $simd_m:ident, $eval_tail:ident) => {};
}

#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn check_token(token: Token) {
    define_log_token_op!(
        tok_log2,
        Logf,
        LogfError,
        LogfErrorKind,
        Token,
        try_log2_with,
        log2_with_token,
        log2,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        0.0,
        f32::NAN,
        yes
    );
    define_log_token_op!(
        tok_ln,
        Logf,
        LogfError,
        LogfErrorKind,
        Token,
        try_ln_with,
        ln_with_token,
        ln,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        -0.0,
        -1.0,
        yes
    );
    define_log_token_op!(
        tok_log10,
        Logf,
        LogfError,
        LogfErrorKind,
        Token,
        try_log10_with,
        log10_with_token,
        log10,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        f32::from_bits(1),
        f32::INFINITY,
        yes
    );
    define_log_token_op!(
        tok_log1p,
        Log1pf,
        Log1pfError,
        Log1pfErrorKind,
        Token,
        try_log1p_with,
        scalar_with_token,
        scalar,
        LOG1P_POSITIVE_VALID,
        LOG1P_SPECIALS_EXTRA,
        -1.0,
        f32::from_bits(1),
        yes
    );

    all_32_flags!(tok_log2, logf_options_from_bits, token);
    all_32_flags!(tok_ln, logf_options_from_bits, token);
    all_32_flags!(tok_log10, logf_options_from_bits, token);
    all_32_flags!(tok_log1p, log1pf_options_from_bits, token);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn check_token_fp16(token: archmage::Avx512Fp16Token) {
    define_log_token_op!(
        fp16_log2,
        Logf,
        LogfError,
        LogfErrorKind,
        archmage::Avx512Fp16Token,
        try_log2_with,
        log2_with_token,
        log2,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        0.0,
        f32::NAN,
        no
    );
    define_log_token_op!(
        fp16_ln,
        Logf,
        LogfError,
        LogfErrorKind,
        archmage::Avx512Fp16Token,
        try_ln_with,
        ln_with_token,
        ln,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        -0.0,
        -1.0,
        no
    );
    define_log_token_op!(
        fp16_log10,
        Logf,
        LogfError,
        LogfErrorKind,
        archmage::Avx512Fp16Token,
        try_log10_with,
        log10_with_token,
        log10,
        LOG_POSITIVE_VALID,
        LOG_SPECIALS_EXTRA,
        f32::from_bits(1),
        f32::INFINITY,
        no
    );
    define_log_token_op!(
        fp16_log1p,
        Log1pf,
        Log1pfError,
        Log1pfErrorKind,
        archmage::Avx512Fp16Token,
        try_log1p_with,
        scalar_with_token,
        scalar,
        LOG1P_POSITIVE_VALID,
        LOG1P_SPECIALS_EXTRA,
        -1.0,
        f32::from_bits(1),
        no
    );

    all_32_flags!(fp16_log2, logf_options_from_bits, token);
    all_32_flags!(fp16_ln, logf_options_from_bits, token);
    all_32_flags!(fp16_log10, logf_options_from_bits, token);
    all_32_flags!(fp16_log1p, log1pf_options_from_bits, token);
}

#[test]
fn token_logf_scalar_and_vector_widths_match_across_all_32_configs() {
    run_all_summonable_tokens!();

    const L20: u128 = Log1pfOptions::new().backend_dependent().caller().bits();
    const L28: u128 = Log1pfOptions::new()
        .lowp()
        .backend_dependent()
        .caller()
        .bits();

    for &x in &LOG1P_POSITIVE_VALID {
        assert_f32_eq(
            Log1pf::<L20>::scalar(x).unwrap(),
            Log1pf::<L20>::scalar_with_token(archmage::ScalarToken, x).unwrap(),
            "log1p BackendDependent Caller standalone vs ScalarToken",
        );
        assert_f32_eq(
            Log1pf::<L28>::scalar(x).unwrap(),
            Log1pf::<L28>::scalar_with_token(archmage::ScalarToken, x).unwrap(),
            "log1p BackendDependent Lowp Caller standalone vs ScalarToken",
        );
    }
}
