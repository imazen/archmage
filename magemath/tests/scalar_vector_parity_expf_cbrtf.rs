//! Exhaustive const-generic scalar vs vector parity for `magemath::expf` (`exp2`, `exp`)
//! and `magemath::cbrtf` (`cbrt`) across all 32 valid option encodings and all tokens.
#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/scalar_vector_parity.rs"]
mod parity;

use magemath::{
    Execution, Portability, SimdMath, Validation,
    cbrtf::{self as c, Cbrtf, CbrtfError, CbrtfErrorKind, CbrtfOptions, CbrtfPrecision},
    expf::{self as e, Expf, ExpfError, ExpfErrorKind, ExpfOptions, ExpfPrecision},
};
use parity::{assert_array_res_eq, assert_f32_eq, assert_scalar_res_eq};

const EXP2_BOUNDED_VALID: [f32; 16] = [
    -200.0,
    -126.0,
    -40.5,
    -1.75,
    -0.5,
    -0.0,
    0.0,
    f32::from_bits(1),
    f32::from_bits(0x8000_0001),
    0.25,
    0.99,
    1.0,
    2.5,
    42.125,
    127.25,
    200.0,
];

const EXP2_FULL_EXTRA: [f32; 4] = [
    f32::NEG_INFINITY,
    f32::INFINITY,
    f32::NAN,
    f32::from_bits(0x7fc0_1234),
];

const EXP_BOUNDED_VALID: [f32; 16] = [
    -f32::MAX,
    -150.0,
    -87.3,
    -20.5,
    -1.75,
    -0.5,
    -0.0,
    0.0,
    f32::from_bits(1),
    f32::from_bits(0x8000_0001),
    0.25,
    1.0,
    2.5,
    88.0,
    150.0,
    f32::MAX,
];

const EXP_FULL_EXTRA: [f32; 4] = [
    f32::NEG_INFINITY,
    f32::INFINITY,
    f32::NAN,
    f32::from_bits(0x7fc0_1234),
];

const CBRT_BOUNDED_VALID: [f32; 16] = [
    -1.0e37,
    -1000.0,
    -27.0,
    -2.5,
    -1.0,
    -1.175_494_4e-38,
    -0.0,
    0.0,
    1.175_494_4e-38,
    0.125,
    0.5,
    1.0,
    2.5,
    8.0,
    1000.0,
    1.0e37,
];

const CBRT_FULL_EXTRA: [f32; 10] = [
    f32::from_bits(1),
    f32::from_bits(0x8000_0001),
    f32::from_bits(0x007f_ffff),
    f32::from_bits(0x807f_ffff),
    f32::MAX,
    -f32::MAX,
    2.0e38,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

const fn expf_options_from_bits(f: u32) -> ExpfOptions {
    let base = if f & 1 == 0 {
        ExpfOptions::new()
    } else {
        ExpfOptions::new().full_domain()
    };
    base.execution(if f & 4 == 0 {
        Execution::Upgrade
    } else {
        Execution::Caller
    })
    .precision(if f & 8 == 0 {
        ExpfPrecision::Midp
    } else {
        ExpfPrecision::Lowp
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

const fn cbrtf_options_from_bits(f: u32) -> CbrtfOptions {
    let base = if f & 1 == 0 {
        CbrtfOptions::new()
    } else {
        CbrtfOptions::new().full_domain()
    };
    base.execution(if f & 4 == 0 {
        Execution::Upgrade
    } else {
        Execution::Caller
    })
    .precision(if f & 8 == 0 {
        CbrtfPrecision::Midp
    } else {
        CbrtfPrecision::Lowp
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
    ($m:ident, $opts_fn:ident $(, $arg:expr)*) => {
        macro_rules! call_one {
            ($f:literal) => {
                $m::<$f, { $opts_fn($f).bits() }>($($arg),*);
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

macro_rules! check_standalone_family {
    (
        $fn_name:ident,
        $op_struct:ident,
        $opts_ty:ty,
        $err_ty:ty,
        $opts_fn:ident,
        $try_default:path,
        $scalar_m:ident,
        $array_m:ident,
        $err_kind:expr,
        $unsupported_width:expr,
        $bounded_valid:ident,
        $full_extra:ident,
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
                $op_struct::<CFG>::$scalar_m(x)
            }
            #[inline(never)]
            fn c_arr<const CFG: u128, const N: usize>(xs: [f32; N]) -> Result<[f32; N], $err_ty> {
                $op_struct::<CFG>::$array_m::<N>(xs)
            }
            #[inline(never)]
            fn def_scalar(x: f32) -> Result<f32, $err_ty> {
                $try_default(x)
            }

            #[inline(never)]
            pub fn run<const F: u32, const CFG: u128>() {
                let opts = $opts_fn(F);
                assert_eq!(<$opts_ty>::from_bits(CFG), Some(opts));

                for &x in &$bounded_valid {
                    let s = c_scalar::<CFG>(x);
                    assert!(s.is_ok(), "expected Ok on bounded valid input {x:?}");
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
                    for &x in &$full_extra {
                        let s = c_scalar::<CFG>(x);
                        assert!(s.is_ok(), "expected Ok on Full extra input {x:?}");
                        assert_scalar_res_eq(
                            c_arr::<CFG, 1>([x]).map(|a| a[0]),
                            s,
                            "Full array::<1> vs scalar",
                        );
                        assert_scalar_res_eq(rt_scalar(x, opts), s, "Full runtime scalar vs const");
                        assert_scalar_res_eq(
                            rt_arr::<1>([x], opts).map(|a| a[0]),
                            s,
                            "Full runtime array::<1> vs const",
                        );
                    }
                }

                for &pack in $bounded_valid.as_chunks::<4>().0 {
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
                for &pack in $bounded_valid.as_chunks::<8>().0 {
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
                    let a_with = c_arr::<CFG, 16>($bounded_valid);
                    let a_rt = rt_arr::<16>($bounded_valid, opts);
                    assert_array_res_eq(a_rt, a_with, "runtime array::<16> vs const array::<16>");
                    let arr = a_with.unwrap();
                    for i in 0..16 {
                        assert_f32_eq(
                            arr[i],
                            c_scalar::<CFG>($bounded_valid[i]).unwrap(),
                            "array::<16> lane vs scalar",
                        );
                    }
                }

                let bad_w_const = c_arr::<CFG, 3>([1.0; 3]).unwrap_err();
                assert_eq!(bad_w_const.kind, $unsupported_width);
                assert_eq!(bad_w_const.lane_mask, 0);
                let bad_w_rt = rt_arr::<3>([1.0; 3], opts).unwrap_err();
                assert_eq!(bad_w_rt, bad_w_const);

                if F & 32 == 0 && F & 1 == 0 {
                    let bads: &[f32] = &$rejected;
                    for &bad in bads {
                        let err = c_scalar::<CFG>(bad).unwrap_err();
                        assert_eq!(err.kind, $err_kind);
                        assert_eq!(err.lane_mask, 1);
                        if F == 0 {
                            assert_eq!(def_scalar(bad).unwrap_err(), err);
                        }
                        assert_eq!(c_arr::<CFG, 1>([bad]).unwrap_err(), err);
                        assert_eq!(rt_scalar(bad, opts).unwrap_err(), err);
                        assert_eq!(rt_arr::<1>([bad], opts).unwrap_err(), err);
                    }
                    let mixed4 = [1.0, bads[0], 0.5, bads[1 % bads.len()]];
                    let err4 = c_arr::<CFG, 4>(mixed4).unwrap_err();
                    assert_eq!(err4.kind, $err_kind);
                    assert_eq!(err4.lane_mask, 0b1010);
                    assert_eq!(rt_arr::<4>(mixed4, opts).unwrap_err(), err4);
                }
            }
        }
    };
}

check_standalone_family!(
    standalone_exp2,
    Expf,
    ExpfOptions,
    ExpfError,
    expf_options_from_bits,
    e::try_exp2,
    exp2,
    exp2_array,
    ExpfErrorKind::InvalidInput,
    ExpfErrorKind::UnsupportedWidth,
    EXP2_BOUNDED_VALID,
    EXP2_FULL_EXTRA,
    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
);

check_standalone_family!(
    standalone_exp,
    Expf,
    ExpfOptions,
    ExpfError,
    expf_options_from_bits,
    e::try_exp,
    exp,
    exp_array,
    ExpfErrorKind::InvalidInput,
    ExpfErrorKind::UnsupportedWidth,
    EXP_BOUNDED_VALID,
    EXP_FULL_EXTRA,
    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
);

check_standalone_family!(
    standalone_cbrt,
    Cbrtf,
    CbrtfOptions,
    CbrtfError,
    cbrtf_options_from_bits,
    c::try_cbrt,
    scalar,
    array,
    CbrtfErrorKind::InvalidInput,
    CbrtfErrorKind::UnsupportedWidth,
    CBRT_BOUNDED_VALID,
    CBRT_FULL_EXTRA,
    [f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]
);

#[test]
fn standalone_scalar_array_and_runtime_match_across_all_32_configs() {
    use standalone_cbrt::run as run_cbrt;
    use standalone_exp::run as run_exp;
    use standalone_exp2::run as run_exp2;
    all_32_flags!(run_exp2, expf_options_from_bits);
    all_32_flags!(run_exp, expf_options_from_bits);
    all_32_flags!(run_cbrt, cbrtf_options_from_bits);
}

macro_rules! define_single_token_op {
    (
        $fn_name:ident,
        $tok_ty:ty,
        $op_struct:ident,
        $err_ty:ty,
        $err_kind:expr,
        $simd_m:ident,
        $tail_m:ident,
        $scalar_m:ident,
        $valid_arr:ident,
        $extra_arr:expr,
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
                $op_struct::<C>::$tail_m(t, x)
            }
            #[inline(never)]
            fn eval_with<const C: u128>(x: f32) -> Result<f32, $err_ty> {
                $op_struct::<C>::$scalar_m(x)
            }

            for &xs in $valid_arr.as_chunks::<4>().0 {
                let v4 = eval4::<CFG>(token, xs).unwrap();
                for i in 0..4 {
                    let tail = eval_tail::<CFG>(token, xs[i]).unwrap();
                    assert_f32_eq(v4[i], tail, "x4 vs _with_token");
                    if F & 16 == 0 {
                        assert_f32_eq(tail, eval_with::<CFG>(xs[i]).unwrap(), "BitExact scalar");
                    }
                }
            }
            for &xs in $valid_arr.as_chunks::<8>().0 {
                let v8 = eval8::<CFG>(token, xs).unwrap();
                for i in 0..8 {
                    let tail = eval_tail::<CFG>(token, xs[i]).unwrap();
                    assert_f32_eq(v8[i], tail, "x8 vs _with_token");
                }
            }
            define_single_token_op!(@x16 $has_x16, $tok_ty, $err_ty, token, CFG, $valid_arr, $simd_m, eval_tail);

            if F & 1 != 0 {
                let extra: [f32; 4] = $extra_arr;
                let v4 = eval4::<CFG>(token, extra).unwrap();
                for i in 0..4 {
                    let tail = eval_tail::<CFG>(token, extra[i]).unwrap();
                    assert_f32_eq(v4[i], tail, "Full x4 vs _with_token");
                    if F & 16 == 0 {
                        assert_f32_eq(
                            tail,
                            eval_with::<CFG>(extra[i]).unwrap(),
                            "Full BitExact scalar",
                        );
                    }
                }
            } else if F & 32 == 0 {
                let err_s = eval_tail::<CFG>(token, $bad0).unwrap_err();
                assert_eq!(err_s.kind, $err_kind);
                assert_eq!(err_s.lane_mask, 1);
                let mixed = [1.0, $bad0, 0.5, $bad1];
                let err_v = eval4::<CFG>(token, mixed).unwrap_err();
                assert_eq!(err_v.kind, $err_kind);
                assert_eq!(err_v.lane_mask, 0b1010);
            }
        }
    };
    (@x16 yes, $tok_ty:ty, $err_ty:ty, $token:ident, $cfg:ident, $arr:ident, $simd_m:ident, $eval_tail:ident) => {
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
                assert_f32_eq(v16[i], tail, "x16 vs _with_token");
            }
        }
    };
    (@x16 no, $tok_ty:ty, $err_ty:ty, $token:ident, $cfg:ident, $arr:ident, $simd_m:ident, $eval_tail:ident) => {};
}

#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn check_token(token: Token) {
    define_single_token_op!(
        tok_exp2,
        Token,
        Expf,
        ExpfError,
        ExpfErrorKind::InvalidInput,
        try_exp2_with,
        exp2_with_token,
        exp2,
        EXP2_BOUNDED_VALID,
        EXP2_FULL_EXTRA,
        f32::NAN,
        f32::NEG_INFINITY,
        yes
    );
    define_single_token_op!(
        tok_exp,
        Token,
        Expf,
        ExpfError,
        ExpfErrorKind::InvalidInput,
        try_exp_with,
        exp_with_token,
        exp,
        EXP_BOUNDED_VALID,
        EXP_FULL_EXTRA,
        f32::INFINITY,
        f32::NAN,
        yes
    );
    define_single_token_op!(
        tok_cbrt,
        Token,
        Cbrtf,
        CbrtfError,
        CbrtfErrorKind::InvalidInput,
        try_cbrt_with,
        scalar_with_token,
        scalar,
        CBRT_BOUNDED_VALID,
        [
            CBRT_FULL_EXTRA[0],
            CBRT_FULL_EXTRA[4],
            CBRT_FULL_EXTRA[7],
            CBRT_FULL_EXTRA[9]
        ],
        f32::NAN,
        f32::MAX,
        yes
    );
    all_32_flags!(tok_exp2, expf_options_from_bits, token);
    all_32_flags!(tok_exp, expf_options_from_bits, token);
    all_32_flags!(tok_cbrt, cbrtf_options_from_bits, token);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn check_token_fp16(token: archmage::Avx512Fp16Token) {
    define_single_token_op!(
        fp16_exp2,
        archmage::Avx512Fp16Token,
        Expf,
        ExpfError,
        ExpfErrorKind::InvalidInput,
        try_exp2_with,
        exp2_with_token,
        exp2,
        EXP2_BOUNDED_VALID,
        EXP2_FULL_EXTRA,
        f32::NAN,
        f32::NEG_INFINITY,
        no
    );
    define_single_token_op!(
        fp16_exp,
        archmage::Avx512Fp16Token,
        Expf,
        ExpfError,
        ExpfErrorKind::InvalidInput,
        try_exp_with,
        exp_with_token,
        exp,
        EXP_BOUNDED_VALID,
        EXP_FULL_EXTRA,
        f32::INFINITY,
        f32::NAN,
        no
    );
    define_single_token_op!(
        fp16_cbrt,
        archmage::Avx512Fp16Token,
        Cbrtf,
        CbrtfError,
        CbrtfErrorKind::InvalidInput,
        try_cbrt_with,
        scalar_with_token,
        scalar,
        CBRT_BOUNDED_VALID,
        [
            CBRT_FULL_EXTRA[0],
            CBRT_FULL_EXTRA[4],
            CBRT_FULL_EXTRA[7],
            CBRT_FULL_EXTRA[9]
        ],
        f32::NAN,
        f32::MAX,
        no
    );
    all_32_flags!(fp16_exp2, expf_options_from_bits, token);
    all_32_flags!(fp16_exp, expf_options_from_bits, token);
    all_32_flags!(fp16_cbrt, cbrtf_options_from_bits, token);
}

#[test]
fn token_scalar_and_vector_widths_match_across_all_32_configs() {
    run_all_summonable_tokens!();

    const E20: u128 = ExpfOptions::new().backend_dependent().caller().bits();
    const E28: u128 = ExpfOptions::new()
        .lowp()
        .backend_dependent()
        .caller()
        .bits();
    const C20: u128 = CbrtfOptions::new().backend_dependent().caller().bits();
    const C28: u128 = CbrtfOptions::new()
        .lowp()
        .backend_dependent()
        .caller()
        .bits();

    for &x in &EXP2_BOUNDED_VALID {
        assert_f32_eq(
            Expf::<E20>::exp2(x).unwrap(),
            Expf::<E20>::exp2_with_token(archmage::ScalarToken, x).unwrap(),
            "exp2 BackendDependent Caller standalone vs ScalarToken",
        );
        assert_f32_eq(
            Expf::<E28>::exp2(x).unwrap(),
            Expf::<E28>::exp2_with_token(archmage::ScalarToken, x).unwrap(),
            "exp2 BackendDependent Lowp Caller standalone vs ScalarToken",
        );
    }
    for &x in &CBRT_BOUNDED_VALID {
        assert_f32_eq(
            Cbrtf::<C20>::scalar(x).unwrap(),
            Cbrtf::<C20>::scalar_with_token(archmage::ScalarToken, x).unwrap(),
            "cbrt BackendDependent Caller standalone vs ScalarToken",
        );
        assert_f32_eq(
            Cbrtf::<C28>::scalar(x).unwrap(),
            Cbrtf::<C28>::scalar_with_token(archmage::ScalarToken, x).unwrap(),
            "cbrt BackendDependent Lowp Caller standalone vs ScalarToken",
        );
    }
}
