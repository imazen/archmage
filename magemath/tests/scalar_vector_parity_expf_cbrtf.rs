//! Exhaustive const-generic scalar vs vector parity for `magemath::expf` (`exp2`, `exp`)
//! and `magemath::cbrtf` (`cbrt`) across all 32 valid option encodings and all tokens.
#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/scalar_vector_parity.rs"]
mod parity;

use magemath::{
    cbrtf::{
        self as c, CbrtfDomain, CbrtfError, CbrtfErrorKind, CbrtfExecution, CbrtfOptions,
        CbrtfPortability, CbrtfPrecision, CbrtfValidation,
    },
    expf::{
        self as e, ExpfDomain, ExpfError, ExpfErrorKind, ExpfExecution, ExpfOptions,
        ExpfPortability, ExpfPrecision, ExpfValidation,
    },
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
    0.99,
    1.0,
    2.5,
    30.125,
    88.0,
    150.0,
];

const EXP_FULL_EXTRA: [f32; 5] = [
    f32::NEG_INFINITY,
    f32::INFINITY,
    f32::NAN,
    f32::MAX,
    -f32::MAX,
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
    ExpfOptions::new()
        .with_domain(if f & 1 == 0 {
            ExpfDomain::Bounded
        } else {
            ExpfDomain::Full
        })
        .with_execution(if f & 4 == 0 {
            ExpfExecution::Automatic
        } else {
            ExpfExecution::Caller
        })
        .with_precision(if f & 8 == 0 {
            ExpfPrecision::Midp
        } else {
            ExpfPrecision::Lowp
        })
        .with_portability(if f & 16 == 0 {
            ExpfPortability::BitExact
        } else {
            ExpfPortability::BackendDependent
        })
        .with_validation(if f & 32 == 0 {
            ExpfValidation::Checked
        } else {
            ExpfValidation::AssumeValid
        })
}

const fn cbrtf_options_from_bits(f: u32) -> CbrtfOptions {
    CbrtfOptions::new()
        .with_domain(if f & 1 == 0 {
            CbrtfDomain::Bounded
        } else {
            CbrtfDomain::Full
        })
        .with_execution(if f & 4 == 0 {
            CbrtfExecution::Automatic
        } else {
            CbrtfExecution::Caller
        })
        .with_precision(if f & 8 == 0 {
            CbrtfPrecision::Midp
        } else {
            CbrtfPrecision::Lowp
        })
        .with_portability(if f & 16 == 0 {
            CbrtfPortability::BitExact
        } else {
            CbrtfPortability::BackendDependent
        })
        .with_validation(if f & 32 == 0 {
            CbrtfValidation::Checked
        } else {
            CbrtfValidation::AssumeValid
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

macro_rules! check_standalone_family {
    (
        $fn_name:ident,
        $opts_ty:ty,
        $err_ty:ty,
        $opts_fn:ident,
        $try_default:path,
        $try_with:ident,
        $try_array_with:ident,
        $try_runtime:ident,
        $try_array_runtime:ident,
        $mod_alias:ident,
        $err_kind:expr,
        $unsupported_width:expr,
        $bounded_valid:ident,
        $full_extra:ident,
        $rejected:expr
    ) => {
        mod $fn_name {
            use super::*;

            // Keep runtime dispatch tables and each const-generic graph in a single
            // frame so `--release` does not inline 32x32 copies into the test loop.
            #[inline(never)]
            fn rt_scalar(x: f32, opts: $opts_ty) -> Result<f32, $err_ty> {
                $mod_alias::$try_runtime(x, opts)
            }
            #[inline(never)]
            fn rt_arr<const N: usize>(xs: [f32; N], opts: $opts_ty) -> Result<[f32; N], $err_ty> {
                $mod_alias::$try_array_runtime::<N>(xs, opts)
            }
            #[inline(never)]
            fn c_scalar<const F: u32>(x: f32) -> Result<f32, $err_ty> {
                $mod_alias::$try_with::<F>(x)
            }
            #[inline(never)]
            fn c_arr<const F: u32, const N: usize>(xs: [f32; N]) -> Result<[f32; N], $err_ty> {
                $mod_alias::$try_array_with::<F, N>(xs)
            }
            #[inline(never)]
            fn def_scalar(x: f32) -> Result<f32, $err_ty> {
                $try_default(x)
            }

            #[inline(never)]
            pub fn run<const F: u32>() {
                let opts = $opts_fn(F);
                assert_eq!(opts.bits(), F);

                for &x in &$bounded_valid {
                    let s = c_scalar::<F>(x);
                    assert!(s.is_ok(), "expected Ok on bounded valid input {x:?}");
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
                    for &x in &$full_extra {
                        let s = c_scalar::<F>(x);
                        assert!(s.is_ok(), "expected Ok on Full extra input {x:?}");
                        assert_scalar_res_eq(
                            c_arr::<F, 1>([x]).map(|a| a[0]),
                            s,
                            "Full array_with::<F, 1> vs _with",
                        );
                        assert_scalar_res_eq(rt_scalar(x, opts), s, "Full runtime scalar vs _with");
                        assert_scalar_res_eq(
                            rt_arr::<1>([x], opts).map(|a| a[0]),
                            s,
                            "Full array_runtime::<1> vs _with",
                        );
                    }
                }

                for &pack in $bounded_valid.as_chunks::<4>().0 {
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
                for &pack in $bounded_valid.as_chunks::<8>().0 {
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
                    let a_with = c_arr::<F, 16>($bounded_valid);
                    let a_rt = rt_arr::<16>($bounded_valid, opts);
                    assert_array_res_eq(a_rt, a_with, "array_runtime::<16> vs array_with::<F, 16>");
                    let arr = a_with.unwrap();
                    for i in 0..16 {
                        assert_f32_eq(
                            arr[i],
                            c_scalar::<F>($bounded_valid[i]).unwrap(),
                            "array_with::<F, 16> lane vs scalar _with",
                        );
                    }
                }

                let bad_w_const = c_arr::<F, 3>([1.0; 3]).unwrap_err();
                assert_eq!(bad_w_const.kind, $unsupported_width);
                assert_eq!(bad_w_const.lane_mask, 0);
                let bad_w_rt = rt_arr::<3>([1.0; 3], opts).unwrap_err();
                assert_eq!(bad_w_rt, bad_w_const);

                if F & 32 == 0 && F & 1 == 0 {
                    let bads: &[f32] = &$rejected;
                    for &bad in bads {
                        let err = c_scalar::<F>(bad).unwrap_err();
                        assert_eq!(err.kind, $err_kind);
                        assert_eq!(err.lane_mask, 1);
                        if F == 0 {
                            assert_eq!(def_scalar(bad).unwrap_err(), err);
                        }
                        assert_eq!(c_arr::<F, 1>([bad]).unwrap_err(), err);
                        assert_eq!(rt_scalar(bad, opts).unwrap_err(), err);
                        assert_eq!(rt_arr::<1>([bad], opts).unwrap_err(), err);
                    }
                    let mixed4 = [1.0, bads[0], 0.5, bads[1 % bads.len()]];
                    let err4 = c_arr::<F, 4>(mixed4).unwrap_err();
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
    ExpfOptions,
    ExpfError,
    expf_options_from_bits,
    e::try_exp2,
    try_exp2_with,
    try_exp2_array_with,
    try_exp2_runtime,
    try_exp2_array_runtime,
    e,
    ExpfErrorKind::InvalidInput,
    ExpfErrorKind::UnsupportedWidth,
    EXP2_BOUNDED_VALID,
    EXP2_FULL_EXTRA,
    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
);

check_standalone_family!(
    standalone_exp,
    ExpfOptions,
    ExpfError,
    expf_options_from_bits,
    e::try_exp,
    try_exp_with,
    try_exp_array_with,
    try_exp_runtime,
    try_exp_array_runtime,
    e,
    ExpfErrorKind::InvalidInput,
    ExpfErrorKind::UnsupportedWidth,
    EXP_BOUNDED_VALID,
    EXP_FULL_EXTRA,
    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX]
);

check_standalone_family!(
    standalone_cbrt,
    CbrtfOptions,
    CbrtfError,
    cbrtf_options_from_bits,
    c::try_cbrt,
    try_cbrt_with,
    try_cbrt_array_with,
    try_cbrt_runtime,
    try_cbrt_array_runtime,
    c,
    CbrtfErrorKind::InvalidInput,
    CbrtfErrorKind::UnsupportedWidth,
    CBRT_BOUNDED_VALID,
    CBRT_FULL_EXTRA,
    [
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN
    ]
);

#[test]
fn standalone_scalar_array_and_runtime_match_across_all_32_configs() {
    use standalone_cbrt::run as run_cbrt;
    use standalone_exp::run as run_exp;
    use standalone_exp2::run as run_exp2;
    all_32_flags!(run_exp2);
    all_32_flags!(run_exp);
    all_32_flags!(run_cbrt);
}

macro_rules! define_single_token_op {
    (
        $fn_name:ident,
        $tok_ty:ty,
        $err_ty:ty,
        $err_kind:expr,
        $mod_id:ident,
        $vec4:ident,
        $vec8:ident,
        $vec16:ident,
        $tail_fn:ident,
        $with_fn:ident,
        $valid_arr:ident,
        $extra_arr:expr,
        $bad0:expr,
        $bad1:expr,
        $has_x16:tt
    ) => {
        #[inline(never)]
        fn $fn_name<const F: u32>(token: $tok_ty) {
            #[inline(never)]
            fn eval4<const FLAGS: u32>(t: $tok_ty, xs: [f32; 4]) -> Result<[f32; 4], $err_ty> {
                use magetypes::simd::generic::f32x4;
                $mod_id::$vec4::<FLAGS, _>(t, f32x4::from_array_t(t, xs)).map(|v| v.to_array())
            }
            #[inline(never)]
            fn eval8<const FLAGS: u32>(t: $tok_ty, xs: [f32; 8]) -> Result<[f32; 8], $err_ty> {
                use magetypes::simd::generic::f32x8;
                $mod_id::$vec8::<FLAGS, _>(t, f32x8::from_array_t(t, xs)).map(|v| v.to_array())
            }
            #[inline(never)]
            fn eval_tail<const FLAGS: u32>(t: $tok_ty, x: f32) -> Result<f32, $err_ty> {
                $mod_id::$tail_fn::<FLAGS, _>(t, x)
            }
            #[inline(never)]
            fn eval_with<const FLAGS: u32>(x: f32) -> Result<f32, $err_ty> {
                $mod_id::$with_fn::<FLAGS>(x)
            }

            for &xs in $valid_arr.as_chunks::<4>().0 {
                let v4 = eval4::<F>(token, xs).unwrap();
                for i in 0..4 {
                    let tail = eval_tail::<F>(token, xs[i]).unwrap();
                    assert_f32_eq(v4[i], tail, "x4 vs _with_token");
                    if F & 16 == 0 {
                        assert_f32_eq(tail, eval_with::<F>(xs[i]).unwrap(), "BitExact _with");
                    }
                }
            }
            for &xs in $valid_arr.as_chunks::<8>().0 {
                let v8 = eval8::<F>(token, xs).unwrap();
                for i in 0..8 {
                    let tail = eval_tail::<F>(token, xs[i]).unwrap();
                    assert_f32_eq(v8[i], tail, "x8 vs _with_token");
                }
            }
            define_single_token_op!(@x16 $has_x16, $tok_ty, $err_ty, token, F, $valid_arr, $mod_id, $vec16, eval_tail);

            if F & 1 != 0 {
                let extra: [f32; 4] = $extra_arr;
                let v4 = eval4::<F>(token, extra).unwrap();
                for i in 0..4 {
                    let tail = eval_tail::<F>(token, extra[i]).unwrap();
                    assert_f32_eq(v4[i], tail, "Full x4 vs _with_token");
                    if F & 16 == 0 {
                        assert_f32_eq(
                            tail,
                            eval_with::<F>(extra[i]).unwrap(),
                            "Full BitExact _with",
                        );
                    }
                }
            } else if F & 32 == 0 {
                let err_s = eval_tail::<F>(token, $bad0).unwrap_err();
                assert_eq!(err_s.kind, $err_kind);
                assert_eq!(err_s.lane_mask, 1);
                let mixed = [1.0, $bad0, 0.5, $bad1];
                let err_v = eval4::<F>(token, mixed).unwrap_err();
                assert_eq!(err_v.kind, $err_kind);
                assert_eq!(err_v.lane_mask, 0b1010);
            }
        }
    };
    (@x16 yes, $tok_ty:ty, $err_ty:ty, $token:ident, $f:ident, $arr:ident, $mod_id:ident, $vec16:ident, $eval_tail:ident) => {
        #[cfg(feature = "w512")]
        {
            #[inline(never)]
            fn eval16<const FLAGS: u32>(t: $tok_ty, xs: [f32; 16]) -> Result<[f32; 16], $err_ty> {
                use magetypes::simd::generic::f32x16;
                $mod_id::$vec16::<FLAGS, _>(t, f32x16::from_array_t(t, xs)).map(|v| v.to_array())
            }
            let v16 = eval16::<$f>($token, $arr).unwrap();
            for i in 0..16 {
                let tail = $eval_tail::<$f>($token, $arr[i]).unwrap();
                assert_f32_eq(v16[i], tail, "x16 vs _with_token");
            }
        }
    };
    (@x16 no, $tok_ty:ty, $err_ty:ty, $token:ident, $f:ident, $arr:ident, $mod_id:ident, $vec16:ident, $eval_tail:ident) => {};
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
    define_single_token_op!(
        tok_exp2,
        Token,
        ExpfError,
        ExpfErrorKind::InvalidInput,
        e,
        try_exp2_f32x4_with,
        try_exp2_f32x8_with,
        try_exp2_f32x16_with,
        try_exp2_with_token,
        try_exp2_with,
        EXP2_BOUNDED_VALID,
        EXP2_FULL_EXTRA,
        f32::NAN,
        f32::NEG_INFINITY,
        yes
    );
    define_single_token_op!(
        tok_exp,
        Token,
        ExpfError,
        ExpfErrorKind::InvalidInput,
        e,
        try_exp_f32x4_with,
        try_exp_f32x8_with,
        try_exp_f32x16_with,
        try_exp_with_token,
        try_exp_with,
        EXP_BOUNDED_VALID,
        [
            EXP_FULL_EXTRA[0],
            EXP_FULL_EXTRA[1],
            EXP_FULL_EXTRA[2],
            EXP_FULL_EXTRA[3]
        ],
        f32::MAX,
        f32::NAN,
        yes
    );
    define_single_token_op!(
        tok_cbrt,
        Token,
        CbrtfError,
        CbrtfErrorKind::InvalidInput,
        c,
        try_cbrt_f32x4_with,
        try_cbrt_f32x8_with,
        try_cbrt_f32x16_with,
        try_cbrt_with_token,
        try_cbrt_with,
        CBRT_BOUNDED_VALID,
        [
            CBRT_FULL_EXTRA[0],
            CBRT_FULL_EXTRA[4],
            CBRT_FULL_EXTRA[7],
            CBRT_FULL_EXTRA[9]
        ],
        f32::from_bits(1),
        f32::MAX,
        yes
    );
    all_32_flags!(tok_exp2, token);
    all_32_flags!(tok_exp, token);
    all_32_flags!(tok_cbrt, token);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn check_token_fp16(token: archmage::Avx512Fp16Token) {
    define_single_token_op!(
        fp16_exp2,
        archmage::Avx512Fp16Token,
        ExpfError,
        ExpfErrorKind::InvalidInput,
        e,
        try_exp2_f32x4_with,
        try_exp2_f32x8_with,
        try_exp2_f32x16_with,
        try_exp2_with_token,
        try_exp2_with,
        EXP2_BOUNDED_VALID,
        EXP2_FULL_EXTRA,
        f32::NAN,
        f32::NEG_INFINITY,
        no
    );
    define_single_token_op!(
        fp16_exp,
        archmage::Avx512Fp16Token,
        ExpfError,
        ExpfErrorKind::InvalidInput,
        e,
        try_exp_f32x4_with,
        try_exp_f32x8_with,
        try_exp_f32x16_with,
        try_exp_with_token,
        try_exp_with,
        EXP_BOUNDED_VALID,
        [
            EXP_FULL_EXTRA[0],
            EXP_FULL_EXTRA[1],
            EXP_FULL_EXTRA[2],
            EXP_FULL_EXTRA[3]
        ],
        f32::MAX,
        f32::NAN,
        no
    );
    define_single_token_op!(
        fp16_cbrt,
        archmage::Avx512Fp16Token,
        CbrtfError,
        CbrtfErrorKind::InvalidInput,
        c,
        try_cbrt_f32x4_with,
        try_cbrt_f32x8_with,
        try_cbrt_f32x16_with,
        try_cbrt_with_token,
        try_cbrt_with,
        CBRT_BOUNDED_VALID,
        [
            CBRT_FULL_EXTRA[0],
            CBRT_FULL_EXTRA[4],
            CBRT_FULL_EXTRA[7],
            CBRT_FULL_EXTRA[9]
        ],
        f32::from_bits(1),
        f32::MAX,
        no
    );
    all_32_flags!(fp16_exp2, token);
    all_32_flags!(fp16_exp, token);
    all_32_flags!(fp16_cbrt, token);
}

#[test]
fn token_scalar_and_vector_widths_match_across_all_32_configs() {
    run_all_summonable_tokens!();

    for &x in &EXP2_BOUNDED_VALID {
        assert_f32_eq(
            e::try_exp2_with::<20>(x).unwrap(),
            e::try_exp2_with_token::<20, _>(archmage::ScalarToken, x).unwrap(),
            "exp2 BackendDependent Caller standalone vs ScalarToken",
        );
        assert_f32_eq(
            e::try_exp2_with::<28>(x).unwrap(),
            e::try_exp2_with_token::<28, _>(archmage::ScalarToken, x).unwrap(),
            "exp2 BackendDependent Lowp Caller standalone vs ScalarToken",
        );
    }
    for &x in &CBRT_BOUNDED_VALID {
        assert_f32_eq(
            c::try_cbrt_with::<20>(x).unwrap(),
            c::try_cbrt_with_token::<20, _>(archmage::ScalarToken, x).unwrap(),
            "cbrt BackendDependent Caller standalone vs ScalarToken",
        );
        assert_f32_eq(
            c::try_cbrt_with::<28>(x).unwrap(),
            c::try_cbrt_with_token::<28, _>(archmage::ScalarToken, x).unwrap(),
            "cbrt BackendDependent Lowp Caller standalone vs ScalarToken",
        );
    }
}
