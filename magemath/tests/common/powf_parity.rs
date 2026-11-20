//! Shared const-generic scalar vs vector parity suite for `magemath::powf`.
#![allow(dead_code)]

#[macro_use]
#[path = "scalar_vector_parity.rs"]
pub mod parity;

pub use magemath::{
    Execution, Portability, SimdMath, Validation,
    powf::{self as p, Powf, PowfError, PowfErrorKind, PowfOptions, PowfPrecision},
};
pub use parity::{assert_array_res_eq, assert_f32_eq, assert_scalar_res_eq};

pub const POWF_CORE_BASES: [f32; 16] = [
    1.0 / 65_536.0,
    1.0 / 1024.0,
    0.015_625,
    0.125,
    0.25,
    0.333_333_34,
    0.5,
    0.75,
    0.999_999_9,
    1.0,
    1.000_000_1,
    1.5,
    2.5,
    8.0,
    256.0,
    65_536.0,
];

pub const POWF_CORE_EXPS: [f32; 5] = [0.0, -0.0, 0.5, 2.2, 4.0];

pub const POWF_FULL_BASES: [f32; 8] = [
    -0.0,
    -2.0,
    f32::from_bits(1),
    f32::from_bits(0x8000_0001),
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    1.0e30,
];

pub const POWF_FULL_EXPS: [f32; 10] = [
    0.0,
    -0.0,
    0.5,
    2.0,
    3.0,
    -2.0,
    -3.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

pub const fn powf_options_from_bits(f: u32) -> PowfOptions {
    let base = if f & 128 != 0 {
        PowfOptions::new().positive_gamma()
    } else {
        match f & 3 {
            0 => PowfOptions::new(),
            1 => PowfOptions::new().admit_zero(),
            2 => PowfOptions::new().gamma(),
            _ => PowfOptions::new().full_domain(),
        }
    };
    let precision = if f & 64 != 0 {
        PowfPrecision::Highp
    } else if f & 8 != 0 {
        PowfPrecision::Lowp
    } else {
        PowfPrecision::Midp
    };
    base.precision(precision)
        .execution(if f & 4 == 0 {
            Execution::Upgrade
        } else {
            Execution::Caller
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

// Non-generic over `const CFG`: materializes the runtime dispatch path once per width.
#[inline(never)]
fn rt_powf_scalar(x: f32, n: f32, opts: PowfOptions) -> Result<f32, PowfError> {
    opts.runtime().scalar(x, n)
}

#[inline(never)]
fn rt_powf_arr<const N: usize>(
    xs: [f32; N],
    n: f32,
    opts: PowfOptions,
) -> Result<[f32; N], PowfError> {
    opts.runtime().array::<N>(xs, n)
}

#[inline(never)]
fn def_powf_scalar(x: f32, n: f32) -> Result<f32, PowfError> {
    p::try_powf(x, n)
}

#[inline(never)]
pub fn c_powf_scalar<const CFG: u128>(x: f32, n: f32) -> Result<f32, PowfError> {
    Powf::<CFG>::scalar(x, n)
}

#[inline(never)]
fn c_powf_arr<const CFG: u128, const N: usize>(
    xs: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    Powf::<CFG>::array::<N>(xs, n)
}

#[inline(never)]
fn c_powf_scalar_tok<const CFG: u128>(x: f32, n: f32) -> Result<f32, PowfError> {
    Powf::<CFG>::scalar_with_token(archmage::ScalarToken, x, n)
}

#[inline(never)]
pub fn standalone_powf<const F: u32, const CFG: u128>() {
    use core::hint::black_box;
    let opts = black_box(powf_options_from_bits(F));
    assert_eq!(PowfOptions::from_bits(CFG), Some(opts));
    let core_bases = black_box(&POWF_CORE_BASES);
    let core_exps = black_box(&POWF_CORE_EXPS);
    let full_bases = black_box(&POWF_FULL_BASES);
    let full_exps = black_box(&POWF_FULL_EXPS);

    for &n in core_exps {
        for &x in core_bases {
            let s = c_powf_scalar::<CFG>(x, n);
            assert!(s.is_ok(), "expected Ok on core powf ({x:?}, {n:?})");
            if F == 0 {
                assert_scalar_res_eq(def_powf_scalar(x, n), s, "try_powf vs Powf::<DEFAULT>");
            }
            assert_scalar_res_eq(
                c_powf_arr::<CFG, 1>([x], n).map(|a| a[0]),
                s,
                "powf array::<1> vs scalar",
            );
            assert_scalar_res_eq(rt_powf_scalar(x, n, opts), s, "powf runtime vs const");
            assert_scalar_res_eq(
                rt_powf_arr::<1>([x], n, opts).map(|a| a[0]),
                s,
                "powf runtime array::<1> vs const",
            );
        }

        for &pack in core_bases.as_chunks::<4>().0 {
            let a_with = c_powf_arr::<CFG, 4>(pack, n);
            let a_rt = rt_powf_arr::<4>(pack, n, opts);
            assert_array_res_eq(a_rt, a_with, "powf runtime array::<4> vs const array");
            let arr = a_with.unwrap();
            for i in 0..4 {
                assert_f32_eq(
                    arr[i],
                    c_powf_scalar::<CFG>(pack[i], n).unwrap(),
                    "powf array::<4> lane vs scalar",
                );
            }
        }

        for &pack in core_bases.as_chunks::<8>().0 {
            let a_with = c_powf_arr::<CFG, 8>(pack, n);
            let a_rt = rt_powf_arr::<8>(pack, n, opts);
            assert_array_res_eq(a_rt, a_with, "powf runtime array::<8> vs const array");
            let arr = a_with.unwrap();
            for i in 0..8 {
                assert_f32_eq(
                    arr[i],
                    c_powf_scalar::<CFG>(pack[i], n).unwrap(),
                    "powf array::<8> lane vs scalar",
                );
            }
        }

        #[cfg(feature = "w512")]
        {
            let a_with = c_powf_arr::<CFG, 16>(*core_bases, n);
            let a_rt = rt_powf_arr::<16>(*core_bases, n, opts);
            assert_array_res_eq(a_rt, a_with, "powf runtime array::<16> vs const array");
            let arr = a_with.unwrap();
            for i in 0..16 {
                assert_f32_eq(
                    arr[i],
                    c_powf_scalar::<CFG>(core_bases[i], n).unwrap(),
                    "powf array::<16> lane vs scalar",
                );
            }
        }
    }

    if (F & 3 == 1) || (F & 3 == 2 && F & 128 == 0) || (F & 3 == 3) {
        let zpack = black_box([0.0, 0.25, 1.0, 16.0]);
        for &n in core_exps {
            let a = c_powf_arr::<CFG, 4>(zpack, n).unwrap();
            for i in 0..4 {
                assert_f32_eq(
                    a[i],
                    c_powf_scalar::<CFG>(zpack[i], n).unwrap(),
                    "powf +0.0 pack vs scalar",
                );
            }
        }
    }

    if F & 3 != 2 {
        let npack = black_box([0.125, 0.5, 2.5, 1024.0]);
        for n in black_box([-0.5_f32, -1.5]) {
            let a = c_powf_arr::<CFG, 4>(npack, n).unwrap();
            for i in 0..4 {
                assert_f32_eq(
                    a[i],
                    c_powf_scalar::<CFG>(npack[i], n).unwrap(),
                    "powf negative exponent pack vs scalar",
                );
            }
        }
    }

    if F & 3 == 3 {
        for &n in full_exps {
            let a = c_powf_arr::<CFG, 8>(*full_bases, n).unwrap();
            let rt = rt_powf_arr::<8>(*full_bases, n, opts).unwrap();
            for i in 0..8 {
                let s = c_powf_scalar::<CFG>(full_bases[i], n).unwrap();
                assert_f32_eq(a[i], s, "powf Full array vs scalar");
                assert_f32_eq(rt[i], s, "powf Full runtime array vs scalar");
            }
        }
    }

    let bad_w = c_powf_arr::<CFG, 3>([1.0; 3], 2.2).unwrap_err();
    assert_eq!(bad_w.kind, PowfErrorKind::UnsupportedWidth);
    assert_eq!(bad_w.lane_mask, 0);
    assert_eq!(rt_powf_arr::<3>([1.0; 3], 2.2, opts).unwrap_err(), bad_w);

    if F & 32 == 0 && F & 3 != 3 {
        assert_eq!(
            c_powf_scalar::<CFG>(2.0, f32::from_bits(1)),
            c_powf_scalar::<CFG>(2.0, 0.0)
        );
        for bad_n in [f32::NAN, f32::INFINITY] {
            let es = c_powf_scalar::<CFG>(2.0, bad_n).unwrap_err();
            assert_eq!(es.kind, PowfErrorKind::InvalidExponent);
            assert_eq!(es.lane_mask, 1);
            if F == 0 {
                assert_eq!(def_powf_scalar(2.0, bad_n).unwrap_err(), es);
            }
            assert_eq!(rt_powf_scalar(2.0, bad_n, opts).unwrap_err(), es);
            let e4 = c_powf_arr::<CFG, 4>([2.0; 4], bad_n).unwrap_err();
            assert_eq!(e4.kind, PowfErrorKind::InvalidExponent);
            assert_eq!(e4.lane_mask, 0b1111);
        }
        if F & 3 == 2 {
            for bad_n in [-0.5_f32, 4.5] {
                let es = c_powf_scalar::<CFG>(2.0, bad_n).unwrap_err();
                assert_eq!(es.kind, PowfErrorKind::InvalidExponent);
                assert_eq!(es.lane_mask, 1);
                let e4 = c_powf_arr::<CFG, 4>([2.0; 4], bad_n).unwrap_err();
                assert_eq!(e4.kind, PowfErrorKind::InvalidExponent);
                assert_eq!(e4.lane_mask, 0b1111);
            }
        }

        let mixed_base = [1.0, -2.0, 0.5, f32::NAN];
        let eb4 = c_powf_arr::<CFG, 4>(mixed_base, 2.2).unwrap_err();
        assert_eq!(eb4.kind, PowfErrorKind::InvalidBase);
        assert_eq!(eb4.lane_mask, 0b1010);
        assert_eq!(rt_powf_arr::<4>(mixed_base, 2.2, opts).unwrap_err(), eb4);

        if F & 3 == 0 || F & 128 != 0 {
            let z_rej = [0.0, 1.0, -0.0, 2.0];
            let ez4 = c_powf_arr::<CFG, 4>(z_rej, 2.2).unwrap_err();
            assert_eq!(ez4.kind, PowfErrorKind::InvalidBase);
            assert_eq!(ez4.lane_mask, 0b0101);
        }

        if F & 3 < 2 {
            let oor = [1.0, 1.0e20, 2.0, 1.0e-20];
            let er4 = c_powf_arr::<CFG, 4>(oor, 10.0).unwrap_err();
            assert_eq!(er4.kind, PowfErrorKind::OutsideApproximationRange);
            assert_eq!(er4.lane_mask, 0b1010);
            let ers = c_powf_scalar::<CFG>(1.0e20, 10.0).unwrap_err();
            assert_eq!(ers.kind, PowfErrorKind::OutsideApproximationRange);
            assert_eq!(ers.lane_mask, 1);
        }
    }

    if F & 16 != 0 && F & 4 != 0 {
        for &x in &POWF_CORE_BASES[..4] {
            assert_f32_eq(
                c_powf_scalar::<CFG>(x, 2.2).unwrap(),
                c_powf_scalar_tok::<CFG>(x, 2.2).unwrap(),
                "powf BackendDependent Caller standalone vs ScalarToken",
            );
        }
    }
}

macro_rules! define_powf_token_op {
    ($fn_name:ident, $tok_ty:ty, $has_x16:tt) => {
        #[inline(never)]
        fn tok_rt4(
            t: $tok_ty,
            xs: [f32; 4],
            n: f32,
            opts: PowfOptions,
        ) -> Result<[f32; 4], PowfError> {
            use magetypes::simd::generic::f32x4;
            opts.runtime()
                .simd(f32x4::from_array_t(t, xs), n)
                .map(|v| v.to_array())
        }
        #[inline(never)]
        fn tok_rt8(
            t: $tok_ty,
            xs: [f32; 8],
            n: f32,
            opts: PowfOptions,
        ) -> Result<[f32; 8], PowfError> {
            use magetypes::simd::generic::f32x8;
            opts.runtime()
                .simd(f32x8::from_array_t(t, xs), n)
                .map(|v| v.to_array())
        }
        define_powf_token_op!(@def_rt16 $has_x16, $tok_ty);

        #[inline(never)]
        fn $fn_name<const F: u32, const CFG: u128>(token: $tok_ty) {
            #[inline(never)]
            fn eval4<const C: u128>(
                t: $tok_ty,
                xs: [f32; 4],
                n: f32,
            ) -> Result<[f32; 4], PowfError> {
                use magetypes::simd::generic::f32x4;
                f32x4::from_array_t(t, xs)
                    .try_powf_with::<C>(n)
                    .map(|v| v.to_array())
            }
            #[inline(never)]
            fn eval8<const C: u128>(
                t: $tok_ty,
                xs: [f32; 8],
                n: f32,
            ) -> Result<[f32; 8], PowfError> {
                use magetypes::simd::generic::f32x8;
                f32x8::from_array_t(t, xs)
                    .try_powf_with::<C>(n)
                    .map(|v| v.to_array())
            }
            #[inline(never)]
            fn eval_tail<const C: u128>(t: $tok_ty, x: f32, n: f32) -> Result<f32, PowfError> {
                Powf::<C>::scalar_with_token(t, x, n)
            }

            let opts = core::hint::black_box(powf_options_from_bits(F));
            let core_bases = core::hint::black_box(&POWF_CORE_BASES);
            let core_exps = core::hint::black_box(&POWF_CORE_EXPS);
            let full_bases = core::hint::black_box(&POWF_FULL_BASES);
            let full_exps = core::hint::black_box(&POWF_FULL_EXPS);

            for &n in core_exps {
                for &xs in core_bases.as_chunks::<4>().0 {
                    let v4 = eval4::<CFG>(token, xs, n).unwrap();
                    let r4 = tok_rt4(token, xs, n, opts).unwrap();
                    for i in 0..4 {
                        let tail = eval_tail::<CFG>(token, xs[i], n).unwrap();
                        assert_f32_eq(v4[i], tail, "powf x4 vs _with_token");
                        assert_f32_eq(r4[i], tail, "powf x4_runtime vs _with_token");
                        if F & 16 == 0 {
                            assert_f32_eq(
                                tail,
                                c_powf_scalar::<CFG>(xs[i], n).unwrap(),
                                "powf BitExact _with_token vs scalar",
                            );
                        }
                    }
                }

                for &xs in core_bases.as_chunks::<8>().0 {
                    let v8 = eval8::<CFG>(token, xs, n).unwrap();
                    let r8 = tok_rt8(token, xs, n, opts).unwrap();
                    for i in 0..8 {
                        let tail = eval_tail::<CFG>(token, xs[i], n).unwrap();
                        assert_f32_eq(v8[i], tail, "powf x8 vs _with_token");
                        assert_f32_eq(r8[i], tail, "powf x8_runtime vs _with_token");
                    }
                }

                define_powf_token_op!(@x16 $has_x16, $tok_ty, token, CFG, opts, n, core_bases, eval_tail);
            }

            if (F & 3 == 1) || (F & 3 == 2 && F & 128 == 0) || (F & 3 == 3) {
                let zpack = core::hint::black_box([0.0, 0.25, 1.0, 16.0]);
                for &n in core_exps {
                    let v4 = eval4::<CFG>(token, zpack, n).unwrap();
                    for i in 0..4 {
                        assert_f32_eq(
                            v4[i],
                            eval_tail::<CFG>(token, zpack[i], n).unwrap(),
                            "powf +0.0 x4 vs _with_token",
                        );
                    }
                }
            }

            if F & 3 == 3 {
                for &n in full_exps {
                    let v8 = eval8::<CFG>(token, *full_bases, n).unwrap();
                    let r8 = tok_rt8(token, *full_bases, n, opts).unwrap();
                    for i in 0..8 {
                        let tail = eval_tail::<CFG>(token, full_bases[i], n).unwrap();
                        assert_f32_eq(v8[i], tail, "powf Full x8 vs _with_token");
                        assert_f32_eq(r8[i], tail, "powf Full x8_runtime vs _with_token");
                        if F & 16 == 0 {
                            assert_f32_eq(
                                tail,
                                c_powf_scalar::<CFG>(full_bases[i], n).unwrap(),
                                "powf Full BitExact _with_token vs scalar",
                            );
                        }
                    }
                }
            } else if F & 32 == 0 {
                let es = eval_tail::<CFG>(token, 2.0, f32::NAN).unwrap_err();
                assert_eq!(es.kind, PowfErrorKind::InvalidExponent);
                assert_eq!(es.lane_mask, 1);
                let ev = eval4::<CFG>(token, [2.0; 4], f32::NAN).unwrap_err();
                assert_eq!(ev.kind, PowfErrorKind::InvalidExponent);
                assert_eq!(ev.lane_mask, 0b1111);

                let mb = [1.0, -2.0, 0.5, f32::NAN];
                let eb = eval4::<CFG>(token, mb, 2.2).unwrap_err();
                assert_eq!(eb.kind, PowfErrorKind::InvalidBase);
                assert_eq!(eb.lane_mask, 0b1010);
                let ebs = eval_tail::<CFG>(token, -2.0, 2.2).unwrap_err();
                assert_eq!(ebs.kind, PowfErrorKind::InvalidBase);
                assert_eq!(ebs.lane_mask, 1);

                if F & 3 < 2 {
                    let oor = [1.0, 1.0e20, 2.0, 1.0e-20];
                    let er = eval4::<CFG>(token, oor, 10.0).unwrap_err();
                    assert_eq!(er.kind, PowfErrorKind::OutsideApproximationRange);
                    assert_eq!(er.lane_mask, 0b1010);
                    let ers = eval_tail::<CFG>(token, 1.0e20, 10.0).unwrap_err();
                    assert_eq!(ers.kind, PowfErrorKind::OutsideApproximationRange);
                    assert_eq!(ers.lane_mask, 1);
                }
            }
        }
    };
    (@def_rt16 yes, $tok_ty:ty) => {
        #[cfg(feature = "w512")]
        #[inline(never)]
        fn tok_rt16(
            t: $tok_ty,
            xs: [f32; 16],
            n: f32,
            opts: PowfOptions,
        ) -> Result<[f32; 16], PowfError> {
            use magetypes::simd::generic::f32x16;
            opts.runtime()
                .simd(f32x16::from_array_t(t, xs), n)
                .map(|v| v.to_array())
        }
    };
    (@def_rt16 no, $tok_ty:ty) => {};
    (@x16 yes, $tok_ty:ty, $token:ident, $cfg:ident, $opts:ident, $n:ident, $core_bases:ident, $eval_tail:ident) => {
        #[cfg(feature = "w512")]
        {
            #[inline(never)]
            fn eval16<const C: u128>(
                t: $tok_ty,
                xs: [f32; 16],
                exp: f32,
            ) -> Result<[f32; 16], PowfError> {
                use magetypes::simd::generic::f32x16;
                f32x16::from_array_t(t, xs)
                    .try_powf_with::<C>(exp)
                    .map(|v| v.to_array())
            }
            let v16 = eval16::<$cfg>($token, *$core_bases, $n).unwrap();
            let r16 = tok_rt16($token, *$core_bases, $n, $opts).unwrap();
            for i in 0..16 {
                let tail = $eval_tail::<$cfg>($token, $core_bases[i], $n).unwrap();
                assert_f32_eq(v16[i], tail, "powf x16 vs _with_token");
                assert_f32_eq(r16[i], tail, "powf x16_runtime vs _with_token");
            }
        }
    };
    (@x16 no, $tok_ty:ty, $token:ident, $cfg:ident, $opts:ident, $n:ident, $core_bases:ident, $eval_tail:ident) => {};
}
