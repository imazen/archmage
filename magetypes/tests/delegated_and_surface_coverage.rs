//! Comprehensive independent-oracle coverage across all 30 generic SIMD types,
//! delegated x86 tokens (`X64V3VnniToken`, `X64V4Token`, `X64V4xToken`,
//! `Avx512Fp16Token`), `ScalarToken`, `X64V3Token`, W512 polyfills, cross-width
//! halves, layout ops, and deprecated constructor forwarders.

use archmage::ScalarToken;
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
use archmage::SimdToken;
use magetypes::numerics::{HalfConvert, IntConvert};
use magetypes::simd::generic::{
    f32x4, f32x8, f64x2, f64x4, i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, i64x2, i64x4, u8x16,
    u8x32, u16x8, u16x16, u32x4, u32x8, u64x2, u64x4,
};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{
    f32x16, f64x8, i8x64, i16x32, i32x16, i64x8, u8x64, u16x32, u32x16, u64x8,
};

macro_rules! check_f32_policy_conversions {
    ($tok:expr, $vec:ident, $lanes:literal) => {{
        const IC_SAT_NE: u32 = IntConvert::new().nearest_even().saturate_all().bits();
        const IC_SAT_TR: u32 = IntConvert::new().trunc().saturate_all().bits();
        const IC_FIN_NE: u32 = IntConvert::new().nearest_even().finite().bits();
        const IC_FIN_TR: u32 = IntConvert::new().trunc().finite().bits();
        const HC_FLUSH: u32 = HalfConvert::new().flush_or_clamp().bits();
        const HC_NORM_ALL: u32 = HalfConvert::new().all_floats().normalize().bits();
        const HC_TRUNC_ALL: u32 = HalfConvert::new().all_floats().normalize().trunc().bits();

        let t = $tok;
        // Ties-to-even (2.5 -> 2, 3.5 -> 4) vs trunc (2.5 -> 2, 3.5 -> 3), plus saturation & NaN -> 0
        let sat_in: [f32; $lanes] = core::array::from_fn(|i| match i % 4 {
            0 => -5.0,
            1 => 2.5,
            2 => 3.5,
            _ => f32::NAN,
        });
        let vsat = $vec::from_array_t(t, sat_in);
        assert_eq!(
            vsat.to_u8_with::<IC_SAT_NE>(),
            core::array::from_fn(|i| match i % 4 {
                0 => 0u8,
                1 => 2,
                2 => 4,
                _ => 0,
            })
        );
        assert_eq!(
            vsat.to_u8_with::<IC_SAT_TR>(),
            core::array::from_fn(|i| match i % 4 {
                0 => 0u8,
                1 => 2,
                2 => 3,
                _ => 0,
            })
        );
        assert_eq!(
            vsat.to_u16_with::<IC_SAT_NE>(),
            core::array::from_fn(|i| match i % 4 {
                0 => 0u16,
                1 => 2,
                2 => 4,
                _ => 0,
            })
        );
        assert_eq!(
            vsat.to_u16_with::<IC_SAT_TR>(),
            core::array::from_fn(|i| match i % 4 {
                0 => 0u16,
                1 => 2,
                2 => 3,
                _ => 0,
            })
        );
        assert_eq!(
            vsat.to_i32_with::<IC_SAT_NE>().to_array(),
            core::array::from_fn(|i| match i % 4 {
                0 => -5i32,
                1 => 2,
                2 => 4,
                _ => 0,
            })
        );
        assert_eq!(
            vsat.to_i32_with::<IC_SAT_TR>().to_array(),
            core::array::from_fn(|i| match i % 4 {
                0 => -5i32,
                1 => 2,
                2 => 3,
                _ => 0,
            })
        );
        assert_eq!(
            vsat.to_u32_with::<IC_SAT_NE>().to_array(),
            core::array::from_fn(|i| match i % 4 {
                0 => 0u32,
                1 => 2,
                2 => 4,
                _ => 0,
            })
        );
        assert_eq!(
            vsat.to_u32_with::<IC_SAT_TR>().to_array(),
            core::array::from_fn(|i| match i % 4 {
                0 => 0u32,
                1 => 2,
                2 => 3,
                _ => 0,
            })
        );
        // Out-of-bounds / NaN rejected under Finite policy
        assert!(vsat.try_to_u8_with::<IC_FIN_NE>().is_err());
        assert!(vsat.try_to_u16_with::<IC_FIN_NE>().is_err());
        assert!(vsat.try_to_i32_with::<IC_FIN_NE>().is_err());
        assert!(vsat.try_to_u32_with::<IC_FIN_NE>().is_err());

        // In-range finite values succeed and match exact rounding
        let fin_in: [f32; $lanes] = core::array::from_fn(|i| match i % 4 {
            0 => 0.0,
            1 => 2.5,
            2 => 3.5,
            _ => 200.25,
        });
        let vfin = $vec::from_array_t(t, fin_in);
        assert_eq!(
            vfin.try_to_u8_with::<IC_FIN_NE>().unwrap(),
            core::array::from_fn(|i| [0u8, 2, 4, 200][i % 4])
        );
        assert_eq!(
            vfin.try_to_u8_with::<IC_FIN_TR>().unwrap(),
            core::array::from_fn(|i| [0u8, 2, 3, 200][i % 4])
        );
        assert_eq!(
            vfin.try_to_u16_with::<IC_FIN_NE>().unwrap(),
            core::array::from_fn(|i| [0u16, 2, 4, 200][i % 4])
        );
        assert_eq!(
            vfin.try_to_i32_with::<IC_FIN_NE>().unwrap().to_array(),
            core::array::from_fn(|i| [0i32, 2, 4, 200][i % 4])
        );
        assert_eq!(
            vfin.try_to_i32_with::<IC_FIN_TR>().unwrap().to_array(),
            core::array::from_fn(|i| [0i32, 2, 3, 200][i % 4])
        );
        assert_eq!(
            vfin.try_to_u32_with::<IC_FIN_NE>().unwrap().to_array(),
            core::array::from_fn(|i| [0u32, 2, 4, 200][i % 4])
        );
        // Large overflow clamps to u16::MAX / u8::MAX
        let big_in: [f32; $lanes] = core::array::from_fn(|i| match i % 4 {
            0 => -1000.0,
            1 => 255.4,
            2 => 65535.4,
            _ => 1e9,
        });
        let vbig = $vec::from_array_t(t, big_in);
        assert_eq!(
            vbig.to_u8_with::<IC_SAT_NE>(),
            core::array::from_fn(|i| [0u8, 255, 255, 255][i % 4])
        );
        assert_eq!(
            vbig.to_u16_with::<IC_SAT_NE>(),
            core::array::from_fn(|i| [0u16, 255, 65535, 65535][i % 4])
        );

        // IEEE binary16 (f16): normal roundtrip, half subnormal flush vs normalize, AllFloats Inf/NaN
        // 0x0001 = 2^-24 (smallest +f16 subnormal), 0x3c00 = 1.0, 0xc100 = -2.5, 0x7bff = 65504.0
        let f16_norm: [u16; $lanes] = core::array::from_fn(|i| [0x8000, 0x3c00, 0xc100, 0x7bff][i % 4]);
        let vf16 = $vec::from_f16_t(t, &f16_norm);
        assert_eq!(
            vf16.to_array().map(f32::to_bits),
            core::array::from_fn(|i| [-0.0f32, 1.0, -2.5, 65504.0][i % 4].to_bits())
        );
        assert_eq!(vf16.to_f16_bits(), f16_norm);
        assert_eq!(vf16.try_to_f16_bits_with::<HC_FLUSH>().unwrap(), f16_norm);

        let f16_sub: [u16; $lanes] = core::array::from_fn(|i| [0x0001, 0x8001, 0x7c00, 0x3c00][i % 4]);
        // FlushOrClamp flushes subnormals 0x0001 / 0x8001 to ±0.0
        let f16_sub_finite: [u16; $lanes] = core::array::from_fn(|i| [0x0001, 0x8001, 0x3c00, 0xbc00][i % 4]);
        let flushed = $vec::from_f16_with_t::<HC_FLUSH>(t, &f16_sub_finite).to_array();
        assert_eq!(flushed[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(flushed[1].to_bits(), (-0.0f32).to_bits());
        // Normalize + AllFloats preserves 2^-24, -2^-24, +Inf, 1.0
        let norm_dec = $vec::try_from_f16_with_t::<HC_NORM_ALL>(t, &f16_sub).unwrap();
        assert_eq!(norm_dec.to_array()[0], f32::from_bits(0x3380_0000)); // 2^-24
        assert_eq!(norm_dec.to_array()[1], f32::from_bits(0xb380_0000)); // -2^-24
        assert_eq!(norm_dec.to_array()[2], f32::INFINITY);
        assert_eq!(norm_dec.to_f16_bits_with::<HC_NORM_ALL>(), f16_sub);
        assert!(norm_dec.try_to_f16_bits_with::<HC_FLUSH>().is_err());
        assert!($vec::try_from_f16_with_t::<HC_FLUSH>(t, &f16_sub).is_err());

        // bfloat16: round-to-nearest-even vs trunc + exact widen
        // 1.0 + 3*2^-8 (0x3f81_8000): nearest-even -> 0x3f82 (tie to even), trunc -> 0x3f81
        let bf_in: [f32; $lanes] = core::array::from_fn(|i| match i % 4 {
            0 => -0.0,
            1 => f32::from_bits(0x3f81_8000),
            2 => f32::from_bits(0x3f80_8000), // exact halfway to even 0x3f80
            _ => -12.5,
        });
        let vbf = $vec::from_array_t(t, bf_in);
        let bf_ne = vbf.to_bf16();
        let bf_tr = vbf.to_bf16_with::<HC_TRUNC_ALL>();
        assert_eq!(bf_ne[0], 0x8000);
        assert_eq!(bf_ne[1], 0x3f82);
        assert_eq!(bf_ne[2], 0x3f80);
        assert_eq!(bf_tr[1], 0x3f81);
        assert_eq!(bf_tr[2], 0x3f80);
        assert_eq!(
            $vec::from_bf16_t(t, &bf_ne).to_array().map(f32::to_bits),
            bf_ne.map(|b| (b as u32) << 16)
        );
        assert!(vbf.try_to_bf16_with::<HC_NORM_ALL>().is_ok());
        assert!(vsat.try_to_bf16_with::<HC_FLUSH>().is_err());

        // RGBA alpha broadcast & replace preserving -0.0 and NaN payload bits
        let rgba_a: [f32; $lanes] = core::array::from_fn(|i| match i % 4 {
            0 => 1.0,
            1 => 2.0,
            2 => 3.0,
            _ => -0.0,
        });
        let rgba_b: [f32; $lanes] = core::array::from_fn(|i| match i % 4 {
            0 => 10.0,
            1 => 20.0,
            2 => 30.0,
            _ => f32::from_bits(0x7fc0_0123),
        });
        let va_rgba = $vec::from_array_t(t, rgba_a);
        let vb_rgba = $vec::from_array_t(t, rgba_b);
        assert_eq!(
            va_rgba.broadcast_rgba_alpha().to_array().map(f32::to_bits),
            [(-0.0f32).to_bits(); $lanes]
        );
        assert_eq!(
            va_rgba.with_rgba_alpha(vb_rgba).to_array().map(f32::to_bits),
            core::array::from_fn(|i| if i % 4 == 3 { 0x7fc0_0123 } else { rgba_a[i].to_bits() })
        );
    }};
}

macro_rules! check_i8_u8_shift_boundaries {
    ($tok:expr, $svec:ident, $uvec:ident, $lanes:literal) => {{
        let t = $tok;
        let sb: [i8; $lanes] =
            core::array::from_fn(|i| [i8::MIN, -1, -64, -17, 0, 1, 0x55, i8::MAX][i % 8]);
        let vs = $svec::from_array_t(t, sb);
        macro_rules! check_s_shift {
            ($n:literal) => {{
                assert_eq!(
                    vs.shl_const::<$n>().to_array(),
                    sb.map(|x| x.wrapping_shl($n))
                );
                assert_eq!(
                    vs.shr_arithmetic_const::<$n>().to_array(),
                    sb.map(|x| x >> $n)
                );
                assert_eq!(
                    vs.shr_logical_const::<$n>().to_array(),
                    sb.map(|x| ((x as u8) >> $n) as i8)
                );
            }};
        }
        check_s_shift!(0);
        check_s_shift!(1);
        check_s_shift!(3);
        check_s_shift!(7);

        let ub: [u8; $lanes] =
            core::array::from_fn(|i| [0u8, 1, 0x55, 0x7f, 0x80, 0xaa, 0xfe, 0xff][i % 8]);
        let vu = $uvec::from_array_t(t, ub);
        macro_rules! check_u_shift {
            ($n:literal) => {{
                assert_eq!(
                    vu.shl_const::<$n>().to_array(),
                    ub.map(|x| x.wrapping_shl($n))
                );
                assert_eq!(vu.shr_logical_const::<$n>().to_array(), ub.map(|x| x >> $n));
            }};
        }
        check_u_shift!(0);
        check_u_shift!(1);
        check_u_shift!(3);
        check_u_shift!(7);
    }};
}

macro_rules! check_block_views {
    ($tok:expr, $vec:ident, $elem:ty, $lanes:literal, $a:expr) => {{
        let t = $tok;
        let a: [$elem; $lanes] = $a;
        let mut v = $vec::from_array_t(t, a);
        assert_eq!(*v.as_array(), a);
        v.as_array_mut()[0] = a[1];
        assert_eq!(v.to_array()[0], a[1]);
        v.as_array_mut()[0] = a[0];

        let bytes_copy = *v.as_bytes();
        assert_eq!($vec::from_bytes_t(t, &bytes_copy).to_array(), a);
        assert_eq!($vec::from_bytes_owned_t(t, bytes_copy).to_array(), a);
        v.as_bytes_mut()[0] ^= 0;
        assert_eq!(v.to_array(), a);

        // Aligned slice via vector values guarantees valid alignment
        let b0 = v;
        let mut b1 = v;
        let backing_elems: &[$elem] = core::slice::from_ref(b0.as_array()).as_flattened();
        let casted = $vec::cast_slice_t(t, backing_elems).expect("aligned slice");
        assert_eq!(casted.len(), 1);
        assert_eq!(casted[0].to_array(), a);
        // Wrong length rejected
        assert!($vec::cast_slice_t(t, &backing_elems[..$lanes - 1]).is_none());

        let backing_mut: &mut [$elem] = core::slice::from_mut(b1.as_array_mut()).as_flattened_mut();
        let casted_mut = $vec::cast_slice_mut_t(t, backing_mut).expect("aligned mut slice");
        assert_eq!(casted_mut.len(), 1);
        assert!($vec::cast_slice_mut_t(t, &mut backing_mut[..$lanes - 1]).is_none());

        #[allow(deprecated)]
        {
            assert_eq!($vec::from_bytes(t, &bytes_copy).to_array(), a);
            assert_eq!($vec::from_bytes_owned(t, bytes_copy).to_array(), a);
            assert_eq!($vec::cast_slice(t, backing_elems).unwrap().len(), 1);
            assert_eq!($vec::cast_slice_mut(t, backing_mut).unwrap().len(), 1);
        }
    }};
}

macro_rules! check_f32_transcendentals {
    ($tok:expr, $vec:ident, $lanes:literal) => {{
        let t = $tok;
        let x_arr: [f32; $lanes] = core::array::from_fn(|i| match i % 4 {
            0 => 0.5,
            1 => 1.5,
            2 => 3.25,
            _ => 10.0,
        });
        let vx = $vec::from_array_t(t, x_arr);
        for (got, x) in vx.log2_lowp_unchecked().to_array().into_iter().zip(x_arr) {
            assert!((got - x.log2()).abs() < 2e-2);
        }
        for (got, x) in vx.exp2_lowp_unchecked().to_array().into_iter().zip(x_arr) {
            assert!((got - x.exp2()).abs() / x.exp2() < 2e-2);
        }
        for (got, x) in vx.ln_lowp_unchecked().to_array().into_iter().zip(x_arr) {
            assert!((got - x.ln()).abs() < 2e-2);
        }
        for (got, x) in vx.exp_lowp_unchecked().to_array().into_iter().zip(x_arr) {
            assert!((got - x.exp()).abs() / x.exp() < 3e-2);
        }
        for (got, x) in vx.log10_lowp_unchecked().to_array().into_iter().zip(x_arr) {
            assert!((got - x.log10()).abs() < 2e-2);
        }
        for (got, x) in vx.pow_lowp_unchecked(1.5).to_array().into_iter().zip(x_arr) {
            assert!((got - x.powf(1.5)).abs() / x.powf(1.5) < 3e-2);
        }
        for (got, x) in vx.log2_midp_precise().to_array().into_iter().zip(x_arr) {
            assert!((got - x.log2()).abs() < 1e-5);
        }
        for (got, x) in vx.exp2_midp_precise().to_array().into_iter().zip(x_arr) {
            assert!((got - x.exp2()).abs() / x.exp2() < 1e-5);
        }
        for (got, x) in vx.ln_midp_unchecked().to_array().into_iter().zip(x_arr) {
            assert!((got - x.ln()).abs() < 1e-5);
        }
        for (got, x) in vx.ln_midp_precise().to_array().into_iter().zip(x_arr) {
            assert!((got - x.ln()).abs() < 1e-5);
        }
        for (got, x) in vx.exp_midp_unchecked().to_array().into_iter().zip(x_arr) {
            assert!((got - x.exp()).abs() / x.exp() < 1e-4);
        }
        for (got, x) in vx.exp_midp_precise().to_array().into_iter().zip(x_arr) {
            assert!((got - x.exp()).abs() / x.exp() < 1e-4);
        }
        for (got, x) in vx.log10_midp_unchecked().to_array().into_iter().zip(x_arr) {
            assert!((got - x.log10()).abs() < 1e-5);
        }
        for (got, x) in vx.log10_midp_precise().to_array().into_iter().zip(x_arr) {
            assert!((got - x.log10()).abs() < 1e-5);
        }
        for (got, x) in vx.pow_midp_unchecked(1.5).to_array().into_iter().zip(x_arr) {
            assert!((got - x.powf(1.5)).abs() / x.powf(1.5) < 1e-4);
        }
        for (got, x) in vx.pow_midp_precise(1.5).to_array().into_iter().zip(x_arr) {
            assert!((got - x.powf(1.5)).abs() / x.powf(1.5) < 1e-4);
        }
        let silu_in: [f32; $lanes] = core::array::from_fn(|i| match i % 4 {
            0 => f32::NEG_INFINITY,
            1 => -2.0,
            2 => 0.0,
            _ => 100.0,
        });
        let silu_out = $vec::from_array_t(t, silu_in).silu_midp().to_array();
        assert_eq!(silu_out[0].to_bits(), (-0.0f32).to_bits());
        let expected_neg2 = -2.0_f32 / (1.0 + (2.0_f32).exp());
        assert!((silu_out[1] - expected_neg2).abs() < 1e-5);
        assert_eq!(silu_out[2], 0.0);
        assert_eq!(silu_out[3], 100.0);
    }};
}

macro_rules! check_common_storage {
    ($tok:expr, $vec:ident, $elem:ty, $lanes:literal, $a:expr, $pad:expr) => {{
        let t = $tok;
        let a: [$elem; $lanes] = $a;
        let v = $vec::from_array_t(t, a);
        assert_eq!(v.to_array(), a);
        assert_eq!($vec::load_t(t, &a).to_array(), a);
        assert_eq!($vec::from_slice_t(t, &a[..]).to_array(), a);
        assert_eq!($vec::splat_t(t, a[0]).to_array(), [a[0]; $lanes]);
        assert_eq!($vec::zero_t(t).to_array(), [0 as $elem; $lanes]);

        let padded = $vec::load_padded_t(t, &a[..1], $pad).to_array();
        assert_eq!(&padded[..1], &a[..1]);
        assert_eq!(&padded[1..], &[$pad; $lanes - 1]);

        let mut buf = [0 as $elem; $lanes];
        v.store(&mut buf);
        assert_eq!(buf, a);

        let mut prefix = [$pad; 1];
        v.store_prefix(&mut prefix);
        assert_eq!(prefix[0], a[0]);

        let mut big = [$pad; $lanes + 1];
        big[..$lanes].copy_from_slice(&a);
        let (chunks, rem) = $vec::partition_slice_t(t, &big);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], a);
        assert_eq!(rem, &[$pad; 1]);
        let (chunks_mut, rem_mut) = $vec::partition_slice_mut_t(t, &mut big);
        assert_eq!(chunks_mut.len(), 1);
        assert_eq!(rem_mut.len(), 1);

        let repr = v.into_repr();
        let rebuilt = $vec::from_repr_t(v.token(), repr);
        assert_eq!(rebuilt.to_array(), a);

        #[allow(deprecated)]
        {
            assert_eq!($vec::splat(t, a[1]).to_array(), [a[1]; $lanes]);
            assert_eq!($vec::zero(t).to_array(), [0 as $elem; $lanes]);
            assert_eq!($vec::load(t, &a).to_array(), a);
            assert_eq!($vec::from_array(t, a).to_array(), a);
            assert_eq!($vec::from_slice(t, &a[..]).to_array(), a);
            let (c2, r2) = $vec::partition_slice(t, &big);
            assert_eq!((c2.len(), r2.len()), (1, 1));
            let (cm2, rm2) = $vec::partition_slice_mut(t, &mut big);
            assert_eq!((cm2.len(), rm2.len()), (1, 1));
            assert_eq!($vec::from_repr(t, repr).to_array(), a);
        }
    }};
}

macro_rules! check_f32_vec {
    ($tok:expr, $vec:ident, $ivec:ident, $uvec:ident, $lanes:literal, $bitcast_i:ident, $to_i:ident, $to_i_round:ident, $from_i_alias:ident) => {{
        let t = $tok;
        let a: [f32; $lanes] = core::array::from_fn(|i| match i % 8 {
            0 => -3.75,
            1 => -2.5,
            2 => -0.25,
            3 => 0.5,
            4 => 1.5,
            5 => 2.5,
            6 => 4.25,
            _ => 9.0,
        });
        let b: [f32; $lanes] = core::array::from_fn(|i| match i % 4 {
            0 => 2.0,
            1 => -4.0,
            2 => 0.5,
            _ => 1.25,
        });
        let c: [f32; $lanes] = core::array::from_fn(|i| (i as f32) * 0.5 - 1.0);

        check_common_storage!(t, $vec, f32, $lanes, a, 7.0);

        let va = $vec::from_array_t(t, a);
        let vb = $vec::from_array_t(t, b);
        let vc = $vec::from_array_t(t, c);

        // Arithmetic & assign ops
        assert_eq!((va + vb).to_array(), core::array::from_fn(|i| a[i] + b[i]));
        assert_eq!((va - vb).to_array(), core::array::from_fn(|i| a[i] - b[i]));
        assert_eq!((va * vb).to_array(), core::array::from_fn(|i| a[i] * b[i]));
        assert_eq!((va / vb).to_array(), core::array::from_fn(|i| a[i] / b[i]));
        assert_eq!((-va).to_array(), a.map(|x| -x));
        let mut acc = va;
        acc += vb;
        acc -= vb;
        acc *= vb;
        acc /= vb;
        for i in 0..$lanes {
            assert!((acc.to_array()[i] - a[i]).abs() <= 1e-5);
        }

        // Unary math & rounding against Rust stdlib f32
        assert_eq!(va.abs().to_array(), a.map(f32::abs));
        assert_eq!(va.floor().to_array(), a.map(f32::floor));
        assert_eq!(va.ceil().to_array(), a.map(f32::ceil));
        assert_eq!(va.round().to_array(), a.map(f32::round_ties_even));
        assert_eq!(va.trunc().to_array(), a.map(f32::trunc));
        let pos = va.abs() + $vec::splat_t(t, 1.0);
        for (got, x) in pos.sqrt().to_array().into_iter().zip(pos.to_array()) {
            assert_eq!(got.to_bits(), x.sqrt().to_bits());
        }

        // Extrema & clamp
        assert_eq!(
            va.min(vb).to_array(),
            core::array::from_fn(|i| a[i].min(b[i]))
        );
        assert_eq!(
            va.max(vb).to_array(),
            core::array::from_fn(|i| a[i].max(b[i]))
        );
        assert_eq!(
            va.minimum(vb).to_array(),
            core::array::from_fn(|i| a[i].min(b[i]))
        );
        assert_eq!(
            va.maximum(vb).to_array(),
            core::array::from_fn(|i| a[i].max(b[i]))
        );
        assert_eq!(
            va.min_fast(vb).to_array(),
            core::array::from_fn(|i| a[i].min(b[i]))
        );
        assert_eq!(
            va.max_fast(vb).to_array(),
            core::array::from_fn(|i| a[i].max(b[i]))
        );
        assert_eq!(
            va.clamp($vec::splat_t(t, -1.0), $vec::splat_t(t, 3.0))
                .to_array(),
            a.map(|x| x.clamp(-1.0, 3.0))
        );

        // FMA / FMS / FNMA variants against f64 exact product
        let exact_fma: [f32; $lanes] =
            core::array::from_fn(|i| ((a[i] as f64) * (b[i] as f64) + (c[i] as f64)) as f32);
        let exact_fms: [f32; $lanes] =
            core::array::from_fn(|i| ((a[i] as f64) * (b[i] as f64) - (c[i] as f64)) as f32);
        let exact_fnma: [f32; $lanes] =
            core::array::from_fn(|i| (-(a[i] as f64) * (b[i] as f64) + (c[i] as f64)) as f32);
        assert_eq!(va.mul_add_fused(vb, vc).to_array(), exact_fma);
        assert_eq!(va.mul_sub_fused(vb, vc).to_array(), exact_fms);
        assert_eq!(va.mul_add_portable(vb, vc).to_array(), exact_fma);
        assert_eq!(va.mul_sub_portable(vb, vc).to_array(), exact_fms);
        assert_eq!(va.mul_neg_add_fused(vb, vc).to_array(), exact_fnma);
        for i in 0..$lanes {
            assert!((va.mul_add_relaxed(vb, vc).to_array()[i] - exact_fma[i]).abs() <= 1e-4);
            assert!((va.mul_sub_relaxed(vb, vc).to_array()[i] - exact_fms[i]).abs() <= 1e-4);
            assert!((va.mul_neg_add_relaxed(vb, vc).to_array()[i] - exact_fnma[i]).abs() <= 1e-4);
        }
        #[allow(deprecated)]
        {
            for i in 0..$lanes {
                assert!((va.mul_add(vb, vc).to_array()[i] - exact_fma[i]).abs() <= 1e-4);
                assert!((va.mul_sub(vb, vc).to_array()[i] - exact_fms[i]).abs() <= 1e-4);
            }
        }

        // Reciprocals & rsqrt (approx + refined + portable)
        for (got, x) in pos.rcp_approx().to_array().into_iter().zip(pos.to_array()) {
            assert!((got - 1.0 / x).abs() / (1.0 / x) < 1e-3);
        }
        for (got, x) in pos.recip().to_array().into_iter().zip(pos.to_array()) {
            assert!((got - 1.0 / x).abs() / (1.0 / x) < 1e-5);
        }
        for (got, x) in pos
            .rsqrt_approx()
            .to_array()
            .into_iter()
            .zip(pos.to_array())
        {
            assert!((got - 1.0 / x.sqrt()).abs() / (1.0 / x.sqrt()) < 1e-3);
        }
        for (got, x) in pos.rsqrt().to_array().into_iter().zip(pos.to_array()) {
            assert!((got - 1.0 / x.sqrt()).abs() / (1.0 / x.sqrt()) < 1e-5);
        }
        let r_approx = pos.rcp_approx_portable();
        for (got, x) in r_approx.to_array().into_iter().zip(pos.to_array()) {
            assert!((got - 1.0 / x).abs() / (1.0 / x) < 5e-3);
        }
        for (got, x) in r_approx
            .recip_newton_portable(pos)
            .to_array()
            .into_iter()
            .zip(pos.to_array())
        {
            assert!((got - 1.0 / x).abs() / (1.0 / x) < 1e-4);
        }
        for (got, x) in pos
            .recip_portable()
            .to_array()
            .into_iter()
            .zip(pos.to_array())
        {
            assert!((got - 1.0 / x).abs() / (1.0 / x) < 1e-5);
        }
        let rs_approx = pos.rsqrt_approx_portable();
        for (got, x) in rs_approx.to_array().into_iter().zip(pos.to_array()) {
            assert!((got - 1.0 / x.sqrt()).abs() / (1.0 / x.sqrt()) < 5e-3);
        }
        for (got, x) in rs_approx
            .rsqrt_newton_portable(pos)
            .to_array()
            .into_iter()
            .zip(pos.to_array())
        {
            assert!((got - 1.0 / x.sqrt()).abs() / (1.0 / x.sqrt()) < 1e-4);
        }
        for (got, x) in pos
            .rsqrt_portable()
            .to_array()
            .into_iter()
            .zip(pos.to_array())
        {
            assert!((got - 1.0 / x.sqrt()).abs() / (1.0 / x.sqrt()) < 1e-5);
        }

        // Reductions
        let sum_ref: f32 = a.iter().copied().sum();
        assert!((va.reduce_add() - sum_ref).abs() <= 1e-4);
        assert_eq!(
            va.reduce_min(),
            a.iter().copied().fold(f32::INFINITY, f32::min)
        );
        assert_eq!(
            va.reduce_max(),
            a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        );

        // Comparisons, blend, bitwise ops & assign ops
        let eq = va.simd_eq(vb);
        let ne = va.simd_ne(vb);
        let lt = va.simd_lt(vb);
        let le = va.simd_le(vb);
        let gt = va.simd_gt(vb);
        let ge = va.simd_ge(vb);
        assert_eq!(
            $vec::blend(lt, va, vb).to_array(),
            core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
        );
        assert_eq!((eq | ne).to_array().map(f32::to_bits), [u32::MAX; $lanes]);
        assert_eq!((lt & gt).to_array().map(f32::to_bits), [0u32; $lanes]);
        assert_eq!(
            (le ^ ge).to_array().map(f32::to_bits),
            core::array::from_fn(|i| if a[i] != b[i] { u32::MAX } else { 0 })
        );
        assert_eq!(
            va.not().to_array().map(f32::to_bits),
            a.map(|x| !x.to_bits())
        );
        let mut bitv = va;
        bitv &= vb;
        bitv |= vc;
        bitv ^= va;
        assert_eq!(
            bitv.to_array().map(f32::to_bits),
            core::array::from_fn(
                |i| ((a[i].to_bits() & b[i].to_bits()) | c[i].to_bits()) ^ a[i].to_bits()
            )
        );

        // Conversions & bitcasts with i32 / u32 / u16 / u8 / 4ch
        let bi = va.bitcast_to_i32();
        assert_eq!(bi.to_array(), a.map(|x| x.to_bits() as i32));
        assert_eq!(
            $vec::from_i32_bitcast_t(t, bi).to_array().map(f32::to_bits),
            a.map(f32::to_bits)
        );
        assert_eq!(va.to_i32().to_array(), a.map(|x| x as i32));
        assert_eq!(
            va.to_i32_round().to_array(),
            a.map(|x| x.round_ties_even() as i32)
        );
        assert_eq!(va.to_i32_saturating().to_array(), a.map(|x| x as i32));
        let iv = $ivec::from_array_t(t, core::array::from_fn(|i| (i as i32) * 3 - 10));
        assert_eq!(
            $vec::from_i32_t(t, iv).to_array(),
            iv.to_array().map(|x| x as f32)
        );
        assert_eq!(va.$bitcast_i().to_array(), bi.to_array());
        assert_eq!(va.$to_i().to_array(), a.map(|x| x as i32));
        assert_eq!(
            va.$to_i_round().to_array(),
            a.map(|x| x.round_ties_even() as i32)
        );
        assert_eq!(
            $vec::$from_i_alias(t, iv).to_array(),
            iv.to_array().map(|x| x as f32)
        );
        #[allow(deprecated)]
        {
            assert_eq!(
                $vec::from_i32_bitcast(t, bi).to_array().map(f32::to_bits),
                a.map(f32::to_bits)
            );
            assert_eq!(
                $vec::from_i32(t, iv).to_array(),
                iv.to_array().map(|x| x as f32)
            );
        }

        // u32 / u16 / u8 / 4ch layout conversions
        let uv = $uvec::from_array_t(t, core::array::from_fn(|i| (i as u32) * 17));
        assert_eq!(
            $vec::from_u32_t(t, uv).to_array(),
            uv.to_array().map(|x| x as f32)
        );
        let u16_in: [u16; $lanes] = core::array::from_fn(|i| (i as u16) * 1000);
        assert_eq!(
            $vec::from_u16_t(t, &u16_in).to_array(),
            u16_in.map(f32::from)
        );
        assert_eq!($vec::from_u16_t(t, &u16_in).to_u16(), u16_in);
        let u8_in: [u8; $lanes] = core::array::from_fn(|i| (i as u8) * 11);
        assert_eq!($vec::from_u8_t(t, &u8_in).to_array(), u8_in.map(f32::from));
        assert_eq!($vec::from_u8_t(t, &u8_in).to_u8(), u8_in);

        let rgba = [
            $vec::from_array_t(t, core::array::from_fn(|i| (i as f32) + 0.0)),
            $vec::from_array_t(t, core::array::from_fn(|i| (i as f32) + 100.0)),
            $vec::from_array_t(t, core::array::from_fn(|i| (i as f32) + 200.0)),
            $vec::from_array_t(t, core::array::from_fn(|i| (i as f32) + 300.0)),
        ];
        let deint = $vec::deinterleave_4ch(rgba);
        let reint = $vec::interleave_4ch(deint);
        for k in 0..4 {
            assert_eq!(reint[k].to_array(), rgba[k].to_array());
        }
    }};
}

macro_rules! check_f64_vec {
    ($tok:expr, $vec:ident, $lanes:literal) => {{
        let t = $tok;
        let a: [f64; $lanes] = core::array::from_fn(|i| match i % 4 {
            0 => -3.75,
            1 => -0.5,
            2 => 2.5,
            _ => 9.0,
        });
        let b: [f64; $lanes] = core::array::from_fn(|i| if i % 2 == 0 { 2.0 } else { -4.0 });
        let c: [f64; $lanes] = core::array::from_fn(|i| (i as f64) * 0.25 - 0.5);

        check_common_storage!(t, $vec, f64, $lanes, a, 11.0);

        let va = $vec::from_array_t(t, a);
        let vb = $vec::from_array_t(t, b);
        let vc = $vec::from_array_t(t, c);

        assert_eq!((va + vb).to_array(), core::array::from_fn(|i| a[i] + b[i]));
        assert_eq!((va - vb).to_array(), core::array::from_fn(|i| a[i] - b[i]));
        assert_eq!((va * vb).to_array(), core::array::from_fn(|i| a[i] * b[i]));
        assert_eq!((va / vb).to_array(), core::array::from_fn(|i| a[i] / b[i]));
        assert_eq!((-va).to_array(), a.map(|x| -x));
        let mut acc = va;
        acc += vb;
        acc -= vb;
        acc *= vb;
        acc /= vb;
        for i in 0..$lanes {
            assert!((acc.to_array()[i] - a[i]).abs() <= 1e-12);
        }

        assert_eq!(va.abs().to_array(), a.map(f64::abs));
        assert_eq!(va.floor().to_array(), a.map(f64::floor));
        assert_eq!(va.ceil().to_array(), a.map(f64::ceil));
        assert_eq!(va.round().to_array(), a.map(f64::round_ties_even));
        assert_eq!(va.trunc().to_array(), a.map(f64::trunc));
        let pos = va.abs() + $vec::splat_t(t, 1.0);
        for (got, x) in pos.sqrt().to_array().into_iter().zip(pos.to_array()) {
            assert_eq!(got.to_bits(), x.sqrt().to_bits());
        }

        assert_eq!(
            va.min(vb).to_array(),
            core::array::from_fn(|i| a[i].min(b[i]))
        );
        assert_eq!(
            va.max(vb).to_array(),
            core::array::from_fn(|i| a[i].max(b[i]))
        );
        assert_eq!(
            va.minimum(vb).to_array(),
            core::array::from_fn(|i| a[i].min(b[i]))
        );
        assert_eq!(
            va.maximum(vb).to_array(),
            core::array::from_fn(|i| a[i].max(b[i]))
        );
        assert_eq!(
            va.min_fast(vb).to_array(),
            core::array::from_fn(|i| a[i].min(b[i]))
        );
        assert_eq!(
            va.max_fast(vb).to_array(),
            core::array::from_fn(|i| a[i].max(b[i]))
        );
        assert_eq!(
            va.clamp($vec::splat_t(t, -1.0), $vec::splat_t(t, 3.0))
                .to_array(),
            a.map(|x| x.clamp(-1.0, 3.0))
        );

        let exact_fma: [f64; $lanes] = core::array::from_fn(|i| a[i].mul_add(b[i], c[i]));
        let exact_fms: [f64; $lanes] = core::array::from_fn(|i| a[i].mul_add(b[i], -c[i]));
        let exact_fnma: [f64; $lanes] = core::array::from_fn(|i| (-a[i]).mul_add(b[i], c[i]));
        assert_eq!(va.mul_add_fused(vb, vc).to_array(), exact_fma);
        assert_eq!(va.mul_sub_fused(vb, vc).to_array(), exact_fms);
        assert_eq!(va.mul_add_portable(vb, vc).to_array(), exact_fma);
        assert_eq!(va.mul_sub_portable(vb, vc).to_array(), exact_fms);
        assert_eq!(va.mul_neg_add_fused(vb, vc).to_array(), exact_fnma);
        for i in 0..$lanes {
            assert!((va.mul_add_relaxed(vb, vc).to_array()[i] - exact_fma[i]).abs() <= 1e-12);
            assert!((va.mul_sub_relaxed(vb, vc).to_array()[i] - exact_fms[i]).abs() <= 1e-12);
            assert!((va.mul_neg_add_relaxed(vb, vc).to_array()[i] - exact_fnma[i]).abs() <= 1e-12);
        }

        let sum_ref: f64 = a.iter().copied().sum();
        assert!((va.reduce_add() - sum_ref).abs() <= 1e-12);
        assert_eq!(
            va.reduce_min(),
            a.iter().copied().fold(f64::INFINITY, f64::min)
        );
        assert_eq!(
            va.reduce_max(),
            a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );

        let eq = va.simd_eq(vb);
        let ne = va.simd_ne(vb);
        let lt = va.simd_lt(vb);
        let le = va.simd_le(vb);
        let gt = va.simd_gt(vb);
        let ge = va.simd_ge(vb);
        assert_eq!(
            eq.to_array().map(f64::to_bits),
            core::array::from_fn(|i| if a[i] == b[i] { u64::MAX } else { 0 })
        );
        assert_eq!(
            ne.to_array().map(f64::to_bits),
            core::array::from_fn(|i| if a[i] != b[i] { u64::MAX } else { 0 })
        );
        assert_eq!(
            le.to_array().map(f64::to_bits),
            core::array::from_fn(|i| if a[i] <= b[i] { u64::MAX } else { 0 })
        );
        assert_eq!(
            ge.to_array().map(f64::to_bits),
            core::array::from_fn(|i| if a[i] >= b[i] { u64::MAX } else { 0 })
        );
        assert_eq!(
            $vec::blend(lt, va, vb).to_array(),
            core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
        );
        assert_eq!((lt & gt).to_array().map(f64::to_bits), [0u64; $lanes]);
        assert_eq!(
            va.not().to_array().map(f64::to_bits),
            a.map(|x| !x.to_bits())
        );
        let mut bitv = va;
        bitv &= vb;
        bitv |= vc;
        bitv ^= va;
        assert_eq!(
            bitv.to_array().map(f64::to_bits),
            core::array::from_fn(
                |i| ((a[i].to_bits() & b[i].to_bits()) | c[i].to_bits()) ^ a[i].to_bits()
            )
        );
    }};
}

macro_rules! check_signed_int_vec {
    ($tok:expr, $vec:ident, $elem:ty, $lanes:literal, $has_mul:tt, $has_sat:tt, $has_shifts:tt) => {{
        let t = $tok;
        // Lane 0 has a[0] > b[0], lane 1 has a[1] < b[1], ensuring mixed masks for all $lanes >= 2.
        let a: [$elem; $lanes] = core::array::from_fn(|i| {
            if i == 0 {
                40 as $elem
            } else {
                ((i as i64) * 13 - 40) as $elem
            }
        });
        let b: [$elem; $lanes] = core::array::from_fn(|i| ((i as i64) * 7 - 19) as $elem);
        check_common_storage!(t, $vec, $elem, $lanes, a, 5 as $elem);

        let va = $vec::from_array_t(t, a);
        let vb = $vec::from_array_t(t, b);
        assert_eq!((va + vb).to_array(), core::array::from_fn(|i| a[i].wrapping_add(b[i])));
        assert_eq!((va - vb).to_array(), core::array::from_fn(|i| a[i].wrapping_sub(b[i])));
        assert_eq!((-va).to_array(), a.map(|x| x.wrapping_neg()));
        assert_eq!(va.abs().to_array(), a.map(|x| x.wrapping_abs()));
        assert_eq!(va.min(vb).to_array(), core::array::from_fn(|i| a[i].min(b[i])));
        assert_eq!(va.max(vb).to_array(), core::array::from_fn(|i| a[i].max(b[i])));
        assert_eq!(va.clamp($vec::splat_t(t, -10), $vec::splat_t(t, 10)).to_array(), a.map(|x| x.clamp(-10, 10)));

        let mut acc = va;
        acc += vb;
        acc -= vb;
        assert_eq!(acc.to_array(), a);

        let mut bitv = va;
        bitv &= vb;
        bitv |= va;
        bitv ^= vb;
        assert_eq!(bitv.to_array(), core::array::from_fn(|i| ((a[i] & b[i]) | a[i]) ^ b[i]));
        assert_eq!(va.not().to_array(), a.map(|x| !x));

        let eq = va.simd_eq(vb);
        let ne = va.simd_ne(vb);
        let lt = va.simd_lt(vb);
        let le = va.simd_le(vb);
        let gt = va.simd_gt(vb);
        let ge = va.simd_ge(vb);
        assert_eq!(eq.to_array(), core::array::from_fn(|i| if a[i] == b[i] { -1 as $elem } else { 0 as $elem }));
        assert_eq!(ne.to_array(), core::array::from_fn(|i| if a[i] != b[i] { -1 as $elem } else { 0 as $elem }));
        assert_eq!(le.to_array(), core::array::from_fn(|i| if a[i] <= b[i] { -1 as $elem } else { 0 as $elem }));
        assert_eq!(gt.to_array(), core::array::from_fn(|i| if a[i] > b[i] { -1 as $elem } else { 0 as $elem }));
        assert_eq!(ge.to_array(), core::array::from_fn(|i| if a[i] >= b[i] { -1 as $elem } else { 0 as $elem }));
        assert_eq!(
            $vec::blend(lt, va, vb).to_array(),
            core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
        );
        assert_eq!(lt.any_true(), (0..$lanes).any(|i| a[i] < b[i]));
        assert_eq!(lt.all_true(), (0..$lanes).all(|i| a[i] < b[i]));
        assert!(lt.any_true());
        assert!(!lt.all_true());
        assert!($vec::splat_t(t, -1).all_true());
        let expected_mask: u64 = a
            .iter()
            .enumerate()
            .fold(0u64, |m, (i, &x)| m | (u64::from(x < 0) << i));
        assert_eq!(u64::from(va.bitmask()), expected_mask);
        assert_eq!(
            va.reduce_add(),
            a.into_iter().fold(0 as $elem, |acc, x| acc.wrapping_add(x))
        );

        check_signed_int_vec!(@mul $has_mul, va, vb, a, b);
        check_signed_int_vec!(@sat $has_sat, t, $vec, $elem, $lanes, va, vb, a, b);
        check_signed_int_vec!(@shifts $has_shifts, va, a, $elem);

        // Boundary abs(): MIN wraps to MIN across every backend (matching wrapping_abs)
        let edge_s: [$elem; $lanes] = core::array::from_fn(|i| match i % 6 {
            0 => <$elem>::MIN,
            1 => <$elem>::MIN + 1,
            2 => -1,
            3 => 0,
            4 => 1,
            _ => <$elem>::MAX,
        });
        assert_eq!(
            $vec::from_array_t(t, edge_s).abs().to_array(),
            edge_s.map(|x| x.wrapping_abs())
        );
    }};
    (@mul true, $va:expr, $vb:expr, $a:expr, $b:expr) => {{
        assert_eq!(($va * $vb).to_array(), core::array::from_fn(|i| $a[i].wrapping_mul($b[i])));
    }};
    (@mul false, $va:expr, $vb:expr, $a:expr, $b:expr) => {};
    (@sat true, $t:expr, $vec:ident, $elem:ty, $lanes:literal, $va:expr, $vb:expr, $a:expr, $b:expr) => {{
        assert_eq!($va.saturating_add($vb).to_array(), core::array::from_fn(|i| $a[i].saturating_add($b[i])));
        assert_eq!($va.saturating_sub($vb).to_array(), core::array::from_fn(|i| $a[i].saturating_sub($b[i])));
        let ext_a: [$elem; $lanes] = core::array::from_fn(|i| if i % 2 == 0 { <$elem>::MIN } else { <$elem>::MAX });
        let ext_b: [$elem; $lanes] = core::array::from_fn(|i| if i % 2 == 0 { -1 } else { 1 });
        let vea = $vec::from_array_t($t, ext_a);
        let veb = $vec::from_array_t($t, ext_b);
        assert_eq!(vea.saturating_add(veb).to_array(), ext_a);
        assert_eq!(vea.saturating_sub($vec::from_array_t($t, ext_b.map(|x| -x))).to_array(), ext_a);
    }};
    (@sat false, $t:expr, $vec:ident, $elem:ty, $lanes:literal, $va:expr, $vb:expr, $a:expr, $b:expr) => {};
    (@shifts true, $va:expr, $a:expr, $elem:ty) => {{
        assert_eq!($va.shl_const::<2>().to_array(), $a.map(|x| x.wrapping_shl(2)));
        assert_eq!($va.shr_arithmetic_const::<2>().to_array(), $a.map(|x| x >> 2));
        assert_eq!(
            $va.shr_logical_const::<2>().to_array(),
            $a.map(|x| {
                let bits = core::mem::size_of::<$elem>() * 8;
                let mask: i64 = if bits == 8 {
                    0x3f
                } else if bits == 16 {
                    0x3fff
                } else if bits == 32 {
                    0x3fff_ffff
                } else {
                    0x3fff_ffff_ffff_ffff
                };
                (((x as i64) >> 2) & mask) as $elem
            })
        );
        #[allow(deprecated)]
        {
            assert_eq!($va.shl::<2>().to_array(), $va.shl_const::<2>().to_array());
            assert_eq!($va.shr_arithmetic::<2>().to_array(), $va.shr_arithmetic_const::<2>().to_array());
            assert_eq!($va.shr_logical::<2>().to_array(), $va.shr_logical_const::<2>().to_array());
        }
    }};
    (@shifts false, $va:expr, $a:expr, $elem:ty) => {};
}

macro_rules! check_unsigned_int_vec {
    ($tok:expr, $vec:ident, $elem:ty, $lanes:literal, $has_mul:tt, $has_sat:tt, $has_shifts:tt) => {{
        let t = $tok;
        // Lane 0 has a[0] > b[0], lane 1 has a[1] < b[1], ensuring mixed masks for all $lanes >= 2.
        let a: [$elem; $lanes] = core::array::from_fn(|i| ((i as u64) * 19 + 5) as $elem);
        let b: [$elem; $lanes] = core::array::from_fn(|i| {
            if i == 1 {
                30 as $elem
            } else {
                ((i as u64) * 11 + 3) as $elem
            }
        });
        check_common_storage!(t, $vec, $elem, $lanes, a, 9 as $elem);

        let va = $vec::from_array_t(t, a);
        let vb = $vec::from_array_t(t, b);
        assert_eq!((va + vb).to_array(), core::array::from_fn(|i| a[i].wrapping_add(b[i])));
        assert_eq!((va - vb).to_array(), core::array::from_fn(|i| a[i].wrapping_sub(b[i])));
        assert_eq!(va.min(vb).to_array(), core::array::from_fn(|i| a[i].min(b[i])));
        assert_eq!(va.max(vb).to_array(), core::array::from_fn(|i| a[i].max(b[i])));
        assert_eq!(va.clamp($vec::splat_t(t, 10), $vec::splat_t(t, 50)).to_array(), a.map(|x| x.clamp(10, 50)));

        let mut acc = va;
        acc += vb;
        acc -= vb;
        assert_eq!(acc.to_array(), a);

        let mut bitv = va;
        bitv &= vb;
        bitv |= va;
        bitv ^= vb;
        assert_eq!(bitv.to_array(), core::array::from_fn(|i| ((a[i] & b[i]) | a[i]) ^ b[i]));
        assert_eq!(va.not().to_array(), a.map(|x| !x));

        let eq = va.simd_eq(vb);
        let ne = va.simd_ne(vb);
        let lt = va.simd_lt(vb);
        let le = va.simd_le(vb);
        let gt = va.simd_gt(vb);
        let ge = va.simd_ge(vb);
        assert_eq!(eq.to_array(), core::array::from_fn(|i| if a[i] == b[i] { <$elem>::MAX } else { 0 as $elem }));
        assert_eq!(ne.to_array(), core::array::from_fn(|i| if a[i] != b[i] { <$elem>::MAX } else { 0 as $elem }));
        assert_eq!(le.to_array(), core::array::from_fn(|i| if a[i] <= b[i] { <$elem>::MAX } else { 0 as $elem }));
        assert_eq!(gt.to_array(), core::array::from_fn(|i| if a[i] > b[i] { <$elem>::MAX } else { 0 as $elem }));
        assert_eq!(ge.to_array(), core::array::from_fn(|i| if a[i] >= b[i] { <$elem>::MAX } else { 0 as $elem }));
        assert_eq!(
            $vec::blend(lt, va, vb).to_array(),
            core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
        );
        assert_eq!(lt.any_true(), (0..$lanes).any(|i| a[i] < b[i]));
        assert_eq!(lt.all_true(), (0..$lanes).all(|i| a[i] < b[i]));
        assert!(lt.any_true());
        assert!(!lt.all_true());
        assert!($vec::splat_t(t, <$elem>::MAX).all_true());
        assert!($vec::splat_t(t, <$elem>::MAX).any_true());
        let expected_mask: u64 = a.iter().enumerate().fold(0u64, |m, (i, &x)| {
            let msb = 1u64 << (core::mem::size_of::<$elem>() * 8 - 1);
            m | (u64::from((x as u64) & msb != 0) << i)
        });
        assert_eq!(u64::from(va.bitmask()), expected_mask);
        assert_eq!(
            va.reduce_add(),
            a.into_iter().fold(0 as $elem, |acc, x| acc.wrapping_add(x))
        );

        check_unsigned_int_vec!(@mul $has_mul, va, vb, a, b);
        check_unsigned_int_vec!(@sat $has_sat, va, vb, a, b);
        check_unsigned_int_vec!(@shifts $has_shifts, va, a);
    }};
    (@mul true, $va:expr, $vb:expr, $a:expr, $b:expr) => {{
        assert_eq!(($va * $vb).to_array(), core::array::from_fn(|i| $a[i].wrapping_mul($b[i])));
    }};
    (@mul false, $va:expr, $vb:expr, $a:expr, $b:expr) => {};
    (@sat true, $va:expr, $vb:expr, $a:expr, $b:expr) => {{
        assert_eq!($va.saturating_add($vb).to_array(), core::array::from_fn(|i| $a[i].saturating_add($b[i])));
        assert_eq!($va.saturating_sub($vb).to_array(), core::array::from_fn(|i| $a[i].saturating_sub($b[i])));
    }};
    (@sat false, $va:expr, $vb:expr, $a:expr, $b:expr) => {};
    (@shifts true, $va:expr, $a:expr) => {{
        assert_eq!($va.shl_const::<2>().to_array(), $a.map(|x| x.wrapping_shl(2)));
        assert_eq!($va.shr_logical_const::<2>().to_array(), $a.map(|x| x >> 2));
        #[allow(deprecated)]
        {
            assert_eq!($va.shl::<2>().to_array(), $va.shl_const::<2>().to_array());
            assert_eq!($va.shr_logical::<2>().to_array(), $va.shr_logical_const::<2>().to_array());
        }
    }};
    (@shifts false, $va:expr, $a:expr) => {};
}

macro_rules! run_w128_w256_for_token {
    ($tok_ty:ty, $get_tok:expr) => {{
        #[inline(never)]
        fn check_floats(t: $tok_ty) {
            check_f32_vec!(
                t,
                f32x4,
                i32x4,
                u32x4,
                4,
                bitcast_i32x4,
                to_i32x4,
                to_i32x4_round,
                from_i32x4_t
            );
            check_f32_vec!(
                t,
                f32x8,
                i32x8,
                u32x8,
                8,
                bitcast_i32x8,
                to_i32x8,
                to_i32x8_round,
                from_i32x8_t
            );
            check_f64_vec!(t, f64x2, 2);
            check_f64_vec!(t, f64x4, 4);

            // Cross-width halves (including split + deprecated from_halves)
            let lo = f32x4::from_array_t(t, [1.0, 2.0, 3.0, 4.0]);
            let hi = f32x4::from_array_t(t, [5.0, 6.0, 7.0, 8.0]);
            let wide = f32x8::from_halves_t(t, lo, hi);
            assert_eq!(wide.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            assert_eq!(wide.low().to_array(), [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(wide.high().to_array(), [5.0, 6.0, 7.0, 8.0]);
            let (s_lo, s_hi) = wide.split();
            assert_eq!(s_lo.to_array(), [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(s_hi.to_array(), [5.0, 6.0, 7.0, 8.0]);
            #[allow(deprecated)]
            {
                assert_eq!(f32x8::from_halves(t, lo, hi).to_array(), wide.to_array());
            }
        }

        #[inline(never)]
        fn check_f32_block_and_transcendentals(t: $tok_ty) {
            const IC: u32 = IntConvert::new().bits();
            check_block_views!(t, f32x4, f32, 4, [1.0, -2.5, 3.25, 4.5]);
            check_block_views!(
                t,
                f32x8,
                f32,
                8,
                [1.0, -2.5, 3.25, 4.5, -5.0, 6.0, 7.5, -8.25]
            );
            check_block_views!(t, f64x2, f64, 2, [-1.25, 9.5]);
            check_block_views!(t, f64x4, f64, 4, [-1.25, 2.5, -3.75, 9.5]);
            check_block_views!(
                t,
                i8x16,
                i8,
                16,
                core::array::from_fn(|i| (i as i8) * 5 - 30)
            );
            check_block_views!(t, i32x4, i32, 4, [-10, 0, 7, 99]);
            check_block_views!(t, i32x8, i32, 8, [-10, -5, 0, 1, 2, 7, 42, 99]);
            check_block_views!(t, u32x4, u32, 4, [0, 1, 12345, u32::MAX]);

            // f32x4 interleave, bitcast_ref/mut, RGBA/RGB load/store, transpose
            let mut v4a = f32x4::from_array_t(t, [10.0, 20.0, 30.0, 40.0]);
            let v4b = f32x4::from_array_t(t, [100.0, 110.0, 120.0, 130.0]);
            assert_eq!(v4a.interleave_lo(v4b).to_array(), [10.0, 100.0, 20.0, 110.0]);
            assert_eq!(v4a.interleave_hi(v4b).to_array(), [30.0, 120.0, 40.0, 130.0]);
            let (ilo, ihi) = v4a.interleave(v4b);
            assert_eq!(ilo.to_array(), [10.0, 100.0, 20.0, 110.0]);
            assert_eq!(ihi.to_array(), [30.0, 120.0, 40.0, 130.0]);
            assert_eq!(v4a.bitcast_ref_i32().to_array(), v4a.to_array().map(|x| x.to_bits() as i32));
            let orig0 = v4a.bitcast_mut_i32().to_array()[0];
            assert_eq!(orig0, 10.0_f32.to_bits() as i32);

            let rgba16: [u8; 16] = core::array::from_fn(|i| (i as u8) * 13);
            let (r4, g4, b4, a4) = f32x4::load_rgba_u8_t(t, &rgba16);
            let mut rgba16_out = [0u8; 16];
            f32x4::store_rgba_u8(r4, g4, b4, a4, &mut rgba16_out);
            assert_eq!(rgba16_out, rgba16);
            let mut rgba16_w = [0u8; 16];
            f32x4::store_rgba_u8_with::<IC>(r4, g4, b4, a4, &mut rgba16_w);
            assert_eq!(rgba16_w, rgba16);
            let mut rgb12 = [0u8; 12];
            f32x4::store_rgb_u8(r4, g4, b4, &mut rgb12);
            let mut rgb12_w = [0u8; 12];
            f32x4::store_rgb_u8_with::<IC>(r4, g4, b4, &mut rgb12_w);
            assert_eq!(rgb12, rgb12_w);
            for px in 0..4 {
                assert_eq!(&rgb12[px * 3..px * 3 + 3], &rgba16[px * 4..px * 4 + 3]);
            }
            #[allow(deprecated)]
            {
                let (dr, dg, db, da) = f32x4::load_4_rgba_u8_t(t, &rgba16);
                assert_eq!(f32x4::store_4_rgba_u8(dr, dg, db, da), rgba16);
                let (dr2, dg2, db2, da2) = f32x4::load_4_rgba_u8(t, &rgba16);
                assert_eq!(f32x4::store_4_rgba_u8(dr2, dg2, db2, da2), rgba16);
                assert_eq!(f32x4::from_u8(t, &[1, 2, 3, 4]).to_array(), [1.0, 2.0, 3.0, 4.0]);
            }
            let mut mat4 = [r4, g4, b4, a4];
            f32x4::transpose_4x4(&mut mat4);
            f32x4::transpose_4x4(&mut mat4);
            assert_eq!(mat4[0].to_array(), r4.to_array());

            // f32x8 interleave, bitcast_ref/mut, RGBA/RGB, 8x8 transpose
            let mut v8a = f32x8::from_array_t(t, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            let v8b = f32x8::from_array_t(t, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
            // In-128-bit-lane interleave: [a0,b0,a1,b1,a4,b4,a5,b5] and [a2,b2,a3,b3,a6,b6,a7,b7]
            assert_eq!(
                v8a.interleave_lo(v8b).to_array(),
                [0.0, 10.0, 1.0, 11.0, 4.0, 14.0, 5.0, 15.0]
            );
            assert_eq!(
                v8a.interleave_hi(v8b).to_array(),
                [2.0, 12.0, 3.0, 13.0, 6.0, 16.0, 7.0, 17.0]
            );
            let (i8lo, i8hi) = v8a.interleave(v8b);
            assert_eq!(
                i8lo.to_array(),
                [0.0, 10.0, 1.0, 11.0, 4.0, 14.0, 5.0, 15.0]
            );
            assert_eq!(
                i8hi.to_array(),
                [2.0, 12.0, 3.0, 13.0, 6.0, 16.0, 7.0, 17.0]
            );
            assert_eq!(v8a.bitcast_ref_i32().to_array()[1], 1.0_f32.to_bits() as i32);
            assert_eq!(v8a.bitcast_mut_i32().to_array()[2], 2.0_f32.to_bits() as i32);

            let rgba32: [u8; 32] = core::array::from_fn(|i| (i as u8) * 7);
            let (r8, g8, b8, a8) = f32x8::load_rgba_u8_t(t, &rgba32);
            let mut rgba32_out = [0u8; 32];
            f32x8::store_rgba_u8(r8, g8, b8, a8, &mut rgba32_out);
            assert_eq!(rgba32_out, rgba32);
            let mut rgba32_w = [0u8; 32];
            f32x8::store_rgba_u8_with::<IC>(r8, g8, b8, a8, &mut rgba32_w);
            assert_eq!(rgba32_w, rgba32);
            let mut rgb24 = [0u8; 24];
            f32x8::store_rgb_u8(r8, g8, b8, &mut rgb24);
            let mut rgb24_w = [0u8; 24];
            f32x8::store_rgb_u8_with::<IC>(r8, g8, b8, &mut rgb24_w);
            assert_eq!(rgb24, rgb24_w);
            for px in 0..8 {
                assert_eq!(&rgb24[px * 3..px * 3 + 3], &rgba32[px * 4..px * 4 + 3]);
            }
            let block64: [f32; 64] = core::array::from_fn(|i| i as f32);
            let mut rows8 = f32x8::load_8x8_t(t, &block64);
            f32x8::transpose_8x8(&mut rows8);
            let mut block64_out = [0.0f32; 64];
            f32x8::store_8x8(&rows8, &mut block64_out);
            for r in 0..8 {
                for c in 0..8 {
                    assert_eq!(block64_out[r * 8 + c], block64[c * 8 + r]);
                }
            }
            #[allow(deprecated)]
            {
                let (dr, dg, db, da) = f32x8::load_8_rgba_u8_t(t, &rgba32);
                assert_eq!(f32x8::store_8_rgba_u8(dr, dg, db, da), rgba32);
                let (dr2, dg2, db2, da2) = f32x8::load_8_rgba_u8(t, &rgba32);
                assert_eq!(f32x8::store_8_rgba_u8(dr2, dg2, db2, da2), rgba32);
                assert_eq!(f32x8::from_u8(t, &[1, 2, 3, 4, 5, 6, 7, 8]).to_u8(), [1, 2, 3, 4, 5, 6, 7, 8]);
                assert_eq!(f32x8::load_8x8(t, &block64)[0].to_array(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            }

            check_f32_transcendentals!(t, f32x4, 4);
            check_f32_transcendentals!(t, f32x8, 8);
            check_f32_policy_conversions!(t, f32x4, 4);
            check_f32_policy_conversions!(t, f32x8, 8);
        }

        #[inline(never)]
        fn check_narrow_ints(t: $tok_ty) {
            check_signed_int_vec!(t, i8x16, i8, 16, false, true, true);
            check_signed_int_vec!(t, i8x32, i8, 32, false, true, true);
            check_unsigned_int_vec!(t, u8x16, u8, 16, false, true, true);
            check_unsigned_int_vec!(t, u8x32, u8, 32, false, true, true);
            check_i8_u8_shift_boundaries!(t, i8x16, u8x16, 16);
            check_i8_u8_shift_boundaries!(t, i8x32, u8x32, 32);
            check_signed_int_vec!(t, i16x8, i16, 8, true, true, true);
            check_signed_int_vec!(t, i16x16, i16, 16, true, true, true);
            check_unsigned_int_vec!(t, u16x8, u16, 8, true, true, true);
            check_unsigned_int_vec!(t, u16x16, u16, 16, true, true, true);

            // Boundary-value parity: i16::MIN*i16::MIN wrap in madd_adjacent, MIN/MAX abs_diff, narrow_saturating
            let m16_a = [i16::MIN, i16::MIN, i16::MIN, i16::MAX, i16::MAX, i16::MAX, -1, 1];
            let m16_b = [i16::MIN, i16::MIN, i16::MAX, i16::MIN, i16::MAX, i16::MAX, i16::MIN, i16::MAX];
            let vm16_8a = i16x8::from_array_t(t, m16_a);
            let vm16_8b = i16x8::from_array_t(t, m16_b);
            let expected_madd8: [i32; 4] = core::array::from_fn(|i| {
                ((m16_a[2 * i] as i32) * (m16_b[2 * i] as i32))
                    .wrapping_add((m16_a[2 * i + 1] as i32) * (m16_b[2 * i + 1] as i32))
            });
            assert_eq!(expected_madd8[0], i32::MIN);
            assert_eq!(vm16_8a.madd_adjacent(vm16_8b).to_array(), expected_madd8);
            assert_eq!(
                vm16_8a.abs_diff(vm16_8b).to_array(),
                core::array::from_fn(|i| m16_a[i].abs_diff(m16_b[i]))
            );
            assert_eq!(vm16_8a.abs_diff(vm16_8b).to_array()[2], u16::MAX);
            let n16_lo = [i16::MIN, -129, -128, -1, 0, 127, 128, 255];
            let n16_hi = [256, i16::MAX, -1000, 1000, -128, 127, 0, 255];
            let vn16_lo = i16x8::from_array_t(t, n16_lo);
            let vn16_hi = i16x8::from_array_t(t, n16_hi);
            assert_eq!(
                vn16_lo.narrow_saturating_i8(vn16_hi).to_array(),
                [-128, -128, -128, -1, 0, 127, 127, 127, 127, 127, -128, 127, -128, 127, 0, 127]
            );
            assert_eq!(
                vn16_lo.narrow_saturating_u8(vn16_hi).to_array(),
                [0, 0, 0, 0, 0, 127, 128, 255, 255, 255, 0, 255, 0, 127, 0, 255]
            );
            let m16_16a: [i16; 16] = core::array::from_fn(|i| m16_a[i % 8]);
            let m16_16b: [i16; 16] = core::array::from_fn(|i| m16_b[i % 8]);
            let vm16_16a = i16x16::from_array_t(t, m16_16a);
            let vm16_16b = i16x16::from_array_t(t, m16_16b);
            assert_eq!(
                vm16_16a.madd_adjacent(vm16_16b).to_array(),
                core::array::from_fn(|i| expected_madd8[i % 4])
            );
            assert_eq!(
                vm16_16a.abs_diff(vm16_16b).to_array(),
                core::array::from_fn(|i| m16_16a[i].abs_diff(m16_16b[i]))
            );
            let u8_ext_a: [u8; 16] = core::array::from_fn(|i| if i % 2 == 0 { 0 } else { 255 });
            let u8_ext_b: [u8; 16] = core::array::from_fn(|i| if i % 2 == 0 { 255 } else { 0 });
            assert_eq!(
                u8x16::from_array_t(t, u8_ext_a)
                    .abs_diff(u8x16::from_array_t(t, u8_ext_b))
                    .to_array(),
                [255u8; 16]
            );
            assert_eq!(
                u8x16::from_array_t(t, u8_ext_a)
                    .saturating_add(u8x16::splat_t(t, 1))
                    .to_array(),
                core::array::from_fn(|i| if i % 2 == 0 { 1 } else { 255 })
            );
            assert_eq!(
                u8x16::from_array_t(t, u8_ext_a)
                    .saturating_sub(u8x16::splat_t(t, 1))
                    .to_array(),
                core::array::from_fn(|i| if i % 2 == 0 { 0 } else { 254 })
            );

            // Widen, narrow, abs_diff, sum_abs_diff, pairwise_widen_add, madd_adjacent, uniform shifts
            let s8_16: [i8; 16] = core::array::from_fn(|i| (i as i8).wrapping_mul(11).wrapping_sub(80));
            let vs8_16 = i8x16::from_array_t(t, s8_16);
            assert_eq!(vs8_16.widen_low().to_array(), core::array::from_fn(|i| s8_16[i] as i16));
            assert_eq!(vs8_16.widen_high().to_array(), core::array::from_fn(|i| s8_16[i + 8] as i16));
            let s8_32: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(7).wrapping_sub(90));
            let vs8_32 = i8x32::from_array_t(t, s8_32);
            assert_eq!(vs8_32.widen_low().to_array(), core::array::from_fn(|i| s8_32[i] as i16));
            assert_eq!(vs8_32.widen_high().to_array(), core::array::from_fn(|i| s8_32[i + 16] as i16));

            let u8_16a: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(15).wrapping_add(3));
            let u8_16b: [u8; 16] = core::array::from_fn(|i| 220u8.wrapping_sub((i as u8).wrapping_mul(11)));
            let vu8_16a = u8x16::from_array_t(t, u8_16a);
            let vu8_16b = u8x16::from_array_t(t, u8_16b);
            assert_eq!(vu8_16a.widen_low().to_array(), core::array::from_fn(|i| u8_16a[i] as u16));
            assert_eq!(vu8_16a.widen_high().to_array(), core::array::from_fn(|i| u8_16a[i + 8] as u16));
            assert_eq!(vu8_16a.abs_diff(vu8_16b).to_array(), core::array::from_fn(|i| u8_16a[i].abs_diff(u8_16b[i])));
            assert_eq!(vu8_16a.sum_abs_diff(vu8_16b), (0..16).map(|i| u8_16a[i].abs_diff(u8_16b[i]) as u32).sum::<u32>());
            assert_eq!(vu8_16a.pairwise_widen_add().to_array(), core::array::from_fn(|i| u8_16a[2 * i] as u16 + u8_16a[2 * i + 1] as u16));

            let u8_32a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(5));
            let u8_32b: [u8; 32] = core::array::from_fn(|i| 240u8.wrapping_sub((i as u8).wrapping_mul(5)));
            let vu8_32a = u8x32::from_array_t(t, u8_32a);
            let vu8_32b = u8x32::from_array_t(t, u8_32b);
            assert_eq!(vu8_32a.widen_low().to_array(), core::array::from_fn(|i| u8_32a[i] as u16));
            assert_eq!(vu8_32a.widen_high().to_array(), core::array::from_fn(|i| u8_32a[i + 16] as u16));
            assert_eq!(vu8_32a.abs_diff(vu8_32b).to_array(), core::array::from_fn(|i| u8_32a[i].abs_diff(u8_32b[i])));
            assert_eq!(vu8_32a.sum_abs_diff(vu8_32b), (0..32).map(|i| u8_32a[i].abs_diff(u8_32b[i]) as u32).sum::<u32>());
            assert_eq!(vu8_32a.pairwise_widen_add().to_array(), core::array::from_fn(|i| u8_32a[2 * i] as u16 + u8_32a[2 * i + 1] as u16));

            let s16_8a: [i16; 8] = [-500, -129, -128, 0, 127, 128, 300, 1000];
            let s16_8b: [i16; 8] = [10, -20, 30, -40, 50, -60, 70, -80];
            let vs16_8a = i16x8::from_array_t(t, s16_8a);
            let vs16_8b = i16x8::from_array_t(t, s16_8b);
            assert_eq!(vs16_8a.shl_uniform(3).to_array(), s16_8a.map(|x| x.wrapping_shl(3)));
            assert_eq!(vs16_8a.shr_logical_uniform(3).to_array(), s16_8a.map(|x| ((x as u16) >> 3) as i16));
            assert_eq!(vs16_8a.shr_arithmetic_uniform(3).to_array(), s16_8a.map(|x| x >> 3));
            assert_eq!(vs16_8a.widen_low().to_array(), core::array::from_fn(|i| s16_8a[i] as i32));
            assert_eq!(vs16_8a.widen_high().to_array(), core::array::from_fn(|i| s16_8a[i + 4] as i32));
            assert_eq!(
                vs16_8a.narrow_saturating_i8(vs16_8b).to_array(),
                core::array::from_fn(|i| {
                    let v = if i < 8 { s16_8a[i] } else { s16_8b[i - 8] };
                    v.clamp(i8::MIN as i16, i8::MAX as i16) as i8
                })
            );
            assert_eq!(
                vs16_8a.narrow_saturating_u8(vs16_8b).to_array(),
                core::array::from_fn(|i| {
                    let v = if i < 8 { s16_8a[i] } else { s16_8b[i - 8] };
                    v.clamp(0, 255) as u8
                })
            );
            assert_eq!(vs16_8a.abs_diff(vs16_8b).to_array(), core::array::from_fn(|i| s16_8a[i].abs_diff(s16_8b[i])));
            assert_eq!(
                vs16_8a.madd_adjacent(vs16_8b).to_array(),
                core::array::from_fn(|i| (s16_8a[2 * i] as i32) * (s16_8b[2 * i] as i32) + (s16_8a[2 * i + 1] as i32) * (s16_8b[2 * i + 1] as i32))
            );

            let s16_16a: [i16; 16] = core::array::from_fn(|i| s16_8a[i % 8] + (i as i16));
            let s16_16b: [i16; 16] = core::array::from_fn(|i| s16_8b[i % 8] - (i as i16));
            let vs16_16a = i16x16::from_array_t(t, s16_16a);
            let vs16_16b = i16x16::from_array_t(t, s16_16b);
            assert_eq!(vs16_16a.shl_uniform(2).to_array(), s16_16a.map(|x| x.wrapping_shl(2)));
            assert_eq!(vs16_16a.shr_logical_uniform(2).to_array(), s16_16a.map(|x| ((x as u16) >> 2) as i16));
            assert_eq!(vs16_16a.shr_arithmetic_uniform(2).to_array(), s16_16a.map(|x| x >> 2));
            assert_eq!(vs16_16a.widen_low().to_array(), core::array::from_fn(|i| s16_16a[i] as i32));
            assert_eq!(vs16_16a.widen_high().to_array(), core::array::from_fn(|i| s16_16a[i + 8] as i32));
            assert_eq!(
                vs16_16a.narrow_saturating_i8(vs16_16b).to_array(),
                core::array::from_fn(|i| {
                    let v = if i < 16 { s16_16a[i] } else { s16_16b[i - 16] };
                    v.clamp(i8::MIN as i16, i8::MAX as i16) as i8
                })
            );
            assert_eq!(
                vs16_16a.narrow_saturating_u8(vs16_16b).to_array(),
                core::array::from_fn(|i| {
                    let v = if i < 16 { s16_16a[i] } else { s16_16b[i - 16] };
                    v.clamp(0, 255) as u8
                })
            );
            assert_eq!(vs16_16a.abs_diff(vs16_16b).to_array(), core::array::from_fn(|i| s16_16a[i].abs_diff(s16_16b[i])));
            assert_eq!(
                vs16_16a.madd_adjacent(vs16_16b).to_array(),
                core::array::from_fn(|i| (s16_16a[2 * i] as i32) * (s16_16b[2 * i] as i32) + (s16_16a[2 * i + 1] as i32) * (s16_16b[2 * i + 1] as i32))
            );

            let u16_8a: [u16; 8] = core::array::from_fn(|i| (i as u16) * 1000 + 17);
            let vu16_8a = u16x8::from_array_t(t, u16_8a);
            assert_eq!(vu16_8a.shl_uniform(3).to_array(), u16_8a.map(|x| x.wrapping_shl(3)));
            assert_eq!(vu16_8a.shr_logical_uniform(3).to_array(), u16_8a.map(|x| x >> 3));
            assert_eq!(vu16_8a.widen_low().to_array(), core::array::from_fn(|i| u16_8a[i] as u32));
            assert_eq!(vu16_8a.widen_high().to_array(), core::array::from_fn(|i| u16_8a[i + 4] as u32));
            assert_eq!(vu16_8a.pairwise_widen_add().to_array(), core::array::from_fn(|i| u16_8a[2 * i] as u32 + u16_8a[2 * i + 1] as u32));

            let u16_16a: [u16; 16] = core::array::from_fn(|i| (i as u16) * 700 + 11);
            let vu16_16a = u16x16::from_array_t(t, u16_16a);
            assert_eq!(vu16_16a.shl_uniform(2).to_array(), u16_16a.map(|x| x.wrapping_shl(2)));
            assert_eq!(vu16_16a.shr_logical_uniform(2).to_array(), u16_16a.map(|x| x >> 2));
            assert_eq!(vu16_16a.widen_low().to_array(), core::array::from_fn(|i| u16_16a[i] as u32));
            assert_eq!(vu16_16a.widen_high().to_array(), core::array::from_fn(|i| u16_16a[i + 8] as u32));
            assert_eq!(vu16_16a.pairwise_widen_add().to_array(), core::array::from_fn(|i| u16_16a[2 * i] as u32 + u16_16a[2 * i + 1] as u32));
        }

        #[inline(never)]
        fn check_wide_ints_and_bitcasts(t: $tok_ty) {
            check_signed_int_vec!(t, i32x4, i32, 4, true, false, true);
            check_signed_int_vec!(t, i32x8, i32, 8, true, false, true);
            check_unsigned_int_vec!(t, u32x4, u32, 4, true, false, true);
            check_unsigned_int_vec!(t, u32x8, u32, 8, true, false, true);
            check_signed_int_vec!(t, i64x2, i64, 2, false, false, true);
            check_signed_int_vec!(t, i64x4, i64, 4, false, false, true);
            check_unsigned_int_vec!(t, u64x2, u64, 2, false, false, true);
            check_unsigned_int_vec!(t, u64x4, u64, 4, false, false, true);

            // i32/u32 uniform shifts and saturating narrow to i16/u16 (incl. i32::MIN/MAX corners)
            let s32_4a = [i32::MIN, -32769, -32768, -1];
            let s32_4b = [0, 32767, 65535, i32::MAX];
            let vs32_4a = i32x4::from_array_t(t, s32_4a);
            let vs32_4b = i32x4::from_array_t(t, s32_4b);
            assert_eq!(vs32_4a.shl_uniform(3).to_array(), s32_4a.map(|x| x.wrapping_shl(3)));
            assert_eq!(vs32_4a.shr_logical_uniform(3).to_array(), s32_4a.map(|x| ((x as u32) >> 3) as i32));
            assert_eq!(vs32_4a.shr_arithmetic_uniform(3).to_array(), s32_4a.map(|x| x >> 3));
            assert_eq!(
                vs32_4a.narrow_saturating_i16(vs32_4b).to_array(),
                [i16::MIN, i16::MIN, i16::MIN, -1, 0, i16::MAX, i16::MAX, i16::MAX]
            );
            assert_eq!(
                vs32_4a.narrow_saturating_u16(vs32_4b).to_array(),
                [0, 0, 0, 0, 0, 32767, 65535, 65535]
            );
            let s32_8a: [i32; 8] = [i32::MIN, -32769, -32768, -1, 0, 32767, 32768, i32::MAX];
            let s32_8b: [i32; 8] = [65535, 65536, i32::MIN, i32::MAX, -1, 0, 1, 32767];
            let vs32_8a = i32x8::from_array_t(t, s32_8a);
            let vs32_8b = i32x8::from_array_t(t, s32_8b);
            assert_eq!(vs32_8a.shl_uniform(2).to_array(), s32_8a.map(|x| x.wrapping_shl(2)));
            assert_eq!(vs32_8a.shr_logical_uniform(2).to_array(), s32_8a.map(|x| ((x as u32) >> 2) as i32));
            assert_eq!(vs32_8a.shr_arithmetic_uniform(2).to_array(), s32_8a.map(|x| x >> 2));
            assert_eq!(
                vs32_8a.narrow_saturating_i16(vs32_8b).to_array(),
                core::array::from_fn(|i| {
                    let v = if i < 8 { s32_8a[i] } else { s32_8b[i - 8] };
                    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
                })
            );
            assert_eq!(
                vs32_8a.narrow_saturating_u16(vs32_8b).to_array(),
                core::array::from_fn(|i| {
                    let v = if i < 8 { s32_8a[i] } else { s32_8b[i - 8] };
                    v.clamp(0, u16::MAX as i32) as u16
                })
            );
            let u32_4a = [0u32, 7, 0x8000_0004, u32::MAX];
            let vu32_4a = u32x4::from_array_t(t, u32_4a);
            assert_eq!(vu32_4a.shl_uniform(3).to_array(), u32_4a.map(|x| x.wrapping_shl(3)));
            assert_eq!(vu32_4a.shr_logical_uniform(3).to_array(), u32_4a.map(|x| x >> 3));
            let u32_8a: [u32; 8] = core::array::from_fn(|i| u32_4a[i % 4].wrapping_add(i as u32));
            let vu32_8a = u32x8::from_array_t(t, u32_8a);
            assert_eq!(vu32_8a.shl_uniform(2).to_array(), u32_8a.map(|x| x.wrapping_shl(2)));
            assert_eq!(vu32_8a.shr_logical_uniform(2).to_array(), u32_8a.map(|x| x >> 2));

            // Cross-type bitcasts (value + ref + mut)
            let mut i8_16 = i8x16::splat_t(t, -1);
            assert_eq!(i8_16.bitcast_u8x16().to_array(), [255u8; 16]);
            assert_eq!(i8_16.bitcast_ref_u8x16().to_array(), [255u8; 16]);
            assert_eq!(i8_16.bitcast_mut_u8x16().to_array(), [255u8; 16]);
            let mut u8_16 = u8x16::splat_t(t, 255);
            assert_eq!(u8_16.bitcast_i8x16().to_array(), [-1i8; 16]);
            assert_eq!(u8_16.bitcast_ref_i8x16().to_array(), [-1i8; 16]);
            assert_eq!(u8_16.bitcast_mut_i8x16().to_array(), [-1i8; 16]);
            let mut i8_32 = i8x32::splat_t(t, -2);
            assert_eq!(i8_32.bitcast_u8x32().to_array(), [254u8; 32]);
            assert_eq!(i8_32.bitcast_ref_u8x32().to_array(), [254u8; 32]);
            assert_eq!(i8_32.bitcast_mut_u8x32().to_array(), [254u8; 32]);
            let mut u8_32 = u8x32::splat_t(t, 254);
            assert_eq!(u8_32.bitcast_i8x32().to_array(), [-2i8; 32]);
            assert_eq!(u8_32.bitcast_ref_i8x32().to_array(), [-2i8; 32]);
            assert_eq!(u8_32.bitcast_mut_i8x32().to_array(), [-2i8; 32]);
            let mut i16_8 = i16x8::splat_t(t, -1);
            assert_eq!(i16_8.bitcast_u16x8().to_array(), [u16::MAX; 8]);
            assert_eq!(i16_8.bitcast_ref_u16x8().to_array(), [u16::MAX; 8]);
            assert_eq!(i16_8.bitcast_mut_u16x8().to_array(), [u16::MAX; 8]);
            let mut u16_8 = i16_8.bitcast_u16x8();
            assert_eq!(u16_8.bitcast_i16x8().to_array(), [-1i16; 8]);
            assert_eq!(u16_8.bitcast_ref_i16x8().to_array(), [-1i16; 8]);
            assert_eq!(u16_8.bitcast_mut_i16x8().to_array(), [-1i16; 8]);
            let mut i16_16 = i16x16::splat_t(t, -1);
            assert_eq!(i16_16.bitcast_u16x16().to_array(), [u16::MAX; 16]);
            assert_eq!(i16_16.bitcast_ref_u16x16().to_array(), [u16::MAX; 16]);
            assert_eq!(i16_16.bitcast_mut_u16x16().to_array(), [u16::MAX; 16]);
            let mut u16_16 = i16_16.bitcast_u16x16();
            assert_eq!(u16_16.bitcast_i16x16().to_array(), [-1i16; 16]);
            assert_eq!(u16_16.bitcast_ref_i16x16().to_array(), [-1i16; 16]);
            assert_eq!(u16_16.bitcast_mut_i16x16().to_array(), [-1i16; 16]);
            let i32_4 = i32x4::splat_t(t, 0x3f80_0000);
            assert_eq!(i32_4.bitcast_f32x4().to_array(), [1.0f32; 4]);
            let mut u32_4 = u32x4::splat_t(t, u32::MAX);
            assert_eq!(u32_4.bitcast_to_i32().to_array(), [-1i32; 4]);
            assert_eq!(u32_4.bitcast_i32x4().to_array(), [-1i32; 4]);
            assert_eq!(u32_4.bitcast_ref_i32x4().to_array(), [-1i32; 4]);
            assert_eq!(u32_4.bitcast_mut_i32x4().to_array(), [-1i32; 4]);
            let i32_8 = i32x8::splat_t(t, 0x4000_0000);
            assert_eq!(i32_8.bitcast_f32x8().to_array(), [2.0f32; 8]);
            let mut u32_8 = u32x8::splat_t(t, u32::MAX);
            assert_eq!(u32_8.bitcast_to_i32().to_array(), [-1i32; 8]);
            assert_eq!(u32_8.bitcast_i32x8().to_array(), [-1i32; 8]);
            assert_eq!(u32_8.bitcast_ref_i32x8().to_array(), [-1i32; 8]);
            assert_eq!(u32_8.bitcast_mut_i32x8().to_array(), [-1i32; 8]);
            let mut i64_2 = i64x2::splat_t(t, 1.5_f64.to_bits() as i64);
            assert_eq!(i64_2.bitcast_f64x2().to_array(), [1.5f64; 2]);
            assert_eq!(i64_2.bitcast_ref_f64x2().to_array(), [1.5f64; 2]);
            assert_eq!(i64_2.bitcast_mut_f64x2().to_array(), [1.5f64; 2]);
            let mut u64_2 = u64x2::splat_t(t, u64::MAX);
            assert_eq!(u64_2.bitcast_i64x2().to_array(), [-1i64; 2]);
            assert_eq!(u64_2.bitcast_ref_i64x2().to_array(), [-1i64; 2]);
            assert_eq!(u64_2.bitcast_mut_i64x2().to_array(), [-1i64; 2]);
            let mut i64_4 = i64x4::splat_t(t, (-2.5_f64).to_bits() as i64);
            assert_eq!(i64_4.bitcast_f64x4().to_array(), [-2.5f64; 4]);
            assert_eq!(i64_4.bitcast_ref_f64x4().to_array(), [-2.5f64; 4]);
            assert_eq!(i64_4.bitcast_mut_f64x4().to_array(), [-2.5f64; 4]);
            let mut u64_4 = u64x4::splat_t(t, u64::MAX);
            assert_eq!(u64_4.bitcast_i64x4().to_array(), [-1i64; 4]);
            assert_eq!(u64_4.bitcast_ref_i64x4().to_array(), [-1i64; 4]);
            assert_eq!(u64_4.bitcast_mut_i64x4().to_array(), [-1i64; 4]);
        }

        if let Some(t) = $get_tok {
            check_floats(t);
            check_f32_block_and_transcendentals(t);
            check_narrow_ints(t);
            check_wide_ints_and_bitcasts(t);
        }
    }};
}

#[cfg(feature = "w512")]
macro_rules! run_w512_for_token {
    ($tok_ty:ty, $get_tok:expr) => {{
        #[inline(never)]
        fn check_w512_floats(t: $tok_ty) {
            const IC: u32 = IntConvert::new().bits();
            check_f32_vec!(
                t,
                f32x16,
                i32x16,
                u32x16,
                16,
                bitcast_i32x16,
                to_i32x16,
                to_i32x16_round,
                from_i32x16_t
            );
            check_f64_vec!(t, f64x8, 8);
            let lo = f32x8::from_array_t(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let hi = f32x8::from_array_t(t, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
            let wide = f32x16::from_halves_t(t, lo, hi);
            assert_eq!(wide.low().to_array(), lo.to_array());
            assert_eq!(wide.high().to_array(), hi.to_array());
            let (s_lo, s_hi) = wide.split();
            assert_eq!(s_lo.to_array(), lo.to_array());
            assert_eq!(s_hi.to_array(), hi.to_array());
            #[allow(deprecated)]
            {
                assert_eq!(f32x16::from_halves(t, lo, hi).to_array(), wide.to_array());
            }

            // f32x16 interleave, bitcast_ref/mut, RGBA/RGB store, dual 8x8 & 16x16 transpose, transcendentals
            let mut v16a = wide;
            let v16b = f32x16::splat_t(t, 100.0);
            let (i16lo, i16hi) = v16a.interleave(v16b);
            assert_eq!(v16a.interleave_lo(v16b).to_array(), i16lo.to_array());
            assert_eq!(v16a.interleave_hi(v16b).to_array(), i16hi.to_array());
            assert_eq!(v16a.bitcast_ref_i32().to_array()[0], 1.0_f32.to_bits() as i32);
            assert_eq!(v16a.bitcast_mut_i32().to_array()[1], 2.0_f32.to_bits() as i32);

            let r16 = f32x16::from_array_t(t, core::array::from_fn(|i| (i as f32) * 4.0));
            let g16 = f32x16::from_array_t(t, core::array::from_fn(|i| (i as f32) * 4.0 + 1.0));
            let b16 = f32x16::from_array_t(t, core::array::from_fn(|i| (i as f32) * 4.0 + 2.0));
            let a16 = f32x16::from_array_t(t, core::array::from_fn(|i| (i as f32) * 4.0 + 3.0));
            let mut rgba64 = [0u8; 64];
            f32x16::store_rgba_u8(r16, g16, b16, a16, &mut rgba64);
            let mut rgba64_w = [0u8; 64];
            f32x16::store_rgba_u8_with::<IC>(r16, g16, b16, a16, &mut rgba64_w);
            assert_eq!(rgba64, rgba64_w);
            assert_eq!(rgba64, core::array::from_fn(|i| i as u8));
            let mut rgb48 = [0u8; 48];
            f32x16::store_rgb_u8(r16, g16, b16, &mut rgb48);
            let mut rgb48_w = [0u8; 48];
            f32x16::store_rgb_u8_with::<IC>(r16, g16, b16, &mut rgb48_w);
            assert_eq!(rgb48, rgb48_w);
            for px in 0..16 {
                assert_eq!(&rgb48[px * 3..px * 3 + 3], &rgba64[px * 4..px * 4 + 3]);
            }

            let dual_in: [f32; 128] = core::array::from_fn(|i| i as f32);
            let mut dual = f32x16::load_dual_8x8_t(t, &dual_in);
            f32x16::transpose_dual_8x8(&mut dual);
            let mut dual_out = [0.0f32; 128];
            f32x16::store_dual_8x8(&dual, &mut dual_out);
            for r in 0..8 {
                for c in 0..8 {
                    assert_eq!(dual_out[r * 16 + c], dual_in[c * 16 + r]);
                    assert_eq!(dual_out[r * 16 + 8 + c], dual_in[c * 16 + 8 + r]);
                }
            }
            let mat16_in: [f32; 256] = core::array::from_fn(|i| i as f32);
            let mut mat16 = f32x16::load_16x16_t(t, &mat16_in);
            f32x16::transpose_16x16(&mut mat16);
            let mut mat16_out = [0.0f32; 256];
            f32x16::store_16x16(&mat16, &mut mat16_out);
            for r in 0..16 {
                for c in 0..16 {
                    assert_eq!(mat16_out[r * 16 + c], mat16_in[c * 16 + r]);
                }
            }

            check_f32_transcendentals!(t, f32x16, 16);
            check_f32_policy_conversions!(t, f32x16, 16);
        }

        #[inline(never)]
        fn check_w512_narrow_ints(t: $tok_ty) {
            check_signed_int_vec!(t, i8x64, i8, 64, false, true, true);
            check_unsigned_int_vec!(t, u8x64, u8, 64, false, true, true);
            check_i8_u8_shift_boundaries!(t, i8x64, u8x64, 64);
            check_signed_int_vec!(t, i16x32, i16, 32, true, true, true);
            check_unsigned_int_vec!(t, u16x32, u16, 32, true, true, true);

            // W512 boundary-value parity: madd_adjacent MIN*MIN wrap + abs_diff MIN/MAX + narrow_saturating
            let m16_32a: [i16; 32] = core::array::from_fn(|i| {
                [i16::MIN, i16::MIN, i16::MIN, i16::MAX, i16::MAX, i16::MAX, -1, 1][i % 8]
            });
            let m16_32b: [i16; 32] = core::array::from_fn(|i| {
                [i16::MIN, i16::MIN, i16::MAX, i16::MIN, i16::MAX, i16::MAX, i16::MIN, i16::MAX][i % 8]
            });
            let vm16_32a = i16x32::from_array_t(t, m16_32a);
            let vm16_32b = i16x32::from_array_t(t, m16_32b);
            let exp_madd32: [i32; 16] = core::array::from_fn(|i| {
                ((m16_32a[2 * i] as i32) * (m16_32b[2 * i] as i32))
                    .wrapping_add((m16_32a[2 * i + 1] as i32) * (m16_32b[2 * i + 1] as i32))
            });
            assert_eq!(exp_madd32[0], i32::MIN);
            assert_eq!(vm16_32a.madd_adjacent(vm16_32b).to_array(), exp_madd32);
            assert_eq!(
                vm16_32a.abs_diff(vm16_32b).to_array(),
                core::array::from_fn(|i| m16_32a[i].abs_diff(m16_32b[i]))
            );

            let s8_64: [i8; 64] = core::array::from_fn(|i| (i as i8).wrapping_mul(3).wrapping_sub(90));
            let vs8_64 = i8x64::from_array_t(t, s8_64);
            assert_eq!(vs8_64.widen_low().to_array(), core::array::from_fn(|i| s8_64[i] as i16));
            assert_eq!(vs8_64.widen_high().to_array(), core::array::from_fn(|i| s8_64[i + 32] as i16));

            let u8_64a: [u8; 64] = core::array::from_fn(|i| (i as u8) * 3 + 5);
            let u8_64b: [u8; 64] = core::array::from_fn(|i| 250u8.wrapping_sub((i as u8) * 3));
            let vu8_64a = u8x64::from_array_t(t, u8_64a);
            let vu8_64b = u8x64::from_array_t(t, u8_64b);
            assert_eq!(vu8_64a.widen_low().to_array(), core::array::from_fn(|i| u8_64a[i] as u16));
            assert_eq!(vu8_64a.widen_high().to_array(), core::array::from_fn(|i| u8_64a[i + 32] as u16));
            assert_eq!(vu8_64a.abs_diff(vu8_64b).to_array(), core::array::from_fn(|i| u8_64a[i].abs_diff(u8_64b[i])));
            assert_eq!(vu8_64a.sum_abs_diff(vu8_64b), (0..64).map(|i| u8_64a[i].abs_diff(u8_64b[i]) as u32).sum::<u32>());
            assert_eq!(vu8_64a.pairwise_widen_add().to_array(), core::array::from_fn(|i| u8_64a[2 * i] as u16 + u8_64a[2 * i + 1] as u16));

            let s16_32a: [i16; 32] = core::array::from_fn(|i| (i as i16) * 41 - 500);
            let s16_32b: [i16; 32] = core::array::from_fn(|i| 300 - (i as i16) * 23);
            let vs16_32a = i16x32::from_array_t(t, s16_32a);
            let vs16_32b = i16x32::from_array_t(t, s16_32b);
            assert_eq!(vs16_32a.shl_uniform(2).to_array(), s16_32a.map(|x| x.wrapping_shl(2)));
            assert_eq!(vs16_32a.shr_logical_uniform(2).to_array(), s16_32a.map(|x| ((x as u16) >> 2) as i16));
            assert_eq!(vs16_32a.shr_arithmetic_uniform(2).to_array(), s16_32a.map(|x| x >> 2));
            assert_eq!(vs16_32a.widen_low().to_array(), core::array::from_fn(|i| s16_32a[i] as i32));
            assert_eq!(vs16_32a.widen_high().to_array(), core::array::from_fn(|i| s16_32a[i + 16] as i32));
            assert_eq!(
                vs16_32a.narrow_saturating_i8(vs16_32b).to_array(),
                core::array::from_fn(|i| {
                    let v = if i < 32 { s16_32a[i] } else { s16_32b[i - 32] };
                    v.clamp(i8::MIN as i16, i8::MAX as i16) as i8
                })
            );
            assert_eq!(
                vs16_32a.narrow_saturating_u8(vs16_32b).to_array(),
                core::array::from_fn(|i| {
                    let v = if i < 32 { s16_32a[i] } else { s16_32b[i - 32] };
                    v.clamp(0, 255) as u8
                })
            );
            assert_eq!(vs16_32a.abs_diff(vs16_32b).to_array(), core::array::from_fn(|i| s16_32a[i].abs_diff(s16_32b[i])));
            assert_eq!(
                vs16_32a.madd_adjacent(vs16_32b).to_array(),
                core::array::from_fn(|i| (s16_32a[2 * i] as i32) * (s16_32b[2 * i] as i32) + (s16_32a[2 * i + 1] as i32) * (s16_32b[2 * i + 1] as i32))
            );
            assert_eq!(vs16_32a.bitcast_u16x32().bitcast_i16x32().to_array(), s16_32a);

            let u16_32a: [u16; 32] = core::array::from_fn(|i| (i as u16) * 500 + 7);
            let vu16_32a = u16x32::from_array_t(t, u16_32a);
            assert_eq!(vu16_32a.shl_uniform(2).to_array(), u16_32a.map(|x| x.wrapping_shl(2)));
            assert_eq!(vu16_32a.shr_logical_uniform(2).to_array(), u16_32a.map(|x| x >> 2));
            assert_eq!(vu16_32a.widen_low().to_array(), core::array::from_fn(|i| u16_32a[i] as u32));
            assert_eq!(vu16_32a.widen_high().to_array(), core::array::from_fn(|i| u16_32a[i + 16] as u32));
            assert_eq!(vu16_32a.pairwise_widen_add().to_array(), core::array::from_fn(|i| u16_32a[2 * i] as u32 + u16_32a[2 * i + 1] as u32));
        }

        #[inline(never)]
        fn check_w512_wide_ints(t: $tok_ty) {
            check_signed_int_vec!(t, i32x16, i32, 16, true, false, true);
            check_unsigned_int_vec!(t, u32x16, u32, 16, true, false, true);
            check_signed_int_vec!(t, i64x8, i64, 8, false, false, true);
            check_unsigned_int_vec!(t, u64x8, u64, 8, false, false, true);

            let s32_16a: [i32; 16] = core::array::from_fn(|i| {
                [i32::MIN, -32769, -32768, -1, 0, 32767, 32768, i32::MAX][i % 8]
            });
            let s32_16b: [i32; 16] = core::array::from_fn(|i| {
                [65535, 65536, i32::MIN, i32::MAX, -1, 0, 1, 32767][i % 8]
            });
            let vs32_16a = i32x16::from_array_t(t, s32_16a);
            let vs32_16b = i32x16::from_array_t(t, s32_16b);
            assert_eq!(vs32_16a.shl_uniform(2).to_array(), s32_16a.map(|x| x.wrapping_shl(2)));
            assert_eq!(vs32_16a.shr_logical_uniform(2).to_array(), s32_16a.map(|x| ((x as u32) >> 2) as i32));
            assert_eq!(vs32_16a.shr_arithmetic_uniform(2).to_array(), s32_16a.map(|x| x >> 2));
            assert_eq!(
                vs32_16a.narrow_saturating_i16(vs32_16b).to_array(),
                core::array::from_fn(|i| {
                    let v = if i < 16 { s32_16a[i] } else { s32_16b[i - 16] };
                    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
                })
            );
            assert_eq!(
                vs32_16a.narrow_saturating_u16(vs32_16b).to_array(),
                core::array::from_fn(|i| {
                    let v = if i < 16 { s32_16a[i] } else { s32_16b[i - 16] };
                    v.clamp(0, u16::MAX as i32) as u16
                })
            );
            assert_eq!(i32x16::splat_t(t, 0x3f80_0000).bitcast_f32x16().to_array(), [1.0f32; 16]);
            let u32_16a: [u32; 16] = core::array::from_fn(|i| (i as u32) * 1000 + 3);
            let vu32_16a = u32x16::from_array_t(t, u32_16a);
            assert_eq!(vu32_16a.shl_uniform(2).to_array(), u32_16a.map(|x| x.wrapping_shl(2)));
            assert_eq!(vu32_16a.shr_logical_uniform(2).to_array(), u32_16a.map(|x| x >> 2));
        }

        if let Some(t) = $get_tok {
            check_w512_floats(t);
            check_w512_narrow_ints(t);
            check_w512_wide_ints(t);
        }
    }};
}

#[test]
fn scalar_w128_w256_surface_coverage() {
    run_w128_w256_for_token!(ScalarToken, Some(ScalarToken));
}

#[cfg(feature = "w512")]
#[test]
fn scalar_w512_surface_coverage() {
    run_w512_for_token!(ScalarToken, Some(ScalarToken));
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[test]
fn x86_v3_w128_w256_surface_coverage() {
    run_w128_w256_for_token!(archmage::X64V3Token, archmage::X64V3Token::summon());
    if let Some(t) = archmage::X64V3Token::summon() {
        let vf = f32x4::from_array_t(t, [1.0, -2.0, 3.5, 4.25]);
        assert_eq!(f32x4::from_raw_t(t, vf.raw()).to_array(), vf.to_array());
        let vi = i16x8::splat_t(t, -123);
        assert_eq!(i16x8::from_raw_t(t, vi.raw()).to_array(), vi.to_array());
        let vl = i64x2::from_array_t(t, [i64::MIN, i64::MAX]);
        assert_eq!(i64x2::from_raw_t(t, vl.raw()).to_array(), vl.to_array());
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "w512"))]
#[test]
fn x86_v3_w512_surface_coverage() {
    run_w512_for_token!(archmage::X64V3Token, archmage::X64V3Token::summon());
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[test]
fn x86_v3_vnni_surface_coverage() {
    run_w128_w256_for_token!(archmage::X64V3VnniToken, archmage::X64V3VnniToken::summon());
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[test]
fn x86_v4_w128_w256_surface_coverage() {
    run_w128_w256_for_token!(archmage::X64V4Token, archmage::X64V4Token::summon());
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[test]
fn x86_v4_w512_surface_coverage() {
    run_w512_for_token!(archmage::X64V4Token, archmage::X64V4Token::summon());
    if let Some(t) = archmage::X64V4Token::summon() {
        let vf = f32x16::splat_t(t, 2.5);
        assert_eq!(
            f32x16::<archmage::X64V4Token>::from_raw_t(t, vf.raw()).to_array(),
            [2.5; 16]
        );
        let vd = f64x8::splat_t(t, -3.5);
        assert_eq!(
            f64x8::<archmage::X64V4Token>::from_raw_t(t, vd.raw()).to_array(),
            [-3.5; 8]
        );
        let vi = i32x16::splat_t(t, -77);
        assert_eq!(
            i32x16::<archmage::X64V4Token>::from_raw_t(t, vi.raw()).to_array(),
            [-77; 16]
        );
        let vl = i64x8::splat_t(t, -99);
        assert_eq!(
            i64x8::<archmage::X64V4Token>::from_raw_t(t, vl.raw()).to_array(),
            [-99; 8]
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[test]
fn x86_v4x_w128_w256_surface_coverage() {
    run_w128_w256_for_token!(archmage::X64V4xToken, archmage::X64V4xToken::summon());
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[test]
fn x86_v4x_w512_surface_coverage() {
    run_w512_for_token!(archmage::X64V4xToken, archmage::X64V4xToken::summon());
    if let Some(t) = archmage::X64V4xToken::summon() {
        let vf = f32x16::splat_t(t, 1.25);
        assert_eq!(
            f32x16::<archmage::X64V4xToken>::from_raw_t(t, vf.raw()).to_array(),
            [1.25; 16]
        );
        let vd = f64x8::splat_t(t, 4.75);
        assert_eq!(
            f64x8::<archmage::X64V4xToken>::from_raw_t(t, vd.raw()).to_array(),
            [4.75; 8]
        );
        let vi = i32x16::from_array_t(
            t,
            core::array::from_fn(|i| (i as i32).wrapping_mul(0x1357_2468).wrapping_sub(5)),
        );
        assert_eq!(
            i32x16::<archmage::X64V4xToken>::from_raw_t(t, vi.raw()).to_array(),
            vi.to_array()
        );
        assert_eq!(
            vi.popcnt().to_array(),
            vi.to_array().map(|x| x.count_ones() as i32)
        );
        let vl = i64x8::from_array_t(
            t,
            core::array::from_fn(|i| (i as i64).wrapping_mul(0x1122_3344_5566_7788) ^ -1),
        );
        assert_eq!(
            i64x8::<archmage::X64V4xToken>::from_raw_t(t, vl.raw()).to_array(),
            vl.to_array()
        );
        assert_eq!(
            vl.popcnt().to_array(),
            vl.to_array().map(|x| x.count_ones() as i64)
        );
        let vb = i8x64::from_array_t(t, core::array::from_fn(|i| (i as i8).wrapping_mul(3)));
        assert_eq!(
            vb.popcnt().to_array(),
            vb.to_array().map(|x| x.count_ones() as i8)
        );
        let vs = i16x32::from_array_t(t, core::array::from_fn(|i| (i as i16).wrapping_mul(173)));
        assert_eq!(
            vs.popcnt().to_array(),
            vs.to_array().map(|x| x.count_ones() as i16)
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[test]
fn x86_fp16_w128_w256_surface_coverage() {
    run_w128_w256_for_token!(
        archmage::Avx512Fp16Token,
        archmage::Avx512Fp16Token::summon()
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_w128_w256_surface_coverage() {
    run_w128_w256_for_token!(archmage::NeonToken, archmage::NeonToken::summon());
}

#[cfg(all(target_arch = "aarch64", feature = "w512"))]
#[test]
fn neon_w512_surface_coverage() {
    run_w512_for_token!(archmage::NeonToken, archmage::NeonToken::summon());
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn wasm128_w128_w256_surface_coverage() {
    run_w128_w256_for_token!(archmage::Wasm128Token, archmage::Wasm128Token::summon());
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "w512"))]
#[test]
fn wasm128_w512_surface_coverage() {
    run_w512_for_token!(archmage::Wasm128Token, archmage::Wasm128Token::summon());
}
