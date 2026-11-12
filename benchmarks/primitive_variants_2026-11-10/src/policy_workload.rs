//! Shared kernels for every supported primitive flag set. No runtime flags in
//! the hot loop; native and hosted WASM call the same const instantiations.
#![allow(dead_code)]
macro_rules! minmax {
    ($op:ident,$stream:ident,$chain:ident,$v:ident,$f:ty,$bound:ident,$w:literal,$min:ident,$max:ident) => {
        #[inline(always)]
        pub(crate) fn $op<
            T: magetypes::simd::backends::$bound,
            const O: u32,
            const MAX: bool,
            const LEGACY: bool,
        >(
            t: T,
            a: magetypes::simd::generic::$v<T>,
            b: magetypes::simd::generic::$v<T>,
        ) -> magetypes::simd::generic::$v<T> {
            if LEGACY {
                if MAX { a.max(b) } else { a.min(b) }
            } else if MAX {
                magemath::minmax::$max::<O, _>(t, a, b).unwrap()
            } else {
                magemath::minmax::$min::<O, _>(t, a, b).unwrap()
            }
        }
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(crate) fn $stream<const O: u32, const MAX: bool, const LEGACY: bool, const N: usize>(
            t: Token,
            a: &[$f; N],
            b: &[$f; N],
            out: &mut [$f; N],
        ) {
            use magetypes::simd::generic::$v as V;
            for ((a, b), out) in a
                .as_chunks::<$w>()
                .0
                .iter()
                .zip(b.as_chunks::<$w>().0)
                .zip(out.as_chunks_mut::<$w>().0)
            {
                *out = $op::<_, O, MAX, LEGACY>(t, V::from_array_t(t, *a), V::from_array_t(t, *b))
                    .to_array();
            }
        }
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(crate) fn $chain<const O: u32, const MAX: bool, const LEGACY: bool>(
            t: Token,
            initial: $f,
        ) -> [$f; $w] {
            use magetypes::simd::generic::$v as V;
            let mut x = V::splat_t(t, initial);
            for _ in 0..64 {
                x = $op::<_, O, MAX, LEGACY>(t, x, V::splat_t(t, 2.0)) + V::splat_t(t, 0.03125);
            }
            x.to_array()
        }
    };
}
minmax!(
    mm4,
    stream_mm4,
    chain_mm4,
    f32x4,
    f32,
    F32x4Backend,
    4,
    try_min_f32x4_with,
    try_max_f32x4_with
);
minmax!(
    mm8,
    stream_mm8,
    chain_mm8,
    f32x8,
    f32,
    F32x8Backend,
    8,
    try_min_f32x8_with,
    try_max_f32x8_with
);
#[cfg(feature = "w512")]
minmax!(
    mm16,
    stream_mm16,
    chain_mm16,
    f32x16,
    f32,
    F32x16Backend,
    16,
    try_min_f32x16_with,
    try_max_f32x16_with
);
minmax!(
    mm64x2,
    stream_mm64x2,
    chain_mm64x2,
    f64x2,
    f64,
    F64x2Backend,
    2,
    try_min_f64x2_with,
    try_max_f64x2_with
);
minmax!(
    mm64x4,
    stream_mm64x4,
    chain_mm64x4,
    f64x4,
    f64,
    F64x4Backend,
    4,
    try_min_f64x4_with,
    try_max_f64x4_with
);
#[cfg(feature = "w512")]
minmax!(
    mm64x8,
    stream_mm64x8,
    chain_mm64x8,
    f64x8,
    f64,
    F64x8Backend,
    8,
    try_min_f64x8_with,
    try_max_f64x8_with
);
macro_rules! reciprocal {
    ($op:ident,$stream:ident,$chain:ident,$v:ident,$bound:ident,$w:literal,$rc:ident,$rs:ident) => {
        #[inline(always)]
        pub(crate) fn $op<
            T: magetypes::simd::backends::$bound,
            const O: u32,
            const RSQRT: bool,
            const LEGACY: bool,
        >(
            t: T,
            x: magetypes::simd::generic::$v<T>,
        ) -> magetypes::simd::generic::$v<T> {
            if LEGACY {
                if RSQRT { x.rsqrt() } else { x.recip() }
            } else if RSQRT {
                magemath::recip::$rs::<O, _>(t, x).unwrap()
            } else {
                magemath::recip::$rc::<O, _>(t, x).unwrap()
            }
        }
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(crate) fn $stream<
            const O: u32,
            const RSQRT: bool,
            const LEGACY: bool,
            const N: usize,
        >(
            t: Token,
            a: &[f32; N],
            _b: &[f32; N],
            out: &mut [f32; N],
        ) {
            use magetypes::simd::generic::$v as V;
            for (a, out) in a
                .as_chunks::<$w>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<$w>().0)
            {
                *out = $op::<_, O, RSQRT, LEGACY>(t, V::from_array_t(t, *a)).to_array();
            }
        }
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(crate) fn $chain<const O: u32, const RSQRT: bool, const LEGACY: bool>(
            t: Token,
            initial: f32,
        ) -> [f32; $w] {
            use magetypes::simd::generic::$v as V;
            let mut x = V::splat_t(t, initial);
            for _ in 0..64 {
                x = $op::<_, O, RSQRT, LEGACY>(t, x) + V::splat_t(t, 0.5);
            }
            x.to_array()
        }
    };
}
reciprocal!(
    rc4,
    stream_rc4,
    chain_rc4,
    f32x4,
    F32x4Convert,
    4,
    try_recip_f32x4_with,
    try_rsqrt_f32x4_with
);
reciprocal!(
    rc8,
    stream_rc8,
    chain_rc8,
    f32x8,
    F32x8Convert,
    8,
    try_recip_f32x8_with,
    try_rsqrt_f32x8_with
);
#[cfg(feature = "w512")]
reciprocal!(
    rc16,
    stream_rc16,
    chain_rc16,
    f32x16,
    F32x16Convert,
    16,
    try_recip_f32x16_with,
    try_rsqrt_f32x16_with
);

/// Common ordinary inputs; zero-clamp inputs include signed zeros and subnormals.
pub(crate) fn input(i: usize, zero: bool, recip: bool, rsqrt: bool) -> (f32, f32) {
    if zero {
        return (
            [
                0.0,
                -0.0,
                f32::from_bits(1),
                -f32::from_bits(1),
                0.25,
                -0.25,
                1.0,
                -1.0,
            ][i % 8],
            0.0,
        );
    }
    let mut a = 0.25 + ((i % 64 + 1) as f32) * 0.03125;
    if !rsqrt && i % 3 == 1 {
        a = -a;
    }
    let b = if recip {
        1.0
    } else {
        0.5 + ((i * 17 % 64 + 1) as f32) * 0.03125
    };
    (a, b)
}
pub(crate) fn input64(i: usize, zero: bool) -> (f64, f64) {
    if zero {
        (
            [
                0.0,
                -0.0,
                f64::from_bits(1),
                -f64::from_bits(1),
                0.25,
                -0.25,
                1.0,
                -1.0,
            ][i % 8],
            0.0,
        )
    } else {
        let (a, b) = input(i, false, false, false);
        (f64::from(a), f64::from(b))
    }
}
pub(crate) trait CheckScalar:
    Copy + Send + Sync + core::fmt::Debug + core::ops::Add<Output = Self> + 'static
{
    fn from_f32(x: f32) -> Self;
    fn bits(self) -> u64;
    fn widen(self) -> f64;
    fn narrow(self) -> f32;
    fn pair(i: usize, zero: bool, recip: bool, rsqrt: bool) -> (Self, Self);
}
impl CheckScalar for f32 {
    fn from_f32(x: f32) -> Self {
        x
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn widen(self) -> f64 {
        f64::from(self)
    }
    fn narrow(self) -> f32 {
        self
    }
    fn pair(i: usize, z: bool, r: bool, s: bool) -> (Self, Self) {
        input(i, z, r, s)
    }
}
impl CheckScalar for f64 {
    fn from_f32(x: f32) -> Self {
        f64::from(x)
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn widen(self) -> f64 {
        self
    }
    fn narrow(self) -> f32 {
        self as f32
    }
    fn pair(i: usize, z: bool, _r: bool, _s: bool) -> (Self, Self) {
        input64(i, z)
    }
}

/// Check the selected result policy, not an implementation mirror. All input
/// generation and checks run before timing. `relative_error` is an observed
/// maximum on this fixed input set, not a whole-domain mathematical bound.
pub(crate) fn check_mm<const MAX: bool>(o: u32, legacy: bool, a: f64, b: f64, y: f64) {
    let expected = if MAX { a.max(b) } else { a.min(b) };
    assert_eq!(y, expected, "min/max selected wrong value");
    if !legacy && o & 16 == 0 && a == 0.0 && b == 0.0 {
        let negative = if MAX {
            a.is_sign_negative() && b.is_sign_negative()
        } else {
            a.is_sign_negative() || b.is_sign_negative()
        };
        assert_eq!(y.is_sign_negative(), negative, "ordered zero sign");
    }
}
pub(crate) fn check_rc(o: u32, legacy: bool, rsqrt: bool, x: f32, y: f32) -> f64 {
    let reference = if rsqrt {
        1.0 / f64::from(x).sqrt()
    } else {
        1.0 / f64::from(x)
    };
    let error = ((f64::from(y) - reference) / reference).abs();
    let bound = if legacy {
        1e-6
    } else {
        match (o >> 1) & 3 {
            0 => 2e-7,
            1 => 1e-4,
            2 => 6e-3,
            _ => unreachable!(),
        }
    };
    assert!(
        error <= bound,
        "flags={o:02x} rsqrt={rsqrt} x={x} y={y} error={error}"
    );
    if legacy {
        assert!(
            y.to_bits().abs_diff((reference as f32).to_bits()) <= 4,
            "legacy working-tier ULP bound"
        );
    }
    if !legacy && o & 6 == 0 {
        assert_eq!(
            y.to_bits(),
            (1.0_f32 / if rsqrt { x.sqrt() } else { x }).to_bits()
        );
    }
    error
}
