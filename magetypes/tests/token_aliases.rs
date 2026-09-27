//! Migration spellings are usable without target-feature attributes, including
//! in backend-generic helpers. Existing spellings continue to compile alongside.
#![forbid(unsafe_code)]

use archmage::ScalarToken;
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::*;

fn generic_bits_and_views<T: F32x8Backend>(token: T) {
    let bits = [0x8000_0000, 0x7fc0_1234, 1, 0xff80_0000, 0, 1, 2, 3];
    // A safe function pointer demonstrates that no caller feature context is
    // required. NaN payloads and negative zero must survive unchanged.
    let load: fn(T, &[f32; 8]) -> f32x8<T> = f32x8::load_t;
    let value = load(token, &bits.map(f32::from_bits));
    assert_eq!(value.to_array().map(f32::to_bits), bits);
    assert_eq!(
        f32x8::from_repr_t(token, value.into_repr())
            .to_array()
            .map(f32::to_bits),
        bits
    );
    let mut values = [2.0; 19];
    let (chunks, tail) = f32x8::<T>::partition_slice_mut_t(token, &mut values);
    assert_eq!(chunks.len(), 2);
    assert_eq!(tail.len(), 3);
    for chunk in chunks {
        (f32x8::load_t(token, chunk) * f32x8::splat_t(token, 3.0)).store(chunk);
    }
    tail.fill(7.0);
    assert_eq!(&values[..16], &[6.0; 16]);
    assert_eq!(&values[16..], &[7.0; 3]);
    let (chunks, tail) = f32x8::<T>::partition_slice_t(token, &values);
    assert_eq!(chunks.as_ptr().cast::<f32>(), values.as_ptr());
    assert_eq!(tail, &[7.0; 3]);
}

#[test]
fn generic_helpers_need_only_the_token() {
    generic_bits_and_views(ScalarToken);
}

macro_rules! family {
    ($name:ident, $elem:ty, $lanes:expr) => {{
        let values = core::array::from_fn(|i| (i + 1) as $elem);
        let token = ScalarToken;
        let ctor: fn(ScalarToken, [$elem; $lanes]) -> $name<ScalarToken> = $name::from_array_t;
        assert_eq!(ctor(token, values).to_array(), values);
        assert_eq!($name::from_slice_t(token, &values).to_array(), values);
        assert_eq!($name::load_t(token, &values).to_array(), values);
        assert_eq!($name::zero_t(token).to_array(), [0 as $elem; $lanes]);
        assert_eq!(
            $name::splat_t(token, 3 as $elem).to_array(),
            [3 as $elem; $lanes]
        );
        assert_eq!(
            $name::splat(token, 3 as $elem).to_array(),
            $name::splat_t(token, 3 as $elem).to_array()
        );
    }};
}

#[test]
fn all_vector_families_keep_both_spellings() {
    family!(f32x4, f32, 4);
    family!(f32x8, f32, 8);
    family!(f64x2, f64, 2);
    family!(f64x4, f64, 4);
    family!(i8x16, i8, 16);
    family!(i8x32, i8, 32);
    family!(u8x16, u8, 16);
    family!(u8x32, u8, 32);
    family!(i16x8, i16, 8);
    family!(i16x16, i16, 16);
    family!(u16x8, u16, 8);
    family!(u16x16, u16, 16);
    family!(i32x4, i32, 4);
    family!(i32x8, i32, 8);
    family!(u32x4, u32, 4);
    family!(u32x8, u32, 8);
    family!(i64x2, i64, 2);
    family!(i64x4, i64, 4);
    family!(u64x2, u64, 2);
    family!(u64x4, u64, 4);
    #[cfg(feature = "w512")]
    {
        family!(f32x16, f32, 16);
        family!(f64x8, f64, 8);
        family!(i8x64, i8, 64);
        family!(u8x64, u8, 64);
        family!(i16x32, i16, 32);
        family!(u16x32, u16, 32);
        family!(i32x16, i32, 16);
        family!(u32x16, u32, 16);
        family!(i64x8, i64, 8);
        family!(u64x8, u64, 8);
    }
}

#[test]
fn scalar_single_lane_aliases() {
    use magetypes::simd::scalar::*;
    macro_rules! one {
        ($name:ident, $elem:ty) => {{
            assert_eq!(
                $name::splat_t(ScalarToken, 7 as $elem).to_array(),
                [7 as $elem]
            );
            assert_eq!($name::zero_t(ScalarToken).to_array(), [0 as $elem]);
            assert_eq!(
                $name::from_array_t(ScalarToken, [9 as $elem]).to_array(),
                [9 as $elem]
            );
        }};
    }
    one!(f32x1, f32);
    one!(f64x1, f64);
    one!(i8x1, i8);
    one!(u8x1, u8);
    one!(i16x1, i16);
    one!(u16x1, u16);
    one!(i32x1, i32);
    one!(u32x1, u32);
    one!(i64x1, i64);
    one!(u64x1, u64);
    assert_eq!(f32x1::load_t(ScalarToken, &[2.0]).to_array(), [2.0]);
    assert_eq!(f64x1::load_t(ScalarToken, &[2.0]).to_array(), [2.0]);
}

#[test]
fn conversion_and_block_aliases() {
    let token = ScalarToken;
    let bytes = core::array::from_fn(|i| i as u8);
    let words = u32x4::from_bytes_t(token, &bytes);
    assert_eq!(*words.as_bytes(), bytes);
    assert_eq!(
        u32x4::from_bytes_owned_t(token, bytes).to_array(),
        words.to_array()
    );
    let mut values = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let views = u32x4::cast_slice_mut_t(token, &mut values).unwrap();
    views[1] = u32x4::splat_t(token, 99);
    assert_eq!(values, [1, 2, 3, 4, 99, 99, 99, 99]);
    assert_eq!(u32x4::cast_slice_t(token, &values).unwrap().len(), 2);
    let lo = f32x4::from_u8_t(token, &[0, 1, 2, 3]);
    let hi = f32x4::splat_t(token, 4.0);
    assert_eq!(
        f32x8::from_halves_t(token, lo, hi).to_array(),
        [0., 1., 2., 3., 4., 4., 4., 4.]
    );
    let ints = i32x4::from_array_t(token, [-2, 0, 5, 123]);
    assert_eq!(
        f32x4::from_i32_t(token, ints).to_array(),
        [-2., 0., 5., 123.]
    );
    assert_eq!(
        f32x4::from_i32x4_t(token, ints).to_array(),
        [-2., 0., 5., 123.]
    );
    let bits = i32x4::from_array_t(token, [0, 1, i32::MIN, 0x7fc0_1234]);
    assert_eq!(
        f32x4::from_i32_bitcast_t(token, bits)
            .to_array()
            .map(f32::to_bits),
        [0, 1, 0x8000_0000, 0x7fc0_1234]
    );
    let (r, g, b, a) = f32x4::load_4_rgba_u8_t(token, &bytes);
    assert_eq!(r.to_array(), [0., 4., 8., 12.]);
    assert_eq!(g.to_array(), [1., 5., 9., 13.]);
    assert_eq!(b.to_array(), [2., 6., 10., 14.]);
    assert_eq!(a.to_array(), [3., 7., 11., 15.]);
}

// Exercise the published define syntax, with both spellings in the same body.
#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn defined(token: Token) -> [f32; 8] {
    generic_bits_and_views(token);
    (f32x8::splat_t(token, 2.0) + f32x8::splat(token, 3.0)).to_array()
}

#[test]
fn published_macros_accept_migration_spelling() {
    assert_eq!(
        archmage::incant!(defined(), [v3, neon, wasm128, scalar]),
        [5.0; 8]
    );
    assert_eq!(defined_scalar(ScalarToken), [5.0; 8]);
}
