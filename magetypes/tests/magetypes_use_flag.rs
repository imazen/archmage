//! Local constructors share vector operations while preserving explicit-token APIs.
#![forbid(unsafe_code)]
use archmage::{ScalarToken, incant, magetypes};
use magetypes::simd::generic::{self, local};

#[magetypes(use(f32x4, f32x8, i32x4, i32x8), v3, neon, wasm128, scalar)]
fn exercise(_token: Token, data: &[f32; 8]) -> [f32; 8] {
    let zero = f32x8::zero();
    let v = f32x8::load(data) + f32x8::splat(2.0) + zero;
    let ints: i32x8 = v.to_i32();
    let v = f32x8::from_i32(ints);
    let (lo, hi): (f32x4, f32x4) = v.split();
    let v = f32x8::from_halves(lo, hi);
    let explicit: generic::f32x8<Token> = v.into();
    let v: f32x8 = explicit.into();
    let bits: i32x4 = lo.to_f16();
    assert_eq!(bits.f16_to_f32().to_array(), lo.to_array());
    assert_eq!(v.to_array(), f32x8::from_slice(&v.to_array()).to_array());
    assert_eq!(f32x8::from_repr(v.into_repr()).to_array(), v.to_array());
    let bytes = v.as_bytes();
    assert_eq!(f32x8::from_bytes(bytes).to_array(), v.to_array());
    assert_eq!(f32x8::from_bytes_owned(*bytes).to_array(), v.to_array());
    let mut copy = v.to_array();
    let (chunks, tail) = f32x8::partition_slice_mut(&mut copy);
    assert!(tail.is_empty());
    assert_eq!(chunks.len(), 1);
    let (chunks, tail) = f32x8::partition_slice(&copy);
    assert!(tail.is_empty());
    assert_eq!(chunks[0], v.to_array());
    v.to_array()
}

#[test]
fn local_dispatch_and_mode_preserving_operations() {
    let input = [1., 2., 3., 4., 5., 6., 7., 8.];
    assert_eq!(
        exercise_scalar(ScalarToken, &input),
        [3., 4., 5., 6., 7., 8., 9., 10.]
    );
    assert_eq!(
        incant!(exercise(&input), [v3, neon, wasm128, scalar]),
        [3., 4., 5., 6., 7., 8., 9., 10.]
    );
}

#[magetypes(rite, use(f32x8), define(i32x8), v3, scalar)]
fn mixed(token: Token) -> [f32; 8] {
    let explicit = i32x8::splat(token, 7);
    let contextual: local::i32x8<Token> = explicit.into();
    f32x8::from_i32(contextual).to_array()
}

#[test]
fn legacy_inference_and_mixed_aliases() {
    let v = generic::f32x8::zero(ScalarToken);
    assert_eq!(v.to_array(), [0.; 8]);
    assert_eq!(mixed_scalar(ScalarToken), [7.; 8]);
}

macro_rules! shape {
    ($name:ident, $element:ty, $lanes:expr) => {{
        type V = local::$name<ScalarToken>;
        let values = [3 as $element; $lanes];
        let v = V::load(&values) + V::zero();
        assert_eq!(v.to_array(), values);
        assert_eq!(V::from_array(values).to_array(), values);
        assert_eq!(V::from_slice(&values).to_array(), values);
        assert_eq!(V::splat(3 as $element).to_array(), values);
        assert_eq!(
            V::zero_with_token(ScalarToken).to_array(),
            [0 as $element; $lanes]
        );
        assert_eq!(
            V::splat_with_token(ScalarToken, 3 as $element).to_array(),
            values
        );
        assert_eq!(V::load_with_token(ScalarToken, &values).to_array(), values);
        assert_eq!(
            V::from_array_with_token(ScalarToken, values).to_array(),
            values
        );
        assert_eq!(
            V::from_slice_with_token(ScalarToken, &values).to_array(),
            values
        );
        assert_eq!(
            V::from_repr_with_token(ScalarToken, v.into_repr()).to_array(),
            values
        );
        assert_eq!(
            generic::$name::splat_with_token(ScalarToken, 3 as $element).to_array(),
            values
        );
        let (chunks, tail) = V::partition_slice_with_token(ScalarToken, &values);
        assert_eq!(chunks, &[values]);
        assert!(tail.is_empty());
        let mut copied = values;
        let (chunks, tail) = V::partition_slice_mut_with_token(ScalarToken, &mut copied);
        assert_eq!(chunks, &[values]);
        assert!(tail.is_empty());
        let token_vector: generic::$name<ScalarToken> = v.into();
        let roundtrip: V = token_vector.into();
        assert_eq!(roundtrip.to_array(), values);
        assert_eq!(core::mem::size_of::<V>(), core::mem::size_of_val(&values));
        assert_eq!(core::mem::align_of::<V>(), core::mem::align_of_val(&values));
    }};
}

#[test]
fn every_shape_has_both_constructor_modes() {
    shape!(f32x4, f32, 4);
    shape!(f32x8, f32, 8);
    shape!(f64x2, f64, 2);
    shape!(f64x4, f64, 4);
    shape!(i8x16, i8, 16);
    shape!(i8x32, i8, 32);
    shape!(u8x16, u8, 16);
    shape!(u8x32, u8, 32);
    shape!(i16x8, i16, 8);
    shape!(i16x16, i16, 16);
    shape!(u16x8, u16, 8);
    shape!(u16x16, u16, 16);
    shape!(i32x4, i32, 4);
    shape!(i32x8, i32, 8);
    shape!(u32x4, u32, 4);
    shape!(u32x8, u32, 8);
    shape!(i64x2, i64, 2);
    shape!(i64x4, i64, 4);
    shape!(u64x2, u64, 2);
    shape!(u64x4, u64, 4);
    #[cfg(feature = "w512")]
    {
        shape!(f32x16, f32, 16);
        shape!(f64x8, f64, 8);
        shape!(i8x64, i8, 64);
        shape!(u8x64, u8, 64);
        shape!(i16x32, i16, 32);
        shape!(u16x32, u16, 32);
        shape!(i32x16, i32, 16);
        shape!(u32x16, u32, 16);
        shape!(i64x8, i64, 8);
        shape!(u64x8, u64, 8);
    }
}

#[test]
fn borrowed_views_and_integer_widths_preserve_context() {
    type F = local::f32x4<ScalarToken>;
    type I = local::i32x4<ScalarToken>;
    let mut v = F::from_array([1., 2., 3., 4.]);
    let bits: &I = v.bitcast_ref_i32x4();
    assert_eq!(bits.to_array()[0], 1f32.to_bits() as i32);
    let bits: &mut I = v.bitcast_mut_i32x4();
    bits[1] = 7f32.to_bits() as i32;
    assert_eq!(v.to_array(), [1., 7., 3., 4.]);
    let mut values = [1., 2., 3., 4.];
    let view: &[F] = F::cast_slice(&values).unwrap();
    assert_eq!(view[0].to_array(), values);
    let view: &mut [F] = F::cast_slice_mut(&mut values).unwrap();
    view[0] += F::splat(1.);
    assert_eq!(values, [2., 3., 4., 5.]);
    let bytes = local::u8x16::<ScalarToken>::splat(255);
    let lo: local::u16x8<ScalarToken> = bytes.widen_low();
    let hi: local::u16x8<ScalarToken> = bytes.widen_high();
    assert_eq!(lo.to_array(), [255; 8]);
    assert_eq!(hi.to_array(), [255; 8]);
}

// No feature annotation: a generic backend bound plus a value token is enough.
fn construct_generic<T: magetypes::simd::backends::F32x8Backend>(
    token: T,
    data: &[f32; 8],
) -> local::f32x8<T> {
    let ctor: fn(T, &[f32; 8]) -> local::f32x8<T> = local::f32x8::load_with_token;
    let v = ctor(token, data);
    let explicit = generic::f32x8::from_array_with_token(token, v.to_array());
    let local: local::f32x8<T> = explicit.into();
    local + local::f32x8::zero_with_token(token)
}

#[magetypes(use(f32x8), v3, neon, wasm128, scalar)]
fn generic_token_entry(token: Token, data: &[f32; 8]) -> [f32; 8] {
    let v: f32x8 = construct_generic(token, data);
    v.to_array()
}

#[test]
fn token_construction_in_plain_generic_functions() {
    let data = [1., -2., 3., 4., 5., 6., 7., 8.];
    assert_eq!(construct_generic(ScalarToken, &data).to_array(), data);
    assert_eq!(
        incant!(generic_token_entry(&data), [v3, neon, wasm128, scalar]),
        data
    );
}

#[test]
fn token_view_conversion_and_block_methods() {
    let token = ScalarToken;
    type F = local::f32x4<ScalarToken>;
    type F8 = local::f32x8<ScalarToken>;
    type I = local::i32x4<ScalarToken>;
    let ints = I::from_array_with_token(token, [1, 2, 3, 4]);
    let v = F::from_i32_with_token(token, ints);
    assert_eq!(v.to_array(), [1., 2., 3., 4.]);
    assert_eq!(
        F::from_i32x4_with_token(token, ints).to_array(),
        v.to_array()
    );
    let bits = I::from_array_with_token(token, [1f32.to_bits() as i32; 4]);
    assert_eq!(
        F::from_i32_bitcast_with_token(token, bits).to_array(),
        [1.; 4]
    );
    assert_eq!(
        F::from_bytes_with_token(token, v.as_bytes()).to_array(),
        v.to_array()
    );
    assert_eq!(
        F::from_bytes_owned_with_token(token, *v.as_bytes()).to_array(),
        v.to_array()
    );
    let mut values = v.to_array();
    assert_eq!(
        F::cast_slice_with_token(token, &values).unwrap()[0].to_array(),
        values
    );
    F::cast_slice_mut_with_token(token, &mut values).unwrap()[0] += F::splat_with_token(token, 1.);
    assert_eq!(values, [2., 3., 4., 5.]);
    assert_eq!(
        F8::from_halves_with_token(token, v, v).to_array(),
        [1., 2., 3., 4., 1., 2., 3., 4.]
    );
}

// Glob imports must not reserve the constructor implementation's names in users'
// modules. Two globs make accidental new exports ambiguous at compile time.
#[allow(non_camel_case_types, dead_code)]
mod downstream_names {
    pub struct Context;
    pub struct Explicit;
    pub struct ConstructorMode;
    pub struct core_types;
    pub struct local;
}

#[test]
fn prelude_retains_legacy_types_without_constructor_machinery() {
    use downstream_names::*;
    use magetypes::prelude::*;
    let _: Option<(Context, Explicit, ConstructorMode, core_types, local)> = None;
    let token = ScalarToken::summon().unwrap();
    let value = f32x8::<ScalarToken>::splat(token, 3.0);
    assert_eq!(f32x8::<ScalarToken>::LANES, 8);
    assert_eq!(value.to_array(), [3.0; 8]);
}
