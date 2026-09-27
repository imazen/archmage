//! Raw interchange must retain bits and preserve the backend's capability proof.
#![forbid(unsafe_code)]
#![cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]

use archmage::{SimdToken, arcane};
use magetypes::simd::generic::*;

macro_rules! roundtrip {
    ($token:ty, $ty:ident, $values:expr) => {{
        let values = $values;
        let token = <$token>::from_context();
        let original = $ty::<$token>::from_array(token, values);
        let restored = $ty::<$token>::from_raw(original.raw());
        assert_eq!(restored.to_array(), values);
        // Nested functions do not inherit the surrounding target features.
        fn plain(token: $token, value: $ty<$token>) -> local::$ty<$token> {
            let ctor: fn($token, _) -> local::$ty<$token> = local::$ty::from_raw_with_token;
            ctor(token, value.raw())
        }
        assert_eq!(plain(token, original).to_array(), values);
        assert_eq!(
            $ty::<$token>::from_raw_with_token(token, original.raw()).to_array(),
            values
        );
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
fn x86_roundtrips() {
    use archmage::X64V3Token;
    roundtrip!(X64V3Token, f32x4, [-0.0, 1.0, f32::INFINITY, f32::MIN]);
    roundtrip!(X64V3Token, f64x2, [f64::MIN, f64::MAX]);
    roundtrip!(X64V3Token, i8x16, [i8::MIN; 16]);
    roundtrip!(X64V3Token, u8x16, [u8::MAX; 16]);
    roundtrip!(X64V3Token, i16x8, [i16::MIN; 8]);
    roundtrip!(X64V3Token, u16x8, [u16::MAX; 8]);
    roundtrip!(X64V3Token, i32x4, [i32::MIN; 4]);
    roundtrip!(X64V3Token, u32x4, [u32::MAX; 4]);
    roundtrip!(X64V3Token, i64x2, [i64::MIN; 2]);
    roundtrip!(X64V3Token, u64x2, [u64::MAX; 2]);
    roundtrip!(X64V3Token, f32x8, [f32::MIN; 8]);
    roundtrip!(X64V3Token, f64x4, [f64::MAX; 4]);
    roundtrip!(X64V3Token, i8x32, [i8::MIN; 32]);
    roundtrip!(X64V3Token, u8x32, [u8::MAX; 32]);
    roundtrip!(X64V3Token, i16x16, [i16::MIN; 16]);
    roundtrip!(X64V3Token, u16x16, [u16::MAX; 16]);
    roundtrip!(X64V3Token, i32x8, [i32::MIN; 8]);
    roundtrip!(X64V3Token, u32x8, [u32::MAX; 8]);
    roundtrip!(X64V3Token, i64x4, [i64::MIN; 4]);
    roundtrip!(X64V3Token, u64x4, [u64::MAX; 4]);
    let bits = [0x8000_0000, 0x7fc0_1234, 1, 0xff80_0000];
    let v = f32x4::<X64V3Token>::from_array(X64V3Token::from_context(), bits.map(f32::from_bits));
    assert_eq!(
        f32x4::<X64V3Token>::from_raw(v.raw())
            .to_array()
            .map(f32::to_bits),
        bits
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn native_raw_roundtrips() {
    #[arcane]
    fn entry(_token: archmage::X64V3Token) {
        x86_roundtrips();
    }
    entry(archmage::X64V3Token::summon().expect("run this test on a v3 CPU"));
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_legacy_roundtrips() {
    #[arcane]
    fn entry(token: archmage::NeonToken) {
        let bits = [0x8000_0000, 0x7fc0_1234, 1, 0xff80_0000];
        let a = f32x4::from_array(token, bits.map(f32::from_bits));
        let b = f32x4::from_float32x4_t(token, a.raw());
        assert_eq!(b.to_array().map(f32::to_bits), bits);
        assert_eq!(
            f32x4::<archmage::NeonToken>::from_raw(b.raw())
                .to_array()
                .map(f32::to_bits),
            bits
        );
        let bits64 = [0x8000_0000_0000_0000, 0x7ff8_0000_0000_1234];
        let a = f64x2::from_array(token, bits64.map(f64::from_bits));
        assert_eq!(
            f64x2::from_float64x2_t(token, a.raw())
                .to_array()
                .map(f64::to_bits),
            bits64
        );
        roundtrip!(archmage::NeonToken, i8x16, [i8::MIN; 16]);
        roundtrip!(archmage::NeonToken, u8x16, [u8::MAX; 16]);
        roundtrip!(archmage::NeonToken, i16x8, [i16::MIN; 8]);
        roundtrip!(archmage::NeonToken, u16x8, [u16::MAX; 8]);
        roundtrip!(archmage::NeonToken, i32x4, [i32::MIN; 4]);
        roundtrip!(archmage::NeonToken, u32x4, [u32::MAX; 4]);
        roundtrip!(archmage::NeonToken, i64x2, [i64::MIN; 2]);
        roundtrip!(archmage::NeonToken, u64x2, [u64::MAX; 2]);
    }
    entry(archmage::NeonToken::summon().expect("run this test on a NEON CPU"));
}
