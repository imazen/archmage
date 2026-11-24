#![cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]

use archmage::SimdToken;
use magetypes::WidthDispatch;

#[test]
fn constructors_preserve_lane_order_and_bits_at_each_width() {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    let Some(token) = archmage::X64V3Token::summon() else {
        return;
    };
    #[cfg(target_arch = "aarch64")]
    let token = archmage::NeonToken::summon().expect("AArch64 NEON");
    #[cfg(target_arch = "wasm32")]
    let token = archmage::Wasm128Token::summon().expect("Wasm SIMD128");

    let f32_lanes = [
        0.0_f32,
        -0.0,
        f32::from_bits(0x7f80_0042),
        f32::from_bits(1),
    ];
    assert_eq!(
        token.f32x4_load(&f32_lanes).to_array().map(f32::to_bits),
        f32_lanes.map(f32::to_bits)
    );
    assert_eq!(token.f32x4_zero().to_array().map(f32::to_bits), [0; 4]);
    assert_eq!(
        token.f32x4_splat(-0.0).to_array().map(f32::to_bits),
        [(-0.0_f32).to_bits(); 4]
    );

    let lanes = core::array::from_fn(|i| i16::try_from(i).unwrap() * 101 - 812);
    assert_eq!(token.i16x16_load(&lanes).to_array(), lanes);
    assert_eq!(token.i16x16_zero().to_array(), [0; 16]);
    assert_eq!(token.i16x16_splat(i16::MIN).to_array(), [i16::MIN; 16]);

    let f32_bits = core::array::from_fn(|i| {
        let bits = f32_lanes[i % 4].to_bits();
        if i < 4 {
            bits
        } else {
            bits | ((i / 4) as u32) << 8
        }
    });
    assert_eq!(
        token
            .f32x16_load(&f32_bits.map(f32::from_bits))
            .map(|part| part.to_array().map(f32::to_bits))
            .concat(),
        f32_bits
    );
    assert_eq!(
        token
            .f32x16_splat(-0.0)
            .map(|part| part.to_array().map(f32::to_bits)),
        [[(-0.0_f32).to_bits(); 4]; 4]
    );
    assert_eq!(
        token
            .f32x16_zero()
            .map(|part| part.to_array().map(f32::to_bits)),
        [[0; 4]; 4]
    );

    let bytes = core::array::from_fn(|i| 192 + u8::try_from(i).unwrap());
    assert_eq!(
        token
            .u8x64_load(&bytes)
            .map(|part| part.to_array())
            .concat(),
        bytes
    );
    assert_eq!(
        token.u8x64_splat(255).map(|part| part.to_array()),
        [[255; 16]; 4]
    );
    assert_eq!(token.u8x64_zero().map(|part| part.to_array()), [[0; 16]; 4]);
}
