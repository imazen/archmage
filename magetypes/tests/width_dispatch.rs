#![cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]

use archmage::SimdToken;
use magetypes::WidthDispatch;

fn check_width_dispatch_all_types<T: WidthDispatch>(token: T)
where
    T::F32x4: Copy,
    T::F64x2: Copy,
{
    // Compile-time witness that the trait is object-usable generically; concrete
    // per-target checks below verify exact lane order and bit patterns for all
    // 30 (10 element types × 3 widths) constructors.
    let _ = token.f32x4_zero();
    let _ = token.f64x2_zero();
}

macro_rules! check_native_ctor {
    ($token:expr, $load:ident, $splat:ident, $zero:ident, $lanes:expr, $splat_val:expr, $zero_val:expr) => {{
        let lanes = $lanes;
        assert_eq!($token.$load(&lanes).to_array(), lanes);
        assert_eq!(
            $token.$splat($splat_val).to_array(),
            lanes.map(|_| $splat_val)
        );
        assert_eq!($token.$zero().to_array(), lanes.map(|_| $zero_val));
    }};
}

macro_rules! check_float_ctor {
    ($token:expr, $load:ident, $splat:ident, $zero:ident, $lanes:expr, $splat_val:expr, $bits_fn:expr) => {{
        let lanes = $lanes;
        let to_bits = $bits_fn;
        assert_eq!(
            $token.$load(&lanes).to_array().map(to_bits),
            lanes.map(to_bits)
        );
        assert_eq!(
            $token.$splat($splat_val).to_array().map(to_bits),
            lanes.map(|_| to_bits($splat_val))
        );
        assert_eq!($token.$zero().to_array().map(to_bits), lanes.map(|_| 0));
    }};
}

macro_rules! check_chunked_ctor {
    ($token:expr, $load:ident, $splat:ident, $zero:ident, $lanes:expr, $splat_val:expr, $zero_val:expr, $sub_n:expr) => {{
        let lanes = $lanes;
        assert_eq!(
            $token.$load(&lanes).map(|part| part.to_array()).concat(),
            lanes
        );
        assert_eq!(
            $token.$splat($splat_val).map(|part| part.to_array()),
            [[$splat_val; $sub_n]; 4]
        );
        assert_eq!(
            $token.$zero().map(|part| part.to_array()),
            [[$zero_val; $sub_n]; 4]
        );
    }};
}

macro_rules! check_chunked_float_ctor {
    ($token:expr, $load:ident, $splat:ident, $zero:ident, $lanes:expr, $splat_val:expr, $bits_fn:expr, $sub_n:expr) => {{
        let lanes = $lanes;
        let to_bits = $bits_fn;
        assert_eq!(
            $token
                .$load(&lanes)
                .map(|part| part.to_array().map(to_bits))
                .concat(),
            lanes.map(to_bits)
        );
        assert_eq!(
            $token
                .$splat($splat_val)
                .map(|part| part.to_array().map(to_bits)),
            [[to_bits($splat_val); $sub_n]; 4]
        );
        assert_eq!(
            $token.$zero().map(|part| part.to_array().map(to_bits)),
            [[0; $sub_n]; 4]
        );
    }};
}

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

    check_width_dispatch_all_types(token);

    // 128-bit: all 10 element types
    let f32_4 = [
        0.0_f32,
        -0.0,
        f32::from_bits(0x7f80_0042),
        f32::from_bits(1),
    ];
    let f64_2 = [-0.0_f64, f64::from_bits(0x7ff0_0000_0000_0042)];
    let i8_16: [i8; 16] = core::array::from_fn(|i| (i as i8).wrapping_mul(17).wrapping_sub(64));
    let u8_16: [u8; 16] = core::array::from_fn(|i| 240u8.wrapping_add(i as u8 * 7));
    let i16_8: [i16; 8] = core::array::from_fn(|i| (i as i16) * 4001 - 16000);
    let u16_8: [u16; 8] = core::array::from_fn(|i| 60000u16.wrapping_add((i as u16) * 1234));
    let i32_4 = [i32::MIN, -1, 0, i32::MAX];
    let u32_4 = [0, 1, 0x8000_0001, u32::MAX];
    let i64_2 = [i64::MIN, i64::MAX];
    let u64_2 = [0x0123_4567_89ab_cdef, u64::MAX];

    check_float_ctor!(
        token,
        f32x4_load,
        f32x4_splat,
        f32x4_zero,
        f32_4,
        -0.0_f32,
        f32::to_bits
    );
    check_float_ctor!(
        token,
        f64x2_load,
        f64x2_splat,
        f64x2_zero,
        f64_2,
        -0.0_f64,
        f64::to_bits
    );
    check_native_ctor!(
        token,
        i8x16_load,
        i8x16_splat,
        i8x16_zero,
        i8_16,
        i8::MIN,
        0i8
    );
    check_native_ctor!(
        token,
        u8x16_load,
        u8x16_splat,
        u8x16_zero,
        u8_16,
        255u8,
        0u8
    );
    check_native_ctor!(
        token,
        i16x8_load,
        i16x8_splat,
        i16x8_zero,
        i16_8,
        i16::MIN,
        0i16
    );
    check_native_ctor!(
        token,
        u16x8_load,
        u16x8_splat,
        u16x8_zero,
        u16_8,
        u16::MAX,
        0u16
    );
    check_native_ctor!(
        token,
        i32x4_load,
        i32x4_splat,
        i32x4_zero,
        i32_4,
        i32::MIN,
        0i32
    );
    check_native_ctor!(
        token,
        u32x4_load,
        u32x4_splat,
        u32x4_zero,
        u32_4,
        u32::MAX,
        0u32
    );
    check_native_ctor!(
        token,
        i64x2_load,
        i64x2_splat,
        i64x2_zero,
        i64_2,
        i64::MIN,
        0i64
    );
    check_native_ctor!(
        token,
        u64x2_load,
        u64x2_splat,
        u64x2_zero,
        u64_2,
        u64::MAX,
        0u64
    );

    // 256-bit: all 10 element types
    let f32_8: [f32; 8] = core::array::from_fn(|i| {
        let bits = f32_4[i % 4].to_bits() ^ ((i as u32) << 8);
        f32::from_bits(bits)
    });
    let f64_4 = [
        -0.0_f64,
        0.0,
        f64::from_bits(1),
        f64::from_bits(0x7ff0_0000_0000_0099),
    ];
    let i8_32: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(9).wrapping_sub(120));
    let u8_32: [u8; 32] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8 * 3));
    let i16_16: [i16; 16] = core::array::from_fn(|i| i16::try_from(i).unwrap() * 101 - 812);
    let u16_16: [u16; 16] = core::array::from_fn(|i| 50000u16.wrapping_add((i as u16) * 777));
    let i32_8: [i32; 8] = core::array::from_fn(|i| i32::MIN.wrapping_add((i as i32) * 300_000_001));
    let u32_8: [u32; 8] =
        core::array::from_fn(|i| 0xf000_0000u32.wrapping_add((i as u32) * 0x0102_0304));
    let i64_4 = [i64::MIN, -1, 1, i64::MAX];
    let u64_4 = [0, 1, 0x8000_0000_0000_0001, u64::MAX];

    check_float_ctor!(
        token,
        f32x8_load,
        f32x8_splat,
        f32x8_zero,
        f32_8,
        -0.0_f32,
        f32::to_bits
    );
    check_float_ctor!(
        token,
        f64x4_load,
        f64x4_splat,
        f64x4_zero,
        f64_4,
        -0.0_f64,
        f64::to_bits
    );
    check_native_ctor!(
        token,
        i8x32_load,
        i8x32_splat,
        i8x32_zero,
        i8_32,
        i8::MIN,
        0i8
    );
    check_native_ctor!(
        token,
        u8x32_load,
        u8x32_splat,
        u8x32_zero,
        u8_32,
        255u8,
        0u8
    );
    check_native_ctor!(
        token,
        i16x16_load,
        i16x16_splat,
        i16x16_zero,
        i16_16,
        i16::MIN,
        0i16
    );
    check_native_ctor!(
        token,
        u16x16_load,
        u16x16_splat,
        u16x16_zero,
        u16_16,
        u16::MAX,
        0u16
    );
    check_native_ctor!(
        token,
        i32x8_load,
        i32x8_splat,
        i32x8_zero,
        i32_8,
        i32::MIN,
        0i32
    );
    check_native_ctor!(
        token,
        u32x8_load,
        u32x8_splat,
        u32x8_zero,
        u32_8,
        u32::MAX,
        0u32
    );
    check_native_ctor!(
        token,
        i64x4_load,
        i64x4_splat,
        i64x4_zero,
        i64_4,
        i64::MIN,
        0i64
    );
    check_native_ctor!(
        token,
        u64x4_load,
        u64x4_splat,
        u64x4_zero,
        u64_4,
        u64::MAX,
        0u64
    );

    // 512-bit: all 10 element types ([W128; 4] on V3/NEON/WASM)
    let f32_16: [f32; 16] = core::array::from_fn(|i| {
        let bits = f32_4[i % 4].to_bits();
        f32::from_bits(if i < 4 {
            bits
        } else {
            bits | (((i / 4) as u32) << 8)
        })
    });
    let f64_8: [f64; 8] = core::array::from_fn(|i| {
        let bits = f64_4[i % 4].to_bits();
        f64::from_bits(if i < 4 { bits } else { bits ^ (i as u64) })
    });
    let i8_64: [i8; 64] = core::array::from_fn(|i| (i as i8).wrapping_mul(5).wrapping_sub(127));
    let u8_64: [u8; 64] = core::array::from_fn(|i| 192 + u8::try_from(i).unwrap());
    let i16_32: [i16; 32] = core::array::from_fn(|i| (i as i16) * 513 - 8192);
    let u16_32: [u16; 32] = core::array::from_fn(|i| 40000u16.wrapping_add((i as u16) * 513));
    let i32_16: [i32; 16] = core::array::from_fn(|i| (i as i32) * 100_000_003 - 800_000_000);
    let u32_16: [u32; 16] =
        core::array::from_fn(|i| 0xe000_0000u32.wrapping_add((i as u32) * 0x0101_0101));
    let i64_8: [i64; 8] =
        core::array::from_fn(|i| i64::MIN.wrapping_add((i as i64) * 0x1234_5678_9abc_def1));
    let u64_8: [u64; 8] = core::array::from_fn(|i| (i as u64).wrapping_mul(0x1111_2222_3333_4445));

    check_chunked_float_ctor!(
        token,
        f32x16_load,
        f32x16_splat,
        f32x16_zero,
        f32_16,
        -0.0_f32,
        f32::to_bits,
        4
    );
    check_chunked_float_ctor!(
        token,
        f64x8_load,
        f64x8_splat,
        f64x8_zero,
        f64_8,
        -0.0_f64,
        f64::to_bits,
        2
    );
    check_chunked_ctor!(
        token,
        i8x64_load,
        i8x64_splat,
        i8x64_zero,
        i8_64,
        i8::MIN,
        0i8,
        16
    );
    check_chunked_ctor!(
        token,
        u8x64_load,
        u8x64_splat,
        u8x64_zero,
        u8_64,
        255u8,
        0u8,
        16
    );
    check_chunked_ctor!(
        token,
        i16x32_load,
        i16x32_splat,
        i16x32_zero,
        i16_32,
        i16::MIN,
        0i16,
        8
    );
    check_chunked_ctor!(
        token,
        u16x32_load,
        u16x32_splat,
        u16x32_zero,
        u16_32,
        u16::MAX,
        0u16,
        8
    );
    check_chunked_ctor!(
        token,
        i32x16_load,
        i32x16_splat,
        i32x16_zero,
        i32_16,
        i32::MIN,
        0i32,
        4
    );
    check_chunked_ctor!(
        token,
        u32x16_load,
        u32x16_splat,
        u32x16_zero,
        u32_16,
        u32::MAX,
        0u32,
        4
    );
    check_chunked_ctor!(
        token,
        i64x8_load,
        i64x8_splat,
        i64x8_zero,
        i64_8,
        i64::MIN,
        0i64,
        2
    );
    check_chunked_ctor!(
        token,
        u64x8_load,
        u64x8_splat,
        u64x8_zero,
        u64_8,
        u64::MAX,
        0u64,
        2
    );
}
