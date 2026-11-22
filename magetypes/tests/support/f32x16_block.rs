use magetypes::simd::{backends::F32x16Convert, generic::f32x16};

pub fn check<T: F32x16Convert>(token: T, tier: &str) {
    // Include distinct NaN payloads as well as finite values to detect any
    // arithmetic or lane reordering in operations that only move bits.
    let input_bits = |i: usize| {
        let payload = (i + 1) as u32;
        match i % 4 {
            0 => 0x7f80_0000 | payload,
            1 => 0xffc0_0000 | payload,
            2 => 0x3f80_0000 | payload,
            _ => 0xbf80_0000 | payload,
        }
    };
    let input = |i| f32::from_bits(input_bits(i));
    let a = f32x16::from_array_t(token, core::array::from_fn(input));
    let b = f32x16::from_array_t(token, core::array::from_fn(|i| input(i + 16)));
    let expected =
        |offset| core::array::from_fn(|i| input_bits(i / 4 * 4 + i % 4 / 2 + i % 2 * 16 + offset));
    let bits = |v: f32x16<T>| v.to_array().map(f32::to_bits);
    assert_eq!(
        bits(a.interleave_lo(b)),
        expected(0),
        "{tier} interleave low"
    );
    assert_eq!(
        bits(a.interleave_hi(b)),
        expected(2),
        "{tier} interleave high"
    );
    let (lo, hi) = a.interleave(b);
    assert_eq!(
        (bits(lo), bits(hi)),
        (expected(0), expected(2)),
        "{tier} interleave"
    );

    let dual = core::array::from_fn(input);
    let mut rows = f32x16::load_dual_8x8_t(token, &dual);
    for (row, v) in rows.iter().enumerate() {
        assert_eq!(
            bits(*v),
            core::array::from_fn(|i| input_bits(16 * row + i)),
            "{tier} dual load"
        );
    }
    let copy = f32x16::transpose_dual_8x8_copy(rows);
    f32x16::transpose_dual_8x8(&mut rows);
    let dual_expected = |i| input_bits(i % 8 * 16 + i / 16 + i % 16 / 8 * 8);
    for (row, (in_place, copied)) in rows.into_iter().zip(copy).enumerate() {
        let expected = core::array::from_fn(|i| dual_expected(row * 16 + i));
        assert_eq!(bits(in_place), expected, "{tier} dual in-place transpose");
        assert_eq!(bits(copied), expected, "{tier} dual copying transpose");
    }
    let mut out = [0.0; 128];
    f32x16::store_dual_8x8(&rows, &mut out);
    assert_eq!(
        out.map(f32::to_bits),
        core::array::from_fn(dual_expected),
        "{tier} dual store"
    );

    let square = core::array::from_fn(input);
    let mut rows = f32x16::load_16x16_t(token, &square);
    for (row, v) in rows.iter().enumerate() {
        assert_eq!(
            bits(*v),
            core::array::from_fn(|i| input_bits(16 * row + i)),
            "{tier} square load"
        );
    }
    let copy = f32x16::transpose_16x16_copy(rows);
    f32x16::transpose_16x16(&mut rows);
    let square_expected = |i| input_bits(i % 16 * 16 + i / 16);
    for (row, (in_place, copied)) in rows.into_iter().zip(copy).enumerate() {
        let expected = core::array::from_fn(|i| square_expected(row * 16 + i));
        assert_eq!(bits(in_place), expected, "{tier} square in-place transpose");
        assert_eq!(bits(copied), expected, "{tier} square copying transpose");
    }
    let mut out = [0.0; 256];
    f32x16::store_16x16(&rows, &mut out);
    assert_eq!(
        out.map(f32::to_bits),
        core::array::from_fn(square_expected),
        "{tier} square store"
    );

    let mut v = a;
    let iv = v.bitcast_ref_i32();
    assert_eq!(
        iv.to_array().map(|i| i as u32),
        core::array::from_fn(input_bits),
        "{tier} bitcast ref"
    );
    let iv = v.bitcast_mut_i32();
    for i in 0..16 {
        iv[i] = if i == 0 {
            i32::MIN
        } else {
            input_bits(i + 16) as i32
        };
    }
    assert_eq!(
        bits(v),
        core::array::from_fn(|i| if i == 0 {
            (-0.0f32).to_bits()
        } else {
            input_bits(i + 16)
        }),
        "{tier} bitcast mut",
    );
}
