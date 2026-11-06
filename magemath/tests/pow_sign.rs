//! Independent exact rails plus scalar-oracle regression for the packed sign path.
#![forbid(unsafe_code)]
use magemath::portable::array::pow_f32x8;
#[test]
fn mathematical_sign_and_rails() {
    for (x, y, expected) in [
        (-2.0, 3.0, 0xc1000000),
        (-2.0, 4.0, 0x41800000),
        (-2.0, -3.0, 0xbe000000),
        (-0.0, 3.0, 0x80000000),
        (-0.0, -3.0, 0xff800000),
        (-0.0, 2.0, 0),
        (f32::NEG_INFINITY, 3.0, 0xff800000),
        (f32::NEG_INFINITY, 2.0, 0x7f800000),
        (f32::NEG_INFINITY, -3.0, 0x80000000),
        (-1.0, 16_777_216.0, 0x3f800000),
        (-1.0, 16_777_215.0, 0xbf800000),
        (0.0, 0.0, 0x3f800000),
    ] {
        assert_eq!(pow_f32x8([x; 8], [y; 8])[0].to_bits(), expected, "{x}^{y}");
    }
    assert!(pow_f32x8([-2.0; 8], [2.4; 8])[0].is_nan());
    assert_eq!(pow_f32x8([1.0; 8], [f32::NAN; 8])[0], 1.0);
    assert!(pow_f32x8([1.0; 8], [f32::from_bits(0x7f800001); 8])[0].is_nan());
}
#[test]
fn negative_integer_samples_match_complete_core() {
    let exponents = [
        -149.0,
        -25.0,
        -7.0,
        -3.0,
        -2.0,
        0.0,
        1.0,
        2.0,
        3.0,
        7.0,
        23.0,
        127.0,
        16_777_215.0,
        16_777_216.0,
        33_554_432.0,
        2.4,
    ];
    for i in 1..2048 {
        let x = -f32::from_bits(0x3e000000 + i * 24637);
        for ys in exponents.as_chunks::<8>().0 {
            let out = pow_f32x8([x; 8], *ys);
            for (got, y) in out.into_iter().zip(ys) {
                let want = zenmath::scalar::powf(x, *y);
                assert!(
                    got.to_bits() == want.to_bits() || got.is_nan() && want.is_nan(),
                    "{x}^{y}: {got:x?} != {want:x?}"
                );
            }
        }
    }
}
