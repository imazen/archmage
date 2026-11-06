#![forbid(unsafe_code)]
#[inline(never)]
pub fn compile_only_floor(x: f32) -> f32 {
    #[cfg(feature = "accelerated")]
    {
        zenmath::simd::scalar::floorf(x)
    }
    #[cfg(not(feature = "accelerated"))]
    {
        zenmath::scalar::reference::floorf(x)
    }
}
#[inline(never)]
pub fn compile_only_exp(x: [f32; 4]) -> [f32; 4] {
    #[cfg(feature = "accelerated")]
    {
        zenmath::simd::array::exp_f32x4(x)
    }
    #[cfg(not(feature = "accelerated"))]
    {
        x.map(zenmath::scalar::reference::expf)
    }
}
fn main() {
    let x: f32 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "-0.25".to_owned())
        .parse()
        .unwrap();
    let want = zenmath::scalar::reference::floorf(x);
    assert_eq!(
        compile_only_floor(std::hint::black_box(x)).to_bits(),
        want.to_bits()
    );
    let input = [x, x + 0.5, -x, 1.0];
    let got = compile_only_exp(std::hint::black_box(input));
    for i in 0..4 {
        let expected = zenmath::scalar::reference::expf(input[i]);
        assert!(got[i].is_nan() && expected.is_nan() || got[i].to_bits() == expected.to_bits());
    }
    std::hint::black_box(got);
}
