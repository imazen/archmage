//! Directed legacy exceptional-value observations; output bits are not accuracy claims.
#![allow(deprecated)] // Deliberately validates the retained Magetypes compatibility implementation.
#![forbid(unsafe_code)]
use archmage::SimdToken;
#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn dump(token: Token) {
    use magetypes::simd::generic::f32x4;
    for base in [
        0.0_f32,
        -1.0,
        1.0,
        f32::INFINITY,
        f32::from_bits(0x7fc0_0123),
        f32::from_bits(0x7f80_0001),
    ] {
        for n in [0.0_f32, 2.4] {
            let v = f32x4::splat_t(token, std::hint::black_box(base));
            let n = std::hint::black_box(n);
            let bits = [
                v.exp2_midp(),
                v.exp_midp(),
                v.exp2_lowp(),
                v.exp_lowp(),
                v.pow_midp(n),
                v.pow_lowp(n),
                v.sigmoid_midp(),
            ]
            .map(|x| x.to_array()[0].to_bits());
            println!(
                "{} {:08x} {:08x} {:08x?}",
                Token::NAME,
                base.to_bits(),
                n.to_bits(),
                bits
            );
        }
    }
}
fn main() {
    println!(
        "# token base exponent [exp2_midp exp_midp exp2_lowp exp_lowp pow_midp pow_lowp sigmoid_midp]"
    );
    dump_scalar(archmage::ScalarToken);
    #[cfg(target_arch = "x86_64")]
    dump_v3(archmage::X64V3Token::summon().expect("requires V3"));
    #[cfg(target_arch = "aarch64")]
    dump_neon(archmage::NeonToken::summon().expect("requires NEON"));
    #[cfg(target_arch = "wasm32")]
    dump_wasm128(archmage::Wasm128Token::summon().expect("requires SIMD128"));
}
