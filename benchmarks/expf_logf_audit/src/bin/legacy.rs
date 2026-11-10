//! Frozen-source comparison protocol: compile this unchanged at each revision.
use magetypes::simd::generic::f32x4;

#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn evaluate(token: Token, xs: [f32; 4]) -> [[f32; 4]; 8] {
    let x = f32x4::from_array_t(token, xs);
    [
        x.exp2_midp().to_array(),
        x.exp2_lowp().to_array(),
        x.exp2_midp_unchecked().to_array(),
        x.log2_midp().to_array(),
        x.log2_lowp().to_array(),
        x.log2_midp_unchecked().to_array(),
        x.exp_midp().to_array(),
        x.ln_midp().to_array(),
    ]
}

fn main() {
    let context = std::env::args().nth(1).unwrap_or_else(|| "scalar".into());
    let mut inputs: Vec<u32> = vec![
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x3f800000, 0x3f7fffff, 0x3f800001,
        0xbf800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001,
    ];
    let mut state = 0x12345678u32;
    for _ in 0..16384 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        inputs.push(state);
    }
    for xs in inputs.chunks_exact(4) {
        let a = [xs[0], xs[1], xs[2], xs[3]].map(f32::from_bits);
        let out = match context.as_str() {
            "scalar" => evaluate_scalar(archmage::ScalarToken, a),
            #[cfg(target_arch = "x86_64")]
            "v3" => evaluate_v3(archmage::X64V3Token::summon().expect("V3 required"), a),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            "v4" => evaluate_v4(archmage::X64V4Token::summon().expect("V4 required"), a),
            #[cfg(target_arch = "aarch64")]
            "neon" => evaluate_neon(archmage::NeonToken::summon().expect("NEON required"), a),
            #[cfg(target_arch = "wasm32")]
            "wasm" => evaluate_wasm128(archmage::Wasm128Token::summon().expect("SIMD required"), a),
            _ => panic!("unsupported context: {context}"),
        };
        for lane in 0..4 {
            print!("{:08x}", xs[lane]);
            for op in &out {
                print!("\t{:08x}", op[lane].to_bits());
            }
            println!();
        }
    }
}
