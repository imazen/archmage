//! Actual controlled rounding outputs for a uniform-exponent f32x4 study.
#![forbid(unsafe_code)]
use std::io::{self, BufRead};

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn relaxed_probe(token: archmage::Wasm128Token) -> u32 {
    use magetypes::simd::generic::f32x4;
    let v=|bits|f32x4::splat_t(token,std::hint::black_box(f32::from_bits(bits)));
    v(0x3f800001).mul_add_relaxed(v(0x3f7ffffe),v(0xbf800000)).to_array()[0].to_bits()
}

fn main() {
    #[cfg(target_arch="wasm32")]
    {
        use archmage::SimdToken;
        let expected=match std::env::var("POWF_EXPECT_RELAXED_FUSED").unwrap().as_str(){"true"=>0xa8800000,"false"=>0,_=>panic!("explicit engine rounding required")};
        let actual=relaxed_probe(archmage::Wasm128Token::summon().unwrap());
        assert_eq!(actual,expected,"actual relaxed provider rounding");
        eprintln!("relaxed provider cancellation bits {actual:08x}");
    }
    let native = match std::env::args().nth(1).as_deref() {
        Some("scalar") => 0,
        Some("native") => 1,
        _ => panic!("explicit scalar or native context required"),
    };
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 3);
        let x = u32::from_str_radix(fields[0], 16).unwrap();
        let n = u32::from_str_radix(fields[1], 16).unwrap();
        let domain = fields[2].parse().unwrap();
        print!("{x:08x} {n:08x} {domain}");
        for method in 0..4 {
            let out = archmage_powf_study::rounding_evaluate(
                f32::from_bits(x), f32::from_bits(n), method, domain, native,
            );
            print!(" {out:016x}");
        }
        // Preserve the actual legacy scalar/native methods for a direct
        // same-input, same-context check; no std result supplies oracle bits.
        for method in [1, 3] {
            let out = archmage_powf_study::replacement_evaluate(
                f32::from_bits(x), f32::from_bits(n), method, domain, native,
            );
            print!(" {out:016x}");
        }
        println!();
    }
}
