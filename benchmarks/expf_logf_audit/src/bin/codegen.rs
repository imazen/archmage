//! Named V3 consumer probes, keeping the frozen main graph independent.
#![forbid(unsafe_code)]
#[path = "../frozen.rs"]
#[allow(dead_code)]
mod frozen;
#[cfg(target_arch = "x86_64")]
mod probes {
    use super::frozen::x4::Frozen;
    use archmage::{SimdToken, X64V3Token};
    use magetypes::simd::generic::f32x4 as V;

    macro_rules! probe {
        ($name:ident, $x:ident, $token:ident, $body:expr) => {
            #[archmage::arcane]
            fn $name($token: X64V3Token, input: [f32; 4]) -> [f32; 4] {
                let $x = V::from_array_t($token, input);
                $body.to_array()
            }
        };
    }
    probe!(main_log2_midp, x, token, x.old_log2_midp(token));
    probe!(pr147_log2_midp, x, token, x.log2_midp());
    probe!(main_log2_lowp, x, token, x.old_log2_lowp(token));
    probe!(pr147_log2_lowp, x, token, x.log2_lowp());
    probe!(main_exp2_midp, x, token, x.old_exp2_midp(token));
    probe!(pr147_exp2_midp, x, token, x.exp2_midp());
    probe!(main_exp2_lowp, x, token, x.old_exp2_lowp(token));
    probe!(pr147_exp2_lowp, x, token, x.exp2_lowp());

    pub fn run() {
        let token = X64V3Token::summon().expect("V3 required");
        let input = std::hint::black_box([0.5, 0.75, 1.25, 2.]);
        for f in [
            main_log2_midp,
            pr147_log2_midp,
            main_log2_lowp,
            pr147_log2_lowp,
            main_exp2_midp,
            pr147_exp2_midp,
            main_exp2_lowp,
            pr147_exp2_lowp,
        ] {
            std::hint::black_box(f(token, input));
        }
    }
}
fn main() {
    #[cfg(target_arch = "x86_64")]
    probes::run();
    #[cfg(not(target_arch = "x86_64"))]
    panic!("this consumer codegen probe requires x86_64 V3");
}
