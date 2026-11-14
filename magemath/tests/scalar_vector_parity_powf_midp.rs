//! Exhaustive const-generic scalar vs vector parity for `magemath::powf` MIDP (all 32 encodings).
#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/powf_parity.rs"]
mod powf_parity;

macro_rules! midp_flags {
    ($m:ident $(, $arg:expr)*) => {
        $m::<0>($($arg),*);
        $m::<1>($($arg),*);
        $m::<2>($($arg),*);
        $m::<4>($($arg),*);
        $m::<5>($($arg),*);
        $m::<6>($($arg),*);
        $m::<16>($($arg),*);
        $m::<17>($($arg),*);
        $m::<18>($($arg),*);
        $m::<20>($($arg),*);
        $m::<21>($($arg),*);
        $m::<22>($($arg),*);
        $m::<32>($($arg),*);
        $m::<33>($($arg),*);
        $m::<34>($($arg),*);
        $m::<36>($($arg),*);
        $m::<37>($($arg),*);
        $m::<38>($($arg),*);
        $m::<48>($($arg),*);
        $m::<49>($($arg),*);
        $m::<50>($($arg),*);
        $m::<52>($($arg),*);
        $m::<53>($($arg),*);
        $m::<54>($($arg),*);
        $m::<130>($($arg),*);
        $m::<134>($($arg),*);
        $m::<146>($($arg),*);
        $m::<150>($($arg),*);
        $m::<162>($($arg),*);
        $m::<166>($($arg),*);
        $m::<178>($($arg),*);
        $m::<182>($($arg),*);
    };
}

#[archmage::magetypes(
    v4x(cfg(avx512)),
    v4(cfg(avx512)),
    v3,
    neon,
    wasm128_relaxed_fma,
    wasm128,
    scalar
)]
fn check_token(token: Token) {
    use powf_parity::*;
    define_powf_token_op!(tok_powf, Token, yes);
    midp_flags!(tok_powf, token);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn check_token_fp16(token: archmage::Avx512Fp16Token) {
    use powf_parity::*;
    define_powf_token_op!(fp16_powf, archmage::Avx512Fp16Token, no);
    midp_flags!(fp16_powf, token);
}

#[test]
fn powf_midp_scalar_and_vector_parity_across_all_32_configs() {
    use powf_parity::standalone_powf;
    midp_flags!(standalone_powf);
    run_all_summonable_tokens!();
}
