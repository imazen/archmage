//! Exhaustive const-generic scalar vs vector parity for `magemath::powf` LOWP (all 32 encodings).
#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/powf_parity.rs"]
mod powf_parity;

macro_rules! lowp_flags {
    ($m:ident $(, $arg:expr)*) => {
        $m::<8>($($arg),*);
        $m::<9>($($arg),*);
        $m::<10>($($arg),*);
        $m::<12>($($arg),*);
        $m::<13>($($arg),*);
        $m::<14>($($arg),*);
        $m::<24>($($arg),*);
        $m::<25>($($arg),*);
        $m::<26>($($arg),*);
        $m::<28>($($arg),*);
        $m::<29>($($arg),*);
        $m::<30>($($arg),*);
        $m::<40>($($arg),*);
        $m::<41>($($arg),*);
        $m::<42>($($arg),*);
        $m::<44>($($arg),*);
        $m::<45>($($arg),*);
        $m::<46>($($arg),*);
        $m::<56>($($arg),*);
        $m::<57>($($arg),*);
        $m::<58>($($arg),*);
        $m::<60>($($arg),*);
        $m::<61>($($arg),*);
        $m::<62>($($arg),*);
        $m::<138>($($arg),*);
        $m::<142>($($arg),*);
        $m::<154>($($arg),*);
        $m::<158>($($arg),*);
        $m::<170>($($arg),*);
        $m::<174>($($arg),*);
        $m::<186>($($arg),*);
        $m::<190>($($arg),*);
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
    lowp_flags!(tok_powf, token);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn check_token_fp16(token: archmage::Avx512Fp16Token) {
    use powf_parity::*;
    define_powf_token_op!(fp16_powf, archmage::Avx512Fp16Token, no);
    lowp_flags!(fp16_powf, token);
}

#[test]
fn powf_lowp_scalar_and_vector_parity_across_all_32_configs() {
    use powf_parity::standalone_powf;
    lowp_flags!(standalone_powf);
    run_all_summonable_tokens!();
}
