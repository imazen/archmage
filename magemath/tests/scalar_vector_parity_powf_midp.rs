//! Exhaustive const-generic scalar vs vector parity for `magemath::powf` MIDP (all 32 encodings).
#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/powf_parity.rs"]
mod powf_parity;

macro_rules! midp_flags {
    ($m:ident $(, $arg:expr)*) => {
        macro_rules! call_one {
            ($f:literal) => {
                $m::<$f, { powf_parity::powf_options_from_bits($f).bits() }>($($arg),*);
            };
        }
        call_one!(0);
        call_one!(1);
        call_one!(2);
        call_one!(4);
        call_one!(5);
        call_one!(6);
        call_one!(16);
        call_one!(17);
        call_one!(18);
        call_one!(20);
        call_one!(21);
        call_one!(22);
        call_one!(32);
        call_one!(33);
        call_one!(34);
        call_one!(36);
        call_one!(37);
        call_one!(38);
        call_one!(48);
        call_one!(49);
        call_one!(50);
        call_one!(52);
        call_one!(53);
        call_one!(54);
        call_one!(130);
        call_one!(134);
        call_one!(146);
        call_one!(150);
        call_one!(162);
        call_one!(166);
        call_one!(178);
        call_one!(182);
    };
}

#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
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
