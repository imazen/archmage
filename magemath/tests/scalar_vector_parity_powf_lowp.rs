//! Exhaustive const-generic scalar vs vector parity for `magemath::powf` LOWP (all 32 encodings).
#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/powf_parity.rs"]
mod powf_parity;

macro_rules! lowp_flags {
    ($m:ident $(, $arg:expr)*) => {
        macro_rules! call_one {
            ($f:literal) => {
                $m::<$f, { powf_parity::powf_options_from_bits($f).bits() }>($($arg),*);
            };
        }
        call_one!(8);
        call_one!(9);
        call_one!(10);
        call_one!(12);
        call_one!(13);
        call_one!(14);
        call_one!(24);
        call_one!(25);
        call_one!(26);
        call_one!(28);
        call_one!(29);
        call_one!(30);
        call_one!(40);
        call_one!(41);
        call_one!(42);
        call_one!(44);
        call_one!(45);
        call_one!(46);
        call_one!(56);
        call_one!(57);
        call_one!(58);
        call_one!(60);
        call_one!(61);
        call_one!(62);
        call_one!(138);
        call_one!(142);
        call_one!(154);
        call_one!(158);
        call_one!(170);
        call_one!(174);
        call_one!(186);
        call_one!(190);
    };
}

#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
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
