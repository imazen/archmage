//! Test for #[arcane] on associated functions (no self receiver) inside impl blocks.
//!
//! This is the pattern used by mozjpeg-rs: SIMD functions inside an impl block
//! that take the struct as an explicit parameter (not a self receiver) because
//! #[arcane] needs the token as the first parameter.
//!
//! Default sibling expansion fails here: the macro generates `__arcane_fn(...)`
//! (bare call) but the sibling is an inherent method requiring
//! `Self::__arcane_fn(...)`. The proc macro can't detect impl scope.
//!
//! Fix: use `#[arcane(nested)]` which generates an inner function instead of a
//! sibling, avoiding the scope issue entirely.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
mod inherent_no_receiver {
    use archmage::{SimdToken, X64V3Token, arcane};

    struct Encoder {
        buffer: Vec<u8>,
    }

    impl Encoder {
        fn new() -> Self {
            Self { buffer: Vec::new() }
        }

        /// Associated function with #[arcane(nested)] — token first, struct
        /// passed explicitly. This is the mozjpeg-rs pattern for SIMD methods
        /// where the token must be the first parameter.
        ///
        /// `nested` is required here because default sibling expansion generates
        /// a bare `__arcane_process(...)` call that can't resolve inside an impl
        /// block (it's a method, not a free function).
        #[arcane(nested)]
        fn process(token: X64V3Token, encoder: &mut Encoder, data: &[u8; 8]) {
            // Call another arcane associated function from within this one
            let sum = Encoder::compute(token, data);
            encoder.buffer.push(sum);
        }

        /// Another associated function with #[arcane(nested)] — pure computation,
        /// no self.
        #[arcane(nested)]
        fn compute(token: X64V3Token, data: &[u8; 8]) -> u8 {
            let mut sum: u16 = 0;
            for &b in data {
                sum = sum.wrapping_add(b as u16);
            }
            (sum & 0xFF) as u8
        }

        fn get_buffer(&self) -> &[u8] {
            &self.buffer
        }
    }

    #[test]
    fn inherent_associated_fn_no_receiver() {
        if let Some(token) = X64V3Token::summon() {
            let mut enc = Encoder::new();
            Encoder::process(token, &mut enc, &[1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(enc.get_buffer(), &[36]);
        }
    }

    #[test]
    fn inherent_associated_fn_standalone() {
        if let Some(token) = X64V3Token::summon() {
            let result = Encoder::compute(token, &[10, 20, 30, 40, 0, 0, 0, 0]);
            assert_eq!(result, 100);
        }
    }
}
