//! The `#[arcane]` wrapper spends its one `unsafe` block on the sibling the
//! macro generated. Rust's name resolution is what keeps that true: a glob
//! import of another `__arcane_<fn>` loses to the local item, an inherent
//! `__arcane_<fn>` method wins over a trait method of that name, and an
//! inherent associated function wins over a trait's. The duplicate-definition
//! and explicit-import cases are rustc errors, and a parameter named like the
//! sibling (the one local that could shadow it) is a macro error; all three
//! are pinned in `tests/soundness/sibling_*_exploit.rs`.
//!
//! The macro reads the token type by name, so the cases are stamped out per
//! architecture with the baseline token that `summon()` always returns.
#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

macro_rules! sibling_resolution_cases {
    ($token:ident) => {
        use archmage::prelude::*;

        const IMPOSTOR: f32 = -1.0;

        mod impostor {
            use archmage::$token;
            /// Same name and signature as the generated sibling of `kernel`.
            pub unsafe fn __arcane_kernel(_token: $token, _data: &[f32; 4]) -> f32 {
                super::IMPOSTOR
            }
            /// Same name as the generated sibling of `Plane::sum`.
            pub trait Sum {
                fn __arcane_sum(&self, _token: $token) -> f32 {
                    super::IMPOSTOR
                }
            }
            /// Same name as the generated sibling of `Plane::filled`.
            pub trait Build {
                fn __arcane_filled(_token: $token, _value: f32) -> super::Plane {
                    super::Plane([super::IMPOSTOR; 4])
                }
            }
        }

        #[allow(unused_imports)]
        use impostor::*;

        #[arcane]
        fn kernel(_token: $token, data: &[f32; 4]) -> f32 {
            data.iter().sum()
        }

        pub struct Plane([f32; 4]);

        impl impostor::Sum for Plane {}
        impl impostor::Build for Plane {}

        impl Plane {
            #[arcane]
            fn sum(&self, _token: $token) -> f32 {
                self.0.iter().sum()
            }

            #[arcane(in_impl)]
            fn filled(_token: $token, value: f32) -> Self {
                Self([value; 4])
            }
        }

        #[test]
        fn glob_imported_impostor_loses_to_the_generated_sibling() {
            let token = $token::summon().expect("baseline token is always available");
            assert_eq!(kernel(token, &[1.0, 2.0, 3.0, 4.0]), 10.0);
        }

        #[test]
        fn trait_items_named_like_the_sibling_lose_to_the_inherent_ones() {
            let token = $token::summon().expect("baseline token is always available");
            let plane = Plane::filled(token, 2.0);
            assert_eq!(plane.0, [2.0; 4]);
            assert_eq!(plane.sum(token), 8.0);
        }
    };
}

#[cfg(target_arch = "x86_64")]
sibling_resolution_cases!(X64V1Token);
#[cfg(target_arch = "aarch64")]
sibling_resolution_cases!(NeonToken);
