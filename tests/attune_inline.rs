#![forbid(unsafe_code)]
#![deny(warnings)]

use archmage::{ScalarToken, attune};

mod implementation {
    use super::*;

    pub struct Kernel;

    impl Kernel {
        #[attune(scalar, inline(default))]
        pub fn identity<T: Copy>(&self, value: T) -> T {
            value
        }
    }

    #[attune(inline(default), make(pub _v3, pub _scalar, _v3_t, _scalar_t, _))]
    pub(crate) fn sum(values: &[u32]) -> u32 {
        values.iter().copied().sum()
    }
}

pub use implementation::Kernel;

#[attune(inline(default), make(_v3, _scalar, _))]
fn sum_twice(values: &[u32]) -> u32 {
    2 * archmage::attuned!(implementation::sum(values), [_v3, _scalar])
}

#[attune(wrap, inline(default))]
fn scalar_proof(token: ScalarToken, value: u32) -> u32 {
    let _ = token;
    value + 1
}

#[attune(scalar, inline(none))]
fn identity(value: u32) -> u32 {
    value
}

#[test]
fn visibility_policy_preserves_generics_dispatch_and_context_calls() {
    assert_eq!(Kernel.identity(17u64), 17);
    assert_eq!(implementation::sum(&[1, 2, 3]), 6);
    assert_eq!(sum_twice(&[1, 2, 3]), 12);
    assert_eq!(implementation::sum_scalar_t(ScalarToken, &[4, 5]), 9);
    assert_eq!(scalar_proof(ScalarToken, 9), 10);
    assert_eq!(identity(7), 7);
}
