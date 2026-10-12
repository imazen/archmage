//! Definition/call contracts for the unified frontend. The legacy suites remain
//! the compatibility oracle and are intentionally unchanged.
#![forbid(unsafe_code)]
#![deny(warnings)]

use archmage::{attune, attuned};

#[attune(make(_v3, _v3_t, _scalar, _scalar_t, _))]
fn sum<const N: usize>((bias, values): (u32, [u32; N])) -> u32 {
    bias + values.into_iter().sum::<u32>()
}

#[attune(make(_v3, _v3_t, _scalar, _scalar_t, _))]
fn composition(value: u32) -> u32 {
    // A qualified invocation must be consumed as a whole by the body rewriter.
    ::archmage::attuned!(sum((value, [1, 2, 3])), [_v3, _scalar])
}

#[attune(make(_v3, _v3_t, _scalar, _scalar_t, _),
    names(_v3 = renamed_direct, _v3_t = renamed_proof))]
fn renamed(value: String) -> usize {
    value.len()
}

#[attune(v3)]
fn feature_helper(value: u32) -> u32 {
    value + 1
}

#[attune]
fn inferred_v3(value: u32) -> u32 {
    feature_helper(value)
}

#[attune(wrap)]
fn boundary(token: archmage::X64V3Token, value: u32) -> u32 {
    let _ = token;
    inferred_v3(value)
}

struct Kernel(u32);
impl Kernel {
    #[attune(make(_v3, _v3_t, _scalar, _scalar_t, _))]
    fn add<const N: usize>(&self, values: [u32; N]) -> u32 {
        self.0 + values.into_iter().sum::<u32>()
    }

    #[attune(make(_v3_t, _), in_impl)]
    fn associated(value: u32) -> u32 {
        value * 2
    }
}

#[test]
fn generated_dispatch_and_context_composition() {
    assert_eq!(sum((4, [1, 2, 3])), 10);
    assert_eq!(composition(4), 10);
    assert_eq!(Kernel(4).add([1, 2, 3]), 10);
    assert_eq!(Kernel::associated(6), 12);
    assert_eq!(attuned!(sum((4, [1, 2, 3])), [_v3, _scalar]), 10);
    assert_eq!(sum_scalar((4, [1, 2, 3])), 10);
}

#[test]
fn renamed_calls_and_move_only_arguments() {
    use std::cell::Cell;
    let evaluations = Cell::new(0);
    assert_eq!(
        attuned!(
            renamed({
                evaluations.set(evaluations.get() + 1);
                String::from("abc")
            }),
            [_v3, _scalar],
            names(_v3_t = renamed_proof)
        ),
        3
    );
    assert_eq!(evaluations.get(), 1);
    assert_eq!(renamed(String::from("abcd")), 4);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn explicit_boundary_and_supplied_proof() {
    if let Some(token) = <archmage::X64V3Token as archmage::SimdToken>::summon() {
        assert_eq!(boundary(token, 4), 5);
        assert_eq!(sum_v3_t(token, (4, [1, 2, 3])), 10);
        assert_eq!(
            attuned!(sum((4, [1, 2, 3])), [_v3, _scalar], using(token)),
            10
        );
    }
}
