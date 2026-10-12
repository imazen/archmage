#![forbid(unsafe_code)]
#![deny(warnings)]

use archmage::attune;

#[attune(make(_v3_t, _))]
#[track_caller]
fn location() -> u32 {
    std::panic::Location::caller().line()
}

#[attune(make(_v3_t, _))]
#[expect(unused_variables, reason = "the body intentionally ignores this input")]
fn ignored_input(value: u32) -> u32 {
    42
}

#[attune(make(_))]
fn dispatcher_only(value: u32) -> u32 {
    value + 1
}

macro_rules! declare {
    ($name:ident) => {
        #[attune(make(all))]
        fn $name(value: u32) -> u32 {
            value + 2
        }
    };
}
declare!(from_template);

#[test]
fn attributes_follow_the_operation_and_forwarding_chain() {
    let line = line!() + 1;
    assert_eq!(location(), line);
    assert_eq!(ignored_input(9), 42);
    assert_eq!(dispatcher_only(9), 10);
    assert_eq!(from_template(9), 11);
}

#[cfg(target_arch = "x86_64")]
mod traits {
    use super::*;
    struct Counter(u32);
    trait Kernel {
        fn apply<T: archmage::HasX64V2>(&mut self, token: T, value: u32) -> u32;
    }
    impl Kernel for Counter {
        #[attune(wrap, in_trait, _self = Counter)]
        fn apply<T: archmage::HasX64V2>(&mut self, token: T, value: u32) -> u32 {
            let _ = token;
            self.0 += value;
            self.0
        }
    }

    #[test]
    fn nested_wrapper_preserves_generic_proof_and_self() {
        if let Some(token) = <archmage::X64V2Token as archmage::SimdToken>::summon() {
            let mut counter = Counter(3);
            assert_eq!(counter.apply(token, 4), 7);
            assert_eq!(counter.0, 7);
        }
    }
}
