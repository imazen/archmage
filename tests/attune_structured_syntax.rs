#![forbid(unsafe_code)]
#![deny(warnings)]

use archmage::{ScalarToken, attune};

#[attune(
    _*(pub),
    _*_t(pub),
    dispatch(pub),
    +v4x(cfg(avx512)),
    inline(hint),
)]
fn identity<T: Copy>(value: T) -> T {
    value
}

#[attune(dispatch, +v4x(cfg(avx512)), -v4)]
pub fn private_variants(value: u32) -> u32 {
    value + 1
}

#[attune(_scalar_t(pub(crate), inline(always)), inline(never))]
fn wrapped(value: u32) -> u32 {
    value + 2
}

#[attune(_scalar, names(_scalar = renamed), inline(none))]
fn original(value: u32) -> u32 {
    value + 3
}

#[attune(_scalar_t)]
fn composed(value: u32) -> u32 {
    archmage::attuned!(identity(value), [_scalar])
}

pub struct Kernel(pub u32);

impl Kernel {
    #[attune(_scalar_t(pub), dispatch(pub))]
    fn add(&self, value: u32) -> u32 {
        self.0 + value
    }

    #[attune(in_impl, _scalar_t(pub))]
    fn associated<T: Copy>(value: T) -> T {
        value
    }
}

#[test]
fn structured_outputs_compile_and_run_without_unsafe() {
    assert_eq!(identity(7u32), 7);
    assert_eq!(identity_scalar(8u32), 8);
    assert_eq!(identity_scalar_t(ScalarToken, 9u32), 9);
    assert_eq!(private_variants(9), 10);
    assert_eq!(wrapped_scalar_t(ScalarToken, 9), 11);
    assert_eq!(renamed(9), 12);
    assert_eq!(composed_scalar_t(ScalarToken, 13), 13);
    assert_eq!(Kernel(10).add(2), 12);
    assert_eq!(Kernel(10).add_scalar_t(ScalarToken, 3), 13);
    assert_eq!(Kernel::associated_scalar_t(ScalarToken, 14), 14);
}

#[cfg(target_arch = "x86_64")]
#[attune(v3)]
fn arbitrary_name(value: u32) -> u32 {
    identity_v3(value)
}

#[cfg(target_arch = "x86_64")]
#[attune]
pub fn check_v3_t(value: u32) -> u32 {
    arbitrary_name(value)
}

#[test]
#[cfg(target_arch = "x86_64")]
fn context_spelling_keeps_the_written_name() {
    use archmage::SimdToken;
    if let Some(token) = archmage::X64V3Token::summon() {
        assert_eq!(check_v3_t(token, 15), 15);
        assert_eq!(identity_v3_t(token, 16), 16);
    }
}
