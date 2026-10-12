#![forbid(unsafe_code)]
#![deny(warnings)]

use archmage::{attune, attuned};

#[attune(make(all, -_v4))]
pub fn portable(value: u32) -> u32 {
    value + 3
}

#[attune(make(pub(crate) inline(never) _v3, _v3_t, _scalar, _scalar_t, _))]
fn controlled(value: u32) -> u32 {
    value + 5
}

#[attune(make(_*, _*_t, -_v4, +v4x(avx512), _))]
fn optional_wide(value: u32) -> u32 {
    value + 7
}

#[attune(make(_v2, _v2_t, _scalar, _scalar_t, _))]
fn upgrades(value: u32) -> u32 {
    archmage::reattune!(controlled(value), [_v3, _scalar])
}

#[attune(make(_v3_t))]
fn sparse(value: u32) -> u32 {
    archmage::attuned!(controlled(value), [_v3_t])
}

#[attune(make(_v3, _v3_t, _scalar, _scalar_t, _))]
fn positioned(value: u32, proof: Token) -> u32 {
    let _: Token = proof;
    value + 11
}

#[attune(make(_v3_t, _))]
fn positioned_caller(value: u32) -> u32 {
    archmage::attuned!(positioned(value, Token), [_v3, _scalar])
}

#[attune(wrap)]
fn generic<T: archmage::HasX64V2 + Extra>(token: T, value: u32) -> T::Output {
    let _ = token;
    T::convert(value)
}

#[cfg(target_arch = "x86_64")]
trait Extra {
    type Output;
    fn convert(value: u32) -> Self::Output;
}
#[cfg(target_arch = "x86_64")]
impl Extra for archmage::X64V2Token {
    type Output = u32;
    fn convert(value: u32) -> Self::Output {
        value + 1
    }
}
#[cfg(target_arch = "x86_64")]
impl Extra for archmage::X64V3Token {
    type Output = u64;
    fn convert(value: u32) -> Self::Output {
        u64::from(value) + 2
    }
}

#[test]
fn wildcard_and_reselection_contracts() {
    assert_eq!(portable(1), 4);
    assert_eq!(attuned!(portable(1)), 4);
    assert_eq!(controlled(1), 6);
    assert_eq!(optional_wide(1), 8);
    assert_eq!(upgrades(1), 6);
    assert_eq!(portable_scalar(1), 4);
    assert_eq!(positioned(1, archmage::ScalarToken), 12);
    assert_eq!(positioned_caller(1), 12);
    assert_eq!(attuned!(positioned(1, Token), [_v3, _scalar]), 12);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn sparse_proof_and_associated_generic_returns() {
    if let Some(token) = <archmage::X64V2Token as archmage::SimdToken>::summon() {
        let result: u32 = generic(token, 3);
        assert_eq!(result, 4);
    }
    if let Some(token) = <archmage::X64V3Token as archmage::SimdToken>::summon() {
        let result: u64 = generic(token, 3);
        assert_eq!(result, 5);
        assert_eq!(sparse_v3_t(token, 3), 8);
        assert_eq!(positioned_v3_t(1, token), 12);
        assert_eq!(
            attuned!(positioned(1, Token), [_v3, _scalar], using(token)),
            12
        );
    }
}
