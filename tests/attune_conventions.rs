#![forbid(unsafe_code)]
#![deny(warnings)]

#[cfg(target_arch = "x86_64")]
use archmage::SimdToken;
use archmage::{IntoConcreteToken, ScalarToken, attune, attuned};

#[attune(make(dispatch, +v4(avx512), -neon))]
pub fn dispatched<T: Copy>(value: T) -> T {
    value
}

#[attune(make(_, -v3, -v4))]
pub fn removed(value: u32) -> u32 {
    value + 1
}

#[attune]
fn choice_v3_t() -> u32 {
    3
}

#[attune]
fn choice_scalar_t() -> u32 {
    0
}

fn choice_scalar() -> u32 {
    0
}

#[attune]
fn lower_v2_t() -> u32 {
    2
}

#[attune]
fn direct_only_v3() -> u32 {
    3
}

#[attune]
pub fn covered_v3_t() -> u32 {
    attuned!(direct_only(), [_v3])
}

#[attune]
pub fn positioned_v3_t(value: u32, proof: Token) -> u32 {
    let _ = proof;
    value
}

#[attune(scalar)]
fn inherited<T: IntoConcreteToken>(parent: T) -> u32 {
    attuned!(choice(), [_v3, _scalar])
}

#[attune(scalar)]
fn borrowed<T>(parent: &T) -> u32
where
    T: Copy,
    T: IntoConcreteToken,
{
    attuned!(choice(), [_v3, _scalar])
}

#[attune(scalar)]
fn borrowed_mut(parent: &mut impl IntoConcreteToken) -> u32 {
    attuned!(choice(), [_v3, _scalar])
}

#[attune(scalar)]
fn shadowed(parent: impl IntoConcreteToken) -> u32 {
    let parent = 42;
    let __attune_parent_proof = String::from("a local, not proof");
    assert_eq!(parent, 42);
    assert!(!__attune_parent_proof.is_empty());
    let invoke = || attuned!(choice(), [_v3, _scalar]);
    invoke()
}

#[attune(scalar)]
fn overridden(_parent: impl IntoConcreteToken, calls: &std::cell::Cell<u32>) -> u32 {
    attuned!(
        choice(),
        [_v3, _scalar],
        using({
            calls.set(calls.get() + 1);
            ScalarToken
        })
    )
}

#[attune(scalar)]
fn nested_and_reselected(_parent: ScalarToken) -> (u32, u32) {
    fn nested() -> u32 {
        attuned!(choice(), [_v3, _scalar])
    }
    (nested(), archmage::reattune!(choice(), [_v3, _scalar]))
}

#[cfg(target_arch = "x86_64")]
#[attune(scalar)]
fn concrete_downgrade(parent: archmage::X64V3Token) -> u32 {
    attuned!(lower(), [_v2])
}

#[cfg(target_arch = "x86_64")]
#[attune(scalar)]
fn concrete_borrowed(parent: &archmage::X64V3Token) -> u32 {
    attuned!(choice(), [_v3])
}

pub struct Kernel(pub u32);
impl Kernel {
    #[attune]
    pub fn method_v3_t(&self, value: u32) -> u32 {
        self.0 + value
    }

    #[attune(in_impl)]
    pub fn associated_v3_t<T: Copy>(value: T) -> T {
        value
    }
}

#[test]
fn dispatcher_modifiers_and_inherited_scalar_proof_are_portable() {
    assert_eq!(dispatched(String::from("text").len()), 4);
    assert_eq!(removed(3), 4);
    assert_eq!(choice_scalar_t(ScalarToken), 0);
    // On SIMD hosts these distinguish inherited proof from CPU rediscovery.
    assert_eq!(inherited(ScalarToken), 0);
    assert_eq!(borrowed(&ScalarToken), 0);
    assert_eq!(borrowed_mut(&mut ScalarToken), 0);
    assert_eq!(shadowed(ScalarToken), 0);
    let calls = std::cell::Cell::new(0);
    assert_eq!(overridden(ScalarToken, &calls), 0);
    assert_eq!(calls.get(), 1);
    let detected = attuned!(choice(), [_v3, _scalar]);
    assert_eq!(nested_and_reselected(ScalarToken), (detected, detected));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn concrete_parent_and_inferred_boundaries_work_with_real_proof() {
    if let Some(proof) = archmage::X64V3Token::summon() {
        assert_eq!(choice_v3_t(proof), 3);
        assert_eq!(inherited(proof), 3);
        assert_eq!(legacy_generic(proof), 3);
        assert_eq!(borrowed(&proof), 3);
        assert_eq!(shadowed(proof), 3);
        assert_eq!(concrete_borrowed(&proof), 3);
        assert_eq!(concrete_downgrade(proof), 2);
        assert_eq!(covered_v3_t(proof), 3);
        assert_eq!(positioned_v3_t(7, proof), 7);
        assert_eq!(marker_v3_t(9, proof), 9);
        assert_eq!(Kernel(4).method_v3_t(proof, 3), 7);
        assert_eq!(Kernel::associated_v3_t(proof, 19u64), 19);
        let calls = std::cell::Cell::new(0);
        assert_eq!(overridden(proof, &calls), 0);
        assert_eq!(calls.get(), 1);
    }
}

#[attune]
pub fn marker_v3_t(value: u32, proof: Token) -> u32 {
    let _: Token = proof;
    attuned!(positioned(value, Token), [_v3_t])
}

#[archmage::arcane]
pub fn legacy_generic<T: archmage::HasX64V2 + IntoConcreteToken>(proof: T) -> u32 {
    legacy_direct(proof)
}

#[archmage::rite]
fn legacy_direct<T: archmage::HasX64V2 + IntoConcreteToken>(proof: T) -> u32 {
    attuned!(choice(), [_v3, _scalar])
}
