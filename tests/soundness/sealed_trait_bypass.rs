// REGRESSION CASE: implementing the sealed supertrait from outside archmage.
//
// `SimdToken` is sealed: its supertrait lives in a private module. If `Sealed`
// were reachable, an external crate could implement it for any type, then
// implement `SimdToken` and fabricate a token. This file must fail to compile;
// tests/soundness_exploits.rs expects E0603 (`Sealed` is private).

use archmage::SimdToken;

// Implement the sealed supertrait for a fake token. `archmage::tokens::Sealed`
// is private, so this must not compile.
struct FakeToken;
impl Clone for FakeToken { fn clone(&self) -> Self { FakeToken } }
impl Copy for FakeToken {}

impl archmage::tokens::Sealed for FakeToken {}
impl archmage::SimdToken for FakeToken {
    const NAME: &'static str = "Fake";
    const TARGET_FEATURES: &'static str = "";
    const ENABLE_TARGET_FEATURES: &'static str = "";
    const DISABLE_TARGET_FEATURES: &'static str = "";
    fn compiled_with() -> Option<bool> { Some(true) }
    fn summon() -> Option<Self> { Some(FakeToken) }
}

fn main() {
    // If it compiled, this would create a "SimdToken" with no CPU check.
    let token = FakeToken::summon().unwrap();
    assert_eq!(token.name(), "Fake");
}
