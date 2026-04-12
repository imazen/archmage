// SOUNDNESS EXPLOIT: The `Sealed` trait is publicly re-exported, breaking the seal.
//
// archmage's SimdToken trait claims to be sealed (cannot be implemented
// outside the crate). But `pub use sealed::Sealed;` in tokens/mod.rs
// makes the Sealed trait publicly accessible. An external crate can
// implement Sealed for any type, and then implement SimdToken.
//
// THIS TEST COMPILING IS A BUG. When fixed, this should fail with
// "trait `Sealed` is private" or similar.

use archmage::SimdToken;

// Implement the "sealed" trait for our fake token — should be impossible
// but isn't because Sealed is pub.
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
    // We can now create arbitrary "SimdToken" instances with no CPU checks.
    let token = FakeToken::summon().unwrap();
    assert_eq!(token.name(), "Fake");
}
