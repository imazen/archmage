//! Scalar/default rite fallbacks use covered calls without vector aliases.
#![forbid(unsafe_code)]

// Successful contextual rewriting consumes every use of this macro.
#[allow(unused_imports)]
use archmage::dispatch_variant;
use archmage::{ScalarToken, incant, magetypes};

#[magetypes(rite, v3, neon, wasm128, scalar)]
fn leaf(_token: Token) -> &'static str {
    core::any::type_name::<Token>()
}

#[magetypes(rite, v3, neon, wasm128, scalar)]
fn tokenless_scalar() -> &'static str {
    incant!(leaf(), [v3, neon, wasm128, scalar])
}

#[magetypes(rite, v3, neon, wasm128, default)]
fn tokenless_default() -> &'static str {
    dispatch_variant!(leaf(), [v3, neon, wasm128, scalar])
}

#[magetypes(rite, v3, neon, wasm128, -scalar)]
fn default_leaf(_token: Token) -> i32 {
    7
}

fn default_leaf_default() -> i32 {
    7
}

#[magetypes(rite, default)]
fn default_to_default() -> i32 {
    incant!(default_leaf(), [v3, neon, wasm128, default])
}

#[magetypes(v3, neon, wasm128, scalar)]
fn boundary(_token: Token) -> &'static str {
    core::any::type_name::<Token>()
}

// A tokenful fallback deliberately retains runtime dispatch through safe
// boundaries. Token parameters need not be first or have a particular name.
#[magetypes(rite, scalar)]
fn tokenful_placeholder(x: i32, proof: Token) -> (&'static str, i32) {
    let _ = proof;
    (incant!(boundary(), [v3, neon, wasm128, scalar]), x)
}

#[magetypes(rite, scalar)]
fn tokenful_concrete(_: ScalarToken) -> &'static str {
    incant!(boundary(), [v3, neon, wasm128, scalar])
}

#[test]
fn tokenless_fallbacks_select_only_covered_callees() {
    let scalar = core::any::type_name::<ScalarToken>();
    assert_eq!(tokenless_scalar_scalar(), scalar);
    assert_eq!(tokenless_default_default(), scalar);
    assert_eq!(default_to_default_default(), 7);
}

#[test]
fn tokenful_fallbacks_retain_runtime_boundary_dispatch() {
    let expected = incant!(boundary(), [v3, neon, wasm128, scalar]);
    assert_eq!(tokenful_placeholder_scalar(42, ScalarToken), (expected, 42));
    assert_eq!(tokenful_concrete_scalar(ScalarToken), expected);
}
