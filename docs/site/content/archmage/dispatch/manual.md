+++
title = "Manual Dispatch"
weight = 1
+++

Prefer `incant!` for routine dispatch. Manual dispatch is useful when a caller
has an additional algorithm or workload decision. It must guard references to
architecture-specific functions, even though token types exist on all targets.

This is a manual-dispatch adaptation of the
[zenfilters gain loop](@/magetypes/examples/generic-kernels.md). The source uses
macro dispatch; the explicit branches here demonstrate the equivalent call-site
requirements, not a second recommended production framework.

```rust
use archmage::prelude::*;

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::load(token, chunk) * factor).store(chunk);
    }
    for value in tail { *value *= gain; }
}

pub fn apply_gain(plane: &mut [f32], gain: f32) {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        return gain_impl_v3(t, plane, gain);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        return gain_impl_neon(t, plane, gain);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        return gain_impl_wasm128(t, plane, gain);
    }
    gain_impl_scalar(ScalarToken, plane, gain);
}
let mut plane = [2.0; 11];
apply_gain(&mut plane, 0.5);
assert_eq!(plane, [1.0; 11]);
```

`stub` has been removed and is rejected by the macro parser. Use `incant!`
or explicit call-site guards; do not resurrect unreachable stubs as fallback code.

Do not add a second detection cache or a cached enum solely to avoid `summon()`:
token detection already caches where appropriate. A caller-owned token can
move detection outside a larger operation; the loop still needs the correct
feature context. Test behavior when tiers are unavailable or disabled.
