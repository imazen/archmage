+++
title = "2. Stay in Context"
description = "Compose helpers without selecting a CPU tier again"
weight = 2
+++

Give internal helpers the default direct variants with `#[attune(_*)]`. Call
them with `attuned!` inside another attuned function.

*Legacy equivalents:* `#[rite(v3, neon, wasm128, scalar)]` generates direct
helpers; `incant!` fills the call-selection role, using legacy family names.

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[attune(_*)]
fn scale_one(value: f32, factor: f32) -> f32 {
    value * factor
}

#[attune(dispatch)]
pub fn scale(values: &mut [f32], factor: f32) {
    for value in values {
        *value = attuned!(scale_one(*value, factor));
    }
}

let mut values = [2.0; 11];
scale(&mut values, 0.5);
assert_eq!(values, [1.0; 11]);
```

The V3 body calls `scale_one_v3`; the NEON body calls `scale_one_neon`; the
scalar body calls `scale_one_scalar`. These calls stay in a covered feature
context and do not probe the CPU again. The helper bodies carry an inline
hint by default; LLVM still decides whether to inline them.

`_*` generates suffixed functions, so there is no ordinary `scale_one` function
in this example. `attuned!` resolves that family name. A missing applicable
variant is a compile error. Keep the default tier set for both caller and helper
until you need a sparse family or explicit gates.

## Choose the surface you need

| Attribute | Callable surface | Legacy counterpart |
| --- | --- | --- |
| `#[attune(dispatch)]` | Ordinary `work(...)`; per-tier bodies stay private | `#[autoversion(...)]`, with different exposed variants |
| `#[attune(_*)]` | Direct `work_v3(...)`, `work_neon(...)`, and other default variants | Multi-tier `#[rite(...)]` |
| `#[attune(_*, dispatch)]` | Both direct variants and ordinary `work(...)` | Direct `#[rite(...)]` helpers plus a separate dispatcher |
| `#[attune(all)]` | Those forms plus token-taking `work_v3_t(...)` and other proof wrappers | Combined `#[rite]`, `#[arcane]`, and dispatch surfaces; no single legacy attribute |

For an ordinary caller, use the dispatcher. Outside an attuned context,
`attuned!` needs the family's `_t` proof wrappers; direct variants alone are
not sufficient. With `all`, the same family can be used from either context.

## One known tier

`#[attune(v3)]` adds V3 features to a function without changing its name.
Bare `#[attune]` also recognizes names ending in `_v3`. These direct functions
must be called from a matching or stronger feature context.

*Legacy equivalent:* `#[rite(v3)]` keeps the written name; bare `#[attune]`
adds suffix inference.

Passing a token to a direct function does not itself create that caller context.

If rustc reports a target-feature call error, check which function owns the
feature context. Use a dispatcher from ordinary code, or an attuned body that
covers the helper's tier. Do not add an `unsafe` block just to bypass the error.

For deliberate reselection, existing proof tokens, or sparse families, continue
with the [full call reference](@/attunement/reference.md#calls-and-proofs).

*Legacy note for `reattune!`:* there is no separate legacy spelling for this
explicit upgrade; nested `incant!` normally follows the caller's covered tier.

**Next:** [Add vectors](@/attunement/vectors.md).

[Inspect the helper-call expansion](@/attunement/expansions.md#covered_calls)
to see how calls inside each tier retain their context.
