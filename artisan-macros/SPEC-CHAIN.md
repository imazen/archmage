# SPEC: `#[chain]`

Normative specification for the `#[chain]` attribute macro in artisan-macros.

Status: **draft** — implementation exists in `src/lib.rs`, awaiting review.

## Synopsis

```rust
#[chain(
    <arch> = [ <tier_fn> = "<features>", ... ],
    ...,
    default = <fallback_fn>,
)]
<vis> fn <name>(<params>) <return> {}
```

Applied to a function with an empty body. The macro reads the function signature, discards the empty body, and emits:

1. An entry function with the same signature, whose body is a compile-time arch switch with per-arch compile-time feature elision.
2. One trampoline per tier, each with its own `AtomicU8` cache and a thread-local override check.
3. A test-hook surface (per-chain enum, `force_max_tier` function, RAII scope guard, thread-local cell) behind `#[cfg(any(test, feature = "artisan_test_hooks"))]`.

## Input grammar

Formal grammar in EBNF:

```
chain_args   := entry ( "," entry )* ","?
entry        := arch_entry | default_entry
arch_entry   := arch_ident "=" "[" tier_entry ( "," tier_entry )* ","? "]"
default_entry := "default" "=" ident
tier_entry   := ident "=" lit_str
arch_ident   := "x86_64" | "x86" | "aarch64" | "arm" | "wasm32"
```

- Tier order within an arch list is **highest-to-lowest**: the first tier is tried first; cache-miss falls through to the next.
- Exactly one `default` entry is required.
- Arch idents must appear at most once per chain.
- Tier function idents must be unique within an arch (they may repeat across arches, though this is unusual).
- The annotated function body must be an empty block `{}`. A non-empty body is a compile error.
- The annotated function must be a safe, non-async, non-generic `fn`. Methods with `self` receivers are rejected.

## Expansion

For a chain declared as

```rust
#[chain(
    x86_64 = [ F_a = "feats_a", F_b = "feats_b" ],
    aarch64 = [ F_n = "feats_n" ],
    default = F_d,
)]
pub fn NAME(p1: T1, p2: T2) -> R {}
```

the macro emits approximately:

```rust
// Entry function — compile-time arch switch.
pub fn NAME(p1: T1, p2: T2) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(all(target_feature = "feats_a_1", target_feature = "feats_a_2", ...))]
        { unsafe { F_a(p1, p2) } }
        #[cfg(not(all(target_feature = "feats_a_1", ...)))]
        { __artisan_NAME__x86_64__F_a__chain(p1, p2) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "feats_n")]
        { unsafe { F_n(p1, p2) } }
        #[cfg(not(target_feature = "feats_n"))]
        { __artisan_NAME__aarch64__F_n__chain(p1, p2) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { F_d(p1, p2) }
}

// Per-tier trampolines.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(non_snake_case)]
fn __artisan_NAME__x86_64__F_a__chain(p1: T1, p2: T2) -> R {
    // Thread-local override (test-hooks only).
    #[cfg(any(test, feature = "artisan_test_hooks"))]
    { /* see SPEC-TEST-HOOKS.md for the exact shape */ }

    use ::core::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    match CACHE.load(Ordering::Relaxed) {
        2u8 => unsafe { F_a(p1, p2) },
        1u8 => __artisan_NAME__x86_64__F_b__chain(p1, p2),
        _ => {
            let ok = ::std::is_x86_feature_detected!("feats_a_1")
                  && ::std::is_x86_feature_detected!("feats_a_2")
                  && /* ... */;
            CACHE.store(if ok { 2u8 } else { 1u8 }, Ordering::Relaxed);
            if ok { unsafe { F_a(p1, p2) } } else { __artisan_NAME__x86_64__F_b__chain(p1, p2) }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(non_snake_case)]
fn __artisan_NAME__x86_64__F_b__chain(p1: T1, p2: T2) -> R {
    // ...same shape; `F_b` on cache-hit, `F_d(...)` on miss (default fallback)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(non_snake_case)]
fn __artisan_NAME__aarch64__F_n__chain(p1: T1, p2: T2) -> R {
    // ...same shape; `F_n` on cache-hit, `F_d(...)` on miss
}

// Test-hook declarations (see SPEC-TEST-HOOKS.md).
```

### Mangling convention

- Trampoline name: `__artisan_{chain_fn}__{arch}__{tier_fn}__chain`.
- Thread-local module: `__artisan_{chain_fn}_hooks`.
- Tier enum: `{CapitalizedChainFn}Tier`.
- Scope guard: `{CapitalizedChainFn}Scope`.
- Force function: `{chain_fn}_force_max_tier`.

All emitted with `#[allow(non_snake_case)]` where needed.

### Cache semantics

- One `AtomicU8` cache per tier, named `CACHE` and scoped to the trampoline function (private to the expansion; two trampolines never share a cache even if they carry the same feature set).
- Values: `0` = empty / unchecked; `1` = false / CPU lacks features; `2` = true / CPU has features.
- Accessed with `Ordering::Relaxed`. The cached value is idempotent (CPUID is a pure function on tier-1 targets), so racing threads writing the same result is benign.
- Initialised to `0`. First reader performs the check, stores the result, and proceeds.

### Per-arch detection macro

| Arch | Runtime detection macro |
|---|---|
| `x86_64`, `x86` | `::std::is_x86_feature_detected!` |
| `aarch64` | `::std::is_aarch64_feature_detected!` |
| `arm` | `::std::is_arm_feature_detected!` |
| `wasm32` | **not supported in runtime dispatch** — see below |

### wasm32 limitation

wasm32 has no runtime feature detection. A `#[chain]` entry with `wasm32 = [ ... ]` fails compilation with a clear error. On wasm32, put SIMD tier functions on the compile-time path only: annotate with `#[cpu_tier(enable = "simd128")]` and check `#[cfg(target_feature = "simd128")]` in the default path to pick between the SIMD tier and a scalar fallback manually. A future revision may add `chain` support for wasm32 via compile-time-only dispatch.

## Feature-string duplication

The feature string appears in **two places**:

1. `#[cpu_tier(enable = "...")]` on the tier function definition. Drives `#[target_feature]`.
2. `#[chain(<arch> = [<tier> = "...", ...])]` at the chain site. Drives `is_*_feature_detected!` and `#[cfg(target_feature = "...")]`.

**The two strings must match exactly.** The current draft does not enforce equality at macro expansion time (it cannot — a proc-macro cannot read attributes from a sibling function). Mismatches are user bugs. Failure modes:

- `cpu_tier` features ⊂ `chain` features: the runtime CPUID check verifies more than `#[target_feature]` enabled. LLVM never emits instructions for features the function wasn't compiled with, so this is safe but wastes detection bandwidth.
- `cpu_tier` features ⊃ `chain` features: the runtime check verifies less than the function requires. On a CPU that passes the check but lacks the extra features, LLVM-emitted instructions for the extra features trap with SIGILL. **This is unsound and is a user bug.**

A future revision may add a compile-time equality check by emitting a `const FEATURES_chain_NAME_TIER: &str = "..."` from `#[cpu_tier]` and a `const _: () = assert!(FEATURES_chain_NAME_TIER == "...")` from `#[chain]`. Out of scope for the current draft.

## Compile-time elision

The entry function's body fast-paths via `#[cfg(all(target_feature = "..."))]`. When the build target fully covers a tier's features (e.g. `-Ctarget-cpu=haswell` or `-Ctarget-feature=+avx2,+fma,...`), the top-tier elision block fires at compile time and the entry function becomes a single direct call. The `AtomicU8` cache, CPUID check, and trampoline chain are dead code eliminated.

**Current draft elides only the top tier per arch.** A fuller implementation elides every tier (with mutually exclusive `cfg(not(all(...)))` clauses to maintain single-match semantics). The top-tier elision covers the common case (build for native + top tier exactly matches); lower-tier elision is an optimisation for mid-range build targets. Not required for correctness.

## Validation rules

| Condition | Error |
|---|---|
| No `default` entry | `missing required \`default = <fn_name>\` entry` |
| Multiple `default` entries | `multiple \`default = ...\` entries` |
| Arch listed twice | `arch \`<arch>\` listed more than once in #[chain]` |
| Tier function repeated in one arch | `tier function \`<fn>\` listed more than once under arch \`<arch>\`` |
| Empty tier list for an arch | `arch \`<arch>\` has an empty tier list; remove it or add tiers` |
| Empty feature list for a tier | `empty feature list for tier \`<fn>\`` |
| `unsafe fn` | `#[chain] expects a safe \`fn\`, not \`unsafe fn\`` |
| `async fn` | `#[chain] does not support \`async fn\` in the current draft` |
| Generic fn | `#[chain] does not support generic functions in the current draft` |
| `self` receiver | `#[chain] cannot be applied to methods with \`self\` receivers; apply to a free function` |
| Destructuring param pattern | `#[chain] expects simple \`name: Type\` parameters (no destructuring)` |
| `ref` / `mut` binding / subpattern | `#[chain] expects simple \`name: Type\` parameters (no \`ref\`, no \`mut\` binding, no subpatterns)` |
| wasm32 in chain | `wasm32 has no runtime feature detection; SIMD tiers on wasm must rely on compile-time features only. ...` |
| Unsupported arch | `unsupported arch \`<arch>\` in #[chain]: expected x86_64, x86, aarch64, or arm` |

## Soundness claims

1. **Every `unsafe` block in the expansion is proc-macro-generated**, inside spans traceable to artisan-macros. Downstream `#![forbid(unsafe_code)]` compiles generated code without modification.
2. **Every `unsafe` block is preceded by a runtime CPUID check or a compile-time `cfg(target_feature)` check** verifying the features required by the called function's `#[target_feature]`.
3. **Cache-hit `2u8` implies the check previously succeeded** — no `unsafe` call is reached without feature verification.
4. **Cache-miss on first call performs the check before dispatch** — the first dispatch on any cache is never unsafe-without-verification.
5. **Default fallback is always reachable** — every chain bottoms out at the user's `default` fn, which has no feature requirements.
6. **Constant-CPUID axiom** (from DESIGN.md): tier-1 Rust targets do not change CPU features during process lifetime. The cache is a memoisation of a pure function.

## Open questions (for review)

- **Lower-tier compile-time elision.** Currently only top tier per arch elides; lower tiers always take the runtime path. Add full elision with mutually exclusive `cfg(not(all(...)))` clauses?
- **Feature-string equality check.** Implement the `const FEATURES_*` + `assert!` mechanism to catch cpu_tier ↔ chain mismatches at compile time?
- **Generic function support.** The current draft rejects generics for simplicity. Add support by emitting generic trampolines and routing `arg_idents` through? Trampoline monomorphisation cost may matter.
- **`async fn` support.** Probably straightforward — trampoline returns the future. Design question: should the trampoline be `async`, or should it return an `impl Future` that does the dispatch on first poll?
- **Trampoline mangling collision.** Two `#[chain]` declarations with the same `chain_fn` name in the same module (e.g. via `use ... as ...`) would collide. The double-underscore prefix plus the arch and tier_fn segments make this unlikely in practice; is it worth hashing?
