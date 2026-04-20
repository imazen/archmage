# artisan-macros design

Auditable, convention-forward proc-macros for sound SIMD tier dispatch.

The entire crate is intended to be readable top-to-bottom in a single sitting. Nothing is hidden behind a TOML registry, an xtask codegen step, or a body-walking rewriter. Users own the feature strings and suffixes; the macros only wire up the attributes and the dispatch chain.

## Goals

1. **Sound by construction** — every `unsafe { }` emitted is co-located with the CPUID check that justifies it, inside the macro's own definition span.
2. **`#![forbid(unsafe_code)]`-compatible downstream** — `unsafe` tokens in the expansion trace to this crate, not the user's.
3. **Auditable** — the proc-macro code fits in a single file, with expansion shapes documented inline. No cross-file indirection.
4. **Convention-forward** — explicit over clever. No auto-inference of features from suffixes, no hidden tier registry, no body cloning with placeholder tokens. What you write is what runs.
5. **Zero runtime dependency** — expansions reference only `core::sync::atomic` and the stdlib feature-detection macros.

## Non-goals

- Token-as-proof type system (archmage owns that layer)
- Body-level rewriting or placeholder substitution (archmage's `incant!` / `#[magetypes]`)
- Scalar auto-vectorization cloning (archmage's `#[autoversion]`)
- A built-in tier registry with LLVM microarchitecture levels baked in

If you need any of the above, use archmage. This crate is for users who want the smallest possible dispatch surface they can read, own, and fork.

## The trampoline-chain model

Per-call-site dispatch trees are replaced by a *chain* of tier trampolines, where each trampoline's failure case is "recurse into the next-lower tier." The compile-time entry function is just an arch switch that calls the highest trampoline for the current architecture.

```text
compute(data)
 └─ #[cfg(target_arch = "x86_64")] compute__chain_v3(data)
     ├─ V3_CACHE hit-yes  → unsafe { compute_v3(data) }
     ├─ V3_CACHE hit-no   → compute__chain_v2(data)
     │                       ├─ V2_CACHE hit-yes → unsafe { compute_v2(data) }
     │                       ├─ V2_CACHE hit-no  → compute_scalar(data)
     │                       └─ V2_CACHE empty   → detect+store, then yes/no arm
     └─ V3_CACHE empty    → detect+store, then yes/no arm
```

Every chain bottom-out is the user-supplied `default` / scalar function. There is no separate dispatcher fn at the call site; the chain IS the dispatcher.

### Why this shape

- **Call sites are a single function call.** Inlining and devirtualization happen naturally.
- **Compile-time elision is trivial.** The entry fn body is `#[cfg(target_feature = "avx2,fma,...")] { return unsafe { compute_v3(data) }; }` before the runtime chain — when the ambient target covers a tier, the runtime chain is dead code.
- **Each trampoline reads as one unit.** Cache check, one detect-and-store arm, two terminal arms. ~12 lines expanded per tier, same shape every time.
- **Adding a tier is local.** You insert a new trampoline in the chain; no other dispatch site changes.
- **Removing a tier is local.** Delete the trampoline, wire its predecessor to the next one down. Nothing else moves.

## Tri-state cache, one per feature set

Each tier in the chain gets an `AtomicU8` cache. Values:

| Value | Meaning |
|---|---|
| `0` | Empty — feature detection has not run yet |
| `1` | False — CPU does NOT have this feature set |
| `2` | True — CPU has this feature set |

The cache is **per feature set**, not per tier name. If two tiers in the same crate happen to have identical feature strings (rare but legal), they share a cache via a deterministic name mangle. This is the only codegen cleverness in the crate and it's explicit.

### Soundness of the cache

- **Constant-CPUID axiom** — on tier-1 Rust targets, CPU features do not change during process lifetime. The cache memoizes a pure function. Relaxed atomic ordering is correct because the value is idempotent — racing stores write the same bits.
- **No safe API to fabricate "true."** The public test-hooks API can only write `1` (force-unavailable), never `2` (force-available). Fabricating availability would call an intrinsic on a CPU that can't execute it — SIGILL. Disabling a tier is always sound.
- **No visible initialization ordering.** The cache is inside the macro expansion, private to the function. Callers cannot observe the transition from `0` to `1`/`2`.

## `#[cpu_tier]` — thin target_feature wrapper

```rust
#[cpu_tier(enable = "avx2,fma,bmi1,bmi2,f16c,lzcnt,popcnt,movbe")]
fn compute_v3(data: &[f32]) -> f32 {
    // hand-written AVX2+FMA body
}
```

Expands to:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,bmi1,bmi2,f16c,lzcnt,popcnt,movbe")]
#[inline]
fn compute_v3(data: &[f32]) -> f32 {
    // body unchanged
}
```

That's it. No token parameter, no wrapper, no body inspection. The function is the user's; the macro only attaches attributes. The `#[target_feature]` function is `fn` (not `unsafe fn`) per the Rust 2024 edition rule — `#![forbid(unsafe_code)]`-compatible.

**Arch inference is the default.** The macro scans the feature list for any unambiguous name (e.g., `avx2`, `avx512f`, `bmi2` → x86_64; `neon`, `dotprod`, `i8mm`, `sve` → aarch64; `simd128` → wasm32) and picks the matching `#[cfg(target_arch = "...")]`. Features that appear on multiple architectures (`aes`, `sha2`, `crc`) are ignored for inference — if every listed feature is ambiguous, inference fails with a clear compile error pointing at the user-supplied `arch = "..."` override. Inference failure is a compile error, not a silent default; ambiguity never ships a misinferred arch.

The inference table is a `const` in the macro source, kept small by design. Users who want to disable inference entirely for a specific tier pass `arch = "x86_64"` explicitly.

**No tier name baked in.** The `v3` is just the suffix on the function name, which is meaningful only to the `chain!` macro's reader. `compute_avx2_bmi2` would work equally well. Suffix conventions are the user's prerogative.

## `#[chain]` — declare a trampoline chain

`chain` is an attribute macro applied to a function with an empty body. The attribute supplies the per-arch tier lists with their feature strings inline; the macro discards the empty body and fills in the entry dispatcher.

```rust
#[chain(
    x86_64 = [
        compute_v3 = "avx2,fma,bmi1,bmi2,f16c,lzcnt,popcnt,movbe",
        compute_v2 = "sse4.2,popcnt",
    ],
    aarch64 = [
        compute_neon = "neon",
    ],
    wasm32 = [
        compute_wasm128 = "simd128",
    ],
    default = compute_scalar,
)]
/// Public entry. Calls the highest-tier trampoline for the host arch.
pub fn compute(data: &[f32]) -> f32 {}
```

The macro:

1. Parses the attribute: each `arch = [ fn_name = "features", ... ]` list plus a single `default = fn_name` fallback.
2. Parses the function signature (args, return type, visibility, doc attrs).
3. Discards the empty `{}` body.
4. Emits one trampoline per tier per arch, chained to its successor.
5. Emits the entry function body with compile-time arch dispatch and compile-time feature-elision fast paths.
6. Emits one `AtomicU8` cache per tier.

Tier order in each arch list is **highest to lowest** — the first entry is tried first; cache-miss falls through to the next.

### Feature-set resolution: inline at the chain site

The feature string lives in two places: `#[cpu_tier(enable = "...")]` on the tier function, and `= "..."` inside the `#[chain]` list. Both must match. This is explicit and auditable — every feature string is visible at both the definition site (where the `#[target_feature]` gets attached) and the dispatch site (where the runtime detection happens).

**Mismatch is a user bug, not a macro bug.** If `#[cpu_tier(enable = "avx2,fma")]` declares a narrower set than `#[chain(... compute_v3 = "avx2,fma,bmi2", ...)]`, the runtime check passes on a CPU that has AVX2+FMA but lacks BMI2, and the function runs — but only with AVX2+FMA enabled in `#[target_feature]`, so whatever BMI2 codegen LLVM would have used is absent. The converse (cpu_tier wider than chain) causes the reverse: LLVM may emit BMI2 instructions the detection check didn't verify, potentially SIGILL on CPUs that have AVX2+FMA but not BMI2.

A future macro revision may emit a `const` from `#[cpu_tier]` and compile-time-assert equality from `#[chain]`, turning mismatches into compile errors. For now: two sites, one string, user copy-pastes.

## Expansion shape

For the call above, `#[chain]` emits approximately:

```rust
pub fn compute(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // Compile-time elision: ambient target covers v3? jump straight in.
        #[cfg(all(
            target_feature = "avx2", target_feature = "fma", target_feature = "bmi1",
            target_feature = "bmi2", target_feature = "f16c", target_feature = "lzcnt",
            target_feature = "popcnt", target_feature = "movbe",
        ))]
        { return unsafe { compute_v3(data) }; }

        #[cfg(not(all( /* v3 features */ )))]
        { return compute__chain_v3(data); }
    }
    #[cfg(target_arch = "aarch64")]
    { return compute__chain_neon(data); }
    #[cfg(target_arch = "wasm32")]
    { return compute__chain_wasm128(data); }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
    { compute_scalar(data) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn compute__chain_v3(data: &[f32]) -> f32 {
    use core::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    match CACHE.load(Ordering::Relaxed) {
        2 => unsafe { compute_v3(data) },
        1 => compute__chain_v2(data),
        _ => {
            let ok = std::is_x86_feature_detected!("avx2")
                  && std::is_x86_feature_detected!("fma")
                  && /* ... same features, splatted from the same const ... */;
            CACHE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
            if ok { unsafe { compute_v3(data) } } else { compute__chain_v2(data) }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn compute__chain_v2(data: &[f32]) -> f32 { /* same shape, falls through to compute_scalar */ }

// aarch64, wasm32 chains: analogous
```

Each trampoline: ~12 lines. Each `unsafe { }` block has the CPUID check 3 lines above it. Auditable by eye.

## `#![forbid(unsafe_code)]` compatibility

The `unsafe { }` blocks in the expansion originate from this crate's proc-macro definition spans. Downstream crates with `#![forbid(unsafe_code)]` accept them — the same exemption that archmage relies on for `#[arcane]`.

The `#[target_feature]` function generated by `#[cpu_tier]` is declared `fn`, never `unsafe fn`. Rust 2024 allows this, and `#![forbid(unsafe_code)]` rejects `unsafe fn` declarations even from proc-macros.

## Test hooks (feature-gated, thread-local)

Behind `#[cfg(feature = "test-hooks")]`, each chain exposes a per-thread override layer on top of the (still-correct) atomic cache. The atomic cache continues to hold CPU truth; the thread-local is an orthogonal downgrade override visible only on the setting thread.

```rust
// Generated alongside each #[chain]-annotated fn `compute`:

#[cfg(feature = "test-hooks")]
pub enum ComputeTier { V3, V2, Neon, Default }  // variants from the user's fn suffixes

#[cfg(feature = "test-hooks")]
thread_local! {
    static COMPUTE_FORCED_MAX_TIER: core::cell::Cell<Option<ComputeTier>>
        = const { core::cell::Cell::new(None) };
}

#[cfg(feature = "test-hooks")]
#[must_use = "dropping the scope immediately restores the previous override"]
pub struct ComputeTierScope { prev: Option<ComputeTier> }

#[cfg(feature = "test-hooks")]
pub fn compute_force_max_tier(tier: ComputeTier) -> ComputeTierScope {
    COMPUTE_FORCED_MAX_TIER.with(|c| {
        let prev = c.replace(Some(tier));
        ComputeTierScope { prev }
    })
}

#[cfg(feature = "test-hooks")]
impl Drop for ComputeTierScope {
    fn drop(&mut self) {
        COMPUTE_FORCED_MAX_TIER.with(|c| c.set(self.prev));
    }
}
```

Each trampoline consults the thread-local BEFORE its atomic cache:

```rust
fn compute__chain_v3(data: &[f32]) -> f32 {
    #[cfg(feature = "test-hooks")]
    if let Some(max) = COMPUTE_FORCED_MAX_TIER.with(|c| c.get()) {
        if (max as u8) < (ComputeTier::V3 as u8) {
            return compute__chain_v2(data);
        }
    }
    // ...normal atomic-cache check...
}
```

### Properties

- **Per-thread isolation = per-test isolation.** Cargo runs one test per thread; forcing on thread A cannot affect thread B. No `#[serial]`, no `RUST_TEST_THREADS=1`, no race conditions between parity tests in the same crate.
- **Zero cost in release.** The `#[cfg(feature = "test-hooks")]` block compiles away entirely. No branch on the hot path, no thread-local lookup, no `Option` check.
- **RAII restore.** `ComputeTierScope` implements `Drop`. A panic during a test still unwinds the guard and restores the previous override. Tests cannot leak forced state.
- **Nested overrides compose.** Calling `force_max_tier` inside a test that already has one set saves the outer value in `prev` and restores it on drop. Nested scopes nest correctly.
- **Atomic cache stays correct.** The thread-local never writes to the cache. Production detection is unaffected; test hooks only overlay.

### Rayon / worker-pool caveat

`thread_local!` values do not propagate to rayon worker threads or any spawned thread. A test that calls `force_max_tier` on the main thread and then dispatches from a rayon `par_iter` sees the real-CPU behavior inside the worker, not the forced tier. This is a genuine limitation.

Mitigations:

- Use `serial_test::serial` on tests that dispatch across thread pools, and optionally expose a process-wide "force-max-tier" knob (separate API, clearly labelled). Not exposed by default — process-wide forcing is the thing we are deliberately avoiding.
- Propagate the override manually by calling `force_max_tier` inside the rayon worker closure. Rayon's `scope` API makes this tolerable.

In practice, parity harness tests are single-threaded (they call a function with an input and compare outputs). The rayon case is rare enough to doc-note rather than design around.

### Comparison with archmage

Archmage's `dangerously_disable_token_process_wide()` writes directly into the process-wide atomic cache, racing under parallel tests in the same way a naive artisan implementation would. Adding thread-local overrides to archmage is tracked as a parallel enhancement; the parity-harness test generation in archmage depends on it.

## What's NOT in this crate

- No tier presets. No `v1`, `v2`, `v3`, `v4` constants. Users write `enable = "avx2,fma,..."` because spelling it out is the whole point.
- No token types. Users who want token-as-proof pass archmage tokens into their `#[cpu_tier]` functions themselves.
- No body rewriting. `$body:block` captures verbatim; the macro only attaches attributes.
- No autoversion. Scalar-body cloning is a separate concern; this crate does not have opinions about how you produce the per-tier bodies.
- No parity test harness. That lives in archmage or as a sibling crate — it needs the tier enum `chain!` generates, so it's a natural follow-up.

## Sizing target

- `src/lib.rs` ≤ 500 lines, proc-macro + parsing + codegen.
- No other source files initially. Every concern fits in one file; users fork by copying `lib.rs` + `Cargo.toml`.
- Zero-dep beyond `syn`/`quote`/`proc-macro2`.

## Decisions (resolved 2026-04-20)

- **Feature-set resolution:** Option A — inline at the chain site. Every feature string visible at both `#[cpu_tier]` and `#[chain]`. Mismatch is a user bug with a documented failure mode; future revision may add const-based equality check.
- **`chain` form:** attribute macro on an empty-body function (`#[chain(...)] pub fn compute(...) -> T {}`). Reads the signature from the function; attribute args carry the tier lists.
- **Default keyword:** `default` (not `scalar`). Honest on no-SIMD architectures where there is no "scalar vs vector" split.
- **Arch inference:** enabled by default. Inference failure (all features ambiguous) is a compile error, not a silent default. Users override with explicit `arch = "..."` in `#[cpu_tier]`.

## Remaining open questions

- **Name mangle for chain trampolines:** `compute__chain_v3` is the current sketch. Pick the ugliest double-underscore name the user can't accidentally collide with.
- **Test-hook tier enum shape:** per-chain `ArtisanTier` enum generated as a public type alongside `force_max_tier`. Naming convention for variants (`V3`, `V2`, etc.) vs the user's actual fn suffixes? Probably the fn suffix uppercased — the user chose the name, respect it.
