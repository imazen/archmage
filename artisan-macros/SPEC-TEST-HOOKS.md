# SPEC: test hooks

Normative specification for the thread-local override / `force_max_tier` API emitted by `#[chain]` in artisan-macros.

Status: **draft** — implementation exists in `src/lib.rs`, awaiting review.

## Synopsis

Behind `#[cfg(any(test, feature = "artisan_test_hooks"))]`, every `#[chain]` expansion emits a per-chain API that lets tests downgrade the dispatcher's tier selection on the current thread only, without touching the process-wide atomic cache.

```rust
#[chain( /* ... */ )]
pub fn compute(data: &[f32]) -> f32 {}
```

Emits (test-hook surface only, full expansion in SPEC-CHAIN.md):

```rust
#[cfg(any(test, feature = "artisan_test_hooks"))]
pub enum ComputeTier { /* one variant per tier, in declared order */, Default }

#[cfg(any(test, feature = "artisan_test_hooks"))]
#[must_use = "dropping the scope immediately restores the previous override"]
pub struct ComputeScope { /* opaque */ }

#[cfg(any(test, feature = "artisan_test_hooks"))]
pub fn compute_force_max_tier(tier: ComputeTier) -> ComputeScope;
```

## Semantics

Calling `compute_force_max_tier(t)`:

1. Reads the current thread-local override (`Option<u8>` tier index) into a local.
2. Writes the chosen tier's index (or `Default` → last+1) into the thread-local.
3. Returns a `ComputeScope` containing the previous value.

On `ComputeScope::drop`:

1. Writes the stored previous value back into the thread-local.

Each trampoline in the chain reads the thread-local before its atomic cache. If the override is `Some(max_idx)` and the trampoline's own tier index is strictly less than `max_idx` (i.e., the trampoline is a HIGHER tier than allowed), the trampoline returns the next-tier dispatch directly, skipping both the atomic cache check and the feature detection.

**Tier indexing convention:** the first tier declared in `#[chain]` is index `0` (highest). Subsequent tiers increase by `1`. `Default` is `<tier_count>`. Forcing to tier index `k` means "do not run any tier with index `< k`."

### Example

```rust
#[chain(
    x86_64 = [ fast_v4 = "...", fast_v3 = "..." ],
    aarch64 = [ fast_neon = "..." ],
    default = fast_scalar,
)]
pub fn fast(data: &[f32]) -> f32 {}

// Indices: fast_v4 = 0, fast_v3 = 1, fast_neon = 2, Default = 3
// (Neon is on aarch64 so index 2 only matters on aarch64; on x86_64 its
//  trampoline is #[cfg]-gated out and never compiles.)
```

On x86_64 with AVX-512 ambient:

- `force_max_tier(FastTier::V4)` (idx 0): v4 runs (0 < 0 false, proceed).
- `force_max_tier(FastTier::V3)` (idx 1): v4 skips (0 < 1 → skip), v3 runs.
- `force_max_tier(FastTier::Default)` (idx 3): v4 skips, v3 skips, scalar runs.

## Invariants

1. **Per-thread isolation.** `force_max_tier` on thread A has no effect on thread B. Cargo's test runner assigns one thread per test by default; parallel parity tests do not interfere.
2. **Zero cost in release.** Without `cfg(test)` or `cfg(feature = "artisan_test_hooks")`, every trampoline is compiled without the override block, without the thread-local read, and without the override branch. The type surface (enum, scope, force fn) is not emitted at all.
3. **RAII restoration.** `Drop` restores the previous override. Panic during a test unwinds the scope and restores state. Tests cannot leak forced state across test functions.
4. **Nested overrides compose.** `force_max_tier(V3)` inside a test that already has `force_max_tier(V4)` set stores `V4` as `prev`, overrides to `V3`. On drop, restores `V4`. On outer drop, restores `None`.
5. **Atomic cache is untouched.** The thread-local never writes to the atomic cache. Production detection state remains correct across test runs.
6. **Downgrade-only.** The API exposes `Option<u8>` with tier indices in declared order plus `Default`. It cannot set the override to a variant that does not exist. It cannot fabricate availability of a tier the CPU lacks — all it can do is restrict selection to lower-or-equal tiers.

## Soundness

The test-hook API can only disable tiers. The `force_max_tier` function takes a value of the generated `Tier` enum, whose variants are exactly the tiers declared in `#[chain]` plus `Default`. It cannot be passed a tier the declaration does not include.

Disabling a tier:

- Does NOT cause a `#[target_feature]` function to be called on a CPU that lacks those features. The trampoline skips to the next-tier trampoline, whose own cache check verifies the next tier's features before any `unsafe` call.
- Cannot cause UB by itself — the thread-local controls a choice between tier trampolines, all of which perform their own verification.
- Cannot corrupt other threads' dispatch decisions because the thread-local is per-thread.

Fabricating a tier (writing `max_idx` higher than any declared tier) is not exposed by the API. The only way to get `Default` is via the `Default` variant, which is `<tier_count>` — equal to or higher than every tier index, meaning "skip everything."

## Rayon / worker-pool caveat

`std::thread_local!` values do not propagate to spawned threads. A test that calls `force_max_tier` on the main thread and then dispatches from a rayon `par_iter` worker sees the main thread's override only in code running directly on the main thread. Worker threads have their own thread-local (initialised to `None`) and dispatch via the real-CPU atomic cache.

Mitigations:

- Propagate manually: inside the rayon worker closure, call `force_max_tier` again. The closure's `ComputeScope` restores the override on worker drop.
- Use `serial_test::serial` and a process-wide forcing mechanism (not exposed by default — process-wide forcing is deliberately avoided to keep parallel tests clean).
- Run rayon-using parity tests on `cargo test -- --test-threads=1` or as `#[serial]`-annotated tests.

This is a genuine limitation. In practice, parity harness tests are single-threaded (they call a function with an input and compare outputs) and do not hit this case. The caveat is worth a documentation note, not a design rework.

## Feature gating

The thread-local, enum, scope, and force function are emitted behind `#[cfg(any(test, feature = "artisan_test_hooks"))]`.

- Unit tests inside the crate declaring `#[chain]` get the hooks automatically via `cfg(test)`.
- Downstream integration-test crates (which compile the library as a dependency, not as a test target) need to enable the `artisan_test_hooks` feature on the crate that owns the `#[chain]` declaration. The convention is:

```toml
# In the crate that owns #[chain]:
[features]
artisan_test_hooks = ["artisan-macros/artisan_test_hooks"]
# or, if the crate does not gate its own code on the feature:
artisan_test_hooks = []
```

The trampolines themselves check `cfg(feature = "artisan_test_hooks")` (propagated through the macro expansion, not through `artisan-macros`'s own Cargo features). The declaring crate must have a feature of this name; tests enable it via `cargo test --features artisan_test_hooks`.

## Fit with the archmage parity-harness proposal

The parity harness proposed for archmage (`PARITY_HARNESS_ISSUE_DRAFT.md`) needs per-thread downgrade scoping as a prerequisite. Archmage's current `dangerously_disable_token_process_wide()` writes to the process-wide atomic cache, racing under parallel tests. The artisan-macros design is the correct shape for archmage too. See `ARCHMAGE-THREAD-LOCAL-ISSUE.md` for the proposed archmage refactor.

## Open questions (for review)

- **`Option<Tier>` in the cell vs `Option<u8>` with index discipline.** Current: `Option<u8>`. Simpler, but couples trampoline code to the declaration order. `Option<Tier>` with the enum stored directly is more self-documenting but adds coupling between the trampoline mod and the enum module.
- **Expose an "unforce" API.** Drop of the scope restores; if a test wants to explicitly unforce without dropping, should we expose `release_force_max_tier()`? Probably no — breaks RAII guarantees.
- **Process-wide forcing as a deliberate escape hatch.** Not exposed by default. If a user genuinely needs it (for `#[serial]` rayon cases), should we add `force_max_tier_process_wide()` under a separate feature flag (`artisan_test_hooks_process_wide`)? Probably no — it's a footgun.
- **Test-hook feature name.** Current: `artisan_test_hooks` (underscored). Alternative: `test-hooks` (hyphenated, Cargo convention). Chose underscored to avoid collision with user-chosen feature names; revisit on feedback.
