## Problem

When a user splits a hot kernel into `_scalar`, `_v3`, `_v4`, `_neon` variants (via `#[autoversion]`, `incant!`, or `#[rite(v3, v4, neon)]`), it is very easy to ship a kernel where one tier silently produces different output from another: an off-by-one in a lane, a wrong rounding mode, a transposed shuffle, a NaN-propagation difference, a missed signed-zero case. Archmage has seen many of these bugs while polyfilling cross-arch behaviors — the [Known Cross-Architecture Behavioral Differences table in `CLAUDE.md`](../blob/main/CLAUDE.md) is an ongoing catalogue.

Writing cross-tier parity tests by hand is tedious, error-prone, and gets skipped. Every downstream crate (zenjpeg, zenresize, zenblend, zenpixels-convert, …) reinvents this scaffolding ad-hoc, if at all.

## Proposal

Auto-generate a cross-tier parity test harness from `#[autoversion]` / `incant!` / `#[rite]` multi-tier declarations. These macros already know the tier list and the function signature — they have everything needed to emit a `#[cfg(test)]` parity test.

For each declaration, generate a test that:

1. Iterates over user-supplied inputs.
2. Computes the reference output using the `_scalar` variant.
3. For each tier the runtime CPU actually supports, forces the dispatcher to select that tier and asserts the output matches the reference within an opt-in tolerance.
4. Skips tiers the CPU does not support (calling them would SIGILL). Upgrade-path testing is handled by the CI matrix (see below).

## Soundness of `force_max_tier`

The test hook is gated behind a `test-hooks` Cargo feature, not compiled into release builds. Its only capability is to downgrade the dispatcher's tier selection — it can force the runtime to pick tier `T` or lower, but never to fabricate a tier the CPU lacks. Disabling a tier the CPU supports cannot cause UB; only fabricating a tier the CPU lacks can. Sound by construction.

**Implementation: per-thread override, not atomic cache poke.** The hook is a `thread_local!<Cell<Option<Tier>>>` consulted by each trampoline before its `AtomicU8` cache. Setting the override returns an RAII scope guard that restores the previous value on drop. Properties:

- **Per-thread isolation = per-test isolation.** Cargo runs one test per thread; forcing on thread A cannot affect thread B. No `#[serial]` required, no `RUST_TEST_THREADS=1`, no race between parallel parity tests.
- **Zero cost in release.** The `#[cfg(feature = "test-hooks")]` block compiles away entirely.
- **RAII restore.** Panic during a test still unwinds the scope and restores state.
- **Atomic cache stays correct.** The thread-local never writes to the cache. Production detection is unaffected; test hooks only overlay.

**Rayon / worker-pool caveat.** `thread_local!` values do not propagate to spawned threads. Tests that dispatch from a rayon worker pool see the real-CPU behavior in the worker, not the forced tier. Doc-note mitigation: call `force_max_tier` inside the worker closure, or use `serial_test::serial` for tests that genuinely need process-wide forcing.

**Prerequisite.** Archmage's existing `dangerously_disable_token_process_wide()` writes to the process-wide atomic cache and has the same race-condition problem under parallel tests. Implementing this harness requires first adding a thread-local override layer to archmage's token caches (generated alongside the `AtomicU8` per token). That refactor is independent of the harness itself but must land first; it should be a separate tracking issue.

## API sketch

`#[autoversion]` gains an `inputs` argument:

```rust
#[autoversion(v3, v4, neon, parity_inputs = parity_cases())]
fn dot(_token: SimdToken, a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn parity_cases() -> impl IntoIterator<Item = (Vec<f32>, Vec<f32>)> { /* ... */ }
```

Expands (behind `#[cfg(test)]`) to a generated test that runs each case through every CPU-supported tier and compares to `dot_scalar`.

For `#[rite(v3, v4, neon)]` and `incant!`, a sibling attribute:

```rust
#[parity_test(inputs = cases(), tolerance = ulp(4))]
#[rite(v3, v4, neon, import_intrinsics)]
fn blend(a: f32x4, b: f32x4, t: f32x4) -> f32x4 { /* ... */ }
```

`tolerance` defaults to exact for integers and `ulp(0)` for floats; users opt into `ulp(N)` or `abs(eps)` for kernels where FMA-vs-separate-mul-add divergence is expected and documented.

## SDE / qemu matrix

Forcing a higher CPU than the host has cannot be done in-process. The recommended pattern is a CI matrix job per tier, running under Intel SDE (`sde -hsw --`, `sde -skx --`, `sde -icx --`) or `qemu-user` with `--cross-cpu`. On each job every tier up to the emulated level is "supported" and the harness exercises it. A reference workflow snippet ships in `docs/site/content/archmage/testing/parity-matrix.md`.

## Relationship to existing tests

Complementary, not overlapping:

- `tests/feature_consistency.rs`, `tests/*_intrinsics.rs` — test the token/intrinsic plumbing (archmage's responsibility).
- Parity harness — tests *user* kernels for bit-exactness across tiers.

## Open questions

- **Input ergonomics** — attribute arg (`parity_inputs = expr`), companion attribute on a user fn, or a standalone `parity_test!` macro? Attribute arg is least invasive; companion attribute composes best with `proptest`.
- **`proptest` integration** — auto-derive strategies from signature, or require the user to pass a `Strategy`? Start with explicit, consider derive later.
- **Tolerance defaults** — exact for ints, `ulp(0)` for floats, or `ulp(2)` to accommodate FMA drift without surprising integer users? Leaning exact-by-default.
- **Where does the SDE recipe live** — archmage docs, a `cargo xtask parity-ci` generator, or a workspace template?

## Prior art / non-goals

`multiversion`, `pulp`, and hand-rolled `target_feature` dispatch all lack any parity harness. `proptest` is a plausible input generator but is not itself a tier-aware test. Non-goal: testing tiers the host CPU cannot run without emulation — that is the CI matrix's job, not the harness's.
