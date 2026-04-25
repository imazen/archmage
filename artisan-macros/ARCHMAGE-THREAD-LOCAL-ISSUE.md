# GitHub issue draft: add thread-local override layer to archmage token caches

**Status:** draft, pending review. Intended target: https://github.com/imazen/archmage/issues

---

## Title

Add thread-local override layer to token caches (prerequisite for cross-tier parity harness)

## Body

## Problem

`archmage::dangerously_disable_token_process_wide()` writes to the process-wide atomic cache of a token. Under parallel test execution (cargo's default), two tests that both disable different tokens race — thread A sees thread B's disabled state mid-test, and vice versa. This makes the API effectively unusable for testing tier fallbacks in a crate that runs more than a handful of tests in parallel.

The proposed cross-tier parity harness (#TBD) needs to downgrade tier selection per-test, reliably, without `#[serial]` annotations or `RUST_TEST_THREADS=1`. Without a thread-local override, the harness has no way to isolate parity tests from each other when they force different max tiers.

## Proposal

Add a thread-local override layer on top of every token's atomic cache. The atomic cache remains the source of truth for CPU detection (and continues to use `Relaxed` ordering with idempotent writes). The thread-local is an orthogonal downgrade override.

Each token `T` gains a generated thread-local boolean ("is this token disabled on the current thread?"). `T::summon()` consults the thread-local after its compile-time `#[cfg(target_feature)]` check and before the atomic cache. If the thread-local says "disabled," `summon()` returns `None` without touching the cache.

`dangerously_disable_token_process_wide` is kept as-is for backwards compatibility (and for the genuine use case of disabling a token across a rayon pool), but it is renamed in docs to something like "process-wide forcing — not for parallel tests; see `force_token_disabled` for per-test scoping."

## API shape

```rust
// NEW: thread-local, RAII-scoped, per-test.
pub fn force_token_disabled<T: SimdToken>() -> TokenScope<T>;

#[must_use = "dropping restores the previous setting"]
pub struct TokenScope<T> { /* opaque */ }

impl<T: SimdToken> Drop for TokenScope<T> {
    fn drop(&mut self) { /* restore previous thread-local state */ }
}

// EXISTING: unchanged, but doc-steered away from tests.
pub fn dangerously_disable_token_process_wide<T: SimdToken>(disabled: bool);
```

Codegen change in `xtask/src/token_gen.rs`:

```rust
// Emitted alongside the existing atomic cache per token:
thread_local! {
    static FORCED_DISABLED: Cell<bool> = const { Cell::new(false) };
}
```

`T::summon()` modified to consult the thread-local:

```rust
impl SimdToken for T {
    fn summon() -> Option<Self> {
        // compile-time fast path (unchanged)
        if let Some(b) = Self::compiled_with() {
            return if b { Some(unsafe { Self::forge_token_dangerously() }) } else { None };
        }
        // NEW: thread-local override
        if FORCED_DISABLED.with(|c| c.get()) {
            return None;
        }
        // atomic cache path (unchanged)
        match CACHE.load(Relaxed) {
            2 => Some(unsafe { Self::forge_token_dangerously() }),
            1 => None,
            _ => { /* run detection + store + return */ }
        }
    }
}
```

## Soundness

The thread-local can only DISABLE a token, never fabricate one. Disabling a token the CPU supports cannot cause UB — the code path takes a slower variant. Fabricating a token the CPU lacks would SIGILL; the API does not expose that direction.

All existing sanity properties are preserved:

- Constant-CPUID axiom still holds. CPU features don't change; the cache is still a memoisation of a pure function.
- Cache values `0/1/2` remain correct; the thread-local is an orthogonal override.
- Downstream `#![forbid(unsafe_code)]` unaffected — thread-local machinery is all safe code.

## Migration

- `dangerously_disable_token_process_wide` stays. Existing callers keep working.
- Documentation adds a section pointing tests toward `force_token_disabled` and reserving the process-wide function for rayon-pool use cases and infrastructure tests in `archmage`'s own test suite.
- The cross-tier parity harness (when implemented) uses only `force_token_disabled`.

## Testing

- Unit test: `force_token_disabled::<X64V3Token>()` on thread A doesn't affect `X64V3Token::summon()` on thread B (spawn two threads, verify).
- Unit test: RAII restoration — after scope drop, `summon()` returns the real CPU state.
- Unit test: nested scopes compose correctly.
- Integration test: two parallel `#[test]` functions each force different tokens disabled and verify no interference. This test is the whole point of the refactor.

## Scope

This issue covers the thread-local override layer. The cross-tier parity harness that builds on it is a separate tracking issue (#TBD parity-harness).

## Open questions

- **Name of the new API.** `force_token_disabled` matches the existing `dangerously_disable_token_process_wide` shape. Alternatives: `scoped_disable_token`, `with_token_disabled`, `disable_token_scoped`. No strong preference; whichever matches archmage's naming conventions.
- **Per-token vs per-tier thread-locals.** Current proposal: one thread-local per token. An alternative is a single tier-indexed thread-local (matching the artisan-macros design). Per-token is more aligned with archmage's existing code-gen; per-tier would be a larger refactor.
- **`for_each_token_permutation` compatibility.** The existing iteration test helper uses process-wide disabling. If the helper runs on a single thread, it can use the new scoped API; if it spawns threads, it needs the process-wide version. Decide per helper.
