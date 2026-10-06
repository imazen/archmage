+++
title = "Testing SIMD Dispatch"
weight = 1
+++

Test the public call chain, not only a generic helper invoked with `ScalarToken`.
Otherwise a missing feature-enabled entry or broken dispatcher can escape tests.

Enable `archmage/testable_dispatch` for tests only: in `[dev-dependencies]`, or
behind a dev-only feature such as `_dev = ["archmage/testable_dispatch"]`. It
lets tests disable tiers the build guarantees at compile time, such as SSE2 on
x86-64 or NEON on AArch64. It also makes every `summon()` read a cache,
including `X64V1Token`'s on x86-64, which is otherwise a constant, and stops
`-Ctarget-cpu` from compiling detection away.

Cargo unifies features across the build graph, so a crate that enables it under
`[dependencies]` or `[workspace.dependencies]` turns it on for every build that
includes that crate, published releases included. Dev-dependency features also
apply to `cargo bench` and examples. Before profiling, check with
`cargo tree -e features -i archmage` (add `@<version>` if two archmage versions
are in the graph).

This example exercises the gain loop under the available tier permutations on
the current machine:

```rust
use archmage::prelude::*;
use archmage::testing::{for_each_token_permutation, CompileTimePolicy};
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn gain_impl(token: Token, data: &mut [f32], gain: f32) {
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, data);
    let factor = f32x8::splat_t(token, gain);
    for chunk in chunks { (f32x8::load_t(token, chunk) * factor).store(chunk); }
    for value in tail { *value *= gain; }
}
fn gain(data: &mut [f32], factor: f32) {
    incant!(gain_impl(data, factor), [v3, neon, wasm128, scalar])
}
let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
    for len in [0, 1, 7, 8, 9, 31, 32, 33] {
        let mut data = vec![2.0; len];
        gain(&mut data, 0.5);
        assert_eq!(data, vec![1.0; len], "{perm}");
    }
});
assert!(report.permutations_run >= 1);
```

The helper serializes token-state changes and restores them even if the closure
panics. Other tests that need stable detection should use `lock_token_testing()`.
The state is process-wide: unrelated threads do not automatically participate
in that lock. Avoid manual disable/re-enable sequences that leak state on panic.

`CompileTimePolicy::Warn` records tiers that cannot be disabled because they
are guaranteed by the compilation target. `WarnStderr` also prints warnings;
`Fail` rejects such guarantees. A baseline build exercises more fallback states
than `-Ctarget-cpu=native`. The number of permutations depends on hardware and
build flags; do not assert a fixed count or require at least two everywhere.

This cannot execute ARM on an x86 CPU or invent unsupported AVX-512. Run native
or emulated architecture tests as well, and inspect the actual selected backend.
Use native hardware for performance evidence.

For zen-style kernels include empty/short rows, vector boundaries, nonmultiple
tails, explicit stride/length policy, integer extremes, NaNs/infinities/signed
zeros where relevant, and the algorithm's floating-point tolerance. Separate
compile success, numerical correctness, and optimized codegen claims.

`python3 xtask/check_docs.py` compiles and runs the website's Rust fences directly.
This prevents a stale Markdown example from being hidden by a corrected copy in
a test file. Syntax-only reference notation uses text rather than Rust fences.
