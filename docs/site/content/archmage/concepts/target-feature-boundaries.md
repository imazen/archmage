+++
title = "Target-Feature Boundaries"
weight = 2
aliases = ["archmage/advanced/llvm-boundaries/"]
+++

A CPU token proves that instructions are available. A function's target features
tell the compiler which instructions it may emit. Passing a token to an ordinary
function changes neither its target features nor its compilation baseline.

## Enter once around the loop

The [complete zenfilters example](@/magetypes/examples/generic-kernels.md)
shows both supported arrangements:

```text
public apply_gain
  → incant! selects a tier once
    → #[magetypes] generated feature-enabled body
      → load / multiply / store / scalar tail
```

```text
public gain
  → incant! selects a tier once
    → #[magetypes] generated feature-enabled entry
      → inline gain_kernel<T: F32x8Backend>
        → load / multiply / store / scalar tail
```

Keep the outer row, strip, or batch loop inside the feature-enabled function.
Summoning once but calling a tiny SIMD entry for every pixel still crosses the
boundary repeatedly. Such calls can prevent loop optimization and add call
and register-transfer overhead. There is no universal slowdown multiplier.

## Generics are not dynamic dispatch

Rust monomorphizes `fn kernel<T: F32x8Backend>` for its concrete callers.
A trait bound does not create a trait object or inherently prevent inlining.
The generated caller establishes target features. An inline generic helper
can then optimize in that context. Use `#[inline]`, or `#[inline(always)]`
where inspection justifies the stronger hint; validate the resulting loop.

A generic `#[arcane] fn f<T: HasX64V2>` establishes the features associated with
**HasX64V2**, even if a particular caller passes a V3 token. It does not ask LLVM
to compile the body for every stronger capability of `T`. Use tier generation
when the function should be compiled separately at several feature levels.

## Which calls can inline?

| Caller → callee | Consequence |
|---|---|
| Baseline → V3 entry | Required feature boundary; put substantial work behind it |
| V3 → matching `#[rite]` helper | Features permit inlining; optimizer still decides |
| V4 → V3 helper | Superset permits inlining; explicit `.v3()` provides a V3 token |
| V3 → V4 helper | Stronger features need their own proof and boundary |
| Tier body → ordinary generic helper | Can inline into the tier; a surviving out-of-line helper retains its own compilation context |

`#[rite]` supplies features directly, without the safe outer entry wrapper.
Use it for matched internal calls, not a public baseline `incant!` target.
Nested `incant!` can select the matching sibling/helper within a macro-managed
context; see [dispatch](@/archmage/dispatch/incant.md).

## Prove performance for the caller you ship

Inspect optimized code built with your supported baseline, not only
`-Ctarget-cpu=native`. Confirm dispatch is outside the loop, the hot loop has
no helper calls caused by lost feature context, and array chunk indexing has
no panic path in that loop. Compare equivalent algorithms, tails, and floating
point contracts. Vector width alone does not establish throughput.

Use the repository's `xtask/codegen.py` checks and representative downstream
kernels. Benchmark each ISA on hardware that supports it. QEMU and WASM test
runners can check correctness; they do not substitute for native timing.
