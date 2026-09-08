+++
title = "Compile-Time vs Runtime"
weight = 1
+++

There are three independent decisions: what source exists in the binary, which
instructions each function may use, and which function runs on this machine.

| Mechanism | What it controls |
|---|---|
| `#[cfg(target_arch = "aarch64")]` | Includes source for an architecture |
| Cargo feature such as `avx512` | Includes optional library support; does not detect the CPU |
| `#[cfg(target_feature = "avx2")]` | Tests the compilation target, not a surrounding function attribute |
| `#[target_feature]`, supplied by archmage macros | Enables instructions within a function |
| `Token::summon()` | Obtains proof of available CPU features, or returns `None` |
| `incant!` | Selects and calls a generated or hand-written variant |
| Rust generics | Statically specialize an algorithm for concrete types and constants |

`#[magetypes]` combines per-tier generation with feature-enabled entry functions.
It is not necessary to choose between it and Rust generics. See the
[generic pixel/const-parameter example](@/magetypes/dispatch/types-and-dispatch.md).

Neither a concrete token parameter nor `#[inline(always)]` alone enables SIMD
instructions in an ordinary function. See [target-feature boundaries](@/archmage/concepts/target-feature-boundaries.md)
for the complete caller/helper model.

`-Ctarget-cpu=native` enables the build machine's features throughout the binary.
It can remove detection and some boundaries, but changes the minimum CPU that
can run that binary. It is appropriate for a known deployment, not evidence
that a baseline-dispatched library has good codegen.
