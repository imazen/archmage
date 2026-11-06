+++
title = "Choosing math: contracts, dispatch and compiler versions"
weight = 0
+++

Choose the numerical contract before choosing the fastest implementation. A
correctly rounded function, a polynomial approximation and algebraic arithmetic
solve different problems. Matching a particular libc is a fourth contract: neither
Rust's standard library nor a correctly rounded implementation promises those bits.

## The available choices

| Surface | Contract | Use |
|---|---|---|
| Rust `std` float methods | Method-specific precision; transcendental results may vary by platform and version | A platform baseline when reproducibility is unnecessary |
| Magetypes `_midp` / `_lowp` | Approximation families with documented domains and ISA differences | Vector kernels with an explicit error budget |
| Magetypes `mul_add_fused` | Single rounding | Algorithms that require FMA semantics |
| Magetypes `mul_add_relaxed` | Backend-dependent fused or separate arithmetic | Explicitly accepted rounding differences |
| Draft Zenmath scalar CORE-MATH ports | Correct rounding for the ported operations under their documented floating-point environment | Portable scalar reference-quality math |
| Draft Magemath correct `exp` / `ln` / `log2` / `pow` | Packed f32x8 common paths with accurate scalar repair | Portable vector math with the correct-rounding contract |
| Draft Magemath approximate surfaces | `portable` fuses, `unfused` separates, `relaxed` follows backend rounding | A fixed approximation family with an explicit rounding policy |
| Rust `algebraic_*` | Compiler-selected algebraic transformations; no reproducibility or precision bound | Opt-in kernels that tolerate reassociation and changed exceptional-value behavior |

Zenmath and Magemath are unpublished draft crates in this repository. They are
not changes to the published Magetypes methods. Magemath currently exposes f32x8;
a larger type elsewhere in Magetypes does not imply a matching Magemath API.
The draft scalar set includes f32 `powf`, `expf`, `logf`, `log2f`, `log10f`,
`sqrtf`, `floorf`, `ceilf`, `roundf`, and f64 `log2`. `roundf` uses ties away
from zero; `roundevenf` is a separate operation. Do not benchmark ties-even
vector rounding as though it were Rust's `round`.

See [precision levels](@/magetypes/math/precision.md) and
[ISA differences](@/magetypes/isa-quirks.md) before substituting methods.
Magetypes' existing transcendental NaN and subnormal behavior is not a full
scalar-libm contract. In particular, approximate power is not a general
replacement for `powf` on negative bases and arbitrary exponents.

## Compiler version: keep 1.89 for the existing API

The workspace currently declares Rust 1.89. The
[current `core::f32` documentation](https://doc.rust-lang.org/core/primitive.f32.html),
checked against Rust 1.99.1, lists the five `algebraic_*` methods as stable and
const since **1.98**. It does not expose `sqrt`, `floor`, `ceil`, `round`,
`mul_add`, or the transcendental methods in `core`. Raising the minimum to
1.98 therefore does not remove the corresponding `no_std` implementations.
The [Rust release notes](https://doc.rust-lang.org/releases.html#version-1980-2026-08-20)
record the algebraic API stabilization.

A higher compiler version and a higher minimum supported compiler version are
separate choices. Consumers can compile Magetypes with a newer LLVM today while
the library keeps its 1.89 minimum. A minimum-version increase alone does not
change the arithmetic contract or mark existing operations as algebraic.

Keep the existing minimum for the current surface. If measurements justify an
explicit algebraic API, review that API and its compiler requirement together.
Do not make ordinary `+` or `*` silently acquire algebraic semantics depending
on the compiler version. No manifest or production arithmetic has been changed
for this investigation.

## Telling the compiler that reassociation is allowed

Call the algebraic methods on the operations that may be transformed. A CPU
feature annotation gives LLVM permission to use instructions; it does not give
permission to change floating-point evaluation order. Ordinary vector `+` and
`*` remain ordinary operations even inside an AVX2 or NEON context.

This complete example requires Rust 1.98 or later. The scalar source loop gives
LLVM the algebraic permissions needed to consider a vector reduction:

```rust
fn dot_algebraic(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0_f32, |sum, (&x, &y)| {
        sum.algebraic_add(x.algebraic_mul(y))
    })
}

let result = dot_algebraic(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
std::hint::black_box(result);
```

The optimizer still decides whether to vectorize. Test the actual kernel's LLVM
IR and assembly: look for packed arithmetic, a vector reduction, and unexpected
calls or memory traffic. Put the whole loop in an appropriate Archmage feature
context when targeting wider instructions. `#[attune]` handles architecture
gating for its generated definitions; its dispatcher and `attuned!` handle
matching calls. A hand-written call to a foreign architecture's named variant
still needs a compile-time gate.

The stable algebraic methods take scalar floats, not Magetypes vector values.
An array of lanes can apply these methods elementwise and be optimized back
into packed instructions; that requires checking the resulting code, especially
across load/store and function boundaries. Rust's
[`core::simd::num::SimdFloat`](https://doc.rust-lang.org/core/simd/num/trait.SimdFloat.html)
remains nightly-only and currently supplies no algebraic methods. The underlying
[`fadd_algebraic` intrinsic](https://doc.rust-lang.org/core/intrinsics/fn.fadd_algebraic.html)
is also nightly-only; the scalar methods are the stable entry point.

Algebraic arithmetic may reassociate, replace division with reciprocal
multiplication, combine operations, and disregard signed zero. Its precision
is unspecified. Identical inputs need not produce identical outputs even within
one execution; NaNs and infinities can behave differently. It remains safe Rust,
but it cannot implement the portable-bit contract. Fused multiply-add is a much
narrower choice: it promises one rounding rather than granting general algebraic
rewrites. See Rust's [algebraic-operator contract](https://doc.rust-lang.org/core/primitive.f32.html#algebraic-operators).

For Rust 1.99.0 specifically, the compiler's
[`LLVMRustSetAlgebraicMath`](https://github.com/rust-lang/rust/blob/1.99.0/compiler/rustc_llvm/llvm-wrapper/RustWrapper.cpp#L583)
sets LLVM's `reassoc`, `contract`, `arcp` and `nsz` flags on each affected
instruction. These permit reassociation, contraction, reciprocal transforms and
ignoring signed zero. It does not set `nnan` or `ninf`; do not describe the API
as an assertion that exceptional inputs never occur. These are implementation
details to check in generated IR, not a stronger numerical promise than Rust's
public contract. See [LLVM's flag definitions](https://llvm.org/docs/LangRef.html#fast-math-flags).
