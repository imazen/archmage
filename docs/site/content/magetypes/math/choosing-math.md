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
| [Rust `std` float methods](https://doc.rust-lang.org/std/primitive.f32.html#method.powf) | Method-specific precision; transcendental results may vary by platform and version | A platform baseline when reproducibility is unnecessary |
| Magetypes `_midp` / `_lowp` | Approximation families with documented domains and ISA differences | Vector kernels with an explicit error budget |
| Magetypes `mul_add_fused` | Single rounding | Algorithms that require FMA semantics |
| Magetypes `mul_add_relaxed` | Backend-dependent fused or separate arithmetic | Explicitly accepted rounding differences |
| Draft Zenmath scalar CORE-MATH ports | Correct rounding for the ported operations under their documented floating-point environment | Portable scalar reference-quality math |
| Draft Magemath correct `exp` / `ln` / `log2` / `pow` | Packed f32x8 common paths with accurate scalar repair | Portable vector math with the correct-rounding contract |
| Draft Magemath approximate surfaces | `portable` fuses, `unfused` separates, `relaxed` follows backend rounding | A fixed approximation family with an explicit rounding policy |
| Rust `algebraic_*` | Compiler-selected algebraic transformations; no reproducibility or precision bound | Opt-in kernels that tolerate reassociation and changed exceptional-value behavior |

Zenmath and Magemath are unpublished draft crates in this repository. They are
not changes to the numerical behavior of published Magetypes methods. The draft
deprecates those methods and provides all 35 legacy transcendental names at
widths four, eight and sixteen in Magemath; width sixteen requires `w512`.
The `legacy` family preserves backend behavior, while `portable`, `relaxed`,
and `unfused` retain their explicit rounding and exceptional-value contracts.
The draft scalar set includes f32 `powf`, `expf`, `logf`, `log2f`, `log10f`,
`sqrtf`, `floorf`, `ceilf`, `roundf`, and f64 `log2`. `roundf` uses ties away
from zero; `roundevenf` is a separate operation. Do not benchmark ties-even
vector rounding as though it were Rust's `round`.

The remaining packed API gaps are explicit: no correctly rounded Magemath
`log10`, no generic Magetypes f64 vector `log2`, and no ties-away vector `round`
in either crate at this revision. Approximate f32 `log10` and ties-even vector
rounding are available. A scalar loop inside a feature context is a useful
benchmark control, but is not a missing packed implementation.

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

For a Magetypes value, an explicit lane map expresses the permissions in safe
Rust. This complete example requires Rust 1.98+. The polynomial is an arithmetic
example, not an accurate implementation of `exp`:

```rust
use archmage::prelude::*;

#[attune(_*_t, define(f32x8))]
fn cubic(token: Token, input: [f32; 8]) -> [f32; 8] {
    let value = f32x8::from_array_t(token, input);
    let lanes = value.to_array().map(|x| {
        0.166_666_67_f32.algebraic_mul(x)
            .algebraic_add(0.5).algebraic_mul(x)
            .algebraic_add(1.0).algebraic_mul(x)
            .algebraic_add(1.0)
    });
    f32x8::from_array_t(token, lanes).to_array()
}

let result = attuned!(cubic([0.5; 8]));
std::hint::black_box(result);
```

This gives the compiler permission to transform each lane's arithmetic. It
cannot force a particular instruction sequence. The reduction counterexample
below is why the map should be benchmarked inside its actual consuming loop.

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

## Algebraic methods in an external crate

A separate crate can define its own extension trait and implement it for
Magetypes vectors. Rust permits implementing a local trait for another crate's
type; adding inherent methods to that foreign type is not permitted. Importing
the trait enables method syntax without changing ordinary vector operators.
See [trait coherence](https://doc.rust-lang.org/reference/items/implementations.html#trait-implementation-coherence).

The generic vector's stored token is private, but its public mutable lane
indexing permits a tokenless implementation: take `mut self`, apply each scalar
algebraic operation to the corresponding lanes, then return `self`. This
preserves the original proof and requires neither CPU detection nor unsafe
code. An alternative accepts the caller's token and reconstructs the result
with `from_array_t`. Inlining and optimization determine whether the temporary
lane representation disappears.

A separate consumer of the prototype compiled and ran its five-operation
fixture against published Magetypes 0.9.30 and 0.9.31-beta.2, with no path
patches. The extension library also compiled with `no_std` on Rust 1.98.1.
These checks establish API compatibility; they do not establish performance
on every backend.

The external-consumer experiment also checks optimized code without LTO.
Its V3 f32 polynomial contains packed eight-lane algebraic operations in LLVM
IR. In the actual timed executable, the tokenless extension, explicit-token
extension and consumer-local lane map each use six packed FMA instructions
per 16 values in the main loop, with direct vector loads/stores and no
extension calls or temporary-array stack traffic there. The function's
assertion frame is outside the loop. This is evidence for these compiled
kernels, not a guarantee that every algebraic expression has zero overhead.
The timed strict-WASM modules likewise use packed `f32x4.mul`/`f32x4.add`
in the polynomial loop, without extension calls. Their ordinary separate
multiply/add instructions do not promise fused arithmetic.

The focused external-crate matrix measures 644 cells across Ryzen V3/V4x,
Neoverse-N1 and strict WASM, with LTO disabled and with thin LTO. Every cell
has at least 30 interleaved rounds. Timing covers f32x8 and f64x4 polynomial,
dot, division and remainder workloads; all six implemented vector widths
have exact-representable fixture coverage.

Selected thin-LTO means, in ns/input value:

| Kernel / target | Ordinary vector | Tokenless extension | Consumer-local algebraic map |
|---|---:|---:|---:|
| f32 polynomial / Ryzen V3 | 0.08221 | 0.06407 | 0.06100 |
| f32 dot / Neoverse NEON | 0.13704 | 0.09452 | 0.09065 |
| f32 polynomial / strict WASM | 0.15327 | 0.15915 | 0.13641 |
| f64 polynomial / Ryzen V3 | 0.11546 | 0.13904 | 0.15614 |

The first two extension-minus-ordinary paired 95% intervals are
[-0.03441, -0.00272] and [-0.04814, -0.03275] ns/value. The last two include
zero, as do extension-minus-local-map intervals for all four rows. Thus these
runs support selected gains, not a uniform speedup or a precise overhead bound.
The WASM wrapper allocates per host call, amortized over 256 repetitions;
native setup is outside timing. Native and WASM absolute times include that
difference. Full tables, raw-data provenance and codegen extracts live in
`benchmarks/algebraic_ext_2026-11-06/`.

The separate extension crate can require Rust 1.98 while Magetypes retains its
existing minimum. This does not make algebraic methods part of Magetypes' own
API. A horizontal algebraic reduction also needs an explicit method; ordinary
`reduce_add` does not acquire algebraic permissions from its input values.

## What changed in Magetypes' betas

The benchmark baseline uses the Magetypes source tree from `v0.9.31-beta.2`.
The draft math work has not replaced its transcendental polynomials or changed
its public math behavior. The complete `magetypes/` tree matches that tag in
this study, not merely the polynomial files.

Beta.1 corrected portable f32 FMA's subnormal rounding and supplied narrower
AVX-512 backend coverage. Beta.2 added the explicit `_fused` and `_relaxed`
multiply-add/subtract names, vectorized the software FMA path, added cached
hardware selection for portable scalar-backend f64 FMA, and used a cached
fusion check for relaxed WASM. The exact software fallback was vendored and
the libm dependency removed. These changes affect math kernel costs without
replacing the existing approximation families. See the
[versioned changelog](https://github.com/imazen/archmage/blob/v0.9.31-beta.2/CHANGELOG.md).

In the study's optimized V3 IR, `poly_array_algebraic_f32` lowers its lane map
to `<8 x float>` multiply/add operations carrying all four algebraic flags.
`dot_algebraic_f32` becomes packed eight-lane operations followed by
`llvm.vector.reduce.fadd.v8f32`. The ordinary Magetypes polynomial also uses
packed eight-lane arithmetic, but without those flags. This verifies that the
array representation can carry algebraic permissions through vectorization;
it does not guarantee the same lowering for every kernel or compiler.
The reproducible probes live in `benchmarks/math_matrix_2026-11-06/`.

The codegen probe uses optimized builds with LTO disabled so emitted IR is
post-optimization; timing builds use thin LTO. Its x86 V3 assembly contracts the lane-map polynomial into three
`vfmadd213ps` instructions per eight values. Its hot loop loads inputs and stores
outputs directly, with no temporary array on the stack and no calls. The function
still has a small stack frame for the input-length assertions and cold panic
paths. The algebraic dot loop uses four independent YMM FMA accumulators and
reduces them afterward. This is observed lowering, not an ABI or optimizer
promise.

## Current performance: Intel Core Ultra 7 265K

These are means in **ns/value**, from 60 interleaved rounds per group over 1024
ordinary inputs. Lower is better. The binary targets baseline x86-64; the packed
column enters a V3 feature context once per buffer. Inputs and outputs are
black-boxed, and allocation is outside the measurement. This is throughput,
not single-call latency. No `target-cpu=native` was used.

| Operation | std baseline | std V3 loop | Zenmath correct baseline | Correct packed V3 |
|---|---:|---:|---:|---:|
| f32 exp | 1.218 | 1.248 | 1.704 | 0.849 |
| f32 ln | 1.413 | 1.239 | 1.878 | 1.077 |
| f32 log2 | 1.231 | 1.254 | 1.691 | 1.174 |
| f32 powf, varying exponents | 2.682 | 2.693 | 19.404 | 7.005 |
| f32 log10 | 1.987 | 1.702 | 2.147 | unavailable |
| f32 sqrt | 0.166 | 0.084 | 2.670 | 0.086 |
| f32 floor | 0.892 | 0.029 | 0.946 | 0.035 |
| f32 ceil | 0.850 | 0.043 | 0.674 | 0.052 |
| f32 round, ties away | 0.738 | 0.045 | 1.135 | unavailable |
| f64 log2 | 2.402 | 1.780 | 8.146 | unavailable |

Packed exp/ln/log2/pow are Magemath's correct f32x8 proof slice entry points;
their measured adapter copies the input into the in-place output buffer.
That copy is included for both proof and runtime-dispatch controls; other
candidates write directly to their output buffer. These are complete API
workload measurements rather than isolated polynomial instruction costs;
sqrt/floor/ceil are Magetypes f32x8 methods. There is no accurate packed log10
or f64 log2 API behind the unavailable cells. Scalar sqrt uses the software
Goldschmidt algorithm even in a std-enabled Zenmath build. Rust's scalar-looking
loop may itself auto-vectorize. With the same V3 context, std sqrt/floor/ceil
already matches or beats the explicit-vector means here; std ties-away round
also vectorizes. The baseline-to-V3 improvement should not be credited to a
new math algorithm. Zenmath's software primitive paths remain a clear
optimization opportunity, without requiring an MSRV increase.

The approximate Magetypes V3 f32x8 midp methods measured 0.248 for exp, 0.227
for ln, 0.195 for log2, and 0.218 for log10. Those are different numerical
contracts. Fixed-exponent `pow_midp(2.4)` measured 0.555; it must not be compared
as though it accepted a different exponent in every lane. The packed Midp
PowBehavior path with varying exponents measured 3.597.

The same run's correctly rounded packed once-buffer dispatchers measured 0.862
for exp, 1.088 for ln, 1.187 for log2 and 7.865 for varying-exponent power.
These separate measurements include dispatch; subtracting means is not a
standalone dispatch-overhead benchmark.

The source, paired intervals, input bands and full platform matrix belong to
`benchmarks/math_matrix_2026-11-06/`. This table uses run
`18e58575f2368e39-141766-0bad75c208d51a68`, binary SHA-256
`8c0450faa9bb53343dacf6ad3b8fa75d4dd34aa51c78ee71ded7a30b7ebfd6ce`.
Resource wrapper: 192 seconds, peak RSS 0.04 GiB. Near ties need the paired
intervals; these means alone do not establish a universal ranking.

## Algebraic performance is kernel-dependent

The f32 dot-product controls measured the following means in ns/value over 1024
inputs, 60 interleaved rounds per group. The ordinary loop accumulates in source
order; the other forms may associate differently. These are not interchangeable
bit contracts.

| CPU / context | Ordinary loop | Algebraic loop | Explicit vector, separate | Explicit vector, fused | Lane-array algebraic |
|---|---:|---:|---:|---:|---:|
| Core Ultra 7 265K / V3 | 0.375 | 0.026 | 0.038 | 0.057 | 0.058 |
| Ryzen 7900X / V3 | 0.698 | 0.047 | 0.065 | 0.070 | 0.073 |
| Neoverse-N1 / NEON | 0.705 | 0.091 | 0.122 | 0.093 | 0.349 |

This supports using the scalar algebraic API for reductions when reassociation
is acceptable. It does not support replacing every explicit vector operation
with an array map. The ARM lane-array reduction needs particular care. For the
polynomial, even the direction varies: the Ryzen 7900X V4x ordinary loop
measured 0.051 and its algebraic counterpart 0.067. Consult the paired intervals
and generated code before making a kernel-specific change.

Native timing/algebraic pipelines completed with resource-wrapper peak RSS
0.04 GiB on the 265K, 0.11 GiB on the 7900X, and 0.03 GiB on Neoverse-N1.
The source and raw rounds are retained with the matrix record.

The practical choice is to keep the existing library minimum and use Rust 1.98+
for explicitly algebraic consumer kernels where measurements justify it. An
algebraic vector API could make those permissions easier to express, but it
would need its own reviewed semantics and codegen checks. Do not apply these
permissions to the correct-rounding kernels: their proofs depend on the stated
sequence of rounded operations, including error-recovery steps.

Disassembly of the actual thin-LTO timing executables confirms both the V3
polynomial FMAs and the ARM lane-array reduction's extra shuffles. The ARM
array loop uses four vector multiplies, ten lane rearrangements and eight
two-lane additions per sixteen values; its main loop does not contract to FMA.
The array-minus-fused timing difference has a paired 95% interval of
0.237–0.281 ns/value in this run. The matrix's `CODEGEN.md` identifies the exact
symbols, binary hashes and extracted loops.

Changed association can also produce large result differences. In the retained
V3 cancellation example, repeating `[1e20_f32, 1, -1e20_f32, 1]` 256 times gives
1 with the ordered scalar loop, 512 with the automatic algebraic loop, and 0
with the explicit vector reductions. The real sum of the represented inputs
is 512. This example demonstrates differing evaluation orders; it establishes
no accuracy guarantee for algebraic arithmetic.

## Observed cross-platform std differences

Adding `log10` to the earlier four-function study found differences in Rust std:
258 of 1024 ordinary ARM results and 105 of 1024 WASM results differed from the
x86 result, by at most one representable step. For input bits `0x405523d7`, the
x86 result is `0x3f05c193` and ARM/WASM return `0x3f05c194`. Correct Zenmath
results agree across the checked targets. The input corpus is finite and
compiler/libc/OS vary together, so this is neither an exhaustive bound nor an
isolated ISA experiment. Rust std does not promise portable transcendental
bits; an implementation that happens to agree on another sampled operation
has not acquired that guarantee.

## The same accurate operations on other targets

Ordinary-input throughput, ns/value, written **std / correct packed**. Native
packed entries use the named proof context and include the slice-adapter copy.
These are per-platform comparisons, not a controlled comparison between CPUs.

| Target | exp | ln | log2 | varying-exponent powf |
|---|---:|---:|---:|---:|
| Ryzen 7900X, V3 | 2.382 / 1.558 | 2.465 / 1.575 | 2.245 / 1.706 | 4.972 / 8.642 |
| Neoverse-N1, NEON | 5.871 / 4.593 | 5.998 / 4.958 | 5.982 / 4.969 | 11.224 / 20.244 |
| Strict WASM, Wasmtime 49.0.5 on Ryzen 5900XT | 3.544 / 4.100 | 3.791 / 5.794 | 3.546 / 5.194 | 8.936 / 41.837 |

WASM constructs its case/input/output buffers per host invocation, amortized
over 256 repetitions of 1024 values; native allocation occurs outside timing.
Its absolute times include that wrapper cost. The WASM run took 466 seconds
under the resource wrapper, with peak RSS 0.43 GiB. Native resource summaries
appear above. Correct packed math is not a speed improvement everywhere:
the accurate packed implementations are slower than std in these strict-WASM cells,
and accurate power is slower on all three targets above.

The retained matrix covers 4568 math throughput cells, 180 algebraic cells and
343 representative native feedback cells, each with at least 30 rounds. One
additional cell, metallic general power on WASM special inputs, traps and has
an explicit failure record. Windows and macOS timings are unavailable; no Zen 5
performance is claimed. The feedback study covers exp, ln, log2 and sqrt; it
is not a complete instruction-latency matrix.
