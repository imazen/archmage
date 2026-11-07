# CORE-MATH versus Rust std and magetypes

This comparison interprets “Rust numerics” as the built-in f32 `exp`, `ln`,
`log2`, and `powf` methods. It does not measure the separate Rust `libm` crate
or every numerical crate. Source starts at local draft `100f0d0e`.

All five implementations receive identical 1024-element inputs and include the
same input copy. Exp inputs span [-8,8); other inputs span [0.01,4.01), in a
permuted grid. **Pow uses exponent 2.4 throughout**, matching magetypes' scalar
exponent API. The earlier core-only benchmark alternated +2.4/-2.4, so its pow
numbers are not directly comparable. Mixed groups replace every eighth input
with NaN. The magetypes methods do not promise the same NaN results; their
mixed-group timings are not a comparison of equivalent semantics.

Baseline builds, no `target-cpu=native`. Dispatch occurs once per buffer for
both vector implementations. CORE-MATH uses four f64 intermediates; magetypes
uses eight f32 lanes. The std calls occur per scalar input. Outputs are all
consumed through black_box, with no allocation inside timed iterations.

## Measured time, ns/value (lower is faster)

Core Ultra 7 265K, Rust 1.99.1, glibc 2.43-2ubuntu2.4, pinned CPU 0:

| Operation | CORE scalar | CORE vector | Rust std | magetypes midp | magetypes lowp |
|---|---:|---:|---:|---:|---:|
| exp | 1.727 | 0.855 | 1.571 | 0.273 | 0.165 |
| ln | 2.176 | 1.331 | 1.245 | 0.239 | 0.137 |
| log2 | 1.451 | 1.093 | 1.236 | 0.229 | 0.128 |
| pow(x,2.4) | 18.415 | 7.656 | 2.676 | 0.581 | 0.318 |

An initial unpinned run is retained too. It measured CORE vector/std ln at
1.091/1.265 ns/value, reversing the small difference above. Both Intel runs
flagged timing drift/noisy rounds (31 and 64 respectively). Treat ln as a
close, run-dependent comparison, not a guaranteed speedup or regression.
The much larger pow and approximation differences persist across the runs.
These are throughput means, not latency or universal implementation rankings.

Neoverse-N1, Rust 1.97.1, glibc 2.39-0ubuntu8.9, idle timing host:

| Operation | CORE scalar | CORE vector | Rust std | magetypes midp | magetypes lowp |
|---|---:|---:|---:|---:|---:|
| exp | 6.943 | 5.709 | 5.809 | 1.728 | 1.003 |
| ln | 7.837 | 5.855 | 6.536 | 1.684 | 1.126 |
| log2 | 5.516 | 5.221 | 5.828 | 1.446 | 0.982 |
| pow(x,2.4) | 30.030 | 17.554 | 10.886 | 3.717 | 2.440 |

ARM exp is nearly tied with std. In the ARM mixed groups CORE vector is slower
than std for all four operations; normal-input rankings do not transfer to
exception-heavy data. All groups and mean absolute deviations are in
[results.csv](results.csv); raw JSON retains rounds and paired comparisons
against the scalar CORE baseline.

## What the extra accuracy buys

- **CORE-MATH scalar/vector:** correctly rounded nearest-even results, gradual
  underflow, full special-case handling, portable output bits except NaN
  payload/sign. [Validation](../core_math_vector_2026-11-06/README.md) includes
  independent MPFR samples, upstream hard cases and exhaustive unary parity.
- **Rust std:** these transcendental methods have unspecified precision, varying
  with target and Rust version; they do not promise correctly rounded or
  cross-platform identical results. See the official
  [f32 documentation](https://doc.rust-lang.org/std/primitive.f32.html).
  This benchmark uses the system libm-linked binary, not a pinned compatibility
  implementation. Do not generalize these Linux timings to Windows or macOS.
- **Magetypes midp:** approximation methods with backend-dependent polynomial
  rounding and restricted accuracy domains. The existing exhaustive-band study
  measured maxima of 4.5 ULP for log2, 4.1 for ln, 8.2 for exp on |x|≤10,
  and 27.8 for pow(x,2.4) on [2^-8,2^8]. These are historical measured bounds,
  not error measurements from this timing run. Subnormals and NaNs do not share
  the new core's contract.
- **Magetypes lowp:** faster approximations; the prior study measured up to
  0.557% relative error for exp/pow in their documented domains. Logs have
  absolute-error bounds instead: 6.4e-6 for log2 and 8.5e-6 for ln.

The detailed domains, overflow/clamping behavior and special-value differences
are in [transcendental accuracy](../../docs/transcendentals.md). Magetypes offers
explicit lowp/midp names; it has no single implicit transcendental accuracy
setting. The portable/relaxed approximation surfaces in magemath remain
separate from the new correctly rounded core.

The new core is an accuracy/portability choice, not a replacement justified by
speed alone. Keep approximate kernels for callers that accept their documented
contract. In particular, correctly rounded pow still has a substantial cost
versus std despite its SIMD speedup over our scalar port.

## Reproduce

```sh
just magemath-core-comparison comparison.json
just magemath-core-comparison-summary comparison.json
```

Run through `run-heavy` on an eligible host. The affinity control uses
`taskset -c 0 target/release/examples/core_math_comparison comparison-intel-cpu0.json`.
Clippy for the new example passed with warnings denied; native release builds
and execution completed on x86 and ARM. This adds benchmark code only.

Artifacts and reference binaries:
`/mnt/v/output/archmage/core-math-comparison-2026-11-06/`, mirrored under
`/mnt/tower/output/archmage/core-math-comparison-2026-11-06/` with SHA256 manifest.

Resource-wrapper completion lines:

```text
Intel initial: rc=0 20s | peak-RSS 0.32GiB | min-avail 21066MiB | peak-load 1.90
Intel CPU 0: rc=0 17s | peak-RSS 0.01GiB | min-avail 20664MiB | peak-load 2.69
ARM build: rc=0 19s | peak-RSS 0.41GiB | min-avail 28514MiB | peak-load 7.89
ARM timing: rc=0 18s | peak-RSS 0.01GiB | min-avail 14894MiB | peak-load 0.02
```
