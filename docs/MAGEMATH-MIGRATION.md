# Migrating configurable math to Magemath

Magemath owns the configurable f32 pow, exponential, logarithm and cube-root
families. It depends on Magetypes vectors and Archmage feature proofs. Magetypes
does not depend on Magemath and cannot forward back into it.
Magetypes retains the flat implementations actually published in 0.9.31, with
deprecation metadata for methods that have replacements. Their arithmetic bodies
are unchanged. The unpublished configured modules and receiver delegates exist
only in Magemath; there is no duplicate configured compatibility implementation.

This migration does not replace libm wholesale. It adds no global floating-point
policy and no new polynomial fits. Options remain operation-specific const
builders ending in `.bits()`, specialized through the caller and kernel.

## What moves now

| Existing API | Configurable owner |
|---|---|
| Unpublished configured power API and `try_powf*` delegates | `magemath::powf` |
| Unpublished configured exponentials | `magemath::expf` |
| Unpublished configured logarithms | `magemath::logf` |
| `pow_lowp*`, `pow_midp*` | `magemath::powf::try_powf_f32xN_with` |
| `exp2_lowp*`, `exp2_midp*` | `magemath::expf::try_exp2_f32xN_with` |
| `exp_lowp*`, `exp_midp*` | `magemath::expf::try_exp_f32xN_with` |
| `log2_lowp*`, `log2_midp*` | `magemath::logf::try_log2_f32xN_with` |
| `ln_lowp*`, `ln_midp*` | `magemath::logf::try_ln_f32xN_with` |
| `log10_lowp*`, `log10_midp*` | `magemath::logf::try_log10_f32xN_with` |
| `cbrt_lowp*`, `cbrt_midp*` | `magemath::cbrtf::try_cbrt_f32xN_with` |

`N` is 4 or 8, and 16 with `w512`. Each family also has scalar and array entry
points. The new crate's exp/log/cbrt precision enums contain MIDP and LOWP, without
predecessor-fit selectors. Unpublished Magetypes module paths and configured
receiver methods are removed, rather than deprecated or forwarded.

These seven operation families are deprecated (`cbrt_*` since `0.9.32`, the other
six since `0.9.31`) while the unpublished configurable APIs are rehomed. Sigmoid
and SiLU have no new replacements yet and remain non-deprecated in Magetypes.
No published flat method is removed. The unpublished `magetypes::powf`,
`magetypes::expf`, `magetypes::logf` modules and `try_powf*` receiver methods
are absent from the Magetypes public API. Removing published flat methods needs
a later breaking release.

## Choosing the contract

A deprecated infallible method is not a promise that every new checked profile
accepts the same inputs. Choose the precision and domain explicitly and handle
whole-vector rejection. `AssumeValid` removes numerical validation; it is safe
Rust, but inputs outside its stated preconditions have unspecified numerical
results. It does not erase required results for valid NaNs/zeros in a domain
that admits them.

- **Powf:** retain the existing PositiveNormal, NonNegativeNormal, Gamma,
  PositiveGamma and Full HIGHP contracts. MIDP/LOWP approximate
  `exp2(n*log2(x))`; mathematical error grows with the argument. BitExact HIGHP
  retains its canonical binary64 graph. Backend-dependent HIGHP std delegation
  still requires `std`.
- **Expf:** finite checked inputs reject NaN/infinity but otherwise retain the
  existing clamps. Exp2 MIDP's 1.9-ULP working target applies below 127.5, not to
  arbitrary clamped inputs or the upper rail. Exp and pow have their own
  multiplication/composition errors. LOWP's 0.56% working target is likewise
  range-limited; a finite domain is not a global error guarantee.
- **Logf:** Positive checks positive normal finite inputs. Specials retains the
  historical special classifications and inaccurate subnormal handling. MIDP's
  working target is 4.5 ULP on the normal domain; LOWP uses an absolute-error
  budget and can have large relative error near 1. In particular LOWP log1p
  retains the rational log2 bias near zero and can be negative for tiny positive
  arguments. Use MIDP there; no corrected LOWP small-x graph is added here.
- **Cbrtf:** `Bounded` preserves `±0.0` and normal finite cube roots with
  `|x| < f32::MAX / 3`, rejecting subnormals, `|x| >= f32::MAX / 3`, infinities
  and NaN; `Full` rescales subnormals via `2^24` / `2^-8`, scales large normals
  (`|x| >= 1e36`) via `1/8` and `2`, and propagates `±0.0`, `±inf` and NaN. All
  presets share the in-register `i32` shift-add Kahan bit-hack seed followed by
  one (`Lowp`, <= 259 ULP) or two (`Midp`, <= 3.2 ULP) Halley iterations.

All numerical guarantees assume round-to-nearest and gradual underflow. NaN
payloads/signs and floating exception flags are not portable-result contracts.
The [powf reference](POWF_REFERENCE.md) and
[frozen exp/log audit](https://github.com/imazen/archmage/blob/1662daa4d14f/benchmarks/expf_logf_audit/PR148.md)
distinguish measured samples, proposed bounds and remaining gaps. The narrower
normal-output exp2 experiment is not promoted into a new public range profile
by this refactor.

## Preserve the caller proof, including tails

Scalar and array entries may automatically select a backend once per operation.
Inside a typed hot loop, pass the existing proof to the vector entry point and
use `*_with_token` for tails with the same flags. Do not substitute ScalarToken
in a backend-dependent native tail: it may choose separate arithmetic.

```rust
use archmage::ScalarToken;
use magetypes::simd::generic::f32x4;
use magemath::powf::{PowfExecution, PowfOptions, try_powf_f32x4_with,
                     try_powf_with_token};

const FLAGS: u32 = PowfOptions::PORTABLE_GAMMA_MIDP
    .with_execution(PowfExecution::Caller).bits();
let token = ScalarToken; // A native kernel passes its own existing token here.
let input = f32x4::from_array_t(token, [0.0, 0.25, 0.5, 1.0]);
let output = try_powf_f32x4_with::<FLAGS, _>(token, input, 2.2).unwrap();
let tail = try_powf_with_token::<FLAGS, _>(token, 0.5, 2.2).unwrap();
assert_eq!(output.to_array()[2].to_bits(), tail.to_bits());
```

Use the runtime-named entry points when options really are dynamic. Powf also
provides `try_powf_f32xN_runtime` to retain a typed caller proof with dynamic
options. The old infallible `_unchecked` spellings do not map mechanically to a
profile: select the domain whose assumptions your code actually establishes.

Magemath is `no_std` by default. `std` enables runtime feature detection and
native fallbacks; `w512` enables logical 16-lane f32 vectors; `avx512` adds native
AVX-512 tiers and implies `w512`. No Magetypes dependency is added to Archmage,
and no Magemath dependency is added to Magetypes.

## Reciprocal and square root boundary

Keep basic `sqrt` in Magetypes: it is a vector primitive used by higher-level
math. A policy-driven approximation belongs above it only when a measured
algorithm needs a different contract.

Reciprocal and reciprocal square root are plausible next Magemath families:
there are already estimate, refinement and precise-composition choices with
real ISA-dependent costs. This PR does not change or deprecate those methods.
A future proposal must define its canonical result and domain first. In
particular correctly rounded mathematical rsqrt and rounded `1/sqrt(x)`
composition are different contracts. Existing Magetypes tests and bounds stay
in force until a replacement is implemented and measured.

## Migration validation

`just magemath-check` checks std/no_std, debug/release contracts, strict Clippy,
whole-vector error masks, signed zero, the independent powf fixtures and typed
vector/scalar-tail identity. `just magemath-check-wasm` checks strict WASM.
See the [extraction validation record](../magemath/VALIDATION.md) for actual
completed checks and gaps. The existing CI matrices also run Magemath on x86, ARM64, Windows ARM64, macOS
ARM64, cross targets including i686, and strict WASM. Numerical fits and the
unchanged primitive implementations reuse their existing evidence; this PR
makes no new whole-matrix speed-promotion claim.

The moved log1p test uses a fixed MPFR256-rounded `log1p(0.5)` reference
(`0x3ecf991f`) rather than an approximate std call under Miri. This retains the
native expected bits and exact assertion; no error threshold is relaxed.

## Published implementation boundary

The crates.io 0.9.31 archive records source commit
`6bfd83d744e41259ec758054631dcb0928fc4ea1`, matching tag `v0.9.31`.
`just check-published-math` compares every published flat transcendental body
on f32x4/x8/x16 with hashes taken from that archive. It ignores whitespace,
comments and external deprecation metadata while retaining arithmetic tokens.
It also rejects configured modules or `try_powf*` receiver APIs in Magetypes.
The frozen hashes live in `xtask/published_math_0_9_31.json`; the check requires
no network access in CI. This is source-body preservation, not a new assertion
that different compilers or primitive implementations always produce equal bits.
