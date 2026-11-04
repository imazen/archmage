# magemath design record

Why `magemath` is shaped the way it is, what the measurements said, and the rules
for extending it. Numbers are from [the 2026-11-05 run](../../benchmarks/magemath_portable_cost_2026-11-05/README.md),
which has the hosts, conditions and raw output.

## The contract

`magemath::portable` returns the same bits on every tier, and those bits equal the
`zenmath::scalar` functions. NaN results agree only in being NaN.

magetypes' `f32x8` methods can't promise this, on purpose: `mul_add_relaxed` rounds
once where the hardware fuses and twice on the scalar backend and strict WASM,
because the fused form made those backends 1.5-23x slower
([#116 record](../../benchmarks/transcendentals_fused_mul_add_2026-10-05.md)). So
callers who need matching bits and callers who need the fastest result per tier want
different code, and magemath is the first group's.

The first draft of this crate wrapped the relaxed methods and tolerated 4 ULP on the
scalar tier. Its tests passed on x86 and failed on NEON and WASM:

- **Rounding.** Over 4,853 inputs, the scalar tier differed from the mirrors for
  `log2_midp` (4 results), `exp2_midp` (19), `ln_midp` (2), `exp_midp` (17) and
  `log10_midp` (3), by at most 2 ULP. Every op using `mul_add_relaxed` differed; every
  op without it (`sqrt`, `floor`, `ceil`, `roundeven`, `cbrt_*`) matched exactly.
- **NaN.** `exp2_midp(NaN)` was 2^-126 on x86 and NaN on NEON and WASM, because
  `max`/`min` treat NaN differently per ISA. `log2_midp(NaN)` was -383.4 everywhere.
  The goldens had pinned those values. The full table is in
  [`docs/transcendentals.md`](../../docs/transcendentals.md).

Hence the kernels in `portable::kernels`: the same polynomials as magetypes, every step
`mul_add_fused`, and NaN handled without relying on anything ISA-specific. On x86 and
NEON `mul_add_fused` is the same instruction as `mul_add_relaxed`, and
`kernels_equal_magetypes_where_hardware_fuses` stops the two copies of each polynomial
from drifting apart.

Where the hardware can't fuse, `mul_add_fused` picks a path, and every path gives the same
bits. Relaxed WASM uses the engine's madd once a cached probe shows it fuses, and
otherwise software. f64 on x86-64 with std checks a cached V3 token. f32 outside a
matching tier goes straight to the vectorized software path, with no runtime CPU check,
because a per-call hardware path measured slower (repo `CLAUDE.md`, Dispatch granularity).
That is only reached on CPUs without V3 anyway: the dispatcher picks the v3 tier first.

The result holds for every f32: all 2^32 inputs match the mirrors on scalar, v3, v4x and
NEON, and about 230,000 per op match on WASM (strict, relaxed on engines that fuse
and engines that don't), qemu without FMA, and i686.

## What it costs

Fused in hardware (x86, NEON), `portable` midp is at least as fast as the relaxed method
it matches, because `exp2_midp` drops a clamp. `portable` lowp pays for specials the
relaxed lowp methods don't have.

| time vs the relaxed method, paired 95% CI | Ryzen 9 5900XT, V3 | Neoverse-N1, NEON |
|---|---:|---:|
| `exp2_midp` | -7.5% to -3.9% | -9.4% to -8.6% |
| `log2_midp` | -5.4% to -0.8% | -0.5% to +0.5% |
| `pow_midp(2.4)` | -7.3% to -3.2% | -2.7% to -1.8% |
| `exp2_lowp` | +19.2% to +20.6% | +12.3% to +13.3% |
| `log2_lowp` | +54.4% to +55.4% | +33.8% to +34.9% |

Fused in software, the cost is large, and it is the price of the contract:

| how much slower than the relaxed method | range over those five ops |
|---|---:|
| x86-64 scalar tier (only CPUs without V3) | 2.0-5.0x |
| strict WASM | 10-25x |
| relaxed WASM, engine fuses | 1.1-3.7x |
| relaxed WASM, engine rounds twice | 7.8-22x |

So a WASM build that wants these kernels should enable relaxed SIMD, and a WASM caller that
doesn't need matching bits should use the magetypes methods. On a fusing engine every
multiply-add also reads the cached probe. Whether that explains the remaining gap is
unmeasured, and hoisting the check above a whole kernel would need a magetypes change.

## Module boundary

| layer | bits across tiers | when to use it |
|---|---|---|
| `portable` | identical | default; anything that must reproduce across machines |
| magetypes `f32x8` methods | 1-2 ULP apart where the hardware can't fuse; NaN differs | the fastest result per tier, when bits needn't match |
| `native` (empty) | may differ by design | a measured need neither of the above meets |

Same op names in different modules, so the import path states the contract. A trait
policy (zenmetrics-math's `MathPolicy`) was the alternative. It earns its keep there,
where kernels are generic over several whole math conventions, but here it would put
a generic parameter on every signature to express what one `use` line already says.

A `native` kernel must never be wired into a `portable` dispatcher. "Range-specific" does
not by itself mean `native`: a new polynomial built from fused ops and IEEE operations
has the same bits on every tier and belongs in `portable` under a name stating its
range. `native` is for what can't: ISA-specific instructions whose results differ, such
as `rcp`/`rsqrt` estimates, or table lookups with per-ISA table shapes.

## Edge policies: `_unchecked`, not const generics

| time saved by `_unchecked` against the checked kernel | Ryzen 9 5900XT, V3 | Neoverse-N1, NEON | scalar tier, x86-64 |
|---|---:|---:|---:|
| `exp2_midp` | 30% | 23% | 4% |
| `log2_midp` | 38% | 26% | 11% |
| `pow_midp(2.4)` | 27% | 18% | 11% |
| `exp2_lowp` | 30% | 25% | 0% |
| `log2_lowp` | 34% | 23% | 8% |

Edge handling is a real cost where the hardware fuses, so a way to skip it is justified. A
const generic is not: there are two useful levels, full and none. Finer gates pay off
in zenmetrics-math's fdlibm ports, where each bit guards a separate expensive path.
Even there the kernel gains are smaller than the savings here (f32x16 `EXACT` 2.41 ns
against `POSITIVE_NORMAL` 2.22 ns per element), though that crate reports about 21% in
one in-context case, so measure in the caller before relying on these. A two-valued const
generic is a worse spelling of a second function: it complicates `attuned!` and fn-pointer
use and buys no shared code the second function doesn't already get by calling the first.

So: `kernels::*_unchecked`, the suffix magetypes already uses, at the kernel level only.
A hot loop with known-valid inputs is already composing kernels in its own `#[attune]`
body, so the slice and array layers don't get twins, which also keeps their compile cost
down. In the stated domain `_unchecked` returns the checked kernel's bits on every tier
(`unchecked_kernels_match_mirrors_in_domain`); outside it the result is unspecified but
never undefined behavior.

`exp2_midp` does not clamp. Lanes a clamp would change are overwritten by the
underflow and overflow blends regardless, and without the clamp NaN reaches the result
through `xf = x - xi` on every ISA at no cost: 35 instructions on v3 against the relaxed
method's 37. `exp2_lowp` can't do that, because a clamp is its specified behavior, so
it clamps with comparisons, which keep NaN.

## Slice tails

Whole vectors and the tail share one loop and one inlined kernel; the tail runs in a
padded vector. The alternatives were a separate tail, which doubles the code in each body
(thousands of instructions on the scalar tier), and a tail through the scalar mirror,
which is a second code path for one operation, and slower in most cases:

| `exp2_midp`, ns per call on Ryzen 9 5900XT | one loop, pad (shipped) | loop then pad | loop then mirror |
|---:|---:|---:|---:|
| length 1 | 21.2 | 20.3 | 16.1 |
| length 7 | 23.4 | 23.5 | 68.5 |
| length 8 | 8.67 | 7.66 | 8.08 |
| length 9 | 18.4 | 30.5 | 20.5 |
| length 15 | 18.8 | 30.6 | 73.9 |
| length 1031 | 271 | 288 | 336 |

The shipped shape loses at length 1 and at exact multiples of 8 by about a nanosecond, and
wins or ties otherwise. This also means `magemath` needs `zenmath` only as a test oracle.

## Compile time

| | cold, including dependencies |
|---|---:|
| `zenmath` check / build --release | 0.13 s / 0.14 s |
| `magetypes` check, no `w512` / default features | 3.49 s / 3.83 s |
| `magemath` check / build --release | 3.62 s / 3.84 s |
| `magemath` build --release with non-inline dispatchers | 4.89 s |
| `magemath` check / build --release, `avx512` | 4.57 s / 5.03 s |

| a downstream crate calling | rebuild of that crate alone | stripped binary |
|---|---:|---:|
| nothing | 0.14 s | 363,128 B |
| one slice op | 0.22 s | 376,544 B |
| all 21 slice ops | 1.13 s | 493,560 B |

`magemath` adds 0.13 s of `cargo check` to magetypes, and leaving off `w512` saves more than
that (0.34 s). Everything `#[attune]` generates here is `#[inline]`, so `magemath` emits no
machine code of its own and a crate pays LLVM only for the ops it calls. Non-inline
dispatchers cost every dependent build another 1.05 s, mostly for the scalar tier: on x86-64
with `avx512`, that tier was 94% of the crate's machine code (96,483 of 102,449
instructions), and it runs only on CPUs without V3. No feature gates for lowp or the
activations: with inlining they cost nothing until called, so a gate would save only a
fraction of that 0.13 s.

## Register residency

Checked in optimized assembly on 2026-11-05 (`cargo rustc --release -- --emit=asm`,
`codegen-units=1`), for a function that loads `&[f32; 8]`, runs the kernels and stores:

| v3 function | instructions | `vfmadd` | non-constant memory ops | stack refs | calls |
|---|---:|---:|---:|---:|---:|
| `exp2_midp` | 35 | 6 | 2 | 0 | 0 |
| `f32x8::exp2_midp` (relaxed) | 37 | 6 | 2 | 0 | 0 |
| `exp2_midp_unchecked` | 26 | 6 | 2 | 0 | 0 |
| `log2_midp` | 38 | 4 | 2 | 0 | 0 |
| `f32x8::log2_midp` (relaxed) | 38 | 4 | 2 | 0 | 0 |
| `log2_midp_unchecked` | 27 | 4 | 2 | 0 | 0 |
| `exp2_midp` then `log2_midp` | 66 | 10 | 2 | 0 | 0 |
| `pow_midp` then `pow_midp` | 121 | 20 | 2 | 0 | 0 |
| `array::exp2_midp_f32x8_v3` then `array::log2_midp_f32x8_v3` | 67 | 10 | 2 | 0 | 0 |

The two memory operations are the one load and one store. Each chain is one region
with nothing spilled. NEON looks the same, at twice the width: `exp2_midp` is 76
instructions with 12 `fmla` and one `ldp` and `stp`; `exp2_midp` then `log2_midp` is
141 with 20.

So chain `kernels` from one `#[attune]` body, or call direct variants from a matching
context. A runtime-dispatched op can't share a region with its neighbors, so each
call through a dispatcher reloads and stores.

## Goldens

One canonical set, not one per tier: `portable` promises one answer, and per-tier files
would record a divergence instead of failing on it. `native` kernels, when they exist,
get per-tier goldens because that is their contract.

| layer | what it catches |
|---|---|
| `tests/parity.rs` | any tier differing from the mirrors, by input, with no tolerance |
| `tests/golden.rs`, `edges.txt` | a special-case result moving, readable in the diff |
| `tests/golden.rs`, `digests.txt` | any of about 230,000 results per op moving, across commits and architectures |
| `tests/reference.rs` | accuracy past the published maxima, against an f64 reference |
| `tests/exhaustive.rs` | any of the 2^32 inputs differing (manual; minutes on many cores) |

The first draft checked in 972 KB of `input output` hex lines for 4,853 inputs per op.
Digests cover 47 times the inputs in 814 bytes and need no filesystem access under
wasmtime. A digest can't show which input moved, but `parity.rs` names the input for
any tier difference, so a digest failure alone means the mirror itself moved. Check out
the previous commit's mirror if the edge table doesn't explain it.

## Polynomial fits and range-specific kernels

None yet, deliberately. A new polynomial needs a caller, a named input range, and a
measured gain over `_unchecked` on the same range, because `_unchecked` already takes
the cheap gains. The known weak spot isn't worth chasing: fixing `exp2_midp` on
[127.5, 128) measured 7-12% slower across the whole domain.

When one lands:

- **Separate functions, not const-generic degree or range counts.** A different degree is
  a different coefficient table and a different error contract. Const generics share no
  code there and hide which polynomial you got.
- **Share reduction, not polynomials.** Factor `#[inline(always)]` reductions (`x → (xi, xf)`,
  `x → (n, a)`) out of the existing kernels only when a second polynomial needs one.
- **Checked-in `const` coefficients, never a build-time fit.** A fit in `build.rs` costs
  every downstream build and makes the bits depend on the fitter's version. The fit
  script goes in the repo, and the constants get a comment with range, degree, measured
  error, date and script.
- **Tests.** Reference bounds per band; goldens if the kernel is `portable`, per tier
  if it is `native`.

Existing work to look at first: `linear-srgb` (rational polynomials and LUTs on [0, 1])
and zenjpegai's Cephes-derived `exp`.

## Not done

- `f32x4`, `f32x16`, f64 vectors: waiting on a caller. `kernels` is written for `f32x8`;
  other widths would mean another copy of the bodies, or generating them.
- `zenmath::special` and `zenmath::exact` are empty placeholders.
- No consumer has migrated. zensr's `exp_midp_portable` is the nearest candidate, but it
  deliberately gives NaN the x86 answer (2^-126) so its model goldens hold; moving it to
  `exp_midp` changes that lane's result to NaN.
- CI does not run these crates yet. Before landing, add `just test-magemath`,
  `test-magemath-wasm`, `test-magemath-no-fma` and a native ARM run to `ci.yml`,
  and exercise `windows-11-arm`, macOS ARM64 and i686.
- Timing is not measured on Zen 5, on any AVX-512 tier, or on a WASM engine other than
  wasmtime. So whether `+v4x` earns its place for `f32x8` is unmeasured: an earlier
  revision of `exp2_midp` was 32 instructions there against 40 on v3, and the tier stays
  behind the `avx512` feature until someone times it. Miri has not been run.
- `zenmath` without `std` runs `fmaf` in software on every target. Its midp functions
  took 4.5-8.5x the time of the scalar-tier vector kernel on aarch64, where `std`'s
  `mul_add` is one instruction. A scalar-only consumer on FMA hardware should try the
  `std` feature; that path's bits are verified, its speed is not measured.
