# magemath design record

The [rounding comparison](../../benchmarks/magemath_rounding_cost_2026-11-05/README.md)
records the current three-surface measurements and validation. The earlier
[portable-only record](../../benchmarks/magemath_portable_cost_2026-11-05/README.md)
contains the slice-tail experiment. Its “relaxed” column means magetypes methods;
the current record calls those `method` and uses `relaxed` for this crate's surface.

## The contract

`magemath::portable` returns the same bits on every tier, and those bits equal the
`zenmath::scalar` functions. NaN results agree only in being NaN.

magetypes' `f32x8` methods can't promise this, on purpose: `mul_add_relaxed` rounds
once where the hardware fuses and twice on the scalar backend and strict WASM,
because the fused form made those backends 1.5-23x slower
([#116 record](../../benchmarks/transcendentals_fused_mul_add_2026-10-05.md)). So
callers who need matching bits and callers who need the fastest result per tier want
different contracts. Magemath exposes both through `portable` and `relaxed`.

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

Hence `portable::kernels`: the same polynomials as magetypes, every step
`mul_add_fused`, and NaN handled without relying on anything ISA-specific. On x86 and
NEON `mul_add_fused` is the same instruction as `mul_add_relaxed`, and the parity tests compare in-domain results with the magetypes methods.

Where the hardware can't fuse, `mul_add_fused` picks a path, and every path gives the same
bits. Relaxed WASM uses the engine's madd once a cached probe shows it fuses, and
otherwise software. f64 on x86-64 with std checks a cached V3 token. f32 outside a
matching tier goes straight to the vectorized software path, with no runtime CPU check,
because a per-call hardware path measured slower (repo `CLAUDE.md`, Dispatch granularity).
That is only reached on CPUs without V3 anyway: the dispatcher picks the v3 tier first.

The original portable-only revision was checked for every f32: all 2^32 inputs matched the mirrors on scalar, v3, v4x and
NEON, and about 230,000 per op matched on WASM (strict, relaxed on engines that fuse
and engines that don't), qemu without FMA, and i686.

## What it costs

Separate multiply/add trades native polynomial throughput for removal of software
FMA. Unfused time relative to fused, paired 95% intervals across the five streaming
kernels (exp2/log2 midp and lowp, plus pow_midp(2.4)):

| backend | streaming interval endpoints across operations | exp2 feedback |
|---|---:|---:|
| Ryzen 5900XT V3 | +16.6% to +51.9% | +29.2% to +35.5% |
| Core Ultra 265K V3 | +11.5% to +33.2% | +9.4% to +11.1% |
| Neoverse-N1 NEON | +7.1% to +21.7% | +13.3% to +14.4% |
| Ryzen 5900XT scalar | −76.5% to −50.6% | −48.0% to −40.9% |
| Core Ultra 265K scalar | −79.7% to −50.5% | −50.6% to −49.0% |
| Neoverse-N1 scalar | −3.9% to +3.7% | +1.9% to +2.4% |

The scalar vector backend fuses in software on baseline x86 but in hardware on
std-enabled AArch64. Zenmath without std uses software FMA on both. Its measured
per-lane streaming costs, raw data, conditions and resource summaries are in the
[comparison](../../benchmarks/magemath_rounding_cost_2026-11-05/README.md).
These are workload-specific measurements; `relaxed` names a rounding contract,
not a guarantee that it beats every alternative in every kernel.

## Rounding and accuracy

Accuracy bounds are unchanged across all three surfaces. Changing rounding is
still a bit-contract change: the exhaustive x86 scan found up to 2–3 ULP between
midp logs, 178 ULP for pow_midp(2.4), and 93,472 ULP for pow_lowp(2.4), over all
input patterns. The lowp polynomial has interval-boundary steps; ULP distance
between two approximations is not the same as relative error against the truth.
No tolerance was added to mirror comparisons or golden tests.

## Module boundary

| layer | bits across tiers | when to use it |
|---|---|---|
| `portable` | identical | default; anything that must reproduce across machines |
| `relaxed` | backend-dependent rounding; NaN inputs remain NaN | callers that accept rounding differences |
| `unfused` (prototype) | identical, two roundings per polynomial step | evaluate the alternative portable contract |
| magetypes `f32x8` methods | backend-dependent rounding and specials | existing method API |
| `native` (empty) | may differ by design | a measured need neither of the above meets |

Same op names in different modules, so the import path states the contract. The
private `Step` trait shares polynomial bodies across `Fused`, `Relaxed` and
`Unfused`; it is not exposed in public signatures. Public kernels remain generic
only over the vector backend. A public policy parameter is unnecessary here.

`portable` currently stays fused, matching `zenmath::scalar`. `relaxed` uses
`mul_add_relaxed`: V3/V4x and NEON round once, scalar and strict WASM round twice,
and relaxed WASM follows the engine. The `unfused` experiment uses `a * b + c`
and mirrors `zenmath::scalar::unfused`. The owner chooses whether that becomes
portable's contract. Such a change also changes the scalar mirrors and golden
bits; it is not an implementation-only optimization. No choice has been made.

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

Core Ultra 7 265K, Rust 1.99.0, medians of three cold builds including dependencies:

| public surfaces | check | release build |
|---|---:|---:|
| portable only | 2.49 s | 2.63 s |
| portable + relaxed | 2.56 s | 2.72 s |
| portable + relaxed + unfused prototype | 2.65 s | 2.82 s |
| all three with avx512 | 3.23 s | 3.54 s |

Relaxed adds 0.07 s / 0.09 s; the unfused prototype adds 0.09 s / 0.10 s.
Non-inline dispatchers take 3.68 s for the all-surface release build. A downstream
caller rebuilds in 0.08 s with no calls, 0.16 s with one portable slice op, and
0.84 s with all 21 portable operations. Its stripped binaries are 363,128,
376,544 and 493,528 bytes respectively. The [measurement record](../../benchmarks/magemath_rounding_cost_2026-11-05/README.md#compile-time)
has all repetitions, command and resource summary.

The private Step policy shares bodies; public wrappers add some type checking
and macro expansion. Inline dispatchers keep unused operations out of the caller's
machine code. Feature gates for individual precision levels are not justified by
these costs. Keep the public rounding contract in the module path.

## Register residency

The current V3 and AArch64 assembly probes confirm separate multiply/add in
unfused kernels: no FMA instructions. Exp2 uses 35 instructions fused versus 41
unfused on V3, and 76 versus 81 on NEON. An exp2→log2 chain uses 66 versus 76
on V3 and 141 versus 152 on NEON. Each has one load/store, zero stack references,
and zero calls. Counts, commands and artifacts are in the [assembly record](../../benchmarks/magemath_rounding_cost_2026-11-05/README.md#assembly).

Chain `kernels` in one `#[attune]` body, or call direct variants from a matching
feature context. A runtime dispatcher is a separate entry point, so composing
several dispatched array calls need not retain registers across them.

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
  other widths need a reviewed extension of the shared bodies.
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
