# Powf provider comparisons

**Promotion remains held.** No external provider was added to production.
The existing opt-in policy API and unchanged legacy methods are documented in
[POLICIES.md](../powf_module_2026-11-07/POLICIES.md). This study adds explicit
std, Rust libm, Rust SLEEF, GNU libmvec, Arm and AOCL comparisons. A separate
[glibc source audit and CORE-MATH comparison](GLIBC-HISTORY.md) identifies the
actual algorithm families, runtime dispatch and hard-to-round witnesses.

## Implementations and coverage

| Implementation | Frozen source / actual interface |
|---|---|
| Rust std | f32 `powf`; compiler/platform identity retained |
| Rust libm | 0.2.16, default `arch` feature, `libm::powf` |
| Rust SLEEF | 0.3.3, scalar/vector `pow_u10` and `pow_fast`, default features |
| GNU libmvec | glibc 2.43, `_ZGVbN4vv_powf` and `_ZGVdN8vv_powf` |
| Arm Optimized Routines | v26.07 / 4be260a5, scalar `powf`, Linux AdvSIMD `_ZGVnN4vv_powf` |
| AOCL-LibM OSE | 5.3.2 / 29fd054f, `amd_powf`, `amd_vrs4_powf`, `amd_vrs8_powf`; separate libalmfast scalar symbol |

The libmvec labels describe **vector ABIs**, not a promise that an IFUNC uses
only those instructions. Runtime-dispatched implementations may upgrade.
Logical x16 with an x8 ABI means two x8 calls. AOCL uses its default AVX2 path
on Intel and its CPU dispatcher on the measured Ryzen 9 5900XT. AVX-512 was
not invoked or emulated for timing. No Zen 5 speed claim is available.

Rust SLEEF requires portable SIMD: this standalone, unpublished package uses
nightly-2026-10-06 for every Rust batch-comparison arm. Baseline and explicit
`+avx2,+fma` worker binaries are separate, hashed artifacts. These flags are a
combined feature context; their comparison does not isolate FMA alone.
The production workspace retains stable Rust and has no SLEEF dependency.

Four physical platforms supplied 234 provider/width executions of the unchanged
27,909 domain-tagged pairs, with both broadcasts and mixed lanes. The 32 ordinary
workload traces also agree with the corresponding repeated batch's last lanes:
528 Intel, 528 AMD, 256 M6 and 280 Neoverse provider/workload combinations.
The first Neoverse Rust worker inherited `-C target-cpu=neoverse-n1`; its timing
pilot is CPU-targeted evidence, not a generic distribution baseline. Its data
is retained. A separate numerical-only generic build with actual Cargo flags
`[]` adds 35 executions: all raw outputs and complete trajectories are
byte-identical to the targeted build. No timing was rerun. The other baseline
workers' retained fingerprints have flags `[]`. Provider numerical differences below
are retained as observations, not accepted mathematical behavior.

The separate stable-Rust std/libm study covers twelve execution contexts,
including baseline/matching-feature Intel and M6, i686, no-FMA emulation, strict
WASM, and relaxed WASM with both fusion selections. Rust libm had zero non-NaN
bit or NaN-class differences across those samples. Payload differences remain
in the ledger. Std differs across platforms: M6 versus Linux had 122 non-NaN
bit differences and eight NaN-class differences; WASM versus Linux had eight
NaN-class differences. None of this certifies a whole-domain portability bound.

## Numerical tradeoffs

MPFR-256 supplies independent directed reference enclosures. The complete
accuracy reference covers positive finite bases and finite exponents; Gamma
zero inputs are checked against exact 0/1 results. Other negative/special
inputs retain observed bits and std comparisons without a new accuracy claim.
The production BitExact policy's correctness reference remains its independent
exact-rational graph oracle; std and the libraries are not its mathematical truth.
The complete
inventory retains every provider's bits, broadcast/mixed differences, NaN
payloads, feature-context and platform differences, and rounded MPFR results.
It covers 21,003 differing input records across 234 provider/width executions;
the uncompressed ledger is 447,102,610 bytes, below its 512 MiB budget.
Raw outputs retain all ties as well. See [ARTIFACTS.md](ARTIFACTS.md).

Gamma statistics use the unchanged independent domain oracle: +0 or normal
bases in [2^-16,2^16], normal-or-zero exponents in [0,4]. Each execution has
8,660 positive-input and 38 positive-zero input records. These are samples,
not exhaustive binary-function coverage. The first-pass reports used a broader
Gamma filter that included subnormal exponents; the complete inventory corrects
that filter. Positive-input ULP and exact zero behavior are reported separately.

Selected sampled maxima, positive Gamma inputs, logical x8:

| Provider / execution | ULP upper bound observed | Important wider-domain behavior |
|---|---:|---|
| std / M6 | 0.508686495945 | Platform-dependent bits and signaling-NaN cases |
| Rust libm 0.2.16 / M6 | 0.802617862821 | Incorrect overflow on one unique wider-range pair |
| SLEEF scalar/vector u10 / M6 | 0.572962813079 | Vector u10 incorrectly overflows on wider-range inputs |
| SLEEF scalar/vector fast / M6 | 58.682476531715 | Wider-range class and large-error differences |
| Arm v26.07 scalar / M6 | 0.501247502863 | Cross-platform sample agreement does not prove bit identity |
| AOCL scalar / AMD | 0.500478085130 | Not assumed canonical across CPUs/builds |
| AOCL vector x4/x8 / AMD | 2.159301763400 | Gamma zero inputs and wider-range/subnormal differences |

Concrete witnesses:

- Rust libm: `3f7ffff6 ^ cd000001` returns `7f800000`, while MPFR rounds
  to finite `792abd43`. The same pair appears under two domain tags and lies
  outside Gamma. Std matches MPFR in the measured Intel/M6 cases.
- SLEEF vector u10: `7f7fffff ^ 3f800000` returns infinity rather than
  f32 MAX. M6 has 54 domain-tagged nonfinite mismatches to finite MPFR results.
  Its scalar u10 path does not share those range mismatches on this corpus.
- AOCL vector x4/x8 on AMD: `+0 ^ 0.5` returns `1fb504f2` rather than +0;
  `+0 ^ +0` returns `3f800001` rather than 1. Rejecting subnormals alone would
  not cover this Gamma-zero issue.
- AOCL x4 has 20 unique broadcast/mixed input differences per logical width
  on both x86 platforms: ten non-NaN bit differences and ten NaN-class
  differences. Full peers and outputs are preserved. Arm AdvSIMD's mixed-lane
  differences here are NaN payload differences, not non-NaN or class changes.

Finite-output ULP maxima never conceal incorrect infinity/finite or zero-class
results. The broad-corpus finite-output maxima include 0.660738430685 for SLEEF
u10 scalar, 0.613705696539 for x86 libmvec, and 1.815039556474 for Arm AdvSIMD.
No measured library becomes an arbitrary-input bit-exact backend through finite
startup probes. Scalar and vector functions require separate qualification.

## Timings and admission

The [frozen protocol](PLAN.md) specifies 50 retained interleaved rounds per cell.
The CXX compiler probe rejects the project's unsafe-code policy; the working
Rust harness retains `forbid(unsafe_code)` and uses persistent Unix-socket
workers. Native ABI functions run in the C++ worker. Thus the batch timings
include adapter work and one request/response; they are not single-call latency.
An adapter-matched system C powf and EMPTY requests expose those costs. EMPTY
performs zero math work and is never subtracted from the provider timings.

| Platform / study | Measured coverage | Qualification |
|---|---:|---|
| M6 direct std/libm | 56/56 cells | All resource-flagged; unrestricted placement |
| M6 external providers | 32/32 cells | All resource-flagged; unrestricted placement |
| Ryzen 9 5900XT external providers | 1/32 pilot | 38/50 clean pairs; unreliable; stopped |
| Neoverse-N1 external providers | 1/32 pilot | Unreliable; stopped |
| Intel external providers | 0/32 | Common admission timed out after 300 s behind a live reservation |
| Intel direct std/libm | 0/56 | Same live reservation prevents admission; not retried |

No dirty rounds were discarded, no busy gate was disabled, and no timing was
rerun until favorable. Qualified upper paired 95% <=1.02 remains the cost bound.
All primary ratios compare a named provider with the same cell's Rust std or
system C powf. Other pairwise comparisons are descriptive. **Zero cells qualify
an unqualified provider-promotion claim.**

M6, x8, 4096-value stream, opaque f32 2.4. Zenbench means in **ns per 1,048,576
actual evaluations**, including one request:

```text
SLEEF vector fast       ██          1,793,483 ns
Rust std loop           ██          2,122,773 ns
system C powf           ███         2,771,428 ns
Arm scalar              ███         2,840,881 ns
SLEEF vector u10        █████       5,301,991 ns
Rust libm               ██████████ 13,549,995 ns
SLEEF scalar u10        ███████████████ 20,433,802 ns
```

The SLEEF vector fast/std paired 95% ratio is 0.840862–0.848977; vector
u10/std is 2.491396–2.502921. Their accuracy/range contracts differ. System C
powf itself costs more than the Rust loop here, so adapter/call shape must not
be attributed entirely to the math algorithm.

AMD's flagged x8 pilot, same batch work: libmvec x8 3,420,858 ns, AOCL x4
3,229,143 ns, AOCL x8 3,597,884 ns, Rust std 4,420,349 ns, system C powf
6,012,202 ns. SLEEF vector u10 changes from 9,168,347 ns in the baseline
binary to 5,089,796 ns with AVX2+FMA; all-round descriptive ratio CI is
0.540669–0.573855. It also changes some result bits. These are exploratory
observations, not passing cost claims or evidence for a universal speed ladder.

Resource records: M6 batch matrix `189s | peak-RSS 0.02GiB | min-avail 9883MiB
| peak-load 2.83`; direct matrix `42s | peak-RSS 0.02GiB | min-avail 9933MiB
| peak-load 2.12`. AMD pilot `14s | peak-RSS 0.12GiB | min-avail 48335MiB
| peak-load 0.66`. Full per-run lines, gate observations and all pairs are in
the artifact. macOS RSS uses the wrapper's monitored process-tree guard, not
a cgroup hard-cap claim.

Native availability is explicit: this Arm release's standard vector build is
Linux-only, so M6 measures its scalar routine. The Neoverse host's glibc 2.39
libmvec has no `_ZGVnN4vv_powf` symbol; its symbol table is retained. AOCL is
measured on x86, not substituted under its name on ARM. External-library
exceptional/bit-distributed timing matrices and cold-start math costs remain
unmeasured. Library build logs are diagnostic, not matched cold-build claims.

## Const-policy compile and size diagnostic

[COLD.md](COLD.md) records twelve fresh-target builds. For the same client with
one additional Gamma MIDP call, the const-options version compiles in
16.54–17.08 s versus 27.03–27.55 s with opaque runtime options. GNU `size` text
is 281,218 versus 318,162 bytes. This is a concrete saving from monomorphization;
it does not establish a strict runtime ordering for every workload.

## Reproduction and sources

Use the `powf-external-*`, `powf-providers-*` and `powf-policy-cold-build`
recipes in the root justfile. Numerical runs and builds use the resource
wrapper; benchmark admission remains zenbench's responsibility. Frozen source,
locks, flags, compiler identities, DSO/binary hashes, failures and raw outputs
are retained independently of Cargo targets.

[Rust SLEEF 0.3.3](https://docs.rs/sleef/0.3.3/sleef/),
[Rust libm 0.2.16](https://docs.rs/libm/0.2.16/libm/fn.powf.html),
[Arm Optimized Routines](https://github.com/ARM-software/optimized-routines),
[AOCL-LibM](https://www.amd.com/en/developer/aocl/libm.html),
[AOCL OSE source](https://github.com/amd/aocl-libm-ose),
[GNU vector ABI](https://github.com/bminor/glibc/blob/master/sysdeps/unix/sysv/linux/x86_64/libmvec.abilist).
