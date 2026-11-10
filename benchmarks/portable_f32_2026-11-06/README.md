# Portable f32 math: measured candidate study

Missing: actual Windows/macOS runs and a new global error proof. Windows execution
through the available WSL host failed with interop timeouts; no mac connection was
attempted. Results cover Linux x86-64 baseline/global-V3, the same baseline binary
under no-FMA Nehalem emulation, real Linux AArch64, and strict-SIMD WASM. No production
Archmage/Magetypes/Zenmetrics API or implementation changed. This workspace is unpublished.

The fastest observed accurate choice depends on operation and target. Fpmath is a
credible scalar alternative; CORE Rust kernels retain correctly rounded results in
all checks here. A reused ARM fast-pow path with CORE edge fallbacks approaches std
throughput while retaining identical tested output bits. It is a prototype, not a
newly proven global bound. See [candidate research](RESEARCH.md) for primary sources,
coverage, guarantees and omissions. There is no universal-fastest claim.

The subsequent [complete available-API matrix](../math_matrix_2026-11-06/README.md)
extends these experiments to the requested scalar inventory, existing vector
widths and algebraic kernels, with explicit unavailable and failed cells.

## Coverage of the requested scalar inventory

This is not an all-operations/all-widths benchmark matrix. Of the requested nine
f32 operations plus f64 log2, six have zenmath-versus-std interleaved measurements:

| Requested operations | Scalar vs std | Multiple libraries + packed implementations |
|---|---|---|
| f32 exp, ln, log2, powf | This study and the earlier scalar study | This study; fixed-array/slice kernels, not every vector width/tier |
| f32 log10; f64 log2 | [33-function scalar study](../core_math_rust_2026-11-05/README.md) | Not covered here |
| f32 sqrt, floor, ceil, round | Not measured for zenmath by these studies | Not covered here |

The older [Magetypes fallback benchmark](../../magetypes/benches/nostd_math_perf.rs)
sets up zenbench groups for its own scalar sqrt/floor/round versus std (and FMA),
but does not benchmark zenmath, ceil, or packed versions. It is not a substitute
for the missing comparisons. The [power-policy study](../pow_behavior_2026-11-06/README.md)
adds same-contract scalar/packed policy comparisons and a frozen sign-path
baseline; its correctness matrix is broader than its performance matrix.

## Throughput, ns/value

All candidates receive the same 1024-value arrays, copy, and black-box consumption.
Exp: `-8+16*t`; logs/pow: `.01+4*t`, with `t=(i*719%1024)/1024`. Fixed pow uses2.4;
general pow uses `-4+8*(i*313%1024)/1024`. Dispatch is once per buffer. Allocation
and input generation are outside native timed iterations. Baseline x86-64 builds;
no `target-cpu=native`. The V3 build below is a separate correctness control.

Core Ultra7 265K, CPU0 within enforced performance cpuset0–7:

| Candidate | exp | ln | log2 | pow(x,2.4) | pow(x,y) |
|---|---:|---:|---:|---:|---:|
| Rust std |1.201|1.460|1.209|2.757|2.736|
| CORE Rust scalar |1.689|2.046|1.405|18.111|19.161|
| CORE Rust vector |1.102|1.049|1.303|7.523|7.504|
| fpmath0.2.0 |1.150|1.653|1.994|5.431|6.861|
| Rust libm0.2.16 |2.219|1.930|2.426|16.574|16.804|
| metallic0.3.1 |19.225|11.365|10.711|35.187|38.212|
| micromath2.1.0 |3.647|0.561|0.579|8.960|7.658|
| magetypes midp |0.476|0.414|0.419|0.944|6.661†|
| magetypes lowp |0.367|0.344|0.322|0.720|2.168†|
| ARM fastpath + CORE scalar fallback |—|—|—|3.132|3.060|

Neoverse-N1, quiet timing machine, CPU0:

| Candidate | exp | ln | log2 | pow(x,2.4) | pow(x,y) |
|---|---:|---:|---:|---:|---:|
| Rust std |5.862|6.130|6.392|10.431|10.506|
| CORE Rust scalar |7.140|7.970|5.976|29.873|30.947|
| CORE Rust vector |5.507|5.305|5.252|19.288|16.940|
| fpmath0.2.0 |4.760|6.546|6.882|22.015|22.888|
| Rust libm0.2.16 |12.436|8.284|10.876|68.559|68.470|
| metallic0.3.1 |8.012|8.347|6.298|27.619|28.406|
| micromath2.1.0 |16.172|1.786|1.860|27.861|31.603|
| magetypes midp |2.232|2.232|2.140|4.960|18.694†|
| magetypes lowp |1.678|2.091|1.728|3.609|9.564†|
| ARM fastpath + CORE scalar fallback |—|—|—|11.322|11.771|

Wasmtime49.0.5 on the Core Ultra7 265K, strict SIMD, CPU0:

| Candidate | exp | ln | log2 | pow(x,2.4) | pow(x,y) |
|---|---:|---:|---:|---:|---:|
| Rust std |1.581|1.740|1.809|3.830|3.650|
| CORE Rust scalar |1.747|1.779|1.781|21.491|21.327|
| CORE Rust vector |3.268|3.697|2.742|20.189|15.949|
| fpmath0.2.0 |1.382|1.807|1.841|8.221|8.090|
| Rust libm0.2.16 |2.490|1.239|1.370|19.760|19.969|
| metallic0.3.1 |98.499|77.522|72.078|165.934|176.979|
| micromath2.1.0 |6.916|0.651|1.999|13.164|9.414|
| magetypes midp |1.693|0.963|0.886|3.127|11.124†|
| magetypes lowp |1.047|0.846|0.868|2.020|5.460†|
| ARM fastpath + CORE scalar fallback |—|—|—|3.662|3.817|

† General magetypes pow is an explicitly labeled API adapter: one splat/pow/extract
per lane, because the public approximation accepts a scalar exponent. It is not
optimized packed per-lane-exponent support. Fixed2.4 uses packed vectors; exponent
uniformity is checked once per buffer. `—` means missing method; no substitution.

Zenbench0.1.11 interleaves candidates with std controls (30–60rounds,100ms warmup,
2s/group limit). x86 pow rows use the final prototype/control run; unary rows use
the earlier run. All raw rounds, paired statistics and MADs are retained; these are
exploratory means. Near ties and noisy/drifting rounds do not establish stable
rankings. No rerun was made merely to improve noise.

WASM uses zenbench's in-process Wasmtime API, not a subprocess timer. Each host call
performs256 copies and the same1024-value kernel; initialization and host-call cost
are included and amortized over those256 kernels. Module LTO is off with
`link-dead-code`/`--export-all` to retain a safe Rust extern-C export; native LTO is
thin. The module and host binaries, flags and hashes are retained. Do not infer a
cross-runtime speed ratio from these differently compiled programs.

## Precision: full range and ordinary inputs

The common corpus has275909exp,204353ln,204353log2 and791134pow records. It includes
200000independent directed-MPFR records per operation,460544CORE-MATH pow hard cases,
plus power-of-two/one/subnormal/overflow/underflow neighborhoods and special values.
The complete input/output files are retained, with identical input hashes per target.

Correct percentages are dataset counts, not workload probabilities: hard cases and
random bit patterns are deliberately overrepresented. [accuracy-x86.csv](results/accuracy-x86.csv)
includes overall correctness, finite_pairs/finite_wrong, special mismatches,
signed-zero mismatches, and the100000ordinary random-domain records per operation.
Finite-pair percentages exclude classification failures, which must be read alongside
them. A finite-to-infinity failure is not acceptable merely because the remaining
finite pairs are accurate.

Maximum observed **true-real** ULP over the main corpus and targeted search, x86:

| Candidate | exp | ln | log2 | pow | Special/classification result |
|---|---:|---:|---:|---:|---|
| CORE Rust scalar/vector |0.500000|0.500000|0.500000|0.500000*|matches every CR reference|
| fpmath |0.500000|0.500000|0.500008|0.500001|28pow sNaN precedence differences|
| Rust std |0.501321|0.813981|0.751762|0.653795|WASM differs on sNaN precedence|
| Rust libm |0.844811|0.766996|0.827158|0.846035|one finite pow becomes infinity, plus28sNaN differences|
| metallic |0.500000|0.500000|0.500000|0.500000*|WASM pow traps on a quiet-NaN case|
| ARM fastpath/CORE prototype |—|—|—|0.653795|matches tested specials|

* CORE/metallic pow upper0.5000000000000002 is outward arithmetic slack at halfway
cases, not a misrounding. CORE scalar/vector have zero wrong reference bits on all
main and retained search inputs. Fpmath main-corpus finite pow has34848wrong/689599
finite pairs, concentrated in hard cases; its100000ordinary pow records have zero
wrong bits. Search finds log2 input`3f7e5b9e` with0.500007955trueULP: fpmath is not
correctly rounded, while still meeting the1.0trueULP working target on tested data.

The working screen is at most1representable step from the correctly rounded
reference; that is **not** a1.0trueULP bound. `mpfr.py` uses directed MPFR at128bits,
increasing precision until both endpoints round to the same f32. Error divides
absolute distance to the real result by its binade's f32 spacing, floored at2^-149.
Directed double endpoint conversion plus outward-nextafter subtraction/division
conservatively brackets error even for bad candidates. Nonfinite answers and
signed zeros are separate checks; NaN payload/sign are excluded from bit identity.

The full-range approximation failures are not their ordinary-domain error bounds.
On the100000ordinary inputs (exp`-8+16*t`; logs/pow base`.001+16*t`, pow exponent
`-8+16*u`), the observed errors are:

| Approximation | exp maximum relative error | ln maximum absolute error | log2 maximum absolute error | pow maximum relative error |
|---|---:|---:|---:|---:|
| micromath |0.00238656|0.1177665|0.1699011|1.5559542|
| magetypes midp |5.33331e-7|3.80420e-7|4.48825e-7|2.31566e-6|
| magetypes lowp |0.00556074|2.09007e-6|2.81140e-6|0.00557112|

These are fractions, not percentages. ln/log2 relative error is ill-conditioned
near1; absolute error is more informative for approximations. Their ordinary-domain
trueULP maxima remain in the CSV. None satisfies the full-range1trueULP target.

Two concrete failures must remain distinct from policy:

- Rust libm pow:`x=3f7ffff7,y=cd048f11`; CR/std/fpmath/CORE=`753e803a` finite,
  libm=`7f800000` infinity. This is its29thspecial mismatch, not sNaN handling.
- Metallic WASM pow:`x=00000001,y=7fc00000` (minimum subnormal, quietNaN) traps
  with invalid float-to-integer conversion in`pow::magnitude`. A20-step bisection
  retains the one-record reproducer and failure log. Other candidates continued
  through explicit separate invocations; no failure was silently skipped.

Fpmath/metallic's28pow sNaN differences (`pow(sNaN,0)`/`pow(1,sNaN)`→1 versus
CORE/native std NaN) are a **policy choice**, not finite error and not automatic
disqualification for Rust-like precision. Raw results are retained. No wrapper or
production semantic change was introduced.

## Cross-target equality and adversarial search

Std results, same inputs, NaN payload/sign excluded:

| Comparison against native x86 | exp differences | ln | log2 | pow |
|---|---:|---:|---:|---:|
| Exact same binary, QEMU Nehalem/no FMA |0|0|0|0|
| Separate global x86-64-v3 build |0|0|0|0|
| Real AArch64 Linux |0|0|0|0|
| WASM/Wasmtime40.0.1 |0|0|0|28sNaN policy cases|

Thus this study found no finite std cross-target difference in these environments.
There is no finite counterexample to report. One policy counterexample is
`pow(7f800001,00000000)`: native=`7fc00001`, WASM=`3f800000`.
[Pairwise counts, max steps and examples](results/std-consistency.csv) are explicit.
Std documents unspecified precision; these observations do not establish a Windows,
macOS, future-compiler or future-libc bit contract. ISA/OS/libc/compiler are
confounded except for the same-binary native/Nehalem test.

CORE scalar/vector, fpmath, libm and the pow prototype have equal tested output bits
on native/ARM/WASM (NaN payload/sign excluded). Force-soft-floats libm was separately
built on x86 and did not change these results; it is not an accuracy proof or a
repair for the pow overflow. [consistency.csv](results/consistency.csv) also retains
backend-dependent approximation differences, max separation and examples.

The upstream maximum-error search source is math_accuracy commit`c2b897fb`.
`search.py` independently adapts its max-mode split/sample/retain/exhaust algorithm:
beam12,64samples/cell, exhaustive terminal volume128, seed0x20261106. It evaluated
76001exp,90324ln,85975log2 and195569pow inputs, searching micromath/libm/fpmath/std/
metallic/CORE and the pow prototype. Search maxima are lower bounds on global worst
error, not proofs. The upstream CORE-MATH hard-to-round corpus has a different
purpose: answers near rounding boundaries, not necessarily largest implementation error.

The16unique retained worst inputs (4/op) were **replayed on native/real ARM/WASM**.
Std, CORE, fpmath, libm and the prototype agree bitwise across those replays; the
metallic retained finite pow cases also execute successfully. Approximation
rounding differences remain (lowp exp and midp log2/pow). The compact
[input corpus](results/retained-inputs.csv), [search maxima](results/search-maxima.csv)
and [replay comparisons](results/replay-consistency.csv) are committed as text.

## Recommendation and limits

For documented near-half-ULP scalar accuracy, fpmath is the best supported new
candidate examined here; exp is particularly competitive. For correctly rounded
results, retain CORE. The measured native per-operation route is fpmath/CORE vector
exp depending on target, CORE vector ln/log2, and the ARM fastpath/CORE pow prototype
when the observed1trueULP contract is acceptable. The prototype is slower than std
on native x86/ARM, but materially closer than CORE or fpmath on these workloads.

On WASM, libm ln/log2 are the fastest observed sub1trueULP candidates; their measured
accuracy is not a newly proven global bound. CORE scalar is a stronger-rounding
alternative, near std's measured cost. Fpmath exp and the pow prototype are the
other measured shortlist. Std's finite outputs agree here, but a pinned Rust
implementation is a clearer future portability contract than relying on host std.

Upstream ARM documents0.82ULP for its fast-pow algorithm. That supports investigating
this reuse; it is not a proof automatically inherited by our adapted fallback
composition. Full-range samples, hard cases and worst-search replays support the
prototype; a proof or exhaustive relevant-domain analysis is still absent. No
universal fastest or globally proven1.0trueULP implementation is established by
this study. The optional PowBehavior design proposal did not alter this matrix.

## Provenance, checks and reproduction

Source base`8eacda2f`, unpublished draft. Candidate harness`1f60c7f9`, pow prototype
`d56d0125`, error/search`6425585e`, native/WASM kernels`9c66f530`, replay metrics
`777937c2`. Later host-tool guards do not change candidate arithmetic. The pow
prototype copies the scalar algorithm/tables from zenmetrics-math`glibc.rs` at
`097b5e4de9f8e5322b0f0abd4efb2b7055e05931`, replaces **all** host fallbacks with
zenmath CORE, and retains [ARM licensing](LICENSE-ARM). It is named as an adaptation,
not glibc compatibility. CORE-MATH checkout pin verified:`660067dfd3e751c8f07acc31c899ff410aa7206e`.

x86/WASM compiler Rust1.99.1/LLVM23.1.1; native Ubuntu26.04.1,glibc2.43-2ubuntu2.4.
ARM binaries built Rust1.97.1/LLVM22.1.6 and run Ubuntu24.04.5,glibc2.39-0ubuntu8.9.
The quiet timing host's installed Rust1.96 was not used to build those binaries.
Accuracy engine: Wasmtime40.0.1; in-process timing engine:49.0.5, pinned Cargo.lock.

Startup generation, registry/token validation and soundness checks completed on
an eligible remote host. Harness fmt/clippy passed on native and real ARM; final
all-target clippy also passed on native and wasm32 with wasm-bench enabled and
warnings denied. Direct target builds, independent MPFR validation, no-FMA run,
and retained-worst replay completed. No existing test expectation was relaxed.
No push, PR, release, shared bookmark change, primary checkout edit or dev heavy job.

Run the independent workspace on eligible hosts under`run-heavy` and`~/tmp/heavy.lock`:

```sh
cd benchmarks/portable_f32_2026-11-06
just build
just lint
just results exp RECORDS OUTPUT_DIRECTORY
just candidate pow RECORDS OUTPUT_DIRECTORY arm_pow_core_fallback
just analyze ARTIFACT_ROOT
just search SEARCH_OUTPUT
just wasm-kernel
just wasm-host
just summary ARTIFACT_ROOT
```

The corpus format is three little-endian u32 words: xbits,ybits,correctly-rounded
result bits. Use the retained original files to seed a **fresh** `artifacts/corpus`
before `run-targets.sh`; `boundaries.py` refuses accidental repeated appends. Native
results and Nehalem use the exact same binary; its hash is in`binary-native.sha256`.
The WASM-host executable takes`MODULE OUTPUT_JSON`; the output is normalized by
1024×256values. Export build flags are in the independent justfile.

Full inputs/outputs, raw rounds, search levels, logs, binaries, identities and hashes:
`/mnt/v/output/archmage/portable-f32-2026-11-06/`, mirrored at
`/mnt/tower/output/archmage/portable-f32-2026-11-06/`. `SHA256SUMS` records file identity.

Resource wrapper completion lines (full logs retained):

```text
x86 matrix: rc=0 94s | peak-RSS 0.53GiB | min-avail 22007MiB | peak-load 1.49
full matrix + search: rc=0 115s | peak-RSS 0.60GiB | min-avail 22430MiB | peak-load 0.95
ARM hybrid build: rc=0 8s | peak-RSS 0.33GiB | min-avail 27429MiB | peak-load 6.82
ARM timing: rc=0 11s | peak-RSS 0.01GiB | min-avail 14891MiB | peak-load 0.15
x86 final pow: rc=0 5s | peak-RSS 0.02GiB | min-avail 23000MiB | peak-load 0.32
WASM host build: rc=0 85s | peak-RSS 1.54GiB | min-avail 20541MiB | peak-load 2.75
WASM timing: rc=0 16s | peak-RSS 0.32GiB | min-avail 22562MiB | peak-load 1.50
final error + replay analysis: rc=0 112s | peak-RSS 0.51GiB | min-avail 19072MiB | peak-load 2.90
final native/WASM lint: rc=0 4s | peak-RSS 0.34GiB | min-avail 22862MiB | peak-load 0.64
```
