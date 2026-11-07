# CORE-MATH compiler and portable-cost study

Measured 2026-11-05 on Core Ultra 7 265K (i265), pinned P-core CPU 2.
This is standalone research, not a shipping API or an exhaustive certification.
Missing: f64, native SIMD, ARM/WASM, other CPUs, dynamic rounding modes, errno,
exception flags, and exhaustive hard-case validation of the Rust translation.
No default implementation or compatibility contract changed.

## Sources and contracts

- [CORE-MATH GitLab](https://gitlab.inria.fr/core-math/core-math), full repository
  fetched; source `660067dfd3e751c8f07acc31c899ff410aa7206e`.
- [jdh8/core-math](https://github.com/jdh8/core-math), `a074c66b`, is a safe Rust
  wrapper around C, not a pure-Rust translation. Its sys build defaults to
  `-march=native`; that is not the policy used here. The sys vendor pointer
  inspected was `85430b3ef7d2beb0de16d7c569ffa9bb391615ed`.
- GCC 15.2.0; ICX 2026.2.0 (20261001), installed from Intel's apt repository.
  System libm is Ubuntu glibc 2.43-2ubuntu2.4, binary hash in the artifacts.
- Rust compiler versions are recorded in the artifacts, including the nightly
  used only for the scheduling experiment.

Correct rounding under nearest-even and gradual underflow defines the candidate
portable contract. Hardware FMA and different evaluation orders can implement
that same output contract. They need not be separate numerical APIs. Named
libm compatibility is a different promise, pinned per algorithm and implementation
path. NaNs here compare by NaN-ness, not payload.

## C compiler results

Time is ns per scalar value, including an indirect function call. Lower is faster.
Every row uses the same input buffer across variants. These are throughput
measurements, not dependency-chain latency or SIMD speed. Baseline means
`-march=x86-64`, not guaranteed software FMA. V3 means `-march=x86-64-v3`.

| Function | GCC baseline | GCC V3, contraction | ICX V3, contraction | glibc |
|---|---:|---:|---:|---:|
| exp2f | 1.522 | 1.271 | 1.533 | 0.991 |
| log2f | 1.449 | 1.371 | 1.393 | 1.137 |
| expf | 1.577 | 1.478 | 1.444 | 1.212 |
| logf | 1.494 | 1.396 | 1.544 | 1.190 |
| cbrtf | 2.375 | 2.173 | 2.267 | 2.362 |
| sinf | 2.342 | 2.001 | 1.977 | 1.746 |
| tanhf | 2.525 | 2.298 | 2.298 | 2.534 |
| powf | 7.441 | 5.366 | 5.266 | 2.589 |

Full eight-variant results, including ICX baseline and contraction disabled,
are in `final-summary.csv`. Its confidence intervals are paired mean differences
in ns/value against GCC baseline; a negative interval favors the candidate.
ICX does not uniformly improve on GCC. Contraction-enabled V3 helps several
functions without introducing mismatches in the validation set.

A baseline object can still call a hardware-dispatched libc helper. We verified
system `fma` resolved to offset `0x78530`, containing `vfmadd213sd`, normally.
With `GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA,-FMA4`, it resolved to `0x78550`, the
software implementation; both disassemblies are preserved. GCC baseline powf
measured 7.441 ns/value normally and 25.989 with software helper selection.
Those are separate runs, not a paired confidence interval for the difference.
V3 still emits its own FMA instructions under the tunable: this experiment
isolates helper selection, not a complete emulation of a pre-FMA processor.
The software run also changes glibc's own selected implementations.

## Safe Rust experiment and instruction review

`exp2.rs` translates the pinned upstream exp2f algorithm with exact constant bits,
wrapping integer arithmetic, and `#![forbid(unsafe_code)]`. The original MIT
notice remains in the file. It promises nearest-even results only: replacing
an overflow calculation with infinity and folding hard-case constants does not
preserve C's exception flags or dynamic rounding semantics.

Both baseline and V3 plain Rust versions have no calls and no FMA instructions
inside the mathematical function. Ordinary multiplication/addition remains
unfused. The lookup index mask also removes the bounds-check path.

The baseline Rust conversion uses `cvtss2sd xmm1,xmm0` without first clearing
xmm1, whereas GCC clears its conversion destination. The legacy instruction
merges upper bits and therefore reads its prior destination. This is a candidate
cross-call dependency, not an algorithmic requirement. Rust's V3 version uses
three-operand VEX instructions. An exact integer-bit widening alternative,
valid after the existing range checks establish a normal finite input, changes
only that conversion and retains all output bits in the test set:

| Rust exp2f | Baseline x86-64 | V3 |
|---|---:|---:|
| Ordinary cast | 2.684 | 1.466 |
| Exact bit widening | 1.702 | 1.715 |
| Rust std comparison | 1.145 | 1.143 |

These rows are paired within each Rust executable. C and Rust are separate
harness builds and their absolute costs are not a controlled language comparison.
The bit form improves baseline but loses on V3; do not select it universally.
No Rust intrinsics were needed for this experiment.

`-C llvm-args=-partial-reg-update-clearance=1000` did not improve the original
baseline (2.678 ns/value). Nightly baseline and `-Z tune-cpu=arrowlake` measured
2.674 and 2.683 ns/value respectively, using the earlier timing corpus. These
are exploratory separate runs; the scheduling flag is nightly-only. No claim
that CPU tuning fixes this baseline dependency follows from the experiment.

## Validation and measurement method

Each of eight CORE-MATH configurations passed 200,000 inputs per function:
100,000 random bit patterns/explicit edges and 100,000 dense workload-domain
inputs. exp2 includes its three explicitly tabulated hard cases. MPFR computes
lower/upper directed bounds, increasing precision until both bounds round to the
same f32. All eight functions passed again with software libc FMA selection.
Both Rust forms passed the 200,000 independently generated exp2 vectors on
baseline and V3. This is sampling, not a full-domain proof.

System libm disagreements against the same oracle: exp2 70, log2 179, exp 74,
log 220, cbrt 0, sin 1891, tanh 0, pow 58. Each denominator is 200,000 inputs;
these distribution-specific counts are not population error probabilities.
They establish why correct-rounding and libc-compatibility promises differ.

The timing phase uses 4096 fixed inputs, 128 passes, two warmup rounds and
30 measured rounds with rotating variant order. exp/log/trig/cbrt/tanh inputs
are uniform in [-8,8); log/log2/pow bases use [0.001,16.001), and timed pow
exponent is 2.4. The validation pow exponents vary in [-8,8).
`analyze.py` invokes zenbench's existing `examples/paired_rounds` analyzer for
bootstrap confidence intervals, rather than implementing replacement statistics.
No fast-math or native CPU build flags are used. ICX starts from strict FP mode;
contraction is explicitly on or off. ICX warns that `-ffp-contract=on` overrides
that part of strict mode; the warning is retained in the build log. The study
does not claim dynamic-rounding or exception-flag equivalence. Both compilers' objects are linked by GCC
against the same system libm, avoiding an accidental Intel-runtime comparison.

Resource records (`run-heavy`, 4 GiB hard cap): final C build peak RSS 0.12 GiB;
normal benchmark rc=0, 6 s, peak RSS 0.00 GiB, min available 22818 MiB;
software-FMA benchmark rc=0, 7 s, peak RSS 0.00 GiB, min available 22898 MiB;
Rust compile/check/benchmark rc=0, 1 s, peak RSS 0.12 GiB, min available 22951 MiB.
The wrapper rounds RSS to two decimals: 0.00 GiB does not mean zero memory.
Full logs retain peak load and the exact commands.

## Reproduction and artifacts

Use the `core-math-study-*` recipes in the root justfile under `run-heavy` on
an eligible host. `build.sh UPSTREAM OUTPUT` builds eight shared libraries and
the MPFR driver; `run.sh OUTPUT` validates before timing. `rust.sh OUTPUT`
checks the oracle file and benchmarks both Rust forms. `tuning.sh OUTPUT`
compares compiler scheduling settings. `analyze.py CSV PAIRED_ROUNDS_BINARY`
produces the committed summaries. MPFR development headers must be present at
the prefix specified in build.sh; this run extracted distro development packages
under the study directory, using the installed runtime libraries.

Raw CSVs, oracle/input binary vectors, assembly, source and binary hashes, and
resource logs live at `/mnt/v/output/archmage/core-math-cost-2026-11-05/`, mirrored
under `/mnt/tower/output/archmage/core-math-cost-2026-11-05/`. Generated artifacts
and binary vectors are intentionally outside git. The scripts and small result
summaries are retained in this draft.

## Design conclusion

Keep baseline x86-64 as the portable deployment floor. Use correctly rounded
per-function implementations as default candidates, with feature-proven optimized
paths producing the same final bits. Choose those paths per function and CPU;
there is no universal ICX winner. FMA-heavy pow needs a materially different
cost assessment on hardware without FMA than exp2/log2 do. Preserve existing
Rust-libm ports and pinned glibc algorithms as compatibility choices. Keep the
bounded-error lowp/midp surface explicit. Promoting this research into a default
still requires full hard-case validation, no_std helper ownership, cross-target
verification, and review of the new API.
