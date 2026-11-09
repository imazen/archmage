# Which powf, and which glibc change?

The glibc 2.35→2.39 scalar pow/exp/log comparison does not show a CORE-MATH
replacement. The ten generic source files for binary32 powf, expf, exp2f,
logf, log2f and binary64 pow, exp, exp2, log, log2 differ only in copyright
years between those tags. `glibc_history.py` retains exact hashes and diffs;
its normalization changes only copyright years and rejects an arithmetic
operand change. This source observation does not imply identical machine code,
dispatch, vector implementations, distribution patches or runtime performance.

## Timeline and implementation families

| Change | First release / source evidence |
|---|---|
| Arm scalar binary32 powf rewrite | glibc 2.27; commit 4ea49f4c08eeac1a64a5aba8c9033b604afe8b2c, 2017 |
| New scalar binary64 exp/exp2 | glibc 2.28; e70c176825, 2018 |
| New scalar binary64 log/pow | glibc 2.29; f41b0a43e4 / 424c4f60ed, 2018 |
| x86 binary64 log2 FMA implementation | f6b10ed8e9, August 2023, within the 2.35→2.39 interval |
| Documented CORE-MATH binary32 imports | glibc 2.41 NEWS; these include expm1f, log1pf, log10f and others, but not powf, expf or logf |

The fast scalar powf computes exp2(y·log2(x)) with private inline range
reduction, small tables and double intermediates. Its source documents a
degree-five log approximation, degree-three exp approximation and 0.82-ULP
error estimate. It does not call general public logf and expf in sequence,
and it does not claim correct rounding. Its ordinary path and exceptional
classification are one operation, an appropriate comparison for a top-level
powf dispatcher. Binary64 pow is a different implementation with a higher
precision log and its own error analysis.

The measured Ubuntu library is **2.43-2ubuntu2.4**. Its exact glibc-source
package was downloaded and extracted without replacing any installed library.
Its e_powf.c still belongs to the Arm algorithm family; its SHA-256 is
`edb76f922e4791795c33088709fb36eadc09c61581b63978e6fed6261a29bdd7`.
Compared with 2.39, it has real excess-precision and overflow/rounding changes;
we retain that diff and do not call it byte-identical. The resolved source
mirror used for the historical tag audit had no 2.43 tag, so the installed
source package supplies this separate check.

## Actual x86 dispatch

Both the 2.39 tag and installed 2.43 source select the powf FMA implementation
only when **FMA and AVX2 are usable**. The helper is compiled with `-mfma
-mavx2`; a bare FMA bit cannot authorize it. Glibc's usable-feature machinery
accounts for OS support. The measured Rust worker imports `powf@GLIBC_2.27`,
not the older compatibility symbol.

`glibc_binding.c` resolves that exact symbol with dlvsym and records its
library-relative address. On the measured Intel library the native address is
0xa1ea0, whose disassembly includes scalar FMA. QEMU Haswell with AVX2 disabled
resolves 0x8b720 instead. These offsets identify this retained library only.
Whole-function IFUNC selection keeps feature testing outside the polynomial;
it is not a test before every multiply-add.

GNU libmvec's x86 ABI routines are a separate vector implementation family.
An SSE2 or AVX2 ABI name does not fully describe the IFUNC-selected instructions.
Likewise, Arm AdvSIMD and AOCL vector powf require their own numerical checks;
scalar accuracy cannot be transferred to them by name.

## Separate CORE-MATH comparison

[CORE-PLAN.md](CORE-PLAN.md) was frozen before timing. We built the cr_powf C
source shipped in core-math-sys **1.4.0**, SHA-256
`c9b5f74aeb91fae7dd4f850be349359918cc2e55c3fc7dd7324806be4a92d492`.
The crate build script defaults to `-march=native`; this study bypasses that
build script and explicitly uses x86-64 or Armv8-A. A second x86 DSO uses
AVX+FMA with AVX2 explicitly disabled. Both use the source's explicit fused
operations; the baseline can call the system's dynamically dispatched fma.
These are feature-kernel versus baseline builds, not fused versus unfused math.
Errno support is disabled in CORE-MATH and remains enabled in system libm.
No production dependency or dispatch choice changes.

On the 27,909 tagged pairs, CORE-MATH has zero rounded-MPFR differences for
the positive finite bases / finite exponents covered by the accuracy oracle,
including broadcast and mixed widths 1/4/8/16 on Intel, AMD, M6 and Neoverse.
The FMA-only kernel also passes that check under QEMU with AVX2 unavailable.
The provider listing proves usable AVX+FMA and absent AVX2; translated block
addresses intersected with execution records prove actual FMA execution.
The first trace parser expected mnemonic disassembly, but this QEMU emits
raw OBJD-T bytes. Its failed analysis is preserved; `core_trace.py` decodes
those bytes with objdump and checks executed addresses, with a negative
control rejecting an unexecuted FMA block. Emulation supplies no speed evidence.

The separate CORE-MATH inventory covers 68 provider/width executions, including
the std references and emulated context. Its complete ledger contains 196
differing input records and 1,082,571 uncompressed bytes. Matching CORE-MATH
implementations have zero non-NaN bit or NaN-class differences across these
executions. NaN payloads remain variable: native x86 versus either Arm platform
differs on 46 tagged cases per width; emulated versus native x86 differs on
three. Every raw output and payload difference is retained. The no-FMA SDE
profile also rejects the FMA-only provider, validating that its guard does not
silently admit that unsupported execution context.

### Hard-to-round search

CORE-MATH includes `src/binary32/pow/worst.c`, a finder for results close to
rounding boundaries. That is different from maximizing an approximation's
error. `check.sh --worst` consumes such witnesses; it is not itself the search.
The project's [worst-case notes](https://core-math.gitlabpages.inria.fr/worst.html)
name [BaCSeL](https://gitlab.inria.fr/bacsel/bacsel) as the tool used for most
of its hard-to-round cases. The powf-specific finder is the tool exercised here.
Neither should be described as a proof of an approximate graph's maximum ULP
error merely because its inputs are hard to round.

We ran the unchanged finder with degree two and target m=16 over x in
[0.5,2] and the single f32 exponent `4019999a` (the f32 value of 2.4).
A COUNT build admitted 16,777,216 logical pairs under the predeclared
20-million-pair / 60-second budget, without OpenMP. The search completed in
2 seconds and produced **515 unique witnesses**. Source, commands, stdout,
MPFR/GMP identities and every witness are retained. Witness-file SHA-256:
`02a99ca094ee9bf15581303aa5779d4c6c6be51316aa05ba5d767dd73fcb506d`.

On these 515 witnesses, glibc 2.43 powf differs from rounded MPFR on **110**
inputs, Apple std/C powf on **124**, and the tested CORE-MATH builds on **zero**.
Glibc's largest observed error here is 0.500007618219 ULP: a wrong final bit
can occur while an implementation remains within one ULP. This is useful
adversarial coverage, not an exhaustive accuracy proof for arbitrary pow pairs.
The search resource line is `2s | peak-RSS 0.08GiB | min-avail 29951MiB |
peak-load 1.33`.

### Exploratory cost

Zenbench means, x8 / 4096-element stream / opaque f32 2.4, **ns per 1,048,576
actual evaluations**, including one request and the C adapter. Each comparison
retains all 50 paired rounds. Neither platform qualifies the promotion gate.

| Exact implementation | Ryzen 9 5900XT ns | M6 ns |
|---|---:|---:|
| Rust std loop | 4,383,894 | 2,128,423 |
| System C powf | 5,869,879 | 2,776,519 |
| CORE-MATH baseline | 18,179,281 | 8,113,573 |
| CORE-MATH AVX+FMA, AVX2 disabled | 13,099,727 | not applicable |

The CORE-MATH FMA-only / glibc paired 95% ratio is 2.189882–2.297375 on AMD;
baseline / glibc is 3.039298–3.198755. M6 CORE-MATH / system C is
2.915553–2.931895. AMD's pilot has 37/50 clean pairs and is unreliable;
the full 32-cell M6 matrix is resource-flagged with unrestricted placement.
No dirty pair was discarded and neither was rerun until favorable.
Resource lines: AMD `5s | peak-RSS 0.12GiB | min-avail 48266MiB | peak-load
0.27`; M6 `97s | peak-RSS 0.02GiB | min-avail 9858MiB | peak-load 2.50`.

## Implication for the powf module

Glibc is a strong measured speed reference for a backend-dependent scalar
HIGHP policy. It is not a cross-platform canonical result or a correctly
rounded oracle. CORE-MATH is a distinct option when the intended contract
requires correct rounding; this experiment shows a material cost on these
ordinary workloads. A fixed portable approximation graph remains a third
contract. The const range/validation/precision flags and whole-operation
dispatch should make those choices explicit. Finite startup probes cannot
prove arbitrary-input agreement between std and the canonical graph.

Sources: [2.35 powf](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/ieee754/flt-32/e_powf.c),
[2.39 powf](https://github.com/bminor/glibc/blob/glibc-2.39/sysdeps/ieee754/flt-32/e_powf.c),
[Arm rewrite](https://github.com/bminor/glibc/commit/4ea49f4c08eeac1a64a5aba8c9033b604afe8b2c),
[2.39 IFUNC](https://github.com/bminor/glibc/blob/glibc-2.39/sysdeps/x86_64/fpu/multiarch/ifunc-fma.h),
[2.41 NEWS](https://github.com/bminor/glibc/blob/glibc-2.41/NEWS),
[CORE-MATH project](https://core-math.gitlabpages.inria.fr/),
[pinned Rust sys crate](https://docs.rs/core-math-sys/1.4.0/core_math_sys/).
