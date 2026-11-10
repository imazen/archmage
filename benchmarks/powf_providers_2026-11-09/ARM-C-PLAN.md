# Arm powf.c before a Rust port

Additional owner-directed experiment, frozen before timing. Do not port the
algorithm until the C comparison is complete. All previous studies remain
sealed and unchanged. Source is Arm Optimized Routines v26.07,
4be260a5117480382690c6d8c300bc784e927d76. Build only powf.c, its two data files
and math_errf.c; retain the exact headers, license and commands. Rename the
public entry to arm_powf_reference and disable glibc compatibility aliases.
Verify it is loaded from its own DSO, not interposed by system powf.

Common flags: -O3, -fno-fast-math, -frounding-math, standard excess precision,
WANT_ROUNDING=1 and explicit architecture baseline. The owner additionally
requested a separate -march=native C build and matching Rust native diagnostic;
these remain host-specific and never replace distribution-profile arms.
Primary x86 build: AVX2+FMA, contraction enabled, WANT_ERRNO=1. Also build
AVX+FMA without AVX2, baseline x86-64 with contraction disabled, and a separate
AVX2+FMA WANT_ERRNO=0 diagnostic. On AArch64 compare native fused, separate,
and fused with the portable magic-shift reduction (HAVE_FAST_ROUND/LROUND=0).
RNE and gradual underflow remain required; never toggle the FP environment.

Compare to the actual installed glibc powf on eligible Linux Intel, AMD and
Neoverse hosts. Record package/DSO hashes, CPU, compiler, flags, disassembly
and selected glibc entry. Scalar functions are measured at logical widths
1/4/8/16; these widths do not turn the scalar C function into a vector ABI.

Numerical checks: unchanged 27,909 tagged pairs, broadcast/mixed widths,
independent MPFR-256, all 515 hard-to-round witnesses and full ordinary
stream/chain trajectories. Preserve all bit/class/payload differences. Check
errno and floating exceptions separately for scalar C calls with state reset
before each request. Compare the baseline source against glibc's SSE2 selection
under an explicitly recorded glibc hardware-capability mask as an additional
untimed execution context. Finite samples are not a proof of whole-domain
identity. Zero non-NaN mismatches to the selected matching glibc execution is
the equivalence gate; report every mismatch instead of loosening it.

Timing reuses the frozen 32 cells, 1,048,576 actual evaluations/request, 50
interleaved retained rounds, identical native adapter, system C powf reference
and EMPTY transport controls. Rust std remains a separately named arm. No
subtraction or scaling into single-call latency. A strict pilot precedes a
full matrix on each Linux host; stop that host on failed admission or unreliable
pilot. No favorable retries. A full matrix, if admitted, retains its pilot as
separate evidence. All raw observations and gates stay visible.

Primary cost gate: AVX2+FMA errno1 / system C powf upper paired 95% CI <=1.02
in every mandatory cell, with clean resource gates and verified placement.
Other builds are descriptive comparisons, not substitutes for a failed primary
build. No qualified claim follows from a noisy pilot. If quiet coverage cannot
be obtained, record that measured blocker and hold promotion. The owner's
subsequent instruction authorizes the Rust port after completing this C
comparison, preserving every mismatch and performance limit. Compare C objdump
against cargo asm, investigate bounds checks, calls and branch layout, and
evaluate core::hint::cold_path without weakening numerical expectations.
