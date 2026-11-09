# Rust port after the C comparison

The initial isolated C comparison completed first: all five C profiles match
installed glibc on the 27,909 tagged pairs and 515 hard-to-round witnesses;
errno-enabled C profiles match errno and floating exceptions as well. Intel's
AVX2+FMA errno1 timing ratio CI is 0.992418–1.018908, but 47/50 resource-clean
pairs prevents qualification. Keep that complete run; do not retry its source.

Implement the same unscaled table graph in the unpublished comparison package,
with exact constants extracted through the C compiler. Preserve Arm's license
and revision. No production API or MSRV change. cold_path is stable since 1.95;
its use here stays in this standalone nightly experiment, while the workspace
MSRV remains 1.89. No unsafe Rust. Use Archmage V3 proof at the entire worker
operation entry, not per primitive; baseline fallback remains available.

Full fused, full separate, cold-hinted/unhinted and explicitly restricted Gamma
assumed-input profiles are distinct names. The latter is tested only on inputs
accepted by the unchanged independent Gamma predicate, including +0 and n=0.
It does not acquire a full-domain contract by appearing in an ordinary benchmark.
The port promises RNE/gradual-underflow result bits, not C errno or FP flags.
NaN payloads remain observed differences; NaN classes and non-NaN bits must match
the matching C graph with zero mismatches. Keep MPFR accuracy separate.

Use the unchanged 32 timing cells and all 50 interleaved retained rounds, paired
95% intervals and <=1.02 upper cost gate. Primary comparisons are full fused
Rust versus system C powf and versus the same binary's Rust std. Keep the C
AVX2+FMA errno1 reference and EMPTY adapter controls. All source iterations and
failed attempts are retained; changed implementations are new named experiments,
not retries of an unchanged noisy run. Stop each Linux version's matrix after
an inadmissible pilot; no gate relaxation or favorable rerun.

Build baseline-distribution and explicit owner-requested native Rust binaries,
record actual Cargo flags and prohibit moving a native binary to another CPU
context. Compare frozen C objdump and cargo asm disassembly of actual linked
Rust code. Inspect ordinary path, FMA graph, table bounds checks, calls, return
conversions and branch placement. Profile with hardware counters before/after
performance changes. Keep all codegen and numerical evidence for each version.

M6 extension: the owner authorized direct LAN verification and exploratory
interleaved timings on the available Mac. Rebuild C using Darwin's native
compiler/linker and baseline plus explicitly native Rust. Compare result bits
against the unscaled Arm C reference; retain Apple std as a distinct platform
implementation. Require an explicit NEON token in the portable evaluator.
Preserve the strict placement refusal, then collect the separately labeled
32-cell/50-pair exploratory matrix with all resource observations retained.
Unsupported macOS process confinement excludes it from the 2% promotion gate.
