# External powf providers: frozen batch comparison

Benchmark-only comparison. No production provider admission or change to the
bit-exact contract. Rust SLEEF 0.3.3 requires portable SIMD on nightly; use
nightly-2026-10-06 for every Rust arm in this separate package. Rust libm is
pinned to 0.2.16 with its default arch feature. No target-cpu=native.

Native sources: Arm Optimized Routines v26.07 (4be260a5117480382690c6d8c300bc784e927d76),
AOCL-LibM OSE 5.3.2 (29fd054f383e6c5e2dec2fce781d5220059f1836),
GNU libmvec from the resolved system glibc (record version, DSO hash and ABI).
Use each DSO's explicit function address with local loading, never preload or
interpose the Rust std provider. Retain compiler flags and source/binary hashes.
Native vector widths are explicit; logical x16 from AVX2 is two x8 calls.
No AVX-512 calls on hardware lacking support. Emulation supplies no speed claim.

The preserved CXX probe fails forbid(unsafe_code). The working Rust harness
retains that lint and uses Unix sockets to persistent native/Rust workers.
All timing belongs to zenbench. Report **ns per 1,048,576 actual evaluations**,
including one request/response; do not scale these times into a single-call or
64-element claim. Include an EMPTY request arm for each worker implementation.
Retain direct in-process std/libm measurements as a separate study.

Mandatory batch axes: logical widths 1/4/8/16, 64/4096 reusable input buffers,
stream and normalized chain, opaque f32 exponents 2.4 and 0.5. Width 1 admits
scalar methods; width 4 adds available x4 methods; widths 8/16 add available x8
methods. Unsupported platform/ABI cells are explicitly unavailable, never
replaced by another implementation under their names. This is 32 cells;
repeat each buffer exactly 1,048,576/size times, reset chain for each repeat.
Input and normalization match the direct-provider study. Native and Rust
adapters have different call overhead, which is included and must be disclosed.

Fifty interleaved retained zenbench rounds; unchanged common admission gates,
all-retained-round paired 95% confidence intervals. Predeclare all real-provider
ratios against the same cell's std arm; report other pairwise edges as descriptive.
Upper CI <=1.02 passes the cost tolerance, lower >1.02 fails, overlap is
inconclusive. Missing coverage or resource flags prohibit promotion. One Intel
proper cpuset pilot then full matrix only if clean; one M6 exploratory matrix,
no favorable reruns. Never overlap this lane's builds/tests with its timings.
Batch admission registers only the actual supervisor and worker PIDs.

Numerical study preserves all outputs on the existing 27,909 domain-tagged
pairs, mixed lanes at widths 1/4/8/16 where supported, and direct-study stream/
chain traces. Compare MPFR-256 independently and retain every bit/class/payload
variance in JSONL, including literal/opaque and cross-platform differences.
Full ledgers have a 512 MiB per-host budget; fail visibly and preserve a partial
ledger on overflow. A finite corpus cannot certify a whole-domain bound.

Sources: [Rust SLEEF](https://docs.rs/sleef/0.3.3/sleef/),
[Arm routines](https://github.com/ARM-software/optimized-routines),
[AOCL-LibM](https://www.amd.com/en/developer/aocl/libm.html),
[AOCL source](https://github.com/amd/aocl-libm-ose),
[glibc vector ABI](https://github.com/bminor/glibc/blob/master/sysdeps/unix/sysv/linux/x86_64/libmvec.abilist).
