# Powf debug compiler footprint and chunk-local comparisons

The large memory measurement belongs to Rust compilation, not a large array of
powf outputs. The numerical tail verifier generates 513 base inputs. It now
compares each packed result immediately with scalar/splat results and checks its
remainder, retaining no full reference-output array. All 64 policies, five
exponents, widths 4/8/16 and the original lengths remain. The benchmark still
uses zenbench's 50 interleaved paired rounds.

Forced inlining and const options previously duplicated arithmetic instances
for flags already consumed by domain selection and dispatch. The debug kernel
now shares only the arithmetic-relevant options, with ZERO/GAMMA and the
arithmetic provider still explicit. Literal const arms support the MSRV.
Debug graph frames are shared; optimized builds retain the original flags and
inlining. No coefficient, arithmetic operation, precision or checked range is
changed, and no assertion or stack limit is relaxed.

| Measurement | Time | Peak RSS | Scope |
|---|---:|---:|---|
| Before sharing | 77 s | 9.54 GiB | Historical strict-WASM library unit build/test, one job |
| Shared debug graph | 35 s | 3.29 GiB | Rebuilt strict-WASM library unit crate, one job, 6 GiB cap |
| Chunk-local complete suite | 63 s | 1.97 GiB | Incremental target, two jobs, all Magetypes tests/doctests |
| Fresh CI-equivalent complete suite | 131 s | 3.28 GiB | New target, `CARGO_INCREMENTAL=0`, two jobs, 6 GiB cap |

The fresh run uses the CI command `cargo test -p magetypes --features "std avx512"
--target wasm32-wasip1` with `RUSTFLAGS=-Ctarget-feature=+simd128`. Minimum
available is 39,317 MiB and peak load 2.69. It includes all new integration
binaries and doctests with the default WASM stack limit. These are compiler
resource diagnostics, not a three-repetition cold-build speed comparison.
3.28 GiB is still a substantial compiler footprint; it is not runtime data RAM.

The earlier published strict-WASM job received a runner shutdown signal while
compiling and exited 143 without a test failure. The remote log does not establish
the shutdown's cause. A local 6 GiB cap does reproduce a rustc memcg OOM before
sharing; the kernel record and first attempts remain retained. An additional
API-dispatch framing experiment exceeds a 4 GiB cap and is not adopted.

All 373 emitted x86 complete-loop bodies match before/after this debug-only
fix, retaining instructions, operands, branch positions and calls. The normalizer
uses the existing consumer parser and resolves imported anonymous constants by
contents and alignment; imported named object initializers are also equal.
ThinLTO module identifiers are normalized separately. Earlier reports identifying
60 imported-symbol differences remain alongside the resolved comparisons. An
initial expectation of 670 independently emitted bodies was incorrect: LLVM
coalesces equivalent instantiations; the frozen compiler emits 373.

Native x86 std/no_std chunk-local grids pass (65 s, peak RSS 1.60 GiB); native
M6 debug grids pass (16 s, reported peak RSS 1.11 GiB). A separate M6 release
verification passes before restricting sharing to debug; release subsequently
returns to its original implementation. Full local CI passes all 19 stages in
677 s, with all 24 warning configurations clean afterward. That combined run
reports 764 s, peak RSS 2.45 GiB, minimum available 40,720 MiB and peak load 4.55.
The final debug-only restriction and chunk-local test form additionally pass
the fresh CI-equivalent WASM suite, native grids, warning-denied native Clippy
and the complete-loop comparison. Existing numerical/Miri expectations remain.

All new evidence is additive to the [sealed bounded study](MIDLOW-ARTIFACTS.md).
The initial 2,342-file core and 61-file closing set remain immutable. Promotion
is still held for the documented performance evidence; this compiler fix does
not turn noisy timings into qualified results.
