# Certified gamma finalists: verified evidence

Eight full-domain records passed exact `zenmath::scalar::reference::powf`
comparison, all chunk audits and cross-path hash comparison. There were
403,078,452 input comparisons and388 checksummed chunks. No accepted run has
a FAILURE marker. Production integration belongs to the main lane.

## Exact coverage

Both endpoints are inclusive. Each kind has four paths: scalar piecewise
Estrin5; scalar global Estrin13 decode/Estrin15 encode; the same global
certificate in array4 and array8 V3 contexts.

| Kind | Exact f32 exponent | Input bits | Inputs/path | Raw XXH3-128 | Ordered aggregate |
|---|---|---|---:|---|---|
| decode | `4019999a` | `3db71074..3f800000` | 29945741 | `1ac8deaf0e5dfa2d0d75a859383ce568` | `15c391a5e9898f972ea96b778197f57e` |
| encode | `3ed55555` | `3b475041..3f800000` | 70823872 | `1c229cfcfabcb6d54aa44974481b4db6` | `9f5b4cdffe409740a581cac0a8ab2d9a` |

All four candidate paths and their independent CORE references match both
hash columns within each kind. Outputs are positive finite in these domains,
so no NaN or zero canonicalization changes the raw stream. Scheme: pinned
Zenhash4c4a993b, XXH3-128 seed0, little-endian f32 bits; ordered chunk aggregate
includes the common source/op/domain header and index/start/count/digest.
An aggregate hashes chunk identities, not the raw output stream.

Scalar-certificate ambiguity counts: pieceE5 decode820, encode10416;
globalE13 decode50042, globalE15 encode501924. Vector rows also report
`scalar_certificate_rescues`: this is the scalar certificate count for their
inputs, **not actual vector rescue calls or tails**.

This is not arbitrary-pow-pair, f64 or whole-f32-domain coverage. Smoke and
MPFR sampling covered52 global and12 piecewise combinations; only the eight
finalist paths above received full admitted-domain sweeps.

## Measured throughput

All timings are native r5900xt streams with64/4096 elements, values/bit-uniform
distributions, warmed executable and preallocated buffers. Rust1.99.1,
release thin LTO/codegen-units1, no native CPU flag, actual Cargo rustflags[].
Shared zenbench lock and resource gates stayed enabled. Every group retained
at least50 paired rounds, all pre-round gates clean, unreliable=false.

| Fixed candidate | Decode time/std | Encode time/std |
|---|---:|---:|
| scalar pieceE5 | 0.845205–0.864196 | 0.854393–0.977835 |
| array4 globalE13/E15 | 0.530643–0.604889 | 0.577460–0.685115 |
| array8 globalE13/E15 | 0.530097–0.561014 | 0.576528–0.682879 |

All corresponding all-round paired MeanCi95% intervals are below zero;
10000 bootstrap resamples use every retained pair, with no extra exclusions.
The narrowest scalar margin, encode values/64, is-0.0826943ns/element,
CI[-0.0996249,-0.0587046],100/100pairs retained (built-in analysis excluded10).

Full-mantissa scalar exploration lost to std: coarse degrees8/12/16/20,
then refinement11..17, best decode1.03–1.09x and encode1.18–1.27x std time.
Piecewise scalar degrees3/4/5 improved that. Best-per-group degree selection
remains exploratory; the fixedE5 and fixed global packed results above are
reported separately. Scalar ranking does not transfer to packed code.

Linked assembly confirms packed separate vmulpd/vaddpd and no block/Evaluation
helper calls in all12 array4/8 contexts. Frames200–248bytes and8–16 static
YMM stack references remain. Calls include conditional CORE rescue,
scalar-tail reference and cold length assertions. The source21-vector
coefficient temporary is not a672-byte stack array. The earlier correct
prototype with outlined piecewise helpers was preserved and never timed.

## Whole C0 transfer: distinct contracts

Composition uses actual linear-srgb scalar source
`8900ffc87cd50b249bae19c99d874431e055d888`, including its C0 constants,
thresholds and separate baseline f32 multiply/add. Independent MPFR tests
passed8328 inputs/kind against the **stage-rounded f32 composition**, including
threshold neighbors, signed zero, rails and NaN behavior.

The separate MPFR real curve uses the source's f64 constants/exponents.
Observed upper endpoint bounds on absolute f32-composition error were
2.136719301493173e-7 decode and1.4644575985212072e-7 encode; these are sampled
metrics, not global error proofs or a replacement f32 power oracle. The std
f64 curve control's observed error was2.220446049250313e-16 for both kinds.

C0 timing has8groups64cells, all clean. Scalar pieceE5 wins7/8 groups under
all-round paired analysis. It loses decode/bits/64:1.182868x std time,
+0.176918ns/element,CI[0.169483,0.184023]. That workload mostly takes the
linear branch. Other decode ratios0.842679,0.871042,0.961714; encode ratios
0.805873,0.882663,0.804570,0.880103. Production overhead remains a required
gate; these results do not promise a win for every surrounding transfer.

## Frozen provenance and resources

Final source-manifest SHA256:
`32c98d40f72d6402262a2512085d16850e327058e99c1bf9f5e5e2f86a74979b`.
Final executable SHA256:
`4d0356aa7336a1db26a95a3cb5b765c83b92d784a7208820b26c5f7aaaaf25f5`.
Retained source archive SHA256:
`8bafaed8f013c29c020f9a0e889fd240c90940d4c8b482cad52eaa8f2878c7b7`.
Build identity binds source/archive/compiler/Cargo fingerprints/executable;
122 numerical/shared-build files byte-match the earlier timed vector binary
`68fa480a4e1ceca7dec86d062b80a52aca0f115e5f17fd7327f0c4b173bebe7b`.
The base remains4ccaf1cf with its old features; no later primary feature WIP
was copied. Main lane must add the new correct-rounding feature at integration.

| Work | run-heavy result |
|---|---|
| four initial health checks | rc0,24s,RSS0.58GiB |
| final build/smoke/MPFR/auditor gates | rc0,33s,RSS0.48GiB,min47544MiB,load1.30 |
| coarse scalar timing | rc0,70s,RSS0.12GiB,min47943MiB,load1.11 |
| refined scalar timing | rc0,104s,RSS0.13GiB,min47686MiB,load1.46 |
| piecewise scalar timing | rc0,74s,RSS0.12GiB,min47936MiB,load0.97 |
| packed timing | rc0,62s,RSS0.12GiB,min47920MiB,load1.21 |
| C0 timing | rc0,31s,RSS0.12GiB,min47951MiB,load1.01 |
| eight full-domain finalists | rc0,15s,RSS0.02GiB,min48092MiB,load0.66 |

All compilation/source/cargo/toolchain/temp placement was local. No own
build, copy or sweep overlapped quiet timing. Frozen executables remain outside
Cargo. Prior build/orchestration failures are preserved separately; none was a
numerical failure. Signal regressions preserve exact original diagnostic bytes.
Auditors reject corrupt/missing chunks, mislabeled kind/family/width/source and
FAILURE markers. Incomplete JSON, empty groups, bad scales and sample widths
are rejected by the analysis gates.

Artifact root on r5900xt:
`/home/lilith/tmp/zenmath-gamma-opt-sol/runs/`.
Durable mirror:
`/mnt/tower/output/archmage/zenmath-gamma-opt-2026-11-06/r5900xt/runs/`.
See [ARCHIVE.md](ARCHIVE.md) for byte-manifest/report identities. Full logs,
source archives, generated proofs, all oracle rows, all timing samples/CIs,
per-path chunk/metadata/summary records and retained binaries are preserved.
