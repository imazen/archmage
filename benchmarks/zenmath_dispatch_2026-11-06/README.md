# Zenmath dispatch study

Unpublished implementation study, all changes local drafts. Production API and
selected scalar policy are under review; no speed claim before retained timings.

The generated matrix covers f32 sqrt/floor/ceil/trunc/roundeven/round/FMA and the
four accurate transcendentals, plus f64 primitives/FMA. Power common and special/
rescue mixtures are distinct streaming groups. Buffers have 1, 16, 1,024 and
65,536 values. Feedback has 1,024 dependent scalar steps, including the documented
mapping cost; several mappings converge to fixed points. It is workload feedback,
not a universal instruction-latency measurement. No packed feedback is claimed.

Controls: non-dispatch scalar reference, std baseline, std whole-loop feature
contexts, cached-feature branches per value, typed OnceLock function pointer per
value, explicit held proofs per value, whole-loop proof contexts, accurate packed
four-double kernels and genuine eight-double AVX-512 candidate. Whole-loop context
is distinct from a boundary at every scalar call. Every pointer carries the real
selected proof in its context. Its proof match/indirect call are measured costs;
this does not implement glibc IFUNC. Cache state is inline and needs no heap
allocation; case closures allocate during benchmark setup. Initialization is
outside steady-state after verification/warmup. No token-disabling during a run;
no-FMA validation uses a fresh emulator process.

Native inputs/outputs/case closures allocate outside timed iterations. Strict WASM
allocates its case vector and buffers per host invocation, then repeats the kernel;
repeat count is explicit per size/mode in the host. Absolute cross-platform times
therefore include different wrapper costs. std is a timing control, not the
correct-rounding oracle. Inputs and all non-std stream/feedback adapters are checked
bit-exact against the original scalar algorithm before timing; NaN payloads are
unspecified and signed zero is preserved.

Reproduce source with `just generate` (includes scoped rustfmt). The production
primitive generator has its separate root recipe. Run checks/builds on eligible
remote hosts under run-heavy; never use target-cpu=native. Required-tier caller
configuration must accompany validation/timing. Raw logs/JSON/assembly/binaries
and per-build source identities go under the external artifact directory and NAS
mirror, not git. Exhaustive digest work belongs to the separate digest lane.
