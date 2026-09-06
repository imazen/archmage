# Native ARM add-green codegen

The 16-byte magetypes kernel emits the same instructions as direct NEON on
Apple M4 Pro. Its scalar counterpart emits byte loads, additions, and stores.
Using two NEON vectors per iteration reduces loop overhead further.

| Row bytes | Scalar mean | magetypes 16-byte mean | magetypes 32-byte mean | Direct NEON 16-byte mean |
|---:|---:|---:|---:|---:|
| 16 | 10.9 ns | 10.2 ns | 11.2 ns | 10.2 ns |
| 256 | 32.7 ns | 16.2 ns | 14.2 ns | 16.2 ns |
| 4096 | 635.3 ns | 159.1 ns | 92.4 ns | 160.2 ns |
| 65536 | 10.57 us | 2.15 us | 1.20 us | 2.15 us |

At 65536 bytes, the 32-byte magetypes variant is 8.8x faster than the scalar
row primitive in this run. It is slower on a 16-byte row, where that variant
uses its scalar tail. A codec can retain a 16-byte vector tail after the
32-byte loop; the wide benchmark here deliberately retains a scalar tail.
This is a row-kernel comparison, not a whole-codec speedup.

[Full log](add-green.log) includes 200 interleaved rounds per size, all seven
variants, paired confidence intervals, and variance flags. Inputs, allocation,
and drop are outside the measured region. Every variant goes through the same
opaque function-pointer call with a black-boxed row. An exhaustive offset/length
check runs before timing and compares whole buffers, including untouched guards.
These tests exercise one RGBA row; the caller selects row slices to preserve
image stride. No content-dependent branches or quality parameters exist in the
timed primitive. No calibration constants are derived from this run.

## Codegen evidence

- [Direct NEON](11direct_neon.asm) and
  [magetypes16](11magetypes16.asm) have identical instruction sequences after
  normalizing local branch addresses: `ldr q`, `tbl.16b`, `add.16b`, `str q` in
  the loop. There are no per-vector helper calls or stack spills. Both end in
  the same scalar-tail function.
- [magetypes32](11magetypes32.asm) uses `ldp q`, two shuffles/adds, and `stp q`.
  The polyfill is a paired-vector loop, with no extra representation traffic.
- [The ScalarToken variant](16magetypes_scalar.asm) retains scalar byte work.
  Having a fixed-array expression alone does not force LLVM to vectorize it.

The tested `#[magetypes]` and `#[arcane]` expansions have no observable
abstraction overhead in this kernel. The measured improvement comes from
expressing the byte operation as SIMD and choosing a useful loop width.
This does not establish that every macro use, backend operation, or codec
kernel has equally good codegen. Float transforms and other ARM CPUs need
their own measurements.

## Provenance and reproduction

- Host: Apple M4 Pro / Mac16,11, 24 GiB RAM, Darwin 25.5.0.
- Rust 1.98.0 (`88d9e12ae`), LLVM 22.1.8, aarch64-apple-darwin.
- Date: 2026-09-05 America/Denver / 2026-09-06 UTC.
- Archmage/magetypes 0.9.29, base `80e9748a4b40a45800458e6ccba9f2491e0a009f`.
- Kernel source: `c3e94d85`,
  [arm_codegen.rs](../../magetypes/benches/support/arm_codegen.rs).
  The raw log reports the base commit because the benchmark extension was
  uncommitted during measurement; its kernel bodies were then committed.
- No target-cpu or target-feature overrides. Standard release profile.
- Serialized build/run, `nice -n 19`, four build/Rayon/OpenMP threads.
- `/usr/bin/time -l`: 105.02 s wall, peak RSS 141,066,240 bytes for the
  cargo build/run invocation. No Linux cgroup memory cap on this Mac.

Reproduce with `just bench-arm-codegen-macos`, which saves complete output to
`~/tmp`. The underlying command is:

```sh
cargo bench --locked -p magetypes --bench generic_vs_concrete -- --format=llm
```

Assembly was extracted from
`target/release/deps/generic_vs_concrete-98ed66230db5d2c2` using `otool -tvV`.
The four excerpts retain original addresses and symbols. Each artifact is
under 30 KB. No library implementation or public API changed in this experiment.

## Landing validation

Before pushing, these benchmark/report commits were rebased onto
`2a1c94195954` (the concurrent tier-trait authentication fix). The benchmark
source is now `3249abce27ba`; the measured pre-rebase source remains
`c3e94d85`. Kernel bodies are unchanged by the rebase. The timings and assembly
above were collected before that rebase and have not been relabeled as new
measurements. Post-rebase ARM benchmark clippy with warnings denied and the
required generation, registry, token, and static soundness checks pass.

The [post-rebase smoke run](post-rebase-smoke.log) also passes all seven
variants through alignment/tail parity checks and the 16-byte timing group.
It is a build/execution check, not a repeat of the larger working-set sweep.
