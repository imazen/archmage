# Direct constructor attributes and forwarding: cold compile comparison

Measured on i265 (Intel Core Ultra 7 265K, Linux), rustc 1.98.1 / LLVM 22.1.8.
This compares the already-implemented constructor modes with two generator
changes. It does not compare against the original pre-mode implementation.

- Before: `fed4b14a`, contextual constructors use `#[archmage::rite(tier)]`.
- Direct: registry-derived `#[target_feature]` and `#[inline]`, unchanged forwarding.
- Flattened: direct attributes plus inline expansion of simple value-constructor bodies.

Six serial builds per cell, 72 builds total. Six cases rotate through every
ordering position. Each build uses a fresh Cargo target directory, release mode,
eight jobs, incremental off, and no native-CPU flags or compiler wrappers.
Filesystem caches were not flushed. All 12 consumer lockfiles have the same
SHA-256; all consumers use the same arithmetic/conversion/dispatch kernel.

## Cold total medians

| Features | Consumer API | Before | Direct only | Direct + flattened | Final change |
|---|---|---:|---:|---:|---:|
| default | explicit | 2.850 s | 2.824 s | 2.836 s | -14.8 ms / -0.52% |
| default | local | 2.841 s | 2.823 s | 2.835 s | -6.5 ms / -0.23% |
| avx512 | explicit | 3.062 s | 3.022 s | 3.046 s | -15.9 ms / -0.52% |
| avx512 | local | 3.069 s | 3.032 s | 3.039 s | -29.9 ms / -0.97% |

The final median changes are small, and ranges overlap. Default-feature ranges
are 2.830–2.868 s before versus 2.830–2.853 s after for explicit calls, and
2.828–2.860 s versus 2.824–2.851 s for local calls. AVX-512 ranges are
3.059–3.090 s versus 3.026–3.078 s for explicit calls, and
3.051–3.086 s versus 3.016–3.048 s for local calls.
Do not treat these as a general compile-speed guarantee or extrapolate them to
whole zen repositories, incremental builds, other compilers, or other hosts.

## Magetypes library-unit medians

| Features | Consumer API | Before | Direct only | Direct + flattened |
|---|---|---:|---:|---:|
| default | explicit | 1.265 s | 1.240 s | 1.250 s |
| default | local | 1.260 s | 1.235 s | 1.250 s |
| avx512 | explicit | 1.495 s | 1.440 s | 1.470 s |
| avx512 | local | 1.475 s | 1.460 s | 1.460 s |

Direct attributes alone have lower median total times than the final version in
all four cells. Flattening removes a source-level forwarding call but adds
concrete function-body tokens for rustc to process. The CSV includes Cargo's
frontend and codegen sections; they are reported at 10 ms precision per build.
Consumer-unit medians remain 0.05 s in every cell; individual builds span
0.04–0.06 s at that precision.
No runtime-speed change was measured.

## Safety and maintenance

The generator replaces 1,242 rite attributes across all architecture cfgs,
including native `from_raw` methods. Feature lists come from `token-registry.toml`;
architecture and Cargo gates remain on the existing impls. Every contextual
constructor still forbids unsafe code and obtains its proof with checked
`from_context()` (or `ScalarToken` for scalar code).

The flattening pass expands 894 simple contextual value constructors: 722 SIMD
and 172 scalar. It accepts only a single value expression ending in `new_repr`,
qualifies backend calls using the original generic bound, and rejects blocks
or ambiguous bounds. `new_repr` remains the one representation/token storage
constructor. Multi-step loaders and memory views retain shared helpers, as do
explicit constructors and mode-generic operations. Their algorithms remain
written once in the generator. All 42 method names remain available in the
same supported backend contexts; this change does not add public signatures.

Generator tests cover full registry feature sets, local proof placement,
qualified backend calls, and rejected flattening shapes. x86 runtime constructor
and raw-interop tests passed; AArch64 equivalents passed under QEMU. All-feature
library checks passed for AArch64, WASM32, and i686. The full local
`cargo run -p xtask -- ci` command passed, including soundness/negative-context
checks, no_std tests, unchanged x86/ARM/WASM public-API snapshots, and docs.
Its optional Miri, Docker/cross, and wasmtime execution paths were unavailable;
the separately run AArch64 QEMU tests above cover the changed constructor paths.
This host has no AVX-512F, so native AVX-512 constructors were compiled but not
executed. All 44 generated files match the measured final snapshot byte-for-byte.
Full CI took 146 s under run-heavy (peak RSS 0.73 GiB, minimum available RAM
25,925 MiB, peak load 2.13).

## Reproduction and retained artifacts

- [Harness](../scripts/measure-local-mode-compile.py), now accepting `--middle`
  and `--before-local` for a baseline that already supports both APIs.
- [Raw rows and frontend/codegen sections](constructor_codegen_compile_2026-09-27.csv).
- [Toolchain, commands, hashes, and source identities](constructor_codegen_compile_2026-09-27.meta.json).
- Source snapshots and manifests, consumer lockfiles, full logs, and Cargo
  timing HTML/unit JSON: `/home/lilith/data/archmage/constructor-compile-optimization/2026-09-27/`.
- The direct-only snapshot is generated with the flattening selection disabled;
  its safety documentation and attribute generation match the final candidate.
- run-heavy completed in 212 s: peak RSS 0.40 GiB, minimum available RAM
  26,386 MiB, peak load 2.16. These are guard observations, not per-crate memory
  benchmarks. No build caches were deleted. No cloud/NAS mirror is configured.
