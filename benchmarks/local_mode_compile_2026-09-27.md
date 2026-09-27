# Constructor-mode cold compile comparison

Measured on i265 (Intel Core Ultra 7 265K, Linux), rustc 1.98.1 / LLVM 22.1.8.
Three serial runs per cell, release profile, eight build jobs, incremental off,
no target-cpu=native. Every build uses a fresh Cargo target directory and builds
its dependencies. Filesystem caches were not flushed. All three variants resolve
the same dependency versions and use the same SIMD consumer kernel.

| Cargo features | Before, total | After, explicit API | After, local API | Explicit API increase |
|---|---:|---:|---:|---:|
| default | 2.673 s | 2.862 s | 2.842 s | 0.189 s / 7.1% |
| avx512 | 2.817 s | 3.070 s | 3.067 s | 0.253 s / 9.0% |

Medians. Default total ranges: before 2.654–2.699 s; explicit 2.851–2.865 s;
local 2.833–2.856 s. AVX-512 ranges: before 2.811–2.865 s;
explicit 3.061–3.080 s; local 3.051–3.068 s.

The magetypes library unit itself increased from 1.08 to 1.26 s (16.7%) with
default features, and from 1.24 to 1.49 s (20.2%) with AVX-512 enabled.
The consumer unit was 0.05 s in every run at Cargo's reported precision.
These results measure a small dispatch/arithmetic/conversion consumer, not full
zen repositories. They do not establish a speed difference between explicit and
local calls, or estimate incremental builds or other hosts.

Both APIs use one mode-generic vector implementation. The added frontend work
includes fixed-policy aliases, concrete feature-gated constructor entry points,
and safe conversions between policies. Arithmetic implementations are shared.

## Reproduction and retained evidence

- Harness: [measure-local-mode-compile.py](../scripts/measure-local-mode-compile.py).
- Raw rows: [CSV](local_mode_compile_2026-09-27.csv).
- Toolchain, command, baseline revision, and source-manifest hashes:
  [metadata](local_mode_compile_2026-09-27.meta.json).
- Full source snapshots, per-run build logs, Cargo timing HTML/unit JSON,
  dependency lockfiles, and checksums:
  `/home/lilith/data/archmage/local-mode-compile/2026-09-27/`.
- Baseline includes the pending native raw-constructor fixes, so this comparison
  isolates constructor modes and macro alias selection from those earlier fixes.
- run-heavy: peak RSS 0.40 GiB, minimum available RAM 26,510 MiB; job completed
  in 53 s. These are resource-guard observations, not per-crate memory benchmarks.

Large artifacts are retained outside git; no cloud/NAS mirror is configured on
this host. No target directories or pre-existing caches were deleted.

## Follow-up inspection

The saved Cargo unit timings locate the increase primarily in frontend work.
Median magetypes frontend duration was 1.05 → 1.22 s with default features,
and 1.20 → 1.44 s with AVX-512; codegen was 0.03 → 0.04 s and
0.04 → 0.05 s, respectively. These are sections from the same three-run
measurement, not a new benchmark.

Inspection of the generated public signatures confirms all 42 inventoried
explicit-token method names also have Context-mode signatures. Coverage is the
30 generic vector shapes where their existing backends provide each operation;
backend token traits and standalone scalar-only x1 wrappers were not migrated.

The first optimization experiment should emit `#[target_feature]` and `#[inline]`
from the token registry during xtask generation, instead of invoking `#[rite]`
for each generated context constructor during a consumer build. The tier-only
rite implementation currently emits those attributes and the architecture cfg;
the generated constructor impl already carries the architecture cfg. Preserve
`#[forbid(unsafe_code)]`, compiler rejection tests, and the soundness validator.
The registry must remain the source of feature sets. Savings are not measured.

Other experiments are flattening trivial constructor forwarding layers while
keeping one generator template, and isolating the cost of extra mode trait
bounds. Feature-gating contextual constructors could reduce the surface for
users of only the old API, but introduces a Cargo configuration requirement;
it does not make local-mode users' builds cheaper. None of these alternatives
has been implemented or benchmarked here.

The direct-attribute and forwarding experiments were subsequently implemented
and measured separately. See [the follow-up comparison](constructor_codegen_compile_2026-09-27.md)
for the current generator, retained safety checks, and measured results.
