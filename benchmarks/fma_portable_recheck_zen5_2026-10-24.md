# Portable FMA recheck: published 0.9.30 versus main, Zen 5, 2026-10-24

The [2026-10-12 comparison](fma_libm_isa_comparison_2026-10-12.md) measured main
`a3372cea`, which called libm directly. The implementation that shipped in
0.9.31-beta.1 (`5dd1c917`) is different: it uses the FMA instruction only when
the build guarantees it at compile time (`std` plus `target_feature = "fma"` on
x86, `neon` on AArch64), and software otherwise (f64/TwoSum/round-to-odd for
f32, libm's generic integer `fma_round` for f64). This reruns the same harness
against current main on one machine.

Only Zen 5 was measured. Intel, Neoverse-N1, Apple M4 Pro and WASM were not
rerun.

## Sources

- Published `v0.9.30`: `e2dbab66`, the archived binaries from the 2026-10-12 run
  (SHA-256 `e2fa14b7…` baseline, `5ebdca32…` x86-64-v3, matching that run's
  provenance).
- Current main: `49ae6ae8` (`git archive`), same harness (`fma_isa.rs`), same
  lockfile versions (libm 0.2.16, zenbench 0.1.10), Rust 1.99.0, release,
  `codegen-units=1`, `avx512` enabled.
- AMD Ryzen 9 9950X3D, pinned to CPU 2, under `run-heavy` (`peak-RSS 0.14GiB`,
  `min-avail 42894MiB`, `peak-load 2.27`, 182 s).
- Commands: `python3 run.py zen5-x86 x86 --isas v3,v4 --cpu 2` and
  `python3 run.py zen5-v3 v3 --isas v3,v4 --cpu 2`; normal scenario only, three
  alternating passes per revision. Ratios are medians of three process medians,
  current / published (above 1 is slower).
- Raw logs, CSV and both current binaries: `/mnt/v/output/archmage/fma_recheck_2026-10-24/`
  on `dev` (`comparison.csv` SHA-256 `0a8ba96e…`).

## Results

| Build | Case | f32 ratio range | f64 ratio range |
|---|---|---:|---:|
| global `-Ctarget-cpu=x86-64-v3` | scalar x1…x16, stream and chain | 0.99–1.07× | 0.98–1.02× |
| global `-Ctarget-cpu=x86-64-v3` | native V3 / V4 widths | 0.98–1.07× | 0.96–1.10× |
| baseline x86-64 | scalar x4…x16 | 1.02–1.38× | 3.0–4.5× |
| baseline x86-64 | scalar x1 | 1.14× stream, 2.6–3.3× chain | 4.2× stream, 5.7–5.9× chain |
| baseline x86-64 | scalar inside a V3 function | 1.45–3.6× | 5.7–11.9× |
| baseline x86-64 | native V3 / V4 widths | 0.91–1.10× (f32 and f64) | |

The 8.95× global-V3 regression measured at `a3372cea` is gone: with FMA
guaranteed at compile time, main matches 0.9.30. Baseline builds, the usual
form of a distributed binary, are slower than 0.9.30 because 0.9.30 checked the
CPU at run time (a cached summon per call) and used the FMA instruction, while
main uses software. f64 pays most, because libm's integer `fma_round` is slower
than the f32 round-to-odd path. Scalar types used inside a V3 function also get
software, since the selection is by build configuration and not by the
enclosing function's target features.

Native vector types stay within run-to-run noise (0.91–1.10×); they use the FMA
intrinsic directly.
