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

Native vector types compile to identical instructions in both baseline
binaries (normalized `objdump` of the V3/V4 stream kernels); only their
addresses moved. On dev the baseline-build native streams nevertheless ran
4–9% slower on main in every process, including the `a * b + c` and bare
`mul_add` controls, and the same held with every function and loop aligned to
64 bytes. dev was shared with other agents (peak load 4.23), so those
native-row ratios are not attributed to the change; the quiet-host rerun below
shows no native difference.

## Quiet-host rerun: i265 and r7900x

Same comparison on two otherwise idle hosts, in parallel, with a five-minute
harness (`quick.py`): six groups (scalar f32x8/f64x4, V3 f32x8/f64x4, scalar
inside a V3 function f32x8/f64x4, plus V4 f32x16 on r7900x), three forms
(`a * b + c`, `mul_add`, `mul_add_portable`), shortened rounds, ten
alternating passes per build. Each process records CPU use on every core except
the pinned one and its SMT sibling; the maximum was 4.8% on i265 and 2.7% on
r7900x, and no run was excluded. Builds: baseline x86-64, the same with
`-Cllvm-args=-align-all-functions=6 -Cllvm-args=-align-loops=64`, and
`-Ctarget-cpu=x86-64-v3`, each for published `e2dbab66` and main `49ae6ae8`.

- i265: Intel Core Ultra 7 265K, pinned to P-core 2, 177 s.
- r7900x: AMD Ryzen 9 7900X (Zen 4, AVX-512 double-pumped), pinned to CPU 2, 261 s.

Current / published, median of ten process medians, baseline build:

| Case | i265 | r7900x |
|---|---:|---:|
| scalar f32x8 stream | 1.21× | 1.10× |
| scalar f32x8 chain | 2.35× | 0.99× |
| scalar f64x4 stream | 4.82× | 4.32× |
| scalar f64x4 chain | 7.11× | 4.25× |
| scalar f32x8 inside V3, stream | 1.88× | 1.44× |
| scalar f32x8 inside V3, chain | 3.39× | 2.98× |
| scalar f64x4 inside V3, stream | 7.38× | 6.32× |
| scalar f64x4 inside V3, chain | 7.18× | 6.52× |
| native V3 (and V4 on r7900x), all forms | 0.99–1.01× | see below |
| global x86-64-v3, all cases | 0.99–1.03× | see below |

All software-path rows above have non-overlapping process ranges. The aligned
build's ratios stay within 2.2% of the baseline build's on every i265 row and
on six of eight r7900x rows; r7900x scalar f32x8 chain moves from 0.99× to
0.92× and scalar f64x4 chain from 4.25× to 4.05×. Code placement shifts those
two by a few percent but does not explain the regression.

On i265 every native and global-V3 row is 0.99–1.03× with overlapping ranges.
On r7900x the native stream rows are multimodal per process: the same binary
lands on discrete levels (for V3 f64x4 `a * b + c`: 766, 1015, 1260 or 1504 ns),
in both revisions, so their ratios (0.81–1.26×, all overlapping) reflect which
levels each side drew, not the code. Chains, which touch no memory, are
0.99–1.00× there.

Not covered by the quiet rerun: randomized inputs (measured only on the shared
dev host, where they changed software-path ratios by at most 4%), ARM, and WASM.

Data: `/mnt/v/output/archmage/fma_recheck_2026-10-24/quick/{i265,r7900x}/` on
`dev` (`summary.txt` SHA-256 `34e14b37…` and `bc32f819…`), binaries in
`artifacts-quick/` with `SHA256SUMS.quick`.
