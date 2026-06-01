# magetypes build.rs compile-time overhead — PR#52 vs main vs PR#51

**Question:** does PR#52's `magetypes/build.rs` capability-probe build script
slow down compiles, hot and cold, on x86 native and aarch64 cross?

**Verdict (short):** No meaningful dev-loop tax. Hot (no-op) rebuild overhead
is **zero** on both arches (the `rerun-if-changed` guard works). The only cost
is a small **one-time cold** hit — and on x86 native the probe never even runs.

## Setup

| | |
|---|---|
| Host | AMD Ryzen 9 7950X, x86_64, 32 threads, 128 GB RAM |
| Toolchain | `rustc 1.95.0 (59807616e 2026-04-14)` stable (host default) |
| Profile measured | dev (`cargo build`, unoptimized + debuginfo) — the dev-loop case the concern is about |
| Tool | hyperfine 1.20.0 + `cargo build --timings` |
| Isolation | crate measured = `cargo build -p magetypes`; all deps (archmage, syn, …) WARM. `cargo clean -p magetypes` cleans only magetypes, leaving deps cached. |
| aarch64 | `aarch64-unknown-linux-gnu` cross-compile (target + `aarch64-linux-gnu-gcc` linker installed); NO emulation, just compile |
| Flags | no `-C target-cpu=native`. Host quiet (load avg 0.45). |
| Branches | main `4bc2a95` · PR#51 `e25aa84` (f16-simd-converter, **no build.rs**) · PR#52 `c6efa9f` (capability-gate, **has build.rs**, 162 lines) |

PR#51 is the right baseline for isolating the build script: it carries the same
f16 source changes as PR#52 but has no `build.rs`. So **PR#52 − PR#51 = the
build-script's contribution alone.**

## Matrix (wall-time, mean ± σ)

| Condition | main | PR#51 | PR#52 | PR#52 − PR#51 |
|---|---|---|---|---|
| **COLD x86 native dev** (clean -p, deps warm) | 2.054 s ± 0.013 | 2.117 s ± 0.017 | 2.195 s ± 0.009 | **+78 ms** |
| **HOT x86 native (no-op rebuild)** | 40.3 ms ± 1.3 | 39.2 ms ± 1.1 | 39.5 ms ± 1.1 | **+0.3 ms (noise)** |
| **COLD aarch64 cross dev** (clean -p, deps warm) | 1.880 s ± 0.023 | 1.867 s ± 0.013 | 2.007 s ± 0.026 | **+140 ms** |
| **HOT aarch64 cross (no-op rebuild)** | 39.7 ms ± 0.4 | 39.2 ms ± 0.5 | 39.6 ms ± 1.0 | **+0.4 ms (noise)** |

(5 runs cold, 8 runs hot. A 3-run cold pass agreed within σ: main 2.034/pr51
2.059/pr52 2.144.)

## build-script-build vs build-script-run decomposition (PR#52, from `cargo --timings`)

| Component | x86 native | aarch64 cross |
|---|---|---|
| **build-script-build** (compile `build.rs` → binary) | 80 ms | 80 ms |
| **build-script-run** (execute the script) | ~0 ms | 20 ms |
| magetypes crate compile (for reference) | 2.03 s | 1.84 s |

- `build-script-build` is **80 ms on both arches** — `build.rs` is always
  compiled host-native, identical work regardless of the eventual target. This
  is the dominant component of the cold overhead, and it is paid **once per
  clean** (not per rebuild).
- `build-script-run` is **~0 on x86** — the probe self-gates on
  `CARGO_CFG_TARGET_ARCH` and **early-returns** before spawning any subprocess.
  Confirmed via `cargo build -vv`: no `cargo:rustc-cfg=archmage_has_neon_f16`
  emitted on x86; on aarch64 it is emitted (probe succeeds on 1.95).
- `build-script-run` is **20 ms on aarch64** — the one `rustc --emit=metadata`
  try-compile of the f16 intrinsic snippet.

### Isolated probe subprocess (the exact rustc invocation the script runs)

```
rustc --edition=2021 --crate-type=lib --emit=metadata \
      --target aarch64-unknown-linux-gnu --out-dir … --cap-lints allow probe.rs
```
**22.7 ms ± 0.6 ms** (15 runs) — agrees with the 20 ms build-script-run above.

## Headlines

- **x86-native cold overhead (PR#52 − PR#51): ~78 ms**, essentially all of it
  the 80 ms one-time `build.rs` *compile* (the probe itself does not run on
  x86). Paid once per `cargo clean`.
- **x86-native HOT overhead: 0 ms.** No-op rebuild is identical to no-build-script
  branches (39.5 vs 39.2 ms — within σ). The `rerun-if-changed=build.rs` +
  `rerun-if-env-changed=RUSTC` guard prevents any rerun. **The guard is not broken.**
- **aarch64-cold probe subprocess cost: 22.7 ms** (the rustc-metadata try-compile),
  on top of the same 80 ms build-script compile → ~140 ms total cold delta.
  Fires only when `build.rs` or `RUSTC` changes.
- **aarch64 HOT overhead: 0 ms** (39.6 vs 39.2 ms — within σ). Probe does NOT
  re-fire on no-op.

## Verdict — negligible for the dev loop

For the everyday dev loop (edit → `cargo build`, an *incremental/hot* rebuild),
the build script costs **nothing measurable** on either arch — the rerun guard
is working as designed, both `rerun-if-changed` and the arch self-gate.

The only cost is a **one-time cold** hit after `cargo clean` (or first build):
- **~80 ms** to compile `build.rs` itself (both arches),
- plus **~23 ms** on aarch64 for the probe subprocess (zero on x86).

That ~80–140 ms is invisible next to the ~2 s cold magetypes compile (≈4–7 %
of one clean build, paid once). It does not recur on incremental builds.

**No fix needed.** The guards the PR ships (arch self-gate + rerun-if-changed)
already eliminate the only overhead a build script could plausibly add to the
dev loop. The single remaining one-time cold cost is the `build.rs` *compile*
(80 ms), which is intrinsic to having any build script.

If that 80 ms one-time cold compile were ever judged worth removing, the only
alternatives that avoid a build-script-compile entirely are: (a) gate the cfg
with `rustversion` attribute macros (`#[rustversion::since(1.94)]`) — moves the
"is this intrinsic available" decision to compile time with no subprocess and no
build-script binary, at the cost of being a version check rather than a true
capability probe (the PR's docstring explains why it preferred probing:
robustness against backports/custom toolchains); or (b) ship the cfg via a
`build-dependency` like `autocfg`/`rustc_version` — but that *adds* a
build-script compile of the dep, so it's strictly worse for cold time, not
better. Given the measured cost is sub-100 ms and one-time, **keeping the
hand-rolled zero-dependency probe is the right call.**

## Raw log

`/tmp/archmage_buildscript_bench_2026-05-29.log` (hyperfine output for all 16
conditions + the isolated probe). cargo-timings HTML regenerable via
`cargo build -p magetypes [--target aarch64-unknown-linux-gnu] --timings` on the
PR#52 checkout.
