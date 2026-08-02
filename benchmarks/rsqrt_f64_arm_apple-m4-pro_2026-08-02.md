# NEON f64 `recip`/`rsqrt` — exact vs estimate-and-refine (Apple M4 Pro)

What the f64 exactness fix of 2026-08-01 costs, measured. The f64 NEON backend
refined `vrecpeq_f64`/`vrsqrteq_f64` with **three** fused Newton steps until then,
which missed its documented "full precision" contract by 1 ULP and was replaced
with exact `FDIV` / `FDIV`+`FSQRT`.

The correctness case was already settled (the contract says full precision). This
answers the separate question: **did making it exact make it slower?** On this
core, no — it made it faster in both directions.

## Provenance

| | |
|---|---|
| Box | Apple **M4 Pro**, 12 cores, macOS (Darwin 25.5.0) |
| Toolchain | rustc 1.97.1 (8bab26f4f 2026-07-14), stable |
| Commit | `786cabb` (main) |
| Bench | `magetypes/benches/rsqrt_arm.rs`, `bench_rsqrt_f64` (`f64x2` / `NeonToken`) |
| Command | `cargo bench -p magetypes --bench rsqrt_arm --features std -- f64` |
| Harness | zenbench 0.1.7 (criterion-compat) |
| Workload | N = 1024 f64 (8 KB, L1-resident → compute-bound), 16 passes/iter, load → op → store |

`1_*` is the form that was replaced; `2_*` is what ships now.

## Results

| kernel | mean ±mad | 95% CI | vs replaced |
|---|--:|--:|--:|
| `f64_rcp_full/1_vrecpe_3step_1ulp` | 7.1 ±0.1 µs | [7.1–7.2] | 1.00× |
| **`f64_rcp_full/2_exact_fdiv`** | **2.2 ±0.0 µs** | [2.2–2.2] | **3.2× faster** |
| `f64_rsqrt_full/1_vrsqrte_3step_1ulp` | 8.0 ±0.0 µs | [8.0–8.0] | 1.00× |
| **`f64_rsqrt_full/2_exact_fdiv_fsqrt`** | **6.5 ±0.1 µs** | [6.4–6.7] | **1.23× faster** |

## Takeaways

1. **The f64 exactness fix is a pure win on this core** — bit-exact *and* faster,
   3.2× for `recip` and 1.23× for `rsqrt`. Three Newton steps at f64 is a long
   dependent chain (each step is a multiply plus a fused FRECPS/FRSQRTS), and
   Apple's FDIV/FSQRT beat it comfortably.

2. **`recip` gains far more than `rsqrt`** because its exact form is a single
   FDIV, whereas exact `rsqrt` still pays FSQRT *and* FDIV serially.

3. **This does NOT generalize to other ARM cores, and must not be assumed to.**
   The f32 analogue is *slower* with exact on Neoverse-N1 — rcp 1.39×, rsqrt
   2.15× (`rsqrt_arm_neoverse-n1_2026-06-21.md`). The estimate-and-refine trick
   pays exactly where hardware divide is slow relative to the estimate, which is
   a per-core property.

## Caveats / not measured

- **f64 on Neoverse-N1 (or any server ARM) is NOT measured.** Given the f32
  result there, exact f64 being *slower* on that class of core is plausible.
  Nobody should quote a number for it until it is run — re-run this bench on the
  Hetzner `arm-big` box to fill the gap.
- zenbench reported "0 rounds" (the criterion-compat path does not run the
  interleaved round-robin), same as the Neoverse f32 report. Per-bench spreads
  were tight (±mad ≤ 0.1 µs, CI within ~3%), and the two arms were run
  back-to-back on an otherwise idle machine.
- L1-resident and compute-bound by construction. Memory-bound working sets will
  compress these gaps as both arms converge on bandwidth.
- Callers who want speed over exactness should use `rcp_approx`/`rsqrt_approx`,
  which are unchanged on every backend.
