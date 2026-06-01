# f16 ↔ f32 slice-conversion: NEON-f16 HW vs software vs scalar (Neoverse-N1)

Measures the **production** `F16Convert::f16_to_f32_slice` / `f32_to_f16_slice`
dispatch path (including the per-call `Arm64V2Token::summon()` fp16 check), not an
inlined kernel microbench.

## Why this exists

The aarch64 NEON-f16 hardware backend (`vcvt_f32_f16`/`vcvt_f16_f32`, gated
`#[rustversion::since(1.94)]`) was originally shipped on the *assumption* that a
1-instruction HW conversion beats the branchless software kernel. This benchmark
replaces that assumption with a measurement, and specifically answers: is the HW
path faster than (a) plain scalar and (b) the NEON-software-branchless fallback
that a `NeonToken` uses without it — once the runtime dispatch cost is included?

## Method

- **Box**: Hetzner CAX (Ampere Altra, **Neoverse-N1**), 8 vCPU, idle (load 0.00).
  `/proc/cpuinfo` reports `fphp asimdhp` → fp16 present → HW path engages.
- **Build**: generic `aarch64-unknown-linux-gnu`, `RUSTFLAGS=""` (NO
  `target-cpu=neoverse-n1`) — runtime feature dispatch is what users get.
- **A/B across the `since(1.94)` gate on the SAME hardware** (no source change):
  - rustc **1.96.0** → `NeonToken` takes the HW `vcvt` path.
  - rustc **1.93.1** → HW kernel not compiled → `NeonToken` takes the branchless
    NEON-f32x4 software kernel.
  - `ScalarToken` = pure-scalar branchless baseline in both builds (its numbers
    match across toolchains → methodology sanity-check passes).
- **Harness**: `magetypes/benches/f16_convert.rs` (zenbench, `harness = false`).
  Run: `cargo +<tc> bench -p magetypes --bench f16_convert --features std`.
- archmage commit: `de0f34b` (release/v0.9.25) + this bench.
- NB: zenbench printed `⚠ only 0 rounds`; validity confirmed instead by (i) stable
  ns/elem across 3 orders of magnitude and (ii) the scalar baseline matching
  across the two toolchains. Ratios are decision-grade.

## Results (mean, total ns/µs)

### decode (f16 → f32)
| n | scalar | NEON-software (1.93) | HW `vcvt` (1.96) | HW vs sw-NEON | HW vs scalar |
|--:|--:|--:|--:|--:|--:|
| 16 | 24.5 ns | 14.2 ns | 15.0 ns | 0.95× | 1.6× |
| 256 | 183 ns | 156 ns | 51.9 ns | 3.0× | 3.5× |
| 4096 | 2.9 µs | 2.4 µs | 720 ns | 3.3× | 3.9× |
| 65536 | 46.0 µs | 39.1 µs | 11.5 µs | 3.4× | 4.0× |
| 1048576 | 743 µs | 634 µs | 243 µs | 2.6× | 3.0× |

### encode (f32 → f16)
| n | scalar | NEON-software (1.93) | HW `vcvt` (1.96) | HW vs sw-NEON | HW vs scalar |
|--:|--:|--:|--:|--:|--:|
| 16 | 18.4 ns | 17.4 ns | 15.0 ns | 1.16× | 1.3× |
| 256 | 238 ns | 214 ns | 51.6 ns | 4.1× | 4.6× |
| 4096 | 3.8 µs | 3.4 µs | 725 ns | 4.7× | 5.1× |
| 65536 | 62.8 µs | 54.5 µs | 11.6 µs | 4.7× | 5.2× |
| 1048576 | 1.00 ms | 909 µs | 353 µs | 2.6× | 2.8× |

## Conclusions

1. **Keep the NEON-f16 HW backend.** `vcvt` beats the NEON-software-branchless
   fallback by **2.6–4.7×** at every size ≥ 256, *including* the memory-bound 1 MP
   case (2.6×) — the bandwidth-equalization hypothesis is falsified.
2. **The `summon()` dispatch is negligible.** At n=16 the HW path is 15 ns total,
   within noise of the software path (14.2 ns) and well under scalar (24.5 ns) —
   the runtime bool check never makes HW lose to scalar.
3. **The software-branchless kernel barely beats scalar (~1.2×)** — it does not
   vectorize well. It's only the rustc<1.94 / no-fp16 fallback, so this is a
   low-priority future optimization, not a blocker; the HW path (the common case
   on modern ARM) carries the win.
