# Relaxed-WASM fusion probe: call layout (2026-10-25)

Question: `mul_add_fused` on relaxed-SIMD WASM checks a cached probe and uses
`relaxed_madd` when the engine fuses. The first draft of that probe took
2.2× (chain) and 2.9× (stream) the time of `a * b + c` on a fusing engine,
while bare `relaxed_madd` took 0.65× and 1.0×. Where does the cost come from?

Method: one WASM module, every variant interleaved (400 rounds, rotating
order, median of 8-run batches), on dev (Ryzen 9 9950X3D, busy host) with
wasmtime 40.0.1, with and without `-C cranelift-has-fma=false`. Kernels run
1,024 dependent steps (chain) or 1,024 independent vectors (stream). Full
output: [results.txt](results.txt). Source: [src/main.rs](src/main.rs).

f32x4, ns per kernel:

| Variant | Fusing chain | Fusing stream | Non-fusing chain | Non-fusing stream |
|---|---:|---:|---:|---:|
| `a * b + c` | 1115 | 595 | 1112 | 595 |
| bare `relaxed_madd` | 724 | 595 | 1112 | 595 |
| software path alone | 9891 | 5929 | 9869 | 5925 |
| draft: inline check with a cold probe call, inline software | 2467 | 1756 | 11504 | 6697 |
| check, then one cold call for probe + software | 761 | 1246 | 13201 | 6861 |
| check with no call on either side (cost floor) | 756 | 748 | — | — |
| **shipped: probe and software both inline, no call site** | **762** | **861** | **9857** | **5971** |

f64x2 through magetypes, shipped form (software path behind one
`#[inline(never)]` call, because its exact path makes per-lane calls):
fusing 1739 / 1230, non-fusing 11580 / 6701. Before the f64 software path
moved behind one call (back-to-back build of the same harness): fusing
4089 / 3207, non-fusing 11254 / 6740.

Findings:

- A call site in the caller's loop costs more than the check itself, even
  when the call never runs: wasmtime then keeps loop state in memory. One cold
  call fixed chains but left streams at 2.1×; removing every call site brought
  streams to 1.45×.
- The call-free form costs the same as plain software on engines that round
  twice (9857 vs 9869, 5971 vs 5925), so the probe no longer has a downside.
- f64 keeps one call because its software path calls `fma_exact` per lane;
  routing the whole software path through one call cut fusing-engine time to
  0.43×/0.38× of the per-lane-call layout with no measurable change on engines
  that round twice.
