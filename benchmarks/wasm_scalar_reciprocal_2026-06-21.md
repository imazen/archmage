# WASM & scalar f32 reciprocal `_approx` — bit-hack vs division

Justifies the per-op `_approx` choices on the two backends with **no hardware
reciprocal estimate** (WASM SIMD128, scalar fallback):

- **`rcp_approx` = exact division.** A bit-hack estimate is *not* faster than a
  single reciprocal-division on either backend (it's roughly break-even on WASM
  and ~1.8× **slower** on scalar), and it would be *less* accurate (~16-bit vs
  full ~24-bit). So division wins on both speed and accuracy.
- **`rsqrt_approx` = bit-hack (integer seed + 2 Newton steps, ~17-bit).** The
  exact `1/sqrt` path is sqrt + division, and sqrt is expensive — the bit-hack is
  a large win.

Measured, not assumed. (Earlier in this work `rcp_approx` was briefly a bit-hack
too; the scalar benchmark caught that it was ~1.8× slower than division, so it
was reverted.)

## Provenance

- Host: AMD Ryzen 9 7950X, Linux 6.18 (WSL2)
- rustc 1.96.0; wasmtime 40.0.1
- Harnesses (zenbench has no WASM backend → `std::time::Instant`, L1-resident
  N=2048 f32 stream, load→op→store):
  - `magetypes/examples/wasm_reciprocal_bench.rs`
  - `magetypes/examples/scalar_reciprocal_bench.rs`
- Commands:
  ```sh
  CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime RUSTFLAGS="-C target-feature=+simd128" \
    cargo run -q -p magetypes --release --example wasm_reciprocal_bench \
    --target wasm32-wasip1 --features std
  cargo run -q -p magetypes --release --example scalar_reciprocal_bench --features std
  ```

## WASM (wasmtime, SIMD128), ns/elem

| op | bit-hack ×2 | exact (div / sqrt+div) | ratio |
|----|-------------|------------------------|-------|
| rcp   | 0.137–0.149 | 0.141–0.148 (division) | ~1.0× (no win → division kept) |
| rsqrt | 0.207–0.238 | 0.399–0.419 (sqrt+div) | **1.72–1.94× faster** |

## Scalar (native x86, no-SIMD fallback), ns/elem

| op | bit-hack ×2 | exact (div / sqrt+div) | ratio |
|----|-------------|------------------------|-------|
| rcp   | 0.254–0.261 | 0.143–0.146 (division) | **0.56–0.57× (1.8× slower → division kept)** |
| rsqrt | 0.312–0.334 | 3.54–3.70 (sqrt+div)   | **10.99–11.53× faster** |

## Conclusion

`rsqrt_approx` ships the bit-hack on WASM/scalar (1.9× / 11× faster, ≥17-bit).
`rcp_approx` stays exact division on both (faster *and* more accurate than the
bit-hack). The bit-hack `rsqrt_approx` is verified bit-identical to
`rsqrt_approx_portable` + one `rsqrt_newton_portable` step in
`magetypes/tests/reciprocal_precision.rs`.
