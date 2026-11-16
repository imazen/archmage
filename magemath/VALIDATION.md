# Magemath validation record

This crate was assembled for the 0.9.32 workspace from the magemath sources of
PR #168 (head `7cd3f2bb`): the pow/exp/log families consolidated in `bd83f034`
plus the cube-root family and scalar entry points (`321f42e3`, `0a8d88a0`,
`a8f9280c`). Differences from that source:

- No `Wasm128RelaxedFmaToken` dispatch arm and no compile-time
  `target_feature = "simd128"` gates: WASM arms use `target_arch = "wasm32"`,
  matching the published Magetypes 0.9.31 backend gating.
- Magetypes is unchanged. Its flat approximation methods are neither
  deprecated nor modified, and Magetypes does not depend on Magemath.

Numerical kernels, fits, presets and test expectations are otherwise the
PR #168 versions; no bound or expected value was changed.

## Checks run on this branch

| Check | Host | Result |
|---|---|---|
| `cargo test -p magemath --features "std avx512"` | r7900x (Zen 4, AVX-512) | pass |
| `cargo test -p magemath --no-default-features --features w512` | r7900x | pass |
| `cargo test -p magemath --release --features "std avx512" --test contracts` | r7900x | pass |
| `cargo clippy -p magemath --all-targets --features "std avx512" -- -D warnings` | r7900x | pass |
| `cargo test -p magemath --doc --features std` | r7900x | pass |
| `cargo test -p magemath --features std` | mac (Apple M6, NEON) | pass |
| `cargo test -p magemath --features "std w512" --target wasm32-wasip1`, `+simd128`, wasmtime 40 | r7900x | pass |

i686, Windows ARM64, the no_std WASM run and Miri are covered by the CI
matrix, not claimed here.
