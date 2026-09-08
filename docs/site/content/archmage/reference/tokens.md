+++
title = "Token Reference"
weight = 1
+++

Use feature requirements, not CPU marketing names, to select tokens. The
[token registry](https://github.com/imazen/archmage/blob/main/token-registry.toml)
is the complete list of instruction features and implication edges. This table
is an orientation, not a replacement for those full feature sets.

| Token family | Purpose |
|---|---|
| `X64V1Token` | x86-64 baseline SSE/SSE2 |
| `X64V2Token` | Extended 128-bit x86 tier |
| `X64V3Token` | AVX2/FMA tier, plus the registry's associated requirements |
| `X64V4Token` | AVX-512 foundation tier |
| `X64V4xToken` | Extended AVX-512 operations |
| x86 crypto/GFNI tokens | Explicit additional cryptographic/finite-field feature groups |
| `NeonToken` | AArch64 NEON |
| `Arm64V2Token`, `Arm64V3Token` | Additional ARM instruction groups |
| ARM AES/SHA3/CRC tokens | Narrow feature groups for specialized kernels |
| `Wasm128Token` | WebAssembly SIMD128 compilation support |
| `Wasm128RelaxedToken` | Relaxed SIMD support; distinct numerical semantics |
| `ScalarToken` | Always-available scalar backend |

`Desktop64` aliases V3; `Arm64` aliases NEON; `Server64` aliases V4. Aliases do
not alter feature requirements. Names exist across architectures, while
`summon()` can return `None`. See [tokens and extraction](@/archmage/getting-started/tokens.md)
for safe examples and [logical type support](@/magetypes/types/platform-notes.md)
for the separate backend matrix.

Detection depends on both CPU and OS support. It can also be overridden by the
test facilities when compile-time guarantees permit. Do not teach blanket
“always succeeds on every machine released after year X” rules.
