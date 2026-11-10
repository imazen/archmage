# Powf validation tools

These programs are machine-facing research and regression tools. Human-facing
usage is taught by [the bounded-power example](../../magemath/examples/bounded_powf.rs).
This package is outside the release workspace and is not published. Its `std`,
`w512` and `avx512` features forward to the library being evaluated. A CLI binary
still uses Rust std for I/O when the library's `std` feature is disabled.

| Binary | Input and output | Purpose |
|---|---|---|
| `powf_contract_dump` | Stdin `base_bits exponent_bits domain`; bounded result bits or rejection | MIDP contract/MPFR driver |
| `powf_highp_eval` | Stdin `E id width base_bits exponent_bits ...`; one result row; explicit context argument | Scalar, packed and cross-target HIGHP oracle comparisons |
| `powf_highp_nan_scan` | Explicit context argument; exhaustive NaN-family summary | Opt-in classification scan over 16,777,214 operand encodings |
| `legacy_edge_variance` | Fixed directed inputs; result-bit rows | Before/after legacy exceptional-value ledger |

Build evaluators with `just powf-highp-eval-build`, the full scan with
`just powf-canonical-highp-nan-build`, and replay edge observations with
`just powf-legacy-edge-variance`. `just powf-validation-build` compiles all tools.
The scan runs only when its binary is explicitly invoked. Protocols and numerical
assertions are preserved from their previous locations; frozen historical
artifacts retain the original executable paths.

With the library's `std` feature disabled, baseline x86 cannot use archmage's
std CPU detector. Evaluate `scalar`/`automatic`, or build a matching fixed CPU
profile (for example `RUSTFLAGS=-Ctarget-cpu=x86-64-v3`) for the `native` V3
context. NEON and WASM contexts likewise require their matching target features.

Relocation verification preserves all four source files byte-for-byte. The bounded
and legacy reporters reproduce prior stdout exactly; HIGHP fixture replay and
std/no_std compilation pass on native and cross targets. The NaN scan is compiled
but its full exhaustive run is not repeated for a file move. Evidence on dev is
`/mnt/v/output/archmage/powf-example-cleanup-2026-11-10/`, independently mirrored
and SHA-256 indexed by `CORE.json`.
