+++
title = "WASM SIMD"
weight = 4
+++

Keep the same portable `#[magetypes]` and `incant!` source used on x86 and ARM.
For a SIMD128 WebAssembly artifact:

```sh
rustup target add wasm32-unknown-unknown
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
```

WASM token availability follows the compilation target; there is no CPUID-style
probe of the browser from Rust's `summon()`. A module containing SIMD instructions
requires an engine supporting those instructions. A scalar branch within that
module is not a fallback for an engine that rejects the module during validation.
To support older engines, select between separately built scalar and SIMD
artifacts in the host loader using feature detection.

The [gain](@/archmage/getting-started/first-simd.md),
[byte transform](@/magetypes/examples/byte-transforms.md), and
[generic luma](@/magetypes/dispatch/types-and-dispatch.md) examples carry complete
portable call chains. Fixed [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) shapes use multiple SIMD128 vectors.

For raw intrinsics, use `#[arcane(import_intrinsics)]` with `Wasm128Token` and
consult the combined intrinsic namespace. Available safe memory wrappers take
references, not raw pointer casts. Not every pointer intrinsic has a wrapper.

Relaxed SIMD is a separate deployment and numerical choice. Compile with
`-Ctarget-feature=+simd128,+relaxed-simd` for the intended WASM target and check
engine support. Relaxed arithmetic is not interchangeable with strict
cross-ISA numerical contracts. Keep that choice out of the basic portable
kernel unless an actual measured workload requires it.
