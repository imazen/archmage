+++
title = "Installation"
weight = 1
aliases = ["archmage/reference/features/", "magetypes/getting-started/installation/"]
+++

Use Rust 1.89 or newer. Add both crates for the vector examples:

```toml
[dependencies]
archmage = "0.9"
magetypes = "0.9"
```

Continue with [Your First SIMD Function](@/archmage/getting-started/first-simd.md).
The guide tracks the repository; additions after the latest published release
are identified in [coverage and version scope](@/magetypes/examples/coverage.md).
API links use docs.rs `latest`, which can lag repository changes.

## Features

| Crate / feature | Default | Effect |
|---|---|---|
| archmage `std` | Yes | Standard-library detection/support |
| archmage `macros` | Compatibility no-op | Macros are included unconditionally |
| archmage `avx512` | No | AVX-512 intrinsic-wrapper and macro support |
| archmage `testable_dispatch` | No | Test-only tier-disabling facilities |
| magetypes `std` | Yes | Forwards archmage standard-library support |
| magetypes `w512` | Yes | Logical 512-bit types, including polyfills |
| magetypes `avx512` | No | Native AVX-512 backends; implies `w512` and archmage AVX-512 support |

Token type names are available independently of the CPU on which you compile.
Cargo flags do not prove that the running CPU supports an instruction.

For a library that exposes optional std and native AVX-512 support:

```toml
[dependencies]
archmage = { version = "0.9", default-features = false }
magetypes = { version = "0.9", default-features = false, features = ["w512"] }

[features]
default = ["std"]
std = ["archmage/std", "magetypes/std"]
avx512 = ["archmage/avx512", "magetypes/avx512"]
```

Disable dependency defaults as well as your own if `--no-default-features`
should produce a no_std dependency graph. Remove `w512` when you only need
128/256-bit logical vectors. Detection without std depends on the target and
compile-time guarantees; do not assume all desktop runtime probes remain.

On WASM, build the SIMD artifact with:

```sh
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

See [WASM deployment](@/archmage/advanced/wasm.md) for module support and fallback.
For known local CPU deployments `-Ctarget-cpu=native` is useful; baseline library
benchmarks and binaries distributed to unknown CPUs need their supported baseline.
