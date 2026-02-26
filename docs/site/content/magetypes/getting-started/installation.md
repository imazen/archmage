+++
title = "Installation"
weight = 1
+++

Magetypes depends on archmage for capability tokens. Add both to your `Cargo.toml`:

```toml
[dependencies]
archmage = "0.8"
magetypes = "0.8"
```

Magetypes re-exports what it needs from archmage, but you'll want archmage directly for `SimdToken`, token types, and macros like `#[arcane]`.

## Feature Flags

### magetypes

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Standard library support |
| `avx512` | no | 512-bit types (`f32x16`, `i32x16`, etc.) |

### archmage

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Standard library support |
| `macros` | yes | `#[arcane]`, `#[rite]`, `incant!` |
| `avx512` | no | AVX-512 token support |

Enable AVX-512 types in both crates:

```toml
[dependencies]
archmage = { version = "0.8", features = ["avx512"] }
magetypes = { version = "0.8", features = ["avx512"] }
```

## `no_std`

Both crates support `no_std` with `alloc`:

```toml
[dependencies]
archmage = { version = "0.8", default-features = false, features = ["macros"] }
magetypes = { version = "0.8", default-features = false }
```

## Platform Requirements

### x86-64

Works out of the box. Tokens detect CPU features at runtime via CPUID.

To compile for a known target CPU (detection compiles away):

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
```

### AArch64

NEON is baseline on 64-bit ARM. `NeonToken::summon()` succeeds on all AArch64 hardware.

### WASM

Enable SIMD128 in your build:

```bash
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

## Verify It Works

```rust
use archmage::{Desktop64, SimdToken};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn print_splat<T: F32x8Backend>(token: T, val: f32) {
    let v = f32x8::<T>::splat(token, val);
    println!("f32x8: {:?}", v.to_array());
}

fn main() {
    match Desktop64::summon() {
        Some(token) => print_splat(token, 42.0),
        None => println!("AVX2+FMA not available on this CPU"),
    }
}
```

```bash
cargo run
```
