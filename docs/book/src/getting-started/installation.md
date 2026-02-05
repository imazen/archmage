# Installation

Add archmage to your `Cargo.toml`:

```toml
[dependencies]
archmage = "0.4"
```

For SIMD vector types with natural operators, also add magetypes:

```toml
[dependencies]
archmage = "0.4"
magetypes = "0.4"
```

## Feature Flags

### archmage

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | ✓ | Standard library support |
| `macros` | ✓ | `#[arcane]`, `incant!`, etc. |
| `avx512` | ✗ | AVX-512 token support |

### magetypes

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | ✓ | Standard library support |
| `avx512` | ✗ | 512-bit types (`f32x16`, etc.) |
| `bytemuck` | ✗ | `Pod`/`Zeroable` implementations |

## Platform Requirements

### x86-64

Works out of the box. Tokens detect CPU features at runtime.

For compile-time optimization on known hardware:

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
```

### AArch64

NEON is baseline on 64-bit ARM—`NeonToken::summon()` always succeeds.

### WASM

Enable SIMD128 in your build:

```bash
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

## Verify Installation

```rust
use archmage::{SimdToken, Desktop64, Arm64};

fn main() {
    // Tokens compile everywhere - summon() returns None on unsupported platforms
    match Desktop64::summon() {
        Some(token) => println!("{} available!", token.name()),
        None => println!("No AVX2+FMA"),
    }

    match Arm64::summon() {
        Some(token) => println!("{} available!", token.name()),
        None => println!("No NEON"),
    }
}
```

Run it:

```bash
cargo run
```

On x86-64 (Haswell+/Zen+): "X64V3 available!" and "No NEON".
On AArch64: "No AVX2+FMA" and "Neon available!".
