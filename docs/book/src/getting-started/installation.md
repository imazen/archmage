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
use archmage::SimdToken;

fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::Desktop64;
        match Desktop64::summon() {
            Some(token) => println!("AVX2+FMA: {} available!", token.name()),
            None => println!("AVX2+FMA not available, would use scalar fallback"),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::Arm64;
        // NEON is always available on aarch64
        let token = Arm64::summon().unwrap();
        println!("NEON: {} available!", token.name());
    }
}
```

Run it:

```bash
cargo run
```

On a modern x86-64 machine (Haswell 2013+ or Zen 1+), you'll see "AVX2+FMA: X64V3 available!".
