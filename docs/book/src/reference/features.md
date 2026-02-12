# Feature Flags

Reference for Cargo feature flags in archmage and magetypes.

## archmage Features

### `std` (default)

Enables standard library support.

```toml
[dependencies]
archmage = "0.6"  # std enabled by default
```

Disable for `no_std`:

```toml
[dependencies]
archmage = { version = "0.6", default-features = false }
```

### `macros` (default)

Enables procedural macros: `#[arcane]`, `#[rite]`, `#[magetypes]`, `incant!`, etc.

```toml
# Disable macros (rare)
archmage = { version = "0.6", default-features = false, features = ["std"] }
```

### `avx512`

Enables AVX-512 tokens and 512-bit types.

```toml
archmage = { version = "0.6", features = ["avx512"] }
```

**Unlocks**:
- `X64V4Token` / `Server64` / `Avx512Token`
- `Avx512ModernToken`
- `Avx512Fp16Token`
- `HasX64V4` trait

### `safe_unaligned_simd`

Re-exports `safe_unaligned_simd` crate in the prelude.

```toml
archmage = { version = "0.6", features = ["safe_unaligned_simd"] }
```

Then use:

```rust
use archmage::prelude::*;
// safe_unaligned_simd functions available
```

## magetypes Features

### `std` (default)

Standard library support.

### `avx512`

Enables 512-bit types.

```toml
magetypes = { version = "0.6", features = ["avx512"] }
```

**Unlocks**:
- `f32x16`, `f64x8`
- `i32x16`, `i64x8`
- `i16x32`, `i8x64`
- `u32x16`, `u64x8`, `u16x32`, `u8x64`

## Feature Combinations

### Full-Featured x86

```toml
[dependencies]
archmage = { version = "0.6", features = ["avx512", "safe_unaligned_simd"] }
magetypes = { version = "0.6", features = ["avx512"] }
```

### Minimal no_std

```toml
[dependencies]
archmage = { version = "0.6", default-features = false, features = ["macros"] }
magetypes = { version = "0.6", default-features = false }
```

### Cross-Platform Library

```toml
[dependencies]
archmage = "0.6"
magetypes = "0.6"

[features]
default = ["std"]
std = ["archmage/std", "magetypes/std"]
avx512 = ["archmage/avx512", "magetypes/avx512"]
```

## Cargo Feature vs CPU Feature

Don't confuse Cargo features with CPU features:

| Cargo Feature | Effect |
|---------------|--------|
| `avx512` | Compiles AVX-512 code paths |
| (none) | Code exists but may not be called |

| CPU Feature | Effect |
|-------------|--------|
| AVX-512 | CPU can execute AVX-512 instructions |
| (none) | Runtime fallback to other path |

```rust
// Cargo feature controls compilation
#[cfg(feature = "avx512")]
fn avx512_path(token: X64V4Token, data: &[f32]) { }

// Token controls runtime dispatch
if let Some(token) = X64V4Token::summon() {  // Runtime check
    avx512_path(token, data);
}
```

## RUSTFLAGS

Not Cargo features, but important compiler flags:

### `-Ctarget-cpu=native`

Compile for current CPU:

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
```

Effects:
- `Token::guaranteed()` returns `Some(true)` for supported features
- `summon()` becomes a no-op
- LLVM generates optimal code for your CPU

### `-Ctarget-cpu=<name>`

Compile for specific CPU:

```bash
# Haswell = AVX2+FMA
RUSTFLAGS="-Ctarget-cpu=haswell" cargo build --release

# Skylake-AVX512 = AVX-512
RUSTFLAGS="-Ctarget-cpu=skylake-avx512" cargo build --release
```

### `-Ctarget-feature=+<feature>`

Enable specific features:

```bash
# Just AVX2
RUSTFLAGS="-Ctarget-feature=+avx2" cargo build

# WASM SIMD
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

## docs.rs Configuration

For docs.rs to show all features:

```toml
# Cargo.toml
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

```rust
// lib.rs
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use tokens::X64V4Token;
```
