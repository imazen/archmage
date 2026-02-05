# Token Reference

Complete reference for all archmage tokens.

## x86-64 Tokens

### X64V2Token

**Features**: SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT

**CPUs**: Intel Nehalem (2008)+, AMD Bulldozer (2011)+

```rust
use archmage::{X64V2Token, SimdToken};

if let Some(token) = X64V2Token::summon() {
    // 128-bit SSE operations
}
```

### X64V3Token / Desktop64 / Avx2FmaToken

**Features**: All V2 + AVX, AVX2, FMA, BMI1, BMI2, F16C, MOVBE

**CPUs**: Intel Haswell (2013)+, AMD Zen 1 (2017)+

```rust
use archmage::{X64V3Token, Desktop64, SimdToken};

// These are the same type:
let t1: Option<X64V3Token> = X64V3Token::summon();
let t2: Option<Desktop64> = Desktop64::summon();
```

**Aliases**:
- `Desktop64` — Friendly name for typical desktop/laptop CPUs
- `Avx2FmaToken` — Legacy name (deprecated)

### X64V4Token / Server64 / Avx512Token

**Features**: All V3 + AVX-512F, AVX-512BW, AVX-512CD, AVX-512DQ, AVX-512VL

**CPUs**: Intel Skylake-X (2017)+, AMD Zen 4 (2022)+

**Requires**: `avx512` feature

```rust
#[cfg(feature = "avx512")]
use archmage::{X64V4Token, Server64, SimdToken};

if let Some(token) = X64V4Token::summon() {
    // 512-bit AVX-512 operations
}
```

**Aliases**:
- `Server64` — Friendly name for server CPUs
- `Avx512Token` — Direct alias

### Avx512ModernToken

**Features**: All V4 + VPOPCNTDQ, IFMA, VBMI, VNNI, BF16, VBMI2, BITALG, VPCLMULQDQ, GFNI, VAES

**CPUs**: Intel Ice Lake (2019)+, AMD Zen 4 (2022)+

**Requires**: `avx512` feature

```rust
#[cfg(feature = "avx512")]
if let Some(token) = Avx512ModernToken::summon() {
    // Modern AVX-512 extensions
}
```

### Avx512Fp16Token

**Features**: AVX-512FP16

**CPUs**: Intel Sapphire Rapids (2023)+

**Requires**: `avx512` feature

```rust
#[cfg(feature = "avx512")]
if let Some(token) = Avx512Fp16Token::summon() {
    // Native FP16 operations
}
```

## AArch64 Tokens

### NeonToken / Arm64

**Features**: NEON (always available on AArch64)

**CPUs**: All 64-bit ARM processors

```rust
use archmage::{NeonToken, Arm64, SimdToken};

// Always succeeds on AArch64
let token = NeonToken::summon().unwrap();
```

**Alias**: `Arm64`

### NeonAesToken

**Features**: NEON + AES

**CPUs**: Most ARMv8 processors with crypto extensions

```rust
if let Some(token) = NeonAesToken::summon() {
    // AES acceleration available
}
```

### NeonSha3Token

**Features**: NEON + SHA3

**CPUs**: ARMv8.2+ with SHA3 extension

```rust
if let Some(token) = NeonSha3Token::summon() {
    // SHA3 acceleration available
}
```

### NeonCrcToken

**Features**: NEON + CRC

**CPUs**: Most ARMv8 processors

```rust
if let Some(token) = NeonCrcToken::summon() {
    // CRC32 acceleration available
}
```

## WASM Token

### Simd128Token

**Features**: WASM SIMD128

**Requires**: Compile with `-Ctarget-feature=+simd128`

```rust
use archmage::{Simd128Token, SimdToken};

if let Some(token) = Simd128Token::summon() {
    // WASM SIMD128 operations
}
```

## Universal Token

### ScalarToken

**Features**: None (pure scalar fallback)

**Availability**: Always, on all platforms

```rust
use archmage::{ScalarToken, SimdToken};

// Always succeeds
let token = ScalarToken::summon().unwrap();

// Or construct directly
let token = ScalarToken;
```

## SimdToken Trait

All tokens implement `SimdToken`:

```rust
pub trait SimdToken: Copy + Clone + Send + Sync + 'static {
    const NAME: &'static str;

    /// Compile-time guarantee check
    fn guaranteed() -> Option<bool>;

    /// Runtime detection
    fn summon() -> Option<Self>;

    /// Alias for summon()
    fn attempt() -> Option<Self>;

    /// Legacy alias (deprecated)
    fn try_new() -> Option<Self>;

    /// Unsafe construction (deprecated)
    unsafe fn forge_token_dangerously() -> Self;
}
```

### guaranteed()

Returns:
- `Some(true)` — Feature is compile-time guaranteed (e.g., `-Ctarget-cpu=haswell`)
- `Some(false)` — Wrong architecture (token can never exist)
- `None` — Runtime check needed

```rust
match Desktop64::guaranteed() {
    Some(true) => {
        // summon() will always succeed, check is elided
        let token = Desktop64::summon().unwrap();
    }
    Some(false) => {
        // Wrong arch, use fallback
    }
    None => {
        // Need runtime check
        if let Some(token) = Desktop64::summon() {
            // ...
        }
    }
}
```

### summon()

Performs runtime CPU feature detection. Returns `Some(token)` if features are available.

```rust
if let Some(token) = Desktop64::summon() {
    // CPU supports AVX2+FMA
}
```

## Token Size

All tokens are zero-sized:

```rust
use std::mem::size_of;

assert_eq!(size_of::<X64V3Token>(), 0);
assert_eq!(size_of::<NeonToken>(), 0);
assert_eq!(size_of::<ScalarToken>(), 0);
```

Passing tokens has zero runtime cost.
