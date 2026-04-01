+++
title = "Understanding Tokens"
weight = 3
+++

# Understanding Tokens

Tokens are the core of archmage's safety model. They're zero-sized proof types that demonstrate CPU feature availability. See [`token-registry.toml`](https://github.com/imazen/archmage/blob/main/token-registry.toml) for the complete token-to-feature mapping.

## The Token Hierarchy

### x86-64 Tokens

| Token | Alias | Features | CPUs |
|-------|-------|----------|------|
| `X64V1Token` | `Sse2Token` | SSE, SSE2 (baseline) | All x86-64 |
| `X64V2Token` | — | + SSE4.2, POPCNT | Nehalem 2008+ |
| `X64CryptoToken` | — | V2 + PCLMULQDQ, AES-NI | Westmere 2010+ |
| `X64V3Token` | — | + AVX2, FMA, BMI1, BMI2 | Haswell 2013+, Zen 1+ |
| `X64V3CryptoToken` | — | V3 + VPCLMULQDQ, VAES | Zen 3+ 2020, Alder Lake 2021+ |
| `X64V4Token` | `Server64`, `Avx512Token` | + AVX-512 F/BW/CD/DQ/VL | Skylake-X 2017+, Zen 4+ |
| `X64V4xToken` | — | + VNNI, VBMI, etc. | Ice Lake 2019+, Zen 4+ |
| `Avx512Fp16Token` | — | + AVX-512 FP16 | Sapphire Rapids 2023+ |

### AArch64 Tokens

| Token | Alias | Features |
|-------|-------|----------|
| `NeonToken` | `Arm64` | NEON (baseline, always available) |
| `Arm64V2Token` | — | + CRC, RDM, DotProd, FP16, AES, SHA2 |
| `Arm64V3Token` | — | + FHM, FCMA, SHA3, I8MM, BF16 |
| `NeonAesToken` | — | NEON + AES |
| `NeonSha3Token` | — | NEON + SHA3 |
| `NeonCrcToken` | — | NEON + CRC |

### WASM Tokens

| Token | Features |
|-------|----------|
| `Wasm128Token` | WASM SIMD128 |
| `Wasm128RelaxedToken` | + Relaxed SIMD |

## Summoning Tokens

```rust
use archmage::{X64V3Token, SimdToken};

// Runtime detection
if let Some(token) = X64V3Token::summon() {
    // CPU has AVX2+FMA
    process_simd(token, data);
} else {
    // Fallback
    process_scalar(data);
}
```

## Compile-Time Guarantees

Check if detection is needed:

```rust
use archmage::{X64V3Token, SimdToken};

match X64V3Token::compiled_with() {
    Some(true) => {
        // Compiled with -Ctarget-cpu=haswell or higher
        // summon() will always succeed, check is elided
        let token = X64V3Token::summon().unwrap();
    }
    Some(false) => {
        // Wrong architecture (e.g., running on ARM)
        // summon() will always return None
    }
    None => {
        // Runtime check needed
        if let Some(token) = X64V3Token::summon() {
            // ...
        }
    }
}
```

## ScalarToken: The Fallback

`ScalarToken` always succeeds—it's for fallback paths:

```rust
use archmage::{ScalarToken, SimdToken};

// Always works
let token = ScalarToken::summon().unwrap();

// Or just construct it directly (it's a unit struct)
let token = ScalarToken;
```

## Token Properties

Tokens are:

- **Zero-sized**: No runtime cost to pass around
- **Copy + Clone**: Pass by value freely
- **Send + Sync**: Safe to share across threads
- **'static**: Can be stored in static variables

```rust
// Zero-sized
assert_eq!(std::mem::size_of::<X64V3Token>(), 0);

// Copy
fn takes_token(token: X64V3Token) {
    let copy = token;  // No move, just copy
    use_both(token, copy);
}
```

## Downcasting Tokens

Higher tokens can be used where lower ones are expected:

```rust
#[arcane(import_intrinsics)]
fn needs_v3(token: X64V3Token, data: &[f32]) { /* ... */ }

if let Some(v4) = X64V4Token::summon() {
    // V4 is a superset of V3 — this works and inlines
    needs_v3(v4, &data);
}
```

V4 includes all V3 features, so the token is valid proof.

### Extraction Methods

Every token has methods to extract any lower-tier token it implies. The method name is the short tier name (`.v1()`, `.v2()`, `.v3()`, `.neon()`, etc.). These are guaranteed, infallible, and zero-cost.

```rust
if let Some(v4) = X64V4Token::summon() {
    let v3: X64V3Token = v4.v3();      // guaranteed — V4 implies V3
    let v2: X64V2Token = v4.v2();      // guaranteed — V4 implies V2
    let v1: X64V1Token = v4.v1();      // guaranteed — V4 implies V1
    let crypto = v4.x64_crypto();      // guaranteed — V4 implies crypto
}

if let Some(arm_v3) = Arm64V3Token::summon() {
    let arm_v2 = arm_v3.arm_v2();      // guaranteed — V3 implies V2
    let neon = arm_v3.neon();           // guaranteed — V3 implies NEON
    let aes = arm_v3.neon_aes();       // guaranteed — V3 implies AES
    let sha3 = arm_v3.neon_sha3();     // guaranteed — V3 implies SHA3
}
```

Use extraction when you have a concrete higher token and need to call a function that takes a specific lower token. This is the most common downcasting pattern.

### Extraction vs `IntoConcreteToken::as_*()`

Don't confuse extraction methods with `IntoConcreteToken::as_*()`. They solve different problems:

| | Extraction (`.v2()`, `.neon()`) | `IntoConcreteToken` (`.as_x64v3()`) |
|---|---|---|
| **Returns** | The lower token directly | `Option<ExactToken>` |
| **Hierarchy-aware** | Yes — follows "implies" relationships | No — identity check only |
| **Use case** | You have V4, need to call a V3 function | You have an unknown `T`, need to branch by type |
| **Fails?** | Never — guaranteed by type system | Returns `None` if token doesn't match exactly |

`as_x64v3()` on an `X64V4Token` returns **`None`** — it checks "are you literally an `X64V3Token`?", not "do you support V3 features?". For hierarchy-aware downcasting, use the extraction methods.

## Trait Bounds

For generic code, use tier traits:

```rust
use archmage::HasX64V2;

fn process<T: HasX64V2>(token: T, data: &[f32]) {
    // Works with X64V2Token, X64V3Token, X64V4Token, etc.
}
```

Available traits:
- `HasX64V2` — SSE4.2 tier
- `HasX64V4` — AVX-512 tier (requires `avx512` feature)
- `HasNeon` — NEON baseline
- `HasNeonAes`, `HasNeonSha3` — NEON extensions
- `HasArm64V2` — Modern ARM compute tier
- `HasArm64V3` — Full modern ARM feature set
