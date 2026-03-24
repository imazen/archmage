+++
title = "incant! Macro"
weight = 2
+++

`incant!` automates dispatch to suffixed function variants. Write one call, get automatic fallback through capability tiers.

**Always specify explicit tier lists.** The tier list tells `incant!` exactly which variants exist and in what order to try them.

## Dispatch Flow

{% mermaid() %}
flowchart TD
    CALL["incant!(process(data), [v3, neon, wasm128, scalar])"] --> V3{"X64V3Token::summon()?<br/>(x86)"}
    V3 -->|Some| PV3["process_v3(token, data)"]
    V3 -->|None / wrong arch| NEON{"NeonToken::summon()?<br/>(aarch64)"}
    NEON -->|Some| PN["process_neon(token, data)"]
    NEON -->|None / wrong arch| WASM{"Wasm128Token::summon()?<br/>(wasm32)"}
    WASM -->|Some| PW["process_wasm128(token, data)"]
    WASM -->|None / wrong arch| PS["process_scalar(data)"]

    style CALL fill:#5a3d1e,color:#fff
    style PV3 fill:#2d5a27,color:#fff
    style PN fill:#2d5a27,color:#fff
    style PW fill:#2d5a27,color:#fff
    style PS fill:#1a4a6e,color:#fff
{% end %}

Variants for other architectures are excluded by `#[cfg(target_arch)]` at compile time — you don't need `_neon` when compiling for x86-64, for example. You must define every variant in your tier list, plus `_scalar`.

### Passthrough Mode

{% mermaid() %}
flowchart TD
    CALL["incant!(process(data)<br/>with token, [v3, neon, wasm128, scalar])"] --> CHECK3{"token.as_x64v3()?"}
    CHECK3 -->|Some| PV3["process_v3(v3_token, data)"]
    CHECK3 -->|None| CHECKN{"token.as_neon()?"}
    CHECKN -->|Some| PN["process_neon(neon_token, data)"]
    CHECKN -->|None| CHECKW{"token.as_wasm128()?"}
    CHECKW -->|Some| PW["process_wasm128(wasm_token, data)"]
    CHECKW -->|None| PS["process_scalar(data)"]

    style CALL fill:#5a3d1e,color:#fff
    style PV3 fill:#2d5a27,color:#fff
    style PN fill:#2d5a27,color:#fff
    style PW fill:#2d5a27,color:#fff
    style PS fill:#1a4a6e,color:#fff
{% end %}

Passthrough uses `IntoConcreteToken` to check what the token actually is, without re-summoning.

## Basic Usage

```rust
use archmage::{incant, arcane};

// Define variants with standard suffixes
#[arcane(import_intrinsics)]
fn sum_v3(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = _mm256_loadu_ps(data);
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}

fn sum_scalar(data: &[f32; 8]) -> f32 {
    data.iter().sum()
}

// Dispatch with explicit tier list
pub fn sum(data: &[f32; 8]) -> f32 {
    incant!(sum(data), [v3, neon, wasm128, scalar])
    // Tries: sum_v3 → sum_neon → sum_wasm128 → sum_scalar
}
```

## How It Works

<details>
<summary>Macro expansion (click to expand)</summary>

```rust
// incant!(process(data), [v3, neon, wasm128, scalar]) expands to approximately:
{
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        return process_v3(token, data);
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(token) = NeonToken::summon() {
        return process_neon(token, data);
    }

    #[cfg(target_arch = "wasm32")]
    if let Some(token) = Wasm128Token::summon() {
        return process_wasm128(token, data);
    }

    process_scalar(data)
}
```

</details>

## Tier Suffixes

Always specify which tiers your function supports. Include `scalar` in the tier list — it is required, not implicit. You must also define `fn_scalar(...)`.

| Suffix | Token | Platform |
|--------|-------|----------|
| `_v1` | `X64V1Token` | x86-64 baseline |
| `_v2` | `X64V2Token` | x86-64 SSE4.2 |
| `_x64_crypto` | `X64CryptoToken` | x86-64 V2 + AES-NI |
| `_v3` | `X64V3Token` | x86-64 AVX2+FMA |
| `_v3_crypto` | `X64V3CryptoToken` | x86-64 V3 + VAES |
| `_v4` | `X64V4Token` | x86-64 AVX-512 |
| `_v4x` | `X64V4xToken` | x86-64 AVX-512 extensions |
| `_neon` | `NeonToken` | AArch64 NEON |
| `_neon_aes` | `NeonAesToken` | AArch64 NEON + AES |
| `_neon_sha3` | `NeonSha3Token` | AArch64 NEON + SHA3 |
| `_neon_crc` | `NeonCrcToken` | AArch64 NEON + CRC |
| `_arm_v2` | `Arm64V2Token` | AArch64 modern compute |
| `_arm_v3` | `Arm64V3Token` | AArch64 full modern |
| `_wasm128` | `Wasm128Token` | WASM SIMD128 |
| `_wasm128_relaxed` | `Wasm128RelaxedToken` | WASM Relaxed SIMD |
| `_scalar` | `ScalarToken` | Always required (must be listed explicitly) |

Tier names in the list can use the `_` prefix — `_v3` is identical to `v3`. This matches the suffix pattern on generated function names (`fn_v3`).

Cross-architecture variants are excluded by `#[cfg]` — on x86-64, you need `_v3` and `_scalar`. You don't need `_neon` or `_wasm128` (they're cfg'd out by `incant!`).

## Passthrough Mode

When you already have a token and want to dispatch to specialized variants:

```rust
fn outer<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    // Passthrough: token already obtained, dispatch to best variant
    incant!(process(data) with token, [v3, neon, wasm128, scalar])
}
```

This uses `IntoConcreteToken` to check the token's actual type and dispatch accordingly, without re-summoning.

## Example: Complete Implementation

```rust
use archmage::{arcane, incant, X64V3Token, NeonToken, Wasm128Token, SimdToken};

// AVX2 variant — #[arcane] cfg's this out on non-x86
#[arcane(import_intrinsics)]
fn dot_product_v3(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let mul = _mm256_mul_ps(va, vb);
    let sum = _mm256_hadd_ps(mul, mul);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}

// NEON variant (128-bit, process two halves)
#[arcane(import_intrinsics)]
fn dot_product_neon(token: NeonToken, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let sum1 = {
        let va = vld1q_f32(a.as_ptr());
        let vb = vld1q_f32(b.as_ptr());
        vaddvq_f32(vmulq_f32(va, vb))
    };
    let sum2 = {
        let va = vld1q_f32(a[4..].as_ptr());
        let vb = vld1q_f32(b[4..].as_ptr());
        vaddvq_f32(vmulq_f32(va, vb))
    };
    sum1 + sum2
}

// Scalar fallback
fn dot_product_scalar(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// Public API — explicit tiers
pub fn dot_product(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    incant!(dot_product(a, b), [v3, neon, wasm128, scalar])
}
```

## Multiple x86 Tiers

When you have both AVX-512 and AVX2 implementations:

```rust
pub fn process(data: &mut [f32]) -> f32 {
    incant!(process(data), [v4, v3, neon, wasm128, scalar])
    // Tries: process_v4 → process_v3 → process_neon → process_wasm128 → process_scalar
}
```

The `avx512` cargo feature must be enabled for `_v4` to be compiled.

## Feature-Gated Tiers

Wrap a tier in `cfg(feature)` to conditionally include it based on a Cargo feature:

```rust
pub fn process(data: &mut [f32]) -> f32 {
    incant!(process(data), [v4(cfg(avx512)), v3, neon, scalar])
}
```

The `v4` dispatch arm is wrapped in `#[cfg(feature = "avx512")]` — if the calling crate doesn't define that feature, v4 is silently excluded. The shorthand `v4(avx512)` also works and produces identical output. The `cfg()` form is canonical.

## Tier List Modifiers

Instead of restating the entire default tier list, use `+` and `-` to modify it:

```rust
// Add arm_v2 to the defaults (v4, v3, neon, wasm128, scalar)
incant!(process(data), [+arm_v2])

// Remove tiers you don't need
incant!(process(data), [-neon, -wasm128])

// Make v4 unconditional (overrides the default avx512 gate)
incant!(process(data), [+v4])

// Replace scalar with tokenless default fallback
incant!(process(data), [+default])

// Add a cfg gate to a default tier
incant!(process(data), [+neon(cfg(neon))])

// Combine freely
incant!(process(data), [-neon, -wasm128, +v1])
```

All entries in a tier list must be modifiers (`+`/`-`) or all must be plain names — mixing is a compile error. `+default` replaces `scalar` as the fallback slot.

## When to Use incant!

**Use incant! when:**
- You have multiple platform-specific implementations
- You want automatic fallback through tiers
- Function signatures are similar across variants

**Use manual dispatch when:**
- You need custom fallback logic
- Variants have different signatures
- You want more explicit control
