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

> **`incant!` targets must be safe-wrapped — use `#[arcane]`, not `#[rite]`.** `incant!` calls each variant from a cold (non-`#[target_feature]`) context, so the variant must be safe to call from there. `#[arcane]` (and `#[magetypes]`) generate a safe wrapper; `#[rite]` does not — calling a `#[rite]` variant from `incant!` is a compile error (`E0133`, requires `unsafe`). Reach `#[rite]` helpers *from inside* an `#[arcane]`/`#[magetypes]` variant instead.

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
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(&mut lanes, v);
    lanes.iter().sum::<f32>()
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

## Tokenless Calls (`without token`)

Inside a tier-annotated body, `incant!(foo(args) without token)` rewrites to a **direct, tokenless** call to the caller's own tier variant — `foo_<caller_tier>(args)` — with no summon, no dispatch, and no token threaded through:

```rust
#[rite(v3, neon)]
fn helper(x: f32) -> f32 { x * 2.0 }       // → helper_v3 / helper_neon (tokenless)

#[rite(v3, neon)]
fn outer(x: f32) -> f32 {
    // In the _v3 variant this is `helper_v3(x)`, in _neon it's `helper_neon(x)`
    incant!(helper(x) without token) + 1.0
}
```

This is the tool for composing **tokenless** multi-tier helpers — those generated by `#[rite(v3, neon)]`, `#[autoversion]`, or the `#[magetypes]` scalar/default variant, none of which take a token. Plain `incant!` and `with token` thread a token to the callee; `without token` is for callees that don't have one. It works in every tier body — `#[rite]`, `#[arcane]`, `#[autoversion]`, and `#[magetypes]` (including their scalar/default variants):

```rust
#[arcane]
fn outer(token: X64V3Token, x: f32) -> f32 {
    // The token is in scope but the callee is tokenless — `without token`
    // emits `helper_v3(x)` and simply ignores the token.
    incant!(helper(x) without token) + 1.0
}
```

**Constraints:**

- `without token` always resolves to the **suffixed** name `foo_<tier>`, so the target **must be a multi-tier / suffixed family** (`#[rite(v3, neon)]`, `#[autoversion]`, `#[magetypes]`). A single-tier `#[rite(v3)]` produces a *bare* `helper` (no suffix) — call it directly as `helper(x)` instead.
- It only works **inside** a tier-macro body. In ordinary (cold) code there is no caller tier to match, so it is a compile error — use a normal `incant!(foo(args), [..])` dispatch there.
- A missing `foo_<tier>` for the caller's tier is a compile error (feature mismatch is caught at build time, never at runtime).

## Example: Complete Implementation

```rust
use archmage::{arcane, incant, X64V3Token, NeonToken, Wasm128Token, SimdToken};

// AVX2 variant — #[arcane] cfg's this out on non-x86
#[arcane(import_intrinsics)]
fn dot_product_v3(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let mul = _mm256_mul_ps(va, vb);
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(&mut lanes, mul);
    lanes.iter().sum::<f32>()
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

Plain tier names may be mixed with `+`/`-` modifiers (issue #48). The mode is chosen automatically: any `+` entry makes the list **additive** (start from the defaults; a plain tier is treated as `+tier`), while `-` entries with no `+` make it an **override** of the plain tiers (where `-tier` removes from the set and `-scalar`/`-default` drop the auto-appended fallback — e.g. `[v3, -scalar]` resolves to just `v3`). An all-plain list still replaces the defaults. `+default` replaces `scalar` as the fallback slot.

## Token Position

Use the `Token` marker in args to control where the summoned token is placed:

```rust
// Token-first (default if Token omitted)
incant!(process(Token, data), [v3, scalar])

// Token-last (matches callees with token as last param)
incant!(process(data, Token), [v3, scalar])
```

Without `Token`, the token is prepended to the args. Including `Token` explicitly is recommended — it documents the callee's expected signature and avoids ambiguity.

## Automatic Rewriting (Zero Overhead)

When `incant!` appears inside an `#[arcane]`, `#[rite]`, or `#[autoversion]` function body, the outer macro rewrites it at compile time to a **direct call** — bypassing the runtime dispatcher entirely.

```rust
#[arcane]
fn outer(token: X64V3Token, data: &[f32; 8]) -> f32 {
    // Rewritten to: inner_v3(token, data) — no summon, no dispatch
    incant!(inner(Token, data), [v3, scalar])
}
```

The rewriter handles:

| Situation | Generated code |
|-----------|----------------|
| Exact tier match (V3 → V3) | `inner_v3(token, data)` — direct call |
| Downgrade (V4 → V3) | `inner_v3(token.v3(), data)` — downgrade method |
| Upgrade available (V3, V4 exists) | `if let Some(t) = V4Token::summon() { inner_v4(t, data) } else { inner_v3(token, data) }` |
| Feature-gated upgrade | `#[cfg(feature = "avx512")] { ... summon V4 ... }` |
| Cross-branch (V4 → V3_crypto) | Summon (V4 can't downgrade to V3_crypto) |
| No same-arch tier | `inner_scalar(ScalarToken, data)` |
| `without token` (tokenless callee) | `inner_<tier>(data)` — no token threaded |

The caller's token variable is recognized by name — it can be `token`, `_token`, `my_simd_proof`, anything. The macro finds it by type in the function signature.

```rust
#[arcane]
fn outer(alligator: X64V3Token, x: f32) -> f32 {
    // `alligator` recognized as the token — passed through correctly
    incant!(inner(alligator, x), [v3, scalar])
}
```

A plain `incant!(foo(args))` needs a token in scope to thread to the callee, so a *single-tier* tokenless `#[rite(v3)]` body has nothing to pass and the plain form is left as-is. To call a tokenless **multi-tier** helper from any tier body, use [`without token`](#tokenless-calls-without-token), which threads no token and resolves directly to `foo_<caller_tier>(args)`.

## When to Use incant!

**Use incant! when:**
- You have multiple platform-specific implementations
- You want automatic fallback through tiers
- Function signatures are similar across variants

**Use manual dispatch when:**
- You need custom fallback logic
- Variants have different signatures
- You want more explicit control
