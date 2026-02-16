# Understanding Tokens

Tokens are the core of archmage's safety model. They're zero-sized proof types that demonstrate CPU feature availability.

## The Token Hierarchy

### x86-64 Tokens

| Token | Alias | Features | CPUs |
|-------|-------|----------|------|
| `X64V2Token` | — | SSE4.2, POPCNT | Nehalem 2008+ |
| `X64V3Token` | `Desktop64` | + AVX2, FMA, BMI1, BMI2 | Haswell 2013+, Zen 1+ |
| `X64V4Token` | `Server64`, `Avx512Token` | + AVX-512 F/BW/CD/DQ/VL | Skylake-X 2017+, Zen 4+ |
| `X64V4xToken` | — | + VNNI, VBMI, etc. | Ice Lake 2019+, Zen 4+ |

### AArch64 Tokens

| Token | Alias | Features |
|-------|-------|----------|
| `NeonToken` | `Arm64` | NEON (baseline, always available) |
| `NeonAesToken` | — | + AES instructions |
| `NeonSha3Token` | — | + SHA3 instructions |
| `NeonCrcToken` | — | + CRC instructions |

### WASM Token

| Token | Features |
|-------|----------|
| `Wasm128Token` | WASM SIMD128 |

## Summoning Tokens

```rust
use archmage::{Desktop64, SimdToken};

// Runtime detection
if let Some(token) = Desktop64::summon() {
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
use archmage::{Desktop64, SimdToken};

match Desktop64::guaranteed() {
    Some(true) => {
        // Compiled with -Ctarget-cpu=haswell or higher
        // summon() will always succeed, check is elided
        let token = Desktop64::summon().unwrap();
    }
    Some(false) => {
        // Wrong architecture (e.g., running on ARM)
        // summon() will always return None
    }
    None => {
        // Runtime check needed
        if let Some(token) = Desktop64::summon() {
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
assert_eq!(std::mem::size_of::<Desktop64>(), 0);

// Copy
fn takes_token(token: Desktop64) {
    let copy = token;  // No move, just copy
    use_both(token, copy);
}
```

## Downcasting Tokens

Higher tokens can be used where lower ones are expected:

```rust
#[arcane]
fn needs_v3(token: X64V3Token, data: &[f32]) { /* ... */ }

if let Some(v4) = X64V4Token::summon() {
    // V4 is a superset of V3 — this works and inlines
    needs_v3(v4, &data);
}
```

V4 includes all V3 features, so the token is valid proof.

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
