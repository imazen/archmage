# Understanding Tokens

Tokens are zero-sized proof types. Holding a `Desktop64` token proves the CPU has AVX2+FMA. You can't construct one without passing the runtime check.

## The Basics

```rust
use archmage::{Desktop64, SimdToken};

if let Some(token) = Desktop64::summon() {
    // CPU supports AVX2+FMA
    // token: Desktop64 proves this
    process_simd(token, data);
}
// No token = no SIMD, take scalar path
```

`summon()` does runtime CPU feature detection using CPUID. It returns `Some(token)` if the CPU has the required features, `None` otherwise.

## Available Tokens

### x86-64

| Token | Also Known As | Features | CPUs |
|-------|---------------|----------|------|
| `X64V2Token` | — | SSE4.2 + POPCNT | Intel Nehalem 2008+, AMD Bulldozer 2011+ |
| `X64V3Token` | `Desktop64` | + AVX2 + FMA | Intel Haswell 2013+, AMD Zen 2017+ |
| `X64V4Token` | `Server64` | + AVX-512 | Intel Skylake-X 2017+, AMD Zen 4 2022+ |

**Use `Desktop64` (= `X64V3Token`)** for most code. It's widely available and provides excellent performance.

### ARM

| Token | Also Known As | Features |
|-------|---------------|----------|
| `NeonToken` | `Arm64` | NEON (always available on 64-bit ARM) |
| `NeonAesToken` | — | + AES instructions |
| `NeonSha3Token` | — | + SHA3 instructions |

`Arm64::summon()` always succeeds on 64-bit ARM — NEON is part of the baseline.

### WASM

| Token | Features |
|-------|----------|
| `Simd128Token` | WASM SIMD128 |

Requires compiling with `-Ctarget-feature=+simd128`.

## Token Properties

Tokens are:

- **Zero-sized** — `size_of::<Desktop64>()` is 0
- **Copy + Clone** — pass by value, no moves
- **Send + Sync** — safe across threads
- **Proof of capability** — can't be forged

```rust
// Zero-sized
assert_eq!(std::mem::size_of::<Desktop64>(), 0);

// Copy
fn use_token(token: Desktop64) {
    let copy = token;  // Copy, not move
    both_work(token, copy);
}
```

## Passing Tokens

Tokens cost nothing to pass — they're erased at compile time:

```rust
#[rite]
fn outer(token: Desktop64, data: &[f32]) -> f32 {
    inner(token, data)  // No actual parameter at runtime
}

#[rite]
fn inner(token: Desktop64, data: &[f32]) -> f32 {
    // Use token to construct SIMD types
    f32x8::from_slice(token, &data[..8]).reduce_add()
}
```

## Downcasting

Higher tokens work where lower tokens are expected:

```rust
#[rite]
fn needs_v3(token: X64V3Token, data: &[f32; 8]) -> f32 { ... }

if let Some(v4) = X64V4Token::summon() {
    // V4 is a superset of V3, this works
    needs_v3(v4, &data);
}
```

`X64V4Token` has all features of `X64V3Token` plus AVX-512, so it's valid proof for V3 requirements.

## Cross-Platform

Token types compile everywhere. On unsupported platforms, `summon()` returns `None`:

```rust
// Compiles on ARM, WASM, everywhere
if let Some(token) = Desktop64::summon() {
    // Only runs on x86-64 with AVX2+FMA
    process_avx2(token, data);
} else {
    // Runs everywhere else
    process_scalar(data);
}
```

Check what a token will do at compile time:

```rust
match Desktop64::guaranteed() {
    Some(true) => println!("Guaranteed available (e.g., -Ctarget-cpu=haswell)"),
    Some(false) => println!("Guaranteed unavailable (e.g., compiling for ARM)"),
    None => println!("Runtime check needed"),
}
```

## ScalarToken

A fallback token that always succeeds:

```rust
use archmage::{ScalarToken, SimdToken};

let token = ScalarToken::summon().unwrap();  // Always works
// Or just:
let token = ScalarToken;
```

Used with `incant!` for automatic dispatch with a scalar fallback.

## Caching

`summon()` caches its result — subsequent calls are a single atomic load (~1.3 ns). With `-Ctarget-cpu=native`, the check compiles away entirely.

```
Desktop64::summon() (first call):     2.6 ns
Desktop64::summon() (cached):         1.3 ns
With -Ctarget-cpu=haswell:            0 ns (compiles to unconditional Some)
```
