# Trait Reference

Reference for archmage traits.

## SimdToken

The base trait for all capability tokens.

```rust
pub trait SimdToken: Copy + Clone + Send + Sync + 'static {
    const NAME: &'static str;
    fn guaranteed() -> Option<bool>;
    fn summon() -> Option<Self>;
    fn attempt() -> Option<Self>;
}
```

**Implementors**: All token types

## IntoConcreteToken

Enables compile-time dispatch via type checking.

```rust
pub trait IntoConcreteToken: SimdToken {
    fn as_x64v2(self) -> Option<X64V2Token> { None }
    fn as_x64v3(self) -> Option<X64V3Token> { None }
    fn as_x64v4(self) -> Option<X64V4Token> { None }
    fn as_avx512_modern(self) -> Option<Avx512ModernToken> { None }
    fn as_avx512_fp16(self) -> Option<Avx512Fp16Token> { None }
    fn as_neon(self) -> Option<NeonToken> { None }
    fn as_neon_aes(self) -> Option<NeonAesToken> { None }
    fn as_neon_sha3(self) -> Option<NeonSha3Token> { None }
    fn as_neon_crc(self) -> Option<NeonCrcToken> { None }
    fn as_wasm128(self) -> Option<Simd128Token> { None }
    fn as_scalar(self) -> Option<ScalarToken> { None }
}
```

**Usage**:

```rust
fn dispatch<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    if let Some(t) = token.as_x64v3() {
        process_avx2(t, data)
    } else if let Some(t) = token.as_neon() {
        process_neon(t, data)
    } else {
        process_scalar(data)
    }
}
```

Each concrete token returns `Some(self)` for its own method, `None` for others. The compiler eliminates dead branches.

## Tier Traits

### HasX64V2

Marker trait for tokens that provide x86-64-v2 features (SSE4.2+).

```rust
pub trait HasX64V2: SimdToken {}
```

**Implementors**: `X64V2Token`, `X64V3Token`, `X64V4Token`, `Avx512ModernToken`, `Avx512Fp16Token`

**Usage**:

```rust
fn process<T: HasX64V2>(token: T, data: &[f32]) {
    // Can use SSE4.2 intrinsics
}
```

### HasX64V4

Marker trait for tokens that provide x86-64-v4 features (AVX-512).

```rust
#[cfg(feature = "avx512")]
pub trait HasX64V4: SimdToken {}
```

**Implementors**: `X64V4Token`, `Avx512ModernToken`, `Avx512Fp16Token`

**Requires**: `avx512` feature

### HasNeon

Marker trait for tokens that provide NEON features.

```rust
pub trait HasNeon: SimdToken {}
```

**Implementors**: `NeonToken`, `NeonAesToken`, `NeonSha3Token`, `NeonCrcToken`

### HasNeonAes

Marker trait for tokens that provide NEON + AES features.

```rust
pub trait HasNeonAes: HasNeon {}
```

**Implementors**: `NeonAesToken`

### HasNeonSha3

Marker trait for tokens that provide NEON + SHA3 features.

```rust
pub trait HasNeonSha3: HasNeon {}
```

**Implementors**: `NeonSha3Token`

## Width Traits (Deprecated)

> **Warning**: These traits are misleading and should not be used in new code.

### Has128BitSimd (Deprecated)

Only enables SSE/SSE2 (x86 baseline). Does NOT enable SSE4, AVX, or anything useful beyond baseline.

**Use instead**: `HasX64V2` or concrete tokens

### Has256BitSimd (Deprecated)

Only enables AVX (NOT AVX2, NOT FMA). This is almost never what you want.

**Use instead**: `X64V3Token` or `Desktop64`

### Has512BitSimd (Deprecated)

Only enables AVX-512F. Missing critical AVX-512 extensions.

**Use instead**: `X64V4Token` or `HasX64V4`

## magetypes Traits

### SimdTypes

Associates SIMD types with a token.

```rust
pub trait SimdTypes {
    type F32: SimdFloat;
    type F64: SimdFloat;
    type I32: SimdInt;
    type I64: SimdInt;
    // ...
}
```

**Usage**:

```rust
fn process<T: SimdTypes>(token: T, data: &[f32]) {
    let v = T::F32::splat(1.0);
    // ...
}
```

### WidthDispatch

Provides access to all SIMD widths from any token.

```rust
pub trait WidthDispatch {
    fn w128(&self) -> W128Types;
    fn w256(&self) -> Option<W256Types>;
    fn w512(&self) -> Option<W512Types>;
}
```

## Using Traits Correctly

### Prefer Concrete Tokens

```rust
// GOOD: Concrete token, full optimization
fn process(token: X64V3Token, data: &[f32]) { }

// OK: Trait bound, but optimization boundary
fn process<T: HasX64V2>(token: T, data: &[f32]) { }
```

### Trait Bounds at API Boundaries

```rust
// Public API can be generic
pub fn process<T: IntoConcreteToken>(token: T, data: &[f32]) {
    // But dispatch to concrete implementations
    if let Some(t) = token.as_x64v3() {
        process_avx2(t, data);
    }
}

// Internal implementations use concrete tokens
#[arcane]
fn process_avx2(token: X64V3Token, data: &[f32]) { }
```

### Don't Over-Constrain

```rust
// WRONG: Over-constrained, hard to call
fn process<T: HasX64V2 + HasNeon>(token: T, data: &[f32]) {
    // No token implements both!
}

// RIGHT: Use IntoConcreteToken for multi-platform
fn process<T: IntoConcreteToken>(token: T, data: &[f32]) {
    if let Some(t) = token.as_x64v3() {
        // x86 path
    } else if let Some(t) = token.as_neon() {
        // ARM path
    }
}
```
