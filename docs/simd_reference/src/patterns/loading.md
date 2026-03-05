# Loading Data

Getting data from slices and arrays into SIMD registers.

## Quick Reference

| Pattern | Input | Works in `#[arcane]`? | Notes |
|---------|-------|----------------------|-------|
| `f32x8::load(token, &array)` | `&[f32; 8]` | Yes | Preferred for magetypes |
| `f32x8::from_array(token, array)` | `[f32; 8]` | Yes | Takes by value |
| `f32x8::from_slice(token, slice)` | `&[f32]` | Yes | Panics if `slice.len() < 8` |
| `f32x8::splat(token, 1.0)` | scalar | Yes | Broadcast one value to all lanes |
| `f32x8::zero(token)` | — | Yes | All lanes zero |
| `_mm256_loadu_ps(data)` | `&[f32; 8]` | Yes | safe via `import_intrinsics` (reference-based) |
| `_mm256_loadu_ps(ptr)` | `*const f32` | unsafe | Raw stdarch (pointer-based) |

## Magetypes (recommended)

The simplest path. Magetypes handles the load internally:

```rust
use archmage::{X64V3Token, SimdToken, arcane};
use magetypes::simd::f32x8;

#[arcane(import_intrinsics)]
fn process(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load(token, data);
    v.reduce_add()
}
```

### From a slice

```rust
#[arcane(import_intrinsics)]
fn sum_slice(token: X64V3Token, data: &[f32]) -> f32 {
    let mut total = f32x8::zero(token);
    for chunk in data.chunks_exact(8) {
        total = total + f32x8::from_slice(token, chunk);
    }
    total.reduce_add()
}
```

### From an array by value

```rust
let v = f32x8::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
```

## Safe memory ops (via `import_intrinsics`)

When working with raw intrinsics inside `#[arcane]`, `import_intrinsics` provides safe memory operations that take references instead of raw pointers:

```rust
use archmage::{X64V3Token, arcane};

#[arcane(import_intrinsics)]
fn load_and_square(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    // import_intrinsics provides safe versions: takes &[f32; 8], not *const f32
    let v = _mm256_loadu_ps(data);
    let squared = _mm256_mul_ps(v, v);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, squared);
    out
}
```

## Slice-to-array conversions

When you have a slice but the intrinsic needs an array reference, use `.first_chunk()` or `.try_into()`. Both compile to the same `vmovups` as a direct array reference — see [ASM-Verified Conversions](../asm-verified/conversions.md).

### `.first_chunk::<N>()` (Rust 1.77+)

```rust
#[arcane(import_intrinsics)]
fn load_from_slice(_token: X64V3Token, data: &[f32]) -> __m256 {
    let arr: &[f32; 8] = data.first_chunk().expect("need at least 8 elements");
    _mm256_loadu_ps(arr)
}
```

### `.try_into()`

```rust
#[arcane(import_intrinsics)]
fn load_via_try_into(_token: X64V3Token, data: &[f32]) -> __m256 {
    let arr: &[f32; 8] = data[..8].try_into().unwrap();
    _mm256_loadu_ps(arr)
}
```

Both produce identical assembly — a bounds check followed by `vmovups`. The bounds check is necessary (you're going from a runtime-sized slice to a compile-time-sized reference), but it's a single comparison + branch that the branch predictor handles trivially.

## Integer loads

Same patterns work for integer types. The compiler may use `vmovups` or `vmovdqu` (functionally identical on modern CPUs):

```rust
#[arcane(import_intrinsics)]
fn load_bytes(_token: X64V3Token, data: &[u8; 32]) -> __m256i {
    _mm256_loadu_si256(data)
}
```

## What NOT to do

```rust
// WRONG: Raw pointer load outside unsafe
let v = _mm256_loadu_ps(data.as_ptr());  // Needs unsafe

// WRONG: Aligned load on potentially unaligned data
let v = _mm256_load_ps(data.as_ptr());  // UB if not 32-byte aligned!

// WRONG: summon() + #[arcane] boundary every iteration
for chunk in data.chunks_exact(8) {
    if let Some(token) = X64V3Token::summon() {
        process(token, chunk);  // target-feature boundary per call!
    }
}

// BETTER: summon hoisted, but still a boundary per iteration
if let Some(token) = X64V3Token::summon() {
    for chunk in data.chunks_exact(8) {
        process(token, chunk);  // still calling #[arcane] each iteration
    }
}

// BEST: loop inside #[arcane], #[rite] helpers
if let Some(token) = X64V3Token::summon() {
    process_all(token, data);  // one boundary crossing
}
```
