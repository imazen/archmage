# Store Patterns (ASM-Verified)

All patterns below are verified by `just verify-asm` to produce the expected instructions.

## 256-bit float store (`vmovups`)

### To array reference

```rust
#[arcane]
fn store_to_array(_t: Desktop64, v: __m256, out: &mut [f32; 8]) {
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(out, v);
}
```

**ASM:** `vmovups [rdi], ymm0` — single unaligned store.

### To slice via `.first_chunk_mut()`

```rust
#[arcane]
fn store_first_chunk_mut(_t: Desktop64, v: __m256, out: &mut [f32]) {
    let arr: &mut [f32; 8] = out.first_chunk_mut().unwrap();
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(arr, v);
}
```

**ASM:** Bounds check + `vmovups [rdi], ymm0` — same store instruction, with a bounds check for the slice-to-array conversion.

## Return by value

Returning a `[f32; 8]` from an `#[arcane]` function:

```rust
#[arcane]
fn load_and_return(_t: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
    let doubled = _mm256_add_ps(v, v);
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, doubled);
    out
}
```

The compiler typically optimizes this to write directly to the caller's return slot — no intermediate buffer. The `let mut out = [0.0; 8]` gets eliminated.

## Magetypes store

```rust
use magetypes::simd::f32x8;

#[arcane]
fn store_f32x8(token: Desktop64, v: f32x8, out: &mut [f32; 8]) {
    v.store(token, out);
}
```

**ASM:** Same `vmovups [rdi], ymm0` — magetypes' `store` compiles to the same instruction.

## Key takeaway

Stores mirror loads: array references produce a bare `vmovups`/`vmovdqu`, slices add a bounds check, and magetypes compiles to the same underlying instruction. No abstraction overhead.
