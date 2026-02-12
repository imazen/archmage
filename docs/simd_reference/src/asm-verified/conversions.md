# Slice Conversions (ASM-Verified)

A common concern: "Does converting a slice to an array reference add overhead?" The answer is no. The compiler eliminates the abstraction entirely.

## The three paths

All three produce identical `vmovups` instructions:

### 1. Direct array reference (baseline)

```rust
#[arcane]
fn load_array_ref(_t: Desktop64, data: &[f32; 8]) -> __m256 {
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(data)
}
```

No conversion needed — the type already matches.

### 2. `.first_chunk::<8>()` (Rust 1.77+)

```rust
#[arcane]
fn load_first_chunk(_t: Desktop64, data: &[f32]) -> __m256 {
    let arr: &[f32; 8] = data.first_chunk().unwrap();
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(arr)
}
```

Borrows the first N elements as an array reference. Panics if the slice is too short.

### 3. `data[..8].try_into().unwrap()`

```rust
#[arcane]
fn load_try_into(_t: Desktop64, data: &[f32]) -> __m256 {
    let arr: &[f32; 8] = data[..8].try_into().unwrap();
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(arr)
}
```

Slices the first 8 elements, then converts the slice reference to an array reference.

## Verified: identical ASM

`just verify-asm` confirms that all three paths produce the same load instruction. The only difference is that paths 2 and 3 include a bounds check (a `cmp` + conditional jump) because they take a dynamically-sized slice.

The bounds check is:
```asm
cmp rsi, 8       ; compare slice length to 8
jb  .panic        ; jump to panic if less
vmovups ymm0, [rdi]  ; the actual load (identical across all paths)
```

This is the minimum necessary — you're asserting a runtime-sized slice has at least 8 elements. The branch predictor handles this perfectly in any non-degenerate usage.

## Which to prefer

| Situation | Recommended | Why |
|-----------|-------------|-----|
| Already have `&[f32; 8]` | Direct | No conversion needed |
| Have a slice, processing in chunks | `.first_chunk()` | Clearest intent, one bounds check |
| Have a slice, already sliced | `.try_into()` | Works when you've already indexed |
| Using magetypes | `f32x8::from_slice(token, slice)` | Handles conversion internally |

## Integer conversions

Same story for integers. `.first_chunk()` on a `&[u8]` slice produces the same `vmovdqu` as a direct `&[u8; 32]` reference:

```rust
#[arcane]
fn load_bytes(_t: Desktop64, data: &[u8]) -> __m256i {
    let arr: &[u8; 32] = data.first_chunk().unwrap();
    safe_unaligned_simd::x86_64::_mm256_loadu_si256(arr)
}
```

**ASM:** `vmovups ymm0, [rdi]` (compiler uses `vmovups` for integer loads too — functionally identical to `vmovdqu` on modern CPUs)

## 128-bit conversions

Works the same at 128-bit width:

```rust
#[arcane]
fn load_128(_t: Desktop64, data: &[f32]) -> __m128 {
    let arr: &[f32; 4] = data.first_chunk().unwrap();
    safe_unaligned_simd::x86_64::_mm_loadu_ps(arr)
}
```

**ASM:** `vmovups xmm0, [rdi]`
