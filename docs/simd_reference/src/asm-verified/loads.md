# Load Patterns (ASM-Verified)

All patterns below are verified by `just verify-asm` to produce the expected instructions.

## 256-bit float load (`vmovups`)

### From array reference (baseline)

```rust
#[arcane]
fn load_array_ref(_t: Desktop64, data: &[f32; 8]) -> __m256 {
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(data)
}
```

**ASM:** `vmovups ymm0, [rdi]` — single unaligned load instruction.

### From slice via `.first_chunk()`

```rust
#[arcane]
fn load_first_chunk(_t: Desktop64, data: &[f32]) -> __m256 {
    let arr: &[f32; 8] = data.first_chunk().unwrap();
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(arr)
}
```

**ASM:** `cmp` + `jb` (bounds check) + `vmovups ymm0, [rdi]` — identical load instruction as array ref, with a bounds check that the branch predictor trivially handles.

**Verified identical** to `load_array_ref` (excluding the bounds check prologue).

### From slice via `.try_into()`

```rust
#[arcane]
fn load_try_into(_t: Desktop64, data: &[f32]) -> __m256 {
    let arr: &[f32; 8] = data[..8].try_into().unwrap();
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(arr)
}
```

**ASM:** Same `vmovups` as above. Identical to `first_chunk` approach.

## 256-bit integer load

```rust
#[arcane]
fn load_first_chunk_i(_t: Desktop64, data: &[u8]) -> __m256i {
    let arr: &[u8; 32] = data.first_chunk().unwrap();
    safe_unaligned_simd::x86_64::_mm256_loadu_si256(arr)
}
```

**ASM:** `vmovups ymm0, [rdi]` — the compiler uses `vmovups` for integer loads too. On modern CPUs, `vmovups` and `vmovdqu` are functionally identical for unaligned loads; the compiler picks whichever it prefers.

## 128-bit float load

```rust
#[arcane]
fn load_first_chunk_128(_t: Desktop64, data: &[f32]) -> __m128 {
    let arr: &[f32; 4] = data.first_chunk().unwrap();
    safe_unaligned_simd::x86_64::_mm_loadu_ps(arr)
}
```

**ASM:** `vmovups xmm0, [rdi]` — 128-bit unaligned load.

## Magetypes load

```rust
use magetypes::simd::f32x8;

#[arcane]
fn load_f32x8_from_slice(_t: Desktop64, data: &[f32]) -> f32x8 {
    f32x8::from_slice(_t, data)
}
```

**ASM:** Same `vmovups ymm0, [rdi]` — magetypes' `from_slice` compiles down to the same instruction.

## Key takeaway

All safe load patterns produce the same core instruction. The abstraction cost is zero at the instruction level. The only difference is whether a bounds check is present (slices) or not (array references), and that bounds check is a single `cmp` + conditional jump.

| Pattern | Instruction | Bounds check? |
|---------|-------------|---------------|
| `&[f32; 8]` → `_mm256_loadu_ps` | `vmovups` | No |
| `&[f32]` → `.first_chunk()` → `_mm256_loadu_ps` | `vmovups` | Yes (1 cmp) |
| `&[f32]` → `.try_into()` → `_mm256_loadu_ps` | `vmovups` | Yes (1 cmp) |
| `&[f32]` → `f32x8::from_slice` | `vmovups` | Yes (1 cmp) |
| `&[f32; 8]` → `f32x8::load` | `vmovups` | No |
