# Raw Intrinsics

If you need direct access to SIMD intrinsics instead of magetypes, use `safe_unaligned_simd` for memory operations. It provides safe wrappers that take references instead of raw pointers.

## When to Use Raw Intrinsics

- Porting existing code that uses intrinsics
- Operations not yet in magetypes
- Maximum control over instruction selection
- Learning how SIMD works under the hood

For new code, prefer [magetypes](../magetypes/overview.md) — it's safer and more ergonomic.

## Setup

```toml
[dependencies]
archmage = "0.4"
safe_unaligned_simd = "0.2"
```

## Basic Pattern

```rust
use archmage::{Desktop64, SimdToken, rite};
use std::arch::x86_64::*;
use safe_unaligned_simd::x86_64 as safe_simd;

pub fn sum_f32(data: &[f32]) -> f32 {
    if let Some(token) = Desktop64::summon() {
        sum_f32_avx2(token, data)
    } else {
        data.iter().sum()
    }
}

#[rite]
fn sum_f32_avx2(_token: Desktop64, data: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();

    for chunk in data.chunks_exact(8) {
        // safe_simd takes references, not pointers
        let v = safe_simd::_mm256_loadu_ps(chunk.try_into().unwrap());
        sum = _mm256_add_ps(sum, v);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps::<1>(sum);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<1>(sum64, sum64));
    let result = _mm_cvtss_f32(sum32);

    // Handle remainder
    result + data.chunks_exact(8).remainder().iter().sum::<f32>()
}
```

## safe_unaligned_simd API

The crate provides reference-based versions of load/store intrinsics:

```rust
use safe_unaligned_simd::x86_64 as safe_simd;

#[rite]
fn process(_: Desktop64, src: &[f32; 8], dst: &mut [f32; 8]) {
    // Load from &[f32; 8]
    let v = safe_simd::_mm256_loadu_ps(src);

    let squared = _mm256_mul_ps(v, v);

    // Store to &mut [f32; 8]
    safe_simd::_mm256_storeu_ps(dst, squared);
}
```

All arithmetic intrinsics (`_mm256_add_ps`, `_mm256_mul_ps`, etc.) are already safe inside `#[rite]` functions since Rust 1.85.

## Architecture-Specific Code

Raw intrinsics are architecture-specific. For cross-platform code, use `#[cfg]`:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub fn process(data: &mut [f32]) {
    if let Some(token) = Desktop64::summon() {
        return process_avx2(token, data);
    }

    if let Some(token) = Arm64::summon() {
        return process_neon(token, data);
    }

    process_scalar(data);
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn process_avx2(_: Desktop64, data: &mut [f32]) {
    // AVX2 intrinsics with safe_simd
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn process_neon(_: Arm64, data: &mut [f32]) {
    // NEON intrinsics with safe_simd
}
```

## Common x86 Intrinsics

### Loads and Stores (use safe_simd)

```rust
use safe_unaligned_simd::x86_64 as safe_simd;

// Unaligned load
let v = safe_simd::_mm256_loadu_ps(&array);

// Unaligned store
safe_simd::_mm256_storeu_ps(&mut array, v);
```

### Arithmetic (safe inside #[rite])

```rust
let sum = _mm256_add_ps(a, b);
let diff = _mm256_sub_ps(a, b);
let prod = _mm256_mul_ps(a, b);
let quot = _mm256_div_ps(a, b);
let fma = _mm256_fmadd_ps(a, b, c);  // a*b + c
```

### Comparisons

```rust
let eq = _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b);
let lt = _mm256_cmp_ps::<_CMP_LT_OQ>(a, b);
let le = _mm256_cmp_ps::<_CMP_LE_OQ>(a, b);
```

### Shuffles and Blends

```rust
let blend = _mm256_blend_ps::<0b10101010>(a, b);
let perm = _mm256_permute_ps::<0b00_01_10_11>(a);
```

## NEON Intrinsics

```rust
use std::arch::aarch64::*;
use safe_unaligned_simd::aarch64 as safe_simd;

#[rite]
fn neon_example(_: Arm64, data: &[f32; 4]) -> f32 {
    let v = safe_simd::vld1q_f32(data);
    let squared = vmulq_f32(v, v);
    vaddvq_f32(squared)
}
```

## Debugging Assembly

Use `cargo asm` to verify your code compiles to expected instructions:

```bash
cargo install cargo-show-asm
cargo asm --lib my_crate::sum_f32_avx2
```

Or use [godbolt.org](https://godbolt.org/) with `-C target-feature=+avx2,+fma`.
