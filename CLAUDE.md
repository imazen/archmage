# archmage - Type-Safe SIMD Capability Tokens

Safe SIMD dispatch through capability proof tokens. Isolates `unsafe` to token construction, enabling safe SIMD code at usage sites.

## Quick Start

```bash
cargo test                    # Run tests
cargo test --all-features     # Test with all integrations
cargo clippy --all-features   # Lint
cargo bench --bench tokens    # Benchmark token overhead
```

## Design Philosophy

### The Problem

Using SIMD safely in Rust is hard because:

1. **`#[cfg(target_feature)]` is compile-time** - Evaluated at crate level, not function level
2. **`#[target_feature]` functions require unsafe to call** - Even from multiversioned functions
3. **Runtime dispatch loses type safety** - No proof that features are available
4. **Crates like `safe_arch` don't compose with multiversion** - They use cfg gating

### The Solution: Capability Tokens

Tokens are zero-sized proof types that can only be constructed when a CPU feature is available:

```rust
// Token creation - the ONLY unsafe point
let token = Avx2Token::new_unchecked();  // unsafe

// All operations are safe - token proves availability
let a = ops::load_f32x8(token, &data);   // safe!
let b = ops::shuffle_f32x8::<0x44>(token, a, a);  // safe!
```

**Why this works:**
- Tokens have private fields - can't be constructed without `new_unchecked()`
- `new_unchecked()` is unsafe - caller must prove feature availability
- Operations take token by value/reference - can't call without valid token
- Zero runtime cost - tokens are ZSTs, optimized away

### Integration Layers

archmage provides **composable** integration with other crates:

```
┌─────────────────────────────────────────────────────────┐
│                    User Code                             │
├─────────────────────────────────────────────────────────┤
│   archmage tokens (Avx2Token, NeonToken, etc.)          │
├──────────┬──────────┬──────────┬───────────────────────┤
│   wide   │safe_simd │multivers.│  raw intrinsics       │
│  types   │  loads   │ dispatch │  (std::arch)          │
└──────────┴──────────┴──────────┴───────────────────────┘
```

Each integration is optional and independent:

- **`wide` feature**: Token-gated `wide::f32x8` operations
- **`safe-simd` feature**: Safe load/store via references
- **`multiversion` feature**: Macros for multiversioned token creation
- **`multiversed` feature**: Higher-level multiversion integration

## Architecture

### Directory Structure

```
src/
├── lib.rs              # Main exports, feature gates
├── tokens/
│   ├── mod.rs          # Token trait and common types
│   ├── x86.rs          # x86_64 tokens (Sse2, Sse41, Avx, Avx2, Avx512, Fma)
│   ├── arm.rs          # ARM tokens (Neon, Sve, Sve2)
│   └── wasm.rs         # WASM tokens (Simd128)
├── ops/
│   ├── mod.rs          # Operation traits
│   ├── x86/
│   │   ├── mod.rs
│   │   ├── load_store.rs   # Memory operations
│   │   ├── arithmetic.rs   # Add, mul, fma, etc.
│   │   ├── shuffle.rs      # Shuffle, permute, blend
│   │   └── compare.rs      # Comparisons, masks
│   ├── arm/
│   │   └── ...
│   └── wasm/
│       └── ...
├── integrate/
│   ├── mod.rs
│   ├── wide.rs         # wide crate integration
│   ├── safe_simd.rs    # safe_unaligned_simd integration
│   └── multiversion.rs # multiversion/multiversed macros
└── composite/
    ├── mod.rs
    ├── transpose.rs    # 8x8 matrix transpose
    ├── dot_product.rs  # Vector dot product
    └── interleave.rs   # Interleave/deinterleave
```

### Token Hierarchy (x86_64)

```
Sse2Token (baseline for x86_64)
    │
    ├── Sse3Token
    │   └── Ssse3Token
    │       └── Sse41Token
    │           └── Sse42Token
    │               └── AvxToken
    │                   └── Avx2Token
    │                       └── Avx512Token (various sub-features)
    │
    └── FmaToken (independent, usually paired with Avx2)
```

Combined tokens for common cases:
- `Avx2FmaToken` - AVX2 + FMA (most common for float ops)
- `Avx512FToken` - AVX-512 Foundation

### Token Trait

```rust
/// Marker trait for SIMD capability tokens
pub trait SimdToken: Copy + Clone + Send + Sync + 'static {
    /// Human-readable name for diagnostics
    const NAME: &'static str;

    /// Runtime detection - returns Some(token) if feature available
    fn try_new() -> Option<Self>;

    /// Unchecked construction - caller must ensure feature is available
    ///
    /// # Safety
    /// Caller must guarantee the CPU feature is available, either via:
    /// - Runtime detection (`is_x86_feature_detected!`)
    /// - Compile-time guarantee (`#[target_feature]` or `-C target-cpu`)
    /// - Being inside a multiversioned function variant
    unsafe fn new_unchecked() -> Self;
}
```

### Integration APIs

#### With `wide` crate

```rust
#[cfg(feature = "wide")]
impl Avx2Token {
    /// Load wide::f32x8 from slice
    pub fn load_f32x8_wide(self, data: &[f32; 8]) -> wide::f32x8 {
        // Uses token to prove AVX2 available
        wide::f32x8::from(*data)
    }

    /// Perform FMA on wide types (requires Avx2FmaToken)
    pub fn fma_f32x8_wide(self, _fma: FmaToken, a: f32x8, b: f32x8, c: f32x8) -> f32x8 {
        a.mul_add(b, c)  // Generates vfmadd with token proof
    }
}
```

#### With `safe_unaligned_simd`

```rust
#[cfg(feature = "safe-simd")]
pub mod safe_ops {
    use safe_unaligned_simd::x86_64 as safe_simd;

    /// Safe unaligned load - token proves feature, reference proves valid memory
    pub fn load_f32x8(token: Avx2Token, data: &[f32; 8]) -> __m256 {
        let _ = token;  // Proves AVX2 available
        unsafe { core::mem::transmute(safe_simd::_mm256_loadu_ps(data)) }
    }
}
```

#### With `multiversion`/`multiversed`

```rust
/// Create token inside multiversioned function
///
/// # Safety justification
/// The `#[multiversion]` or `#[multiversed]` macro guarantees that this
/// code path only executes when the CPU supports the required features.
#[macro_export]
macro_rules! avx2_token {
    () => {{
        // SAFETY: Only valid inside multiversioned AVX2 code path
        unsafe { $crate::tokens::Avx2Token::new_unchecked() }
    }};
}

// Usage with multiversed:
#[multiversed]
fn my_kernel(data: &mut [f32]) {
    let token = avx2_token!();  // Single unsafe point
    // ... all ops are safe via token
}
```

## Implementation Plan

### Phase 1: Core Tokens (x86_64)
- [ ] Token trait definition
- [ ] Sse2Token, Sse41Token, AvxToken, Avx2Token, FmaToken
- [ ] Combined tokens (Avx2FmaToken)
- [ ] Runtime detection via `is_x86_feature_detected!`
- [ ] Basic ops: load, store, zero, set1

### Phase 2: Full x86_64 Operations
- [ ] Arithmetic: add, sub, mul, div, fma
- [ ] Shuffle: unpacklo/hi, shuffle, permute, blend
- [ ] Compare: cmp, min, max
- [ ] Bitwise: and, or, xor, andnot
- [ ] Convert: cvt between types

### Phase 3: Integrations
- [ ] `wide` feature integration
- [ ] `safe-simd` feature integration
- [ ] `multiversion` macros
- [ ] `multiversed` macros

### Phase 4: Composite Operations
- [ ] 8x8 transpose
- [ ] Dot product
- [ ] Interleave/deinterleave
- [ ] Horizontal sum

### Phase 5: ARM Support
- [ ] NeonToken
- [ ] SveToken, Sve2Token
- [ ] Neon operations
- [ ] SVE operations

### Phase 6: WASM Support
- [ ] Simd128Token
- [ ] WASM SIMD operations

## Key Decisions

### Why ZST Tokens?
Zero-sized types have no runtime cost. The token is purely a compile-time proof that gets optimized away completely.

### Why `Copy` Tokens?
Tokens should be freely copyable - they're proofs, not resources. This allows passing to multiple operations without borrowing complexity.

### Why Unsafe `new_unchecked`?
The single point of unsafety. All other operations are safe because they require a valid token. This is the "trusted computing base" - if token creation is correct, everything else is safe.

### Why Separate Tokens for Each Feature?
Granularity. Some code needs only AVX, some needs AVX2, some needs FMA. Combined tokens (Avx2FmaToken) exist for common cases but aren't required.

### Why Optional Integrations?
Users shouldn't pay for what they don't use. A no_std embedded project using raw intrinsics shouldn't pull in `wide`. A project already using `wide` shouldn't duplicate types.

## Testing Strategy

1. **Unit tests**: Each token, each operation
2. **Integration tests**: Each optional feature combination
3. **Property tests**: Operations match scalar reference
4. **Assembly tests**: Verify correct instructions generated
5. **Cross-compile tests**: Verify ARM/WASM build

## Benchmarking

Compare against:
- Raw intrinsics (baseline)
- `pulp` crate
- `wide` crate directly
- `safe_arch` crate

Measure:
- Token creation overhead (should be zero)
- Operation overhead vs raw intrinsics (should be zero)
- Composite operation performance

## Intrinsic Safety in Rust 1.92

As of Rust 1.92, intrinsics have nuanced safety rules:

### SAFE in `#[target_feature]` Context

All **value-based operations** are safe when called from a function with the appropriate `#[target_feature(enable = "...")]`:

```rust
#[target_feature(enable = "avx2", enable = "fma")]
fn safe_target_feature_fn() {
    // Creation - SAFE
    let zero = _mm256_setzero_ps();
    let ones = _mm256_set1_ps(1.0);
    let vals = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let undefined = _mm256_undefined_ps();  // even this!

    // Arithmetic - SAFE
    let add = _mm256_add_ps(zero, ones);
    let mul = _mm256_mul_ps(add, ones);
    let fma = _mm256_fmadd_ps(add, ones, zero);

    // Shuffle/Blend/Permute - SAFE
    let shuf = _mm256_shuffle_ps::<0b00_01_10_11>(add, ones);
    let blend = _mm256_blend_ps::<0b10101010>(add, ones);

    // Comparison - SAFE
    let cmp = _mm256_cmp_ps::<0>(add, ones);

    // Conversion - SAFE
    let to_int = _mm256_cvtps_epi32(add);
    let to_float = _mm256_cvtepi32_ps(to_int);

    // Bitwise - SAFE
    let and = _mm256_and_ps(add, ones);
}
```

### UNSAFE: Pointer-Based Operations

**All memory operations with raw pointers remain unsafe**, even in `#[target_feature]` context:

```rust
#[target_feature(enable = "avx2")]
fn still_needs_unsafe() {
    let arr = [1.0f32; 8];
    let mut out = [0.0f32; 8];

    // These ALL require unsafe blocks:
    let load_aligned = unsafe { _mm256_load_ps(arr.as_ptr()) };
    let load_unaligned = unsafe { _mm256_loadu_ps(arr.as_ptr()) };
    unsafe { _mm256_store_ps(out.as_mut_ptr(), load_aligned) };
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), load_unaligned) };

    // Masked and gather operations too:
    let mask = _mm256_set1_epi32(-1);
    let maskload = unsafe { _mm256_maskload_ps(arr.as_ptr(), mask) };
    let idx = _mm256_set1_epi32(0);
    let gather = unsafe { _mm256_i32gather_ps::<4>(arr.as_ptr(), idx) };
}
```

### Why safe_unaligned_simd Works

The `safe_unaligned_simd` crate makes loads/stores safe by using **references instead of raw pointers**:

```rust
// Raw intrinsic (unsafe - pointer could be invalid)
unsafe fn _mm256_loadu_ps(ptr: *const f32) -> __m256;

// safe_unaligned_simd (safe - reference guarantees valid memory)
#[target_feature(enable = "avx")]
fn _mm256_loadu_ps(mem_addr: &[f32; 8]) -> __m256 {
    unsafe { arch::_mm256_loadu_ps(mem_addr.as_ptr()) }
}
```

The reference-based API:
1. **Guarantees valid memory** - References in Rust are always valid
2. **Guarantees size** - `&[f32; 8]` is exactly 8 floats
3. **Zero overhead** - Generates identical `vmovups` instructions

Verified assembly (identical for both APIs):
```asm
vmovups ymm0, ymmword ptr [rsp - 64]  ; unaligned load
vmovups ymmword ptr [rsp - 32], ymm0  ; unaligned store
vzeroupper
ret
```

## Optimized Feature Detection

archmage provides `is_x86_feature_available!` that combines compile-time and runtime detection:

```rust
// Standard approach - ALWAYS runtime, even in multiversioned code
if is_x86_feature_detected!("avx2") { ... }  // Runtime check

// archmage approach - compile-time when possible, runtime fallback
if is_x86_feature_available!("avx2") { ... }
```

Inside a `#[target_feature(enable = "avx2")]` function or when compiled with `-C target-cpu=x86-64-v3`:
```asm
; is_x86_feature_available!("avx2") compiles to:
mov al, 1
ret
```

This eliminates nested dispatch overhead in multiversioned code.

## Related Work

- **simd-compare**: Research project that developed this pattern (`~/work/simd-compare`)
- **pulp**: Runtime dispatch, but unsafe at call sites
- **safe_arch**: Compile-time cfg gating, incompatible with multiversion
- **wide**: Portable types, needs target features for performance
- **safe_unaligned_simd**: Safe load/store via references, zero overhead, integrates with tokens

## Known Limitations

1. **Token creation requires unsafe** - Fundamental, can't be avoided
2. **Macros hide unsafe** - `avx2_token!()` contains unsafe, but justified by multiversion context
3. **No automatic feature detection in const** - `try_new()` can't be const
4. **Some intrinsics genuinely unsafe** - Memory ops with raw pointers can't be made safe

## Error Handling

Tokens use Option for fallible creation:
```rust
if let Some(token) = Avx2Token::try_new() {
    // AVX2 path
} else {
    // Fallback path
}
```

No panics, no unwrap in library code.
