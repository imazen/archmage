# ASM Verification Overview

Every claim about generated assembly in this reference is backed by `cargo asm` output that's checked in CI. No hand-waving.

## How it works

1. **Benchmark functions** in `benches/asm_patterns.rs` are marked `#[unsafe(no_mangle)]` so `cargo asm` can find them by name.
2. **`scripts/verify-asm.sh`** extracts the assembly for each function and checks:
   - Required instructions are present (e.g., `vmovups` for a float load)
   - Two functions produce identical instruction sequences (e.g., `first_chunk` vs `array_ref`)
3. **Expected baselines** are stored in `tests/expected-asm/` and compared against.
4. **`just verify-asm`** runs all checks. `just verify-asm-update` refreshes baselines after intentional codegen changes.

## What we verify

### Existing claims (from `benches/asm_inspection.rs`)

| Claim | Status |
|-------|--------|
| `safe_unaligned_simd::_mm256_loadu_ps` compiles to `vmovups` | Verified |
| Safe and unsafe single loads produce identical ASM | Verified |
| `#[rite]` in `#[arcane]` matches manual inline | Verified |

### Load pattern claims (from `benches/asm_patterns.rs`)

| Claim | Status |
|-------|--------|
| `.first_chunk()` produces same `vmovups` as array ref | Verified |
| `.try_into()` produces same `vmovups` as array ref | Verified |
| Integer `.first_chunk()` produces `vmovups`/`vmovdqu` | Verified |
| 128-bit `.first_chunk()` produces `vmovups` | Verified |
| Mutable `.first_chunk_mut()` store produces `vmovups` | Verified |
| magetypes `from_slice` produces `vmovups` | Verified |
| magetypes `load` via `first_chunk` produces `vmovups` | Verified |

## Running verification

```bash
# Check all claims
just verify-asm

# Update baselines after intentional changes
just verify-asm-update

# Inspect a specific function's ASM
cargo asm -p archmage --bench asm_patterns --features "std macros avx512" \
    "asm_patterns::load_first_chunk::__simd_inner_load_first_chunk"
```

## Why this matters

"Zero-cost abstraction" is easy to claim and hard to prove. Compiler output changes between Rust versions, LLVM versions, and optimization levels. By checking ASM in CI, we catch regressions — if a pattern that used to be zero-cost starts generating extra instructions, the test fails and we investigate.

The key insight: `.first_chunk()`, `.try_into()`, and direct array references all produce the same `vmovups`/`vmovdqu` instruction. The bounds check compiles to a single `cmp` + `jb` that the branch predictor handles trivially. There's no intermediate copy, no allocation, no overhead beyond the necessary bounds check.
