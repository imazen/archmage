# WASM Tokens

## Wasm128Token

**Features:** WASM SIMD128
**Requires:** Compile with `-Ctarget-feature=+simd128`
**Register width:** 128-bit

```rust
use archmage::{Wasm128Token, SimdToken, arcane};

if let Some(token) = Wasm128Token::summon() {
    process_wasm(token, &data);
}
```

WASM SIMD128 provides 128-bit SIMD operations in WebAssembly. Support is available in all major browsers and runtimes (Chrome, Firefox, Safari, Node.js, Wasmtime, Wasmer).

### Detection

Unlike x86 where detection happens at runtime via CPUID, WASM feature availability is a compile-time decision:

```bash
# Build with SIMD128 support
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

When compiled with `+simd128`, `Wasm128Token::summon()` compiles away to always return `Some`. Without the flag, it always returns `None`.

### Type system

WASM uses a single `v128` type for all SIMD operations — floats, integers, any width. The operation determines the interpretation:

```rust
// WASM intrinsic naming: [type]x[lanes]_[op]
f32x4_add(a, b)    // Interpret v128 as 4×f32 and add
i32x4_mul(a, b)    // Interpret v128 as 4×i32 and multiply
i8x16_shuffle(a, b, ...)  // Byte-level shuffle
```

### 256-bit on WASM

Like NEON, WASM only has 128-bit registers. `f32x8` and wider types use polyfills (two 128-bit operations).

### Relaxed SIMD

WASM Relaxed SIMD provides operations where the exact result may vary between engines (browsers/runtimes). Magetypes uses relaxed SIMD for FMA and some transcendentals where the performance gain justifies engine-dependent rounding.

### ScalarToken

`ScalarToken` is always available on all platforms, including WASM. It's the fallback used by `incant!` when no SIMD token is available:

```rust
use archmage::ScalarToken;

// Always succeeds, everywhere
let token = ScalarToken::summon().unwrap();
// Or construct directly
let token = ScalarToken;
```
