# Archmage & Magetypes

**Archmage** makes SIMD programming in Rust safe and ergonomic. You prove CPU feature availability once with a **capability token**, then write clean code that compiles to raw SIMD instructions.

**Magetypes** provides SIMD vector types (`f32x8`, `i32x4`, etc.) with natural Rust operators.

## Quick Example

```rust
use archmage::{Desktop64, SimdToken, arcane};
use magetypes::f32x8;

#[arcane]
fn multiply(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let a = f32x8::from_array(token, *data);
    let b = f32x8::splat(token, 2.0);
    let c = a * b;
    c.to_array()
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        let result = multiply(token, &[1.0; 8]);
        println!("{:?}", result);
    }
}
```

## How It Works

1. **`summon()`** checks CPU features at runtime. Returns `Some(token)` if the CPU supports AVX2+FMA.

2. **The token** is a zero-sized proof type. You can't fake it — it only exists if the check passed.

3. **`#[arcane]`** tells the compiler to use SIMD instructions. The token parameter proves the CPU supports them.

4. **magetypes** gives you `f32x8`, `i32x4`, etc. with natural operators (`+`, `-`, `*`, `/`).

## Zero Overhead

The safety abstractions exist only at compile time. At runtime, you get the exact same assembly as hand-tuned SIMD:

```
Benchmark: 1000 iterations of 8-float vector operations
  Hand-tuned SIMD:        570 ns
  #[rite] in #[arcane]:   572 ns  ← identical
```

The key is using the right pattern: put loops inside `#[arcane]`, use `#[rite]` for helpers. See [Token Hoisting](./concepts/token-hoisting.md).

## Supported Platforms

| Platform | Tokens | Register Width |
|----------|--------|----------------|
| x86-64 | `X64V2Token`, `X64V3Token`/`Desktop64`, `X64V4Token`/`Server64` | 128-512 bit |
| AArch64 | `NeonToken`/`Arm64`, `NeonAesToken`, `NeonSha3Token` | 128 bit |
| WASM | `Simd128Token` | 128 bit |

## Next Steps

- [Installation](./getting-started/installation.md) — Add archmage to your project
- [Your First SIMD Function](./getting-started/first-simd.md) — Complete walkthrough
- [Understanding Tokens](./getting-started/tokens.md) — Learn the token hierarchy
