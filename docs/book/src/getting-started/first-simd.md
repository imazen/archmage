# Your First SIMD Function

Let's write a function that processes arrays of floats 8 at a time using AVX2.

## The Goal

We'll implement a dot product — multiply corresponding elements and sum the results:

```
a = [1, 2, 3, 4, 5, 6, 7, 8]
b = [2, 2, 2, 2, 2, 2, 2, 2]
dot(a, b) = 1*2 + 2*2 + 3*2 + ... + 8*2 = 72
```

## Step 1: Add Dependencies

```toml
[dependencies]
archmage = "0.4"
magetypes = "0.4"
```

## Step 2: Write the Public API

```rust
use archmage::{Desktop64, SimdToken};

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    if let Some(token) = Desktop64::summon() {
        dot_product_simd(token, a, b)
    } else {
        dot_product_scalar(a, b)
    }
}

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
```

This is the **dispatch point**. We check for AVX2+FMA once, then call the appropriate implementation.

## Step 3: Write the SIMD Implementation

```rust
use archmage::{Desktop64, rite};
use magetypes::f32x8;

#[rite]
fn dot_product_simd(token: Desktop64, a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::zero(token);

    // Process 8 floats at a time
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
        let va = f32x8::from_slice(token, chunk_a);
        let vb = f32x8::from_slice(token, chunk_b);
        sum = va.mul_add(vb, sum);  // sum += va * vb (FMA instruction)
    }

    // Reduce 8 lanes to 1
    let mut total = sum.reduce_add();

    // Handle remainder (0-7 elements)
    for (x, y) in remainder_a.iter().zip(remainder_b) {
        total += x * y;
    }

    total
}
```

## What's Happening

**`#[rite]`** tells the compiler this function uses AVX2+FMA instructions. It adds the necessary `#[target_feature]` attribute automatically.

**`f32x8`** is a vector of 8 floats. Operations on it compile to single AVX2 instructions:
- `f32x8::from_slice()` → `vmovups` (load 8 floats)
- `va.mul_add(vb, sum)` → `vfmadd231ps` (fused multiply-add)
- `sum.reduce_add()` → horizontal add sequence

**The token** proves CPU features are available. You can't construct `f32x8` without one.

## Complete Example

```rust
use archmage::{Desktop64, SimdToken, rite};
use magetypes::f32x8;

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    if let Some(token) = Desktop64::summon() {
        dot_product_simd(token, a, b)
    } else {
        dot_product_scalar(a, b)
    }
}

#[rite]
fn dot_product_simd(token: Desktop64, a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::zero(token);

    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
        let va = f32x8::from_slice(token, chunk_a);
        let vb = f32x8::from_slice(token, chunk_b);
        sum = va.mul_add(vb, sum);
    }

    let mut total = sum.reduce_add();
    for (x, y) in remainder_a.iter().zip(remainder_b) {
        total += x * y;
    }
    total
}

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() {
    let a: Vec<f32> = (1..=1000).map(|x| x as f32).collect();
    let b: Vec<f32> = vec![2.0; 1000];

    let result = dot_product(&a, &b);
    println!("dot product: {}", result);
    // dot product: 1001000
}
```

## Key Points

1. **Dispatch once, loop inside** — The `if let Some(token)` check happens once. The loop is inside the SIMD function, not outside calling in.

2. **Use `#[rite]` for SIMD functions** — It enables SIMD instructions. All your SIMD helper functions should use `#[rite]`.

3. **Handle remainders** — When data length isn't a multiple of vector width, process the remainder with scalar code.

4. **Tokens are zero-cost** — They're zero-sized types erased at compile time. Passing them around costs nothing.

## Next: Multiple Helper Functions

When you have multiple SIMD helpers, they all use `#[rite]`:

```rust
#[rite]
fn dot_product_simd(token: Desktop64, a: &[f32], b: &[f32]) -> f32 {
    let products = multiply_vectors(token, a, b);
    sum_vector(token, &products)
}

#[rite]
fn multiply_vectors(token: Desktop64, a: &[f32], b: &[f32]) -> Vec<f32> {
    // ...
}

#[rite]
fn sum_vector(token: Desktop64, data: &[f32]) -> f32 {
    // ...
}
```

All `#[rite]` functions inline into each other with zero overhead because they share the same `#[target_feature]` context.

See [When to Use #\[arcane\]](../concepts/arcane-vs-rite.md) for the rare cases where you need `#[arcane]` instead.
