# How Rust 1.89 brought the safe SIMD story together for Archmage

Archmage requires Rust 1.89+. This isn't arbitrary — 1.89 is the version where Rust's SIMD story finally came together:

- **1.86** stabilized `target_feature_11`: `#[target_feature]` functions can be safe `fn`, and calling between matching `#[target_feature]` contexts is safe. This is what lets `#[arcane]` generate a safe wrapper without `unsafe fn`.
- **1.87** declared value-based `std::arch` intrinsics safe. `_mm256_add_ps` inside a `#[target_feature]` function no longer needs an `unsafe` block.
- **1.88** stabilized `as_chunks` and `as_chunks_mut` on slices — ergonomic SIMD-width chunking without `chunks_exact` + `try_into().unwrap()`.
- **1.89** stabilized all AVX-512 target features and intrinsics — 22 target features, ~857 intrinsic functions, mask types, plus SHA-512/SM3/SM4. Without this, `#[target_feature(enable = "avx512f")]` is a compiler error on stable. [Full inventory →](https://imazen.github.io/archmage/archmage/reference/rust-189-simd/)

Four releases turned Rust SIMD from "unsafe everything" to "zero unsafe with full feature coverage." If you're writing SIMD code in Rust and you're on anything older than 1.89, you're fighting the language instead of using it.
