# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [tests/self_replacement.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/self_replacement.rs#L341) (continued)

- L341 [source]: `#[arcane(_self = Point)]` on `fn compute_length(&self, _token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L347 [source]: `#[arcane(_self = Point)]` on `fn distance_to_origin(&self, _token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L375 [source]: `#[arcane(_self = Point)]` on `fn translate(&mut self, _token: X64V3Token, dx: f32, dy: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L381 [source]: `#[arcane(_self = Point)]` on `fn reset_to_origin(&mut self, _token: X64V3Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L387 [source]: `#[arcane(_self = Point)]` on `fn scale_in_place(&mut self, _token: X64V3Token, factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L419 [source]: `#[arcane(_self = Point)]` on `fn into_negated(self, _token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L424 [source]: `#[arcane(_self = Point)]` on `fn into_scaled(self, _token: X64V3Token, factor: f32) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L455 [source]: `#[arcane(_self = Point)]` on `fn map_coords(&self, _token: X64V3Token, f: fn(f32) -> f32) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L496 [source]: `#[arcane(_self = Shape)]` on `fn scale_shape(&self, _token: X64V3Token, factor: f32) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L528 [source]: `#[arcane(_self = Point)]` on `fn clamp_to_unit(&self, _token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L572 [source]: `#[arcane(_self = Point)]` on `fn try_negate(&self, _token: X64V3Token) -> Result<Self, &'static str>` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L631 [source]: `#[arcane(_self = Color)]` on `fn invert(&self, _token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L636 [source]: `#[arcane(_self = Color)]` on `fn is_dark(&self, _token: X64V3Token) -> bool` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L666 [source]: `#[arcane(_self = Point)]` on `fn replicate(&self, _token: X64V3Token) -> [Self; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L703 [source]: `#[arcane(_self = Point)]` on `fn method_a(&self, _token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L710 [source]: `#[arcane(_self = Point)]` on `fn method_b(&self, _token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L737 [source]: `#[arcane(_self = Point)]` on `fn deep_transform(&self, _token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L776 [source]: `#[arcane(_self = Point)]` on `fn measure_distance(&self, _token: X64V3Token, target: &Point) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L819 [source]: `#[arcane(_self = Expr)]` on `fn simplify(&self, _token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L861 [source]: `#[arcane(_self = Point)]` on `fn collect_into(&self, _token: X64V3Token, dest: &mut Vec<Self>) where Self: Sized + Clone,` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L905 [source]: `#[arcane(_self = Pair::<f32>)]` on `fn swap(&self, _token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L931 [source]: `#[arcane(_self = Point)]` on `fn negate(&self, _token: Desktop64) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L955 [source]: `#[arcane(_self = Point)]` on `fn describe(&self, _token: X64V3Token) -> String` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L999 [source]: `#[arcane(_self = Container)]` on `fn filtered(&self, _token: X64V3Token, min: f32) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1024 [source]: `#[arcane(_self = Point)]` on `fn scale_and_offset(&self, _token: X64V3Token, scale: f32, offset: f32) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/sibling_expansion.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/sibling_expansion.rs#L18)

- L18 [source]: `#[arcane]` on `fn free_fn_double(token: X64V3Token, data: &[f32; 4]) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L47 [source]: `#[arcane]` on `fn sum(&self, token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L52 [source]: `#[arcane]` on `fn scale(&mut self, token: X64V3Token, factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L59 [source]: `#[arcane]` on `fn into_sum(self, token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L65 [source]: `#[arcane]` on `fn doubled(&self, token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L78 [source]: `#[arcane]` on `fn with_offset(&self, token: X64V3Token) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L135 [source]: `#[arcane]` on `fn wildcard_sibling(_: X64V3Token, val: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L149 [source]: `#[arcane]` on `fn alias_sibling(token: Desktop64, val: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L177 [source]: `#[arcane(_self = Point)]` on `fn compute(&self, token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L182 [source]: `#[arcane(_self = Point)]` on `fn transform(&self, token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L206 [source]: `#[arcane]` on `fn many_args_fn( token: X64V3Token, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32, i: f32, ) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L233 [source]: `#[arcane]` on `fn many_args_method( &self, token: X64V3Token, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, ) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L261 [source]: `#[arcane]` on `fn user_inline_fn(token: X64V3Token, data: &[f32; 4]) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/from_context_weaker_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/from_context_weaker_context.rs#L5)

- L5 [expected-failure]: `#[rite(v1)]` on `fn weaker() -> X64V3Token` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/raw_matching_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/raw_matching_context.rs#L4)

- L4 [soundness-control]: `#[archmage::rite(v3)]` on `fn wrap(v: core::arch::x86_64::__m256) -> f32x8<X64V3Token>` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L8 [soundness-control]: `#[archmage::token_target_features(v4)]` on `fn wrap_superset(v: core::arch::x86_64::__m256) -> f32x8<X64V3Token>` [visibility: implicit; method/trait context unresolved, see source] → [P8](migration.md#p8); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/raw_weaker_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/raw_weaker_context.rs#L3)

- L3 [expected-failure]: `#[archmage::rite(v2)]` on `fn wrap(v: core::arch::x86_64::__m256) -> f32x8<X64V3Token>` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/token_aliasing_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/token_aliasing_exploit.rs#L14)

- L14 [expected-failure]: `#[arcane]` on `fn evil(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/token_shadowing_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/token_shadowing_exploit.rs#L16)

- L16 [expected-failure]: `#[arcane]` on `fn evil(_token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/tokenless_misdeclared_callee.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/tokenless_misdeclared_callee.rs#L6)

- L6 [expected-failure]: `#[rite(v2)]` on `fn outer()` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).
- L7 [expected-failure]: `incant!(helper(), [v2, -scalar])` → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/tokenless_uncovered.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/tokenless_uncovered.rs#L2)

- L2 [expected-failure]: `#[rite(v2)]` on `fn outer()` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).
- L3 [expected-failure]: `incant!(helper(), [v3, -scalar])` → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/trait_aliasing_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/trait_aliasing_exploit.rs#L6)

- L6 [expected-failure]: `#[arcane(import_intrinsics)]` on `fn needs_v2<T: archmage::HasX64V2>(_token: T, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/trait_shadowing_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/trait_shadowing_exploit.rs#L13)

- L13 [expected-failure]: `#[arcane(import_intrinsics)]` on `fn evil(_token: impl HasX64V2, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/trait_shadowing_generic_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/trait_shadowing_generic_exploit.rs#L10)

- L10 [expected-failure]: `#[arcane(import_intrinsics)]` on `fn evil<T: HasX64V2>(_token: T, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness/trait_shadowing_rite_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/trait_shadowing_rite_exploit.rs#L10)

- L10 [expected-failure]: `#[rite(import_intrinsics)]` on `fn evil(_token: impl HasX64V2, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness_edge_cases.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness_edge_cases.rs#L34)

- L34 [source]: `#[arcane]` on `fn process_with_token(token: X64V3Token, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L60 [source]: `#[arcane]` on `fn process_with_closure(token: X64V3Token, data: &[f32; 4], transform: impl Fn(f32) -> f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L80 [source]: `#[autoversion]` on `fn sum_squares(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L110 [source]: `#[arcane]` on `fn v3_work(_token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L115 [source]: `#[arcane]` on `fn v4_calls_v3(token: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L208 [source]: `#[rite(v3)]` on `fn rite_helper(a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L213 [source]: `#[arcane]` on `fn arcane_calls_rite(_token: X64V3Token, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L230 [source]: `#[arcane]` on `fn recursive_sum(token: X64V3Token, data: &[f32], acc: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/soundness_exploits.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness_exploits.rs#L257)

- L257 [source]: `#[archmage::arcane]` on `fn sibling(_token: archmage::X64V3Token, value: u32) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L261 [source]: `#[archmage::arcane(nested)]` on `fn nested(_token: archmage::X64V3Token, value: u32) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L285 [source]: `#[arcane(_self = X64V3Token)]` on `fn splat(self, value: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L293 [source]: `#[arcane(_self = X64V3Token)]` on `fn rotate<const N: i32>(&self, value: i32) -> [i32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/test_direct_safe_simd.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/test_direct_safe_simd.rs#L12)

- L12 [source]: `#[arcane(import_intrinsics)]` on `fn test_value_intrinsics(_token: Desktop64, a: __m256, b: __m256) -> __m256` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L21 [source]: `#[arcane(import_intrinsics)]` on `fn test_safe_load(_token: Desktop64, data: &[f32; 8]) -> __m256` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/test_inline_safe_simd.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/test_inline_safe_simd.rs#L7)

- L7 [source]: `#[arcane(import_intrinsics)]` on `pub fn process(_token: Desktop64, data: &mut [[f32; 8]])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/tier_modifiers.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/tier_modifiers.rs#L32)

- L32 [source]: `#[autoversion(+v1)]` on `fn sum_plus_v1(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L60 [source]: `#[magetypes(+v2)]` on `fn mt_plus_v2(_token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L90 [source]: `#[autoversion(+v4)]` on `fn sum_unconditional_v4(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L117 [source]: `#[autoversion(+default)]` on `fn sum_with_default(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L142 [source]: `#[autoversion(-wasm128)]` on `fn sum_no_wasm(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L153 [source]: `#[autoversion(-neon)]` on `fn sum_no_neon(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L164 [source]: `#[autoversion(-wasm128, -neon, +v1)]` on `fn sum_custom(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L196 [source]: `#[arcane]` on `fn gated_incant_v4(_: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L203 [source]: `#[arcane]` on `fn gated_incant_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L214 [source]: `incant!(gated_incant(x), [+v3(cfg(nonexistent_feature))])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L237 [source]: `#[arcane]` on `fn im_v4(_: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L242 [source]: `#[arcane]` on `fn im_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L256 [source]: `incant!(im(x), [-neon, -wasm128, +v1])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L271 [source]: `incant!(im(x), [v3, -neon, -wasm128, +v1])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L288 [source]: `#[magetypes(-wasm128, +v2)]` on `fn mt_mod(_token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L325 [source]: `#[autoversion(v3, +v1)]` on `fn mix_av(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L340 [source]: `#[magetypes(v3, -scalar)]` on `fn only_v3(_token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [tests/tokenless_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/tokenless_context.rs#L6)

- L6 [source]: `#[magetypes(rite, v3, neon, wasm128, scalar)]` on `fn add<const N: usize>(_token: Token, values: &[u32; N]) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L11 [source]: `#[rite(v3, neon, wasm128, scalar)]` on `fn inner<const N: usize>(values: &[u32; N]) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L13 [source]: `incant!(add::<N>(values), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L16 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn entry(_token: Token, values: &[u32; 3]) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L18 [source]: `incant!(inner::<3>(values) without token)` → [P6](migration.md#p6); apply [contract rules](migration-contracts.md#p12).
