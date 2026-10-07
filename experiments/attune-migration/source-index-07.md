# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [tests/expand/arcane/destructured_array.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/destructured_array.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn process(token: X64V3Token, [a, b, c, d]: [f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/destructured_nested.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/destructured_nested.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn process(token: X64V3Token, ((x, _), z): ((f32, f32), f32)) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/destructured_tuple.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/destructured_tuple.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn process(token: X64V3Token, (a, b): (f32, f32)) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/dyn_trait_param.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/dyn_trait_param.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn process(token: X64V3Token, callback: &dyn Fn(f32) -> f32, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/fn_ptr_param.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/fn_ptr_param.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn apply_fn(token: X64V3Token, f: fn(f32) -> f32, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/generic_const.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/generic_const.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn sum_n<const N: usize>(token: X64V3Token, data: &[f32; N]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/generic_lifetime.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/generic_lifetime.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn process<'a>(token: X64V3Token, data: &'a [f32]) -> &'a f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/generic_lifetime_and_type.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/generic_lifetime_and_type.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn first_n<'a, T: Copy, const N: usize>( token: X64V3Token, data: &'a [T; N], ) -> &'a T` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/generic_multi_params.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/generic_multi_params.rs#L5)

- L5 [expansion-input]: `#[arcane]` on `fn add_items<T: Add<Output = T> + Copy>(token: X64V3Token, a: T, b: T) -> T` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/generic_where_clause.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/generic_where_clause.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn sum_slice<T>(token: X64V3Token, data: &[T]) -> f32 where T: Copy + Into<f32>,` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/higher_ranked_trait_bound.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/higher_ranked_trait_bound.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn apply_ref<F>(token: X64V3Token, f: F, data: &[f32]) -> f32 where F: for<'a> Fn(&'a [f32]) -> f32,` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/import_intrinsics.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/import_intrinsics.rs#L3)

- L3 [expansion-input]: `#[arcane(import_intrinsics)]` on `fn process(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/multi_bounds.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/multi_bounds.rs#L5)

- L5 [expansion-input]: `#[arcane]` on `fn fma<T>(token: X64V3Token, a: T, b: T, c: T) -> T where T: Copy + Mul<Output = T> + Add<Output = T>,` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/multiple_wildcards.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/multiple_wildcards.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn process(_: X64V3Token, _: f32, _: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/nested.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/nested.rs#L4)

- L4 [expansion-input]: `#[arcane(nested)]` on `fn process(token: X64V3Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/nested_self.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/nested_self.rs#L5)

- L5 [expansion-input]: `#[arcane(_self = Processor)]` on `fn process(&self, token: X64V3Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/return_impl_trait.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/return_impl_trait.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn make_iter(token: X64V3Token, data: &[f32]) -> impl Iterator<Item = &f32>` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/sibling.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/sibling.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn process(token: X64V3Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/token_first.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_first.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process(token: X64V3Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/token_generic.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_generic.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process<T: HasX64V2>(token: T, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/token_last.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_last.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process(a: f32, b: f32, token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/token_middle.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_middle.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process(a: f32, token: X64V3Token, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/token_neon.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_neon.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process(token: NeonToken, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/token_trait_bound.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_trait_bound.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process(token: impl HasX64V2, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/token_v2.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_v2.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process(token: X64V2Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/token_v3.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_v3.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process(token: X64V3Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/token_v4.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_v4.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process(token: X64V4Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/token_wasm.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_wasm.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process(token: Wasm128Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/unsafe_fn.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/unsafe_fn.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `unsafe fn process(token: X64V3Token, ptr: *const f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/wildcard_token.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/wildcard_token.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn process(_: X64V3Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/basic.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/basic.rs#L3)

- L3 [expansion-input]: `#[autoversion]` on `fn sum(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/cfg_feature.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/cfg_feature.rs#L3)

- L3 [expansion-input]: `#[autoversion(cfg(simd_opt))]` on `fn sum(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/default_tier.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/default_tier.rs#L3)

- L3 [expansion-input]: `#[autoversion(+default)]` on `fn sum(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/explicit_gated.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/explicit_gated.rs#L3)

- L3 [expansion-input]: `#[autoversion(v4(cfg(avx512)), v3, neon, scalar)]` on `fn sum(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/explicit_v3_neon_scalar.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/explicit_v3_neon_scalar.rs#L3)

- L3 [expansion-input]: `#[autoversion(v3, neon, scalar)]` on `fn sum(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/explicit_v3_scalar.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/explicit_v3_scalar.rs#L3)

- L3 [expansion-input]: `#[autoversion(v3, scalar)]` on `fn sum(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/plain_self.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/plain_self.rs#L5)

- L5 [expansion-input]: `#[autoversion]` on `fn apply(&self, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/scalar_token.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/scalar_token.rs#L3)

- L3 [expansion-input]: `#[autoversion]` on `fn process(token: ScalarToken, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/self_type.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/self_type.rs#L5)

- L5 [expansion-input]: `#[autoversion(_self = P)]` on `fn apply(&self, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/tier_modifiers.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/tier_modifiers.rs#L3)

- L3 [expansion-input]: `#[autoversion(+arm_v2, -wasm128)]` on `fn sum(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/autoversion/unsafe_fn.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/unsafe_fn.rs#L3)

- L3 [expansion-input]: `#[autoversion]` on `unsafe fn process(ptr: *const f32, len: usize) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/combinations/arcane_calls_rite.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/combinations/arcane_calls_rite.rs#L3)

- L3 [expansion-input]: `#[rite(v3)]` on `fn normalize(v: &[f32; 4]) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L10 [expansion-input]: `#[arcane]` on `fn process(token: X64V3Token, data: &[f32; 4]) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/combinations/autoversion_chain.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/combinations/autoversion_chain.rs#L3)

- L3 [expansion-input]: `#[autoversion]` on `fn inner(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[autoversion]` on `fn outer(data: &[f32; 4], scale: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/combinations/token_downgrade.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/combinations/token_downgrade.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn v3_helper(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn v4_caller(token: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/combinations/token_upgrade.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/combinations/token_upgrade.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn v4_fast(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn v3_with_upgrade(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/deprecated/autoversion_simdtoken.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/deprecated/autoversion_simdtoken.rs#L5)

- L5 [expansion-input]: `#[autoversion]` on `fn process(_token: SimdToken, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/deprecated/dispatch_variant.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/deprecated/dispatch_variant.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn compute_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `dispatch_variant!(compute(x), [v3, scalar])` → [P8](migration.md#p8); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/deprecated/simd_fn.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/deprecated/simd_fn.rs#L5)

- L5 [expansion-input]: `#[simd_fn]` on `fn process(token: X64V3Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P8](migration.md#p8); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/deprecated/simd_route.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/deprecated/simd_route.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn compute_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `simd_route!(compute(x), [v3, scalar])` → [P8](migration.md#p8); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/deprecated/token_target_features.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/deprecated/token_target_features.rs#L3)

- L3 [expansion-input]: `#[token_target_features]` on `fn process(token: X64V3Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P8](migration.md#p8); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/deprecated/token_target_features_boundary.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/deprecated/token_target_features_boundary.rs#L3)

- L3 [expansion-input]: `#[token_target_features_boundary]` on `fn process(token: X64V3Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P8](migration.md#p8); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/incant/default_tiers.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/incant/default_tiers.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L4 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_neon(_t: NeonToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `incant!(inner(x))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/incant/feature_gated.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/incant/feature_gated.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L4 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_neon(_t: NeonToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
