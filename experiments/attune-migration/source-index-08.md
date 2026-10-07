# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [tests/expand/incant/feature_gated.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/incant/feature_gated.rs#L7) (continued)

- L7 [expansion-input]: `incant!(inner(x), [v3(cfg(avx_opt)), scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/incant/tier_modifiers.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/incant/tier_modifiers.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L4 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_neon(_t: NeonToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `incant!(inner(x), [-neon, -wasm128])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/incant/token_explicit_first.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/incant/token_explicit_first.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L4 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_neon(_t: NeonToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `incant!(inner(Token, x), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/incant/token_explicit_last.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/incant/token_explicit_last.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn inner_v3(x: f32, _t: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `incant!(inner(x, Token), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/incant/token_prepend.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/incant/token_prepend.rs#L3)

- L3 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L4 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_neon(_t: NeonToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `incant!(inner(x), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/arcane_downgrade.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/arcane_downgrade.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `#[arcane]` on `fn outer(token: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L9 [expansion-input]: `incant!(inner(x), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/arcane_exact.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/arcane_exact.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `#[arcane]` on `fn outer(token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L9 [expansion-input]: `incant!(inner(x), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/arcane_named_token.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/arcane_named_token.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L6 [expansion-input]: `#[arcane]` on `fn outer(alligator: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L9 [expansion-input]: `incant!(inner(alligator, x), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/arcane_token_last.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/arcane_token_last.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v3(x: f32, _t: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L6 [expansion-input]: `#[arcane]` on `fn outer(token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L8 [expansion-input]: `incant!(inner(x, Token), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/arcane_upgrade.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/arcane_upgrade.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `#[arcane]` on `fn outer(token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L9 [expansion-input]: `incant!(inner(x), [v4, v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/arcane_upgrade_gated.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/arcane_upgrade_gated.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `#[arcane]` on `fn outer(token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L9 [expansion-input]: `incant!(inner(x), [v4(cfg(avx512)), v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/autoversion_exact.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/autoversion_exact.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `#[autoversion(v3, scalar)]` on `fn outer(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L9 [expansion-input]: `incant!(inner(x), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/autoversion_upgrade.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/autoversion_upgrade.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `#[autoversion(v3, scalar)]` on `fn outer(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L9 [expansion-input]: `incant!(inner(x), [v4, v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/autoversion_upgrade_gated.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/autoversion_upgrade_gated.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v4(_t: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `#[autoversion(v3, scalar)]` on `fn outer(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L9 [expansion-input]: `incant!(inner(x), [v4(cfg(avx512)), v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/cross_branch_no_downgrade.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/cross_branch_no_downgrade.rs#L5)

- L5 [expansion-input]: `#[arcane]` on `fn inner_v3_crypto(_t: X64V3CryptoToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `#[arcane]` on `fn outer(token: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L9 [expansion-input]: `incant!(inner(x), [v3_crypto, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/magetypes_rite_flag.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/magetypes_rite_flag.rs#L3)

- L3 [expansion-input]: `#[magetypes(rite, v3, scalar)]` on `fn helper(_t: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/rite_exact.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/rite_exact.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L6 [expansion-input]: `#[rite]` on `fn outer(token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L8 [expansion-input]: `incant!(inner(x), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/rite_multi.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/rite_multi.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn inner_neon(_t: NeonToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `#[rite(v3, neon)]` on `fn outer(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L9 [expansion-input]: `incant!(inner(x), [v3, neon, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/rite_tokenless_passthrough.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/rite_tokenless_passthrough.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L6 [expansion-input]: `#[rite(v3)]` on `fn outer(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L8 [expansion-input]: `incant!(inner(x), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/scalar_fallback.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/scalar_fallback.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn inner_neon(_t: NeonToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L6 [expansion-input]: `#[arcane]` on `fn outer(token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L8 [expansion-input]: `incant!(inner(x), [neon, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/without_token_from_arcane.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/without_token_from_arcane.rs#L4)

- L4 [expansion-input]: `#[rite(v3, neon)]` on `fn helper(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[arcane]` on `fn outer(token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L8 [expansion-input]: `incant!(helper(x) without token)` → [P6](migration.md#p6); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rewrite/without_token_rite_multi.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rewrite/without_token_rite_multi.rs#L4)

- L4 [expansion-input]: `#[rite(v3, neon)]` on `fn helper(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L5 [expansion-input]: `#[rite(v3, neon)]` on `fn outer(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L7 [expansion-input]: `incant!(helper(x) without token)` → [P6](migration.md#p6); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/import_intrinsics.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/import_intrinsics.rs#L3)

- L3 [expansion-input]: `#[rite(import_intrinsics)]` on `fn helper(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/modifier_minus_scalar.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/modifier_minus_scalar.rs#L3)

- L3 [expansion-input]: `#[rite(v3, -scalar)]` on `fn helper(a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/modifier_mixed.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/modifier_mixed.rs#L3)

- L3 [expansion-input]: `#[rite(v3, +v4, -scalar)]` on `fn compute(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/modifier_plus_multi.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/modifier_plus_multi.rs#L3)

- L3 [expansion-input]: `#[rite(+v3, +neon)]` on `fn compute(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/multi_v3_neon.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/multi_v3_neon.rs#L3)

- L3 [expansion-input]: `#[rite(v3, neon)]` on `fn compute(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/multi_v3_v4_neon.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/multi_v3_v4_neon.rs#L3)

- L3 [expansion-input]: `#[rite(v3, v4, neon)]` on `fn compute(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/multi_with_cfg.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/multi_with_cfg.rs#L3)

- L3 [expansion-input]: `#[rite(v3, neon, cfg(simd_opt))]` on `fn compute(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/multi_with_scalar_and_default.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/multi_with_scalar_and_default.rs#L3)

- L3 [expansion-input]: `#[rite(v3, neon, wasm128, scalar, default)]` on `fn compute(_t: ScalarToken, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/single_default.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/single_default.rs#L3)

- L3 [expansion-input]: `#[rite(default)]` on `fn compute(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/single_scalar.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/single_scalar.rs#L3)

- L3 [expansion-input]: `#[rite(scalar)]` on `fn compute(_t: ScalarToken, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/single_tier.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/single_tier.rs#L3)

- L3 [expansion-input]: `#[rite(v3)]` on `fn helper(a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/single_token.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/single_token.rs#L3)

- L3 [expansion-input]: `#[rite]` on `fn helper(token: X64V3Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/trait_bound.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/trait_bound.rs#L3)

- L3 [expansion-input]: `#[rite]` on `fn helper(token: impl HasX64V2, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/rite/unsafe_fn.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/rite/unsafe_fn.rs#L3)

- L3 [expansion-input]: `#[rite]` on `unsafe fn helper(token: X64V3Token, ptr: *const f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
