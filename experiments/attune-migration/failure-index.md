# Failure and control file index

Status is fixture intent, not a fresh execution result. See migration-contracts.md#p9 for exceptions, harness and dynamic cases.

- [tests/avx512-cfg-tests/v4-import-intrinsics-no-feature/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/avx512-cfg-tests/v4-import-intrinsics-no-feature/src/lib.rs#L1): expected-failure.
- [tests/compile_fail/autoversion_concrete_token.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/autoversion_concrete_token.rs#L1): expected-failure.
- [tests/compile_fail/featureless_simdtoken.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/featureless_simdtoken.rs#L1): expected-failure.
- [tests/compile_fail/missing_scalar.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/missing_scalar.rs#L1): expected-failure.
- [tests/compile_fail/scalar_default_mutual_exclusion.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/scalar_default_mutual_exclusion.rs#L1): expected-failure.
- [tests/compile_fail/scalar_not_in_tier_list.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/scalar_not_in_tier_list.rs#L1): dormant rejection (not currently run).
- [tests/compile_fail/token_aliasing.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/token_aliasing.rs#L1): expected-failure.
- [tests/compile_fail/token_shadowing.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/token_shadowing.rs#L1): expected-failure.
- [tests/compile_fail/unknown_generic_bound.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/unknown_generic_bound.rs#L1): expected-failure.
- [tests/compile_fail/unknown_trait_bound.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/unknown_trait_bound.rs#L1): expected-failure.
- [tests/compile_fail/wrong_token.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/wrong_token.rs#L1): expected-failure.
- [tests/expand/should-fail/autoversion_trait_impl.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/should-fail/autoversion_trait_impl.rs#L1): expected-failure.
- [tests/expand/should-fail/incant_passthrough.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/should-fail/incant_passthrough.rs#L1): expected-failure.
- [tests/expand/should-fail/rite_trait_impl.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/should-fail/rite_trait_impl.rs#L1): expected-failure.
- [tests/soundness/from_context_fn_pointer.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/from_context_fn_pointer.rs#L1): expected-failure.
- [tests/soundness/from_context_missing_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/from_context_missing_context.rs#L1): expected-failure.
- [tests/soundness/from_context_weaker_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/from_context_weaker_context.rs#L1): expected-failure.
- [tests/soundness/from_context_wrong_arch.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/from_context_wrong_arch.rs#L1): expected-failure.
- [tests/soundness/raw_fn_pointer.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/raw_fn_pointer.rs#L1): expected-failure.
- [tests/soundness/raw_matching_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/raw_matching_context.rs#L1): soundness-control.
- [tests/soundness/raw_missing_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/raw_missing_context.rs#L1): expected-failure.
- [tests/soundness/raw_weaker_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/raw_weaker_context.rs#L1): expected-failure.
- [tests/soundness/sealed_trait_bypass.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/sealed_trait_bypass.rs#L1): expected-failure.
- [tests/soundness/token_aliasing_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/token_aliasing_exploit.rs#L1): expected-failure.
- [tests/soundness/token_shadowing_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/token_shadowing_exploit.rs#L1): expected-failure.
- [tests/soundness/tokenless_misdeclared_callee.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/tokenless_misdeclared_callee.rs#L1): expected-failure.
- [tests/soundness/tokenless_uncovered.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/tokenless_uncovered.rs#L1): expected-failure.
- [tests/soundness/trait_aliasing_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/trait_aliasing_exploit.rs#L1): expected-failure.
- [tests/soundness/trait_shadowing_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/trait_shadowing_exploit.rs#L1): expected-failure.
- [tests/soundness/trait_shadowing_generic_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/trait_shadowing_generic_exploit.rs#L1): expected-failure.
- [tests/soundness/trait_shadowing_rite_exploit.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness/trait_shadowing_rite_exploit.rs#L1): expected-failure.
- [tests/ui/unsafe_gather_requires_unsafe.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/ui/unsafe_gather_requires_unsafe.rs#L1): expected-failure.
- [tests/ui/unsafe_load_requires_unsafe.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/ui/unsafe_load_requires_unsafe.rs#L1): expected-failure.
- [tests/ui/unsafe_maskload_requires_unsafe.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/ui/unsafe_maskload_requires_unsafe.rs#L1): expected-failure.
- [tests/ui/unsafe_store_requires_unsafe.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/ui/unsafe_store_requires_unsafe.rs#L1): expected-failure.
