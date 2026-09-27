# Context constructor design probe

This is a standalone compile probe, not an implementation of `use(...)`.
Coverage: one vector shape (`f32x8`), zero construction, addition, conversions,
V3 feature-context checks, and scalar construction. It does not implement the
30-vector API, the macro syntax, NEON/WASM variants, or measure compile-time
or runtime overhead.

Two designs are checked against the actual local archmage/magetypes crates:

- A single generic vector with a defaulted constructor-mode parameter permits
  distinct `zero(token)` and `zero()` functions, but breaks inferred
  `Vector::zero(token)` lookup with E0034. Defaulting the parameter does not
  preserve every old call form.
- A separate `LocalVector<T>` family leaves the old vector untouched. Its V3
  `zero()` requires the matching feature context; a scalar variant needs none.
  Generic arithmetic delegates to the existing vector, and explicit `From`
  conversions preserve the stored token without unsafe code or detection.

Run serially from the repository root:

```sh
~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 tests/design-probes/context-mode/check.py --log-dir ~/data/archmage/context-mode-probe/2026-09-27
```

Choose the memory cap for the actual host. The script saves full diagnostics
and machine-readable results. All six cases passed their stated expectations
with rustc 1.98.1 on x86_64: positive compilation; missing/weaker feature context
E0133; implicit cross-mode assignment E0308; legacy inferred constructor E0034;
and missing feature context for the separate family E0133.

The API-selection proposal is documented in
[the migration analysis](../../../docs/TOKEN-CONTEXT-MIGRATION.md).

The subsequent 29-case standalone matrix identified the compatible shared-core
solution: public aliases must **fix** their mode parameter, rather than default
it. That solution is now implemented by the generator and tested in
`magetypes/tests/magetypes_use_flag.rs`. The defaulted-mode ambiguity above
remains a regression probe for the rejected alias shape.

The harness also compiles a small procedural attribute macro and applies it as
`#[keyword_attribute::accept(use(f32x8))]`. This confirms that Rust permits the
keyword in an attribute's token stream; it does not add `use(...)` to magetypes.
The keyword macro and consumer are separate compiler invocations with saved
logs. Both passed on rustc 1.98.1. The production parser recognizes
`define(...)` and `use(...)`. The real macro integration is tested in
`magetypes/tests/magetypes_use_flag.rs`.
