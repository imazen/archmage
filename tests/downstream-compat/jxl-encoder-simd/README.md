# Published jxl-encoder-simd 0.3.0 compatibility

This fixture uses the **unmodified, exact published** dependency and patches only
archmage, archmage-macros, and magetypes to this checkout.

The x86 build must pass. ARM and WASM currently have one remaining error each:
`f32x4::from_i32x4(vector)` requires `f32x4::from_i32x4(token, vector)` in the
published generic API. The native `.raw()` and constructor restoration repairs
the other errors. The checker deliberately asserts the one known error, its
code, and location; any additional error or unexpected success fails the check.
It reports **known incompatibility**, never a passing ARM/WASM consumer build.

Run `python3 tests/downstream-compat/jxl-encoder-simd/check.py --target TARGET`
with x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu, or wasm32-wasip1.

Rust inherent methods cannot offer both arities under the same name. Restoring
the one-argument form would break callers of the published two-argument generic
method. The consumer needs the token argument added at dequant.rs:421 and :563
and a new consumer release. Track that resolution in
[#117](https://github.com/imazen/archmage/issues/117); when published, replace this
expected-failure fixture with a successful ARM/WASM compilation check.
