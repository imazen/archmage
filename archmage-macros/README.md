# archmage-macros

**[Official guide and examples](https://imazen.github.io/archmage/)** · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/)

Procedural macros for archmage's CPU-capability tokens: primarily
`#[attune]`, `attuned!`, and `reattune!`.

Application code should depend on [archmage](https://crates.io/crates/archmage),
which re-exports these macros and pins the compatible macro version. Add the
[magetypes vector crate](https://docs.rs/magetypes/latest/magetypes/) when using
vector types such as [`f32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x4.html).
`#[attune(_*_t, define(f32x8))]` generates portable per-tier families; the
magetypes crate supplies the vector operations.

Start with the [unified macro guide](https://github.com/imazen/archmage/blob/main/docs/ATTUNE.md),
the [complete SIMD call chain](https://imazen.github.io/archmage/archmage/getting-started/first-simd/),
and [type and const generics](https://imazen.github.io/archmage/magetypes/dispatch/types-and-dispatch/).
For internal feature-enabled helpers, use `#[attune(v3)]` and `X64V3Token::from_context()`: see [rite and from_context()](https://imazen.github.io/archmage/archmage/concepts/rite/).

Macro expansion tests, compile-fail tests, and optimized-code comparisons cover
different obligations. See the [testing guide](https://imazen.github.io/archmage/archmage/testing/dispatch-testing/).

## Legacy attributes

Earlier attributes (`#[arcane]`, `#[rite]`, `#[magetypes]`, `#[autoversion]`,
`incant!`) remain supported without breaking changes. Use matching `archmage`
and `archmage-macros` versions; the exact dependency pin keeps generated code
and runtime APIs paired.

## License

MIT OR Apache-2.0.
