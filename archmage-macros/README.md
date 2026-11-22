# archmage-macros [![CI](https://img.shields.io/github/actions/workflow/status/imazen/archmage/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/archmage/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/archmage-macros?style=flat-square)](https://crates.io/crates/archmage-macros) [![lib.rs](https://img.shields.io/crates/v/archmage-macros?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/archmage-macros) [![docs.rs](https://img.shields.io/docsrs/archmage-macros?style=flat-square)](https://docs.rs/archmage-macros) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://github.com/imazen/archmage/blob/main/MSRV.md) [![license](https://img.shields.io/crates/l/archmage-macros?style=flat-square)](#license)

**[Official guide and examples](https://imazen.github.io/archmage/)** · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/) · [Magemath API](https://docs.rs/magemath/latest/magemath/)

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
For internal feature-enabled helpers, use `#[attune(v3)]` and `X64V3Token::from_context()`: see [feature-enabled helpers and from_context()](https://imazen.github.io/archmage/archmage/concepts/rite/).

Macro expansion tests, compile-fail tests, and optimized-code comparisons cover
different obligations. See the [testing guide](https://imazen.github.io/archmage/archmage/testing/dispatch-testing/).

## Legacy attributes (queued for deprecation)

Earlier attributes (`#[arcane]`, `#[rite]`, `#[magetypes]`, `#[autoversion]`,
`incant!`) are superseded by `#[attune]`, `attuned!`, and `reattune!` and
queued for deprecation in the next release (they remain warning-free in 0.9.33).
Use matching `archmage` and `archmage-macros` versions; the exact dependency
pin keeps generated code and runtime APIs paired.

## License

MIT OR Apache-2.0.
