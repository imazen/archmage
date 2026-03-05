# User Feedback Log

## 2026-03-04

- User reported broken CI docs build (broken doc links in testing.rs). Also requested adding doc build check to `just ci` to prevent publishing without testing docs.
- User requested updating all remaining `safe_unaligned_simd` references in docs/ markdown files to reflect the new combined intrinsics module (`archmage::intrinsics::{arch}`) and `import_intrinsics` pattern

## 2026-03-02

- User requested updating all repo documentation for sibling expansion, cfg-out default, stub param, and analyzing all dependents for breakage
  - Updated 11 files: README, spec, CHANGELOG, site docs (arcane, rite, cross-platform, methods), book docs, crate docs
  - Analyzed 11 dependents: all safe from cfg-out default change
  - Key finding: all dependents properly cfg-guard their #[arcane] functions and use incant! with scalar fallbacks

## 2026-03-01

- User requested CPU survey tool + GitHub Actions workflow: `examples/cpu_survey.rs` (comprehensive CPU feature report) and `.github/workflows/cpu-survey.yml` (12-runner matrix covering all public GH runners)

## 2026-01-31

- User requested implementing plan: hard-fail validation, fix intrinsic bugs, explicit feature checks
  - Add bmi1, f16c, lzcnt checks to all v3+ token summon()
  - Move all x86 magetypes widths (W128, W256) to X64V3Token
  - Fix _mm_srai_epi64 codegen bug (AVX-512F only, polyfill for W128/W256)
  - Fix token_provides_features() and Avx2FmaToken entry
  - Make validation a hard build failure
  - Regenerate everything

- User requested generating `src/tokens/` from token-registry.toml
  - Replaced ~650 lines of handwritten token code with codegen from registry
  - Added display_name, short_name, parent, extraction_aliases, doc fields to registry
  - Created xtask/src/token_gen.rs generator
  - Tokens, traits, stubs all generated; hand-written mod.rs retains only SimdToken trait
  - User OK with API changes ("you can change the api in intuitive or smart ways")
  - User asked about special casing — all special cases are data-driven from registry

- User requested deleting `experiments` and `integrate` folders, updating CLAUDE.md
  - Removed `src/experiments/`, `src/integrate/`, `benches/wide_comparison.rs`
  - Removed `__experiments`, `__wide`, `wide` features and `wide` dependency from Cargo.toml
  - Updated CLAUDE.md: directory structure, CI feature lists, codegen style rules
  - Added rule: ban `writeln!` chains in codegen, use `r#` raw strings + `formatdoc!`

## 2026-02-21

- User requested adding descriptive aliases for proc macros and saturating docs with safety messaging
  - Added `#[token_target_features_boundary]` (alias for `#[arcane]`)
  - Added `#[token_target_features]` (alias for `#[rite]`)
  - Added `dispatch_variant!` (alias for `incant!`)
  - Added Safety Model section to README.md
  - Added crypto tokens to README token table
  - Updated 10+ doc files with alias notes, Rust 1.85 safety messaging, `#![forbid(unsafe_code)]` compatibility, registry links
  - Added alias tests in tests/arcane_macro.rs

## 2026-02-15

- User requested adding i32 backend trait generation to backend_gen.rs
  - I32VecType model + I32x4Backend/I32x8Backend traits
  - F32x4Convert/F32x8Convert conversion traits
  - Implementations for x86 V3, scalar, NEON, WASM platforms
  - Update generate_backend_files() and generate_backends_mod()
2026-02-25T00:30:38-07:00 - User requested comprehensive review and update of mdbook docs (docs/book/src/). Fixed: deprecated guaranteed()→compiled_with(), version 0.6→0.8, incomplete token/trait listings, typos, outdated references.
2026-02-25 - User requested creating all 31 Zola markdown files for the magetypes documentation section. Created: section indexes (9), getting-started (2), types (2), operations (4), conversions (4), math (3), memory (4), cross-platform (2), dispatch (1). Content migrated from docs/book/src/magetypes/ and expanded with proper TOML front matter, `@/` internal links, standalone context.
2026-02-25 - User requested rewriting all magetypes docs to use generic pattern (`f32x8::<T>`, not flat aliases). Also: CI-tested doc examples, CLAUDE.md rule for doc testing. User asked "why do we have to import the backends again?" — discussed nested use path syntax as solution.
- 2026-02-26: Added #[inline(always)] to all generic function definitions in documentation files under docs/site/content/
- 2026-02-28: User requested CI test for testable_dispatch + CompileTimePolicy::Fail. Fixed V1 codegen to respect testable_dispatch (compiled_with/summon were unconditional), added test + CI job + justfile target.
2026-03-01: User requested research on unstable Rust target features across architectures
