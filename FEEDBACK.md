# User Feedback Log

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
