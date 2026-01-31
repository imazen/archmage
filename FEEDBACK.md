# User Feedback Log

## 2026-01-31

- User requested implementing plan: hard-fail validation, fix intrinsic bugs, explicit feature checks
  - Add bmi1, f16c, lzcnt checks to all v3+ token try_new()
  - Move all x86 magetypes widths (W128, W256) to X64V3Token
  - Fix _mm_srai_epi64 codegen bug (AVX-512F only, polyfill for W128/W256)
  - Fix token_provides_features() and Avx2FmaToken entry
  - Make validation a hard build failure
  - Regenerate everything
