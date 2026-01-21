# User Feedback Log

## 2026-01-21

- User requested SSE4.2 as baseline, remove SSE/SSE2/SSE4.1 tokens and traits
- User requested cumulative/explicit feature checks in all try_new() methods
- FMA should be on top of AVX2, not independent (HasFma: HasAvx2)
- NeonAes and NeonSha3 are orthogonal (HasArmSha3 should NOT extend HasArmAes)
- ArmCryptoToken and ArmCrypto3Token are poor aliases - remove them
- Want HasAvx512 and HasModernAvx512 traits
- Want profile traits: HasX64V3/HasDesktop64, HasX64V4/HasServer64, HasArm64
- Keep Avx2FmaToken instead of FmaToken for clarity
- **AVX-512 details were lost - need user to re-provide**
