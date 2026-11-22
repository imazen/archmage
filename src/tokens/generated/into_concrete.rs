//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

use super::{
    Arm64V2Token, Arm64V3Token, Avx512Fp16Token, NeonAesToken, NeonCrcToken, NeonSha3Token,
    NeonToken, Wasm128RelaxedToken, Wasm128Token, X64CryptoToken, X64V1Token, X64V2Token,
    X64V3CryptoToken, X64V3GfniCryptoToken, X64V3Token, X64V3VnniToken, X64V4Token, X64V4xToken,
};
use crate::tokens::{ScalarToken, SimdToken};

/// Recover concrete proofs from a generic SIMD token.
///
/// The `as_*` methods check the token's exact type. The `try_*` methods also
/// accept descendants of the requested token in the registry's parent
/// hierarchy. For example, an [`X64V4Token`] can supply a proof through
/// [`Self::try_v3`], while [`Self::as_x64v3`] returns `None` for it.
///
/// Both forms use the proof already held by the caller. They perform no
/// CPU detection, and monomorphization lets the compiler fold the checks.
/// For a known concrete type, its inherent extraction method (such as
/// `X64V4Token::v3`) gives the parent proof directly on the native target.
///
/// # Example
///
/// Check stronger proofs first when multiple tiers can run the operation.
///
/// ```rust
/// use archmage::{IntoConcreteToken, ScalarToken, SimdToken, X64V4Token};
///
/// fn selected_tier(token: impl IntoConcreteToken) -> &'static str {
///     if let Some(proof) = token.try_v4x() {
///         proof.name()
///     } else if let Some(proof) = token.try_v4() {
///         proof.name()
///     } else if let Some(proof) = token.try_v3() {
///         proof.name()
///     } else if let Some(proof) = token.try_neon() {
///         proof.name()
///     } else if let Some(proof) = token.try_wasm128() {
///         proof.name()
///     } else {
///         "Scalar"
///     }
/// }
///
/// assert_eq!(selected_tier(ScalarToken), "Scalar");
/// if let Some(v4) = X64V4Token::summon() {
///     assert_eq!(selected_tier(v4), X64V4Token::NAME);
///     assert!(v4.as_x64v3().is_none());
///     assert!(v4.try_v3().is_some());
/// }
/// ```
pub trait IntoConcreteToken: SimdToken + Sized {
    /// Returns the proof if this token is exactly [`X64V1Token`].
    #[inline(always)]
    fn as_x64v1(self) -> Option<X64V1Token> {
        None
    }

    /// Returns the proof if this token is exactly [`X64V2Token`].
    #[inline(always)]
    fn as_x64v2(self) -> Option<X64V2Token> {
        None
    }

    /// Returns the proof if this token is exactly [`X64CryptoToken`].
    #[inline(always)]
    fn as_x64_crypto(self) -> Option<X64CryptoToken> {
        None
    }

    /// Returns the proof if this token is exactly [`X64V3Token`].
    #[inline(always)]
    fn as_x64v3(self) -> Option<X64V3Token> {
        None
    }

    /// Returns the proof if this token is exactly [`X64V3CryptoToken`].
    #[inline(always)]
    fn as_x64v3_crypto(self) -> Option<X64V3CryptoToken> {
        None
    }

    /// Returns the proof if this token is exactly [`X64V3GfniCryptoToken`].
    #[inline(always)]
    fn as_x64v3_gfni_crypto(self) -> Option<X64V3GfniCryptoToken> {
        None
    }

    /// Returns the proof if this token is exactly [`X64V3VnniToken`].
    #[inline(always)]
    fn as_x64v3_vnni(self) -> Option<X64V3VnniToken> {
        None
    }

    /// Returns the proof if this token is exactly [`X64V4Token`].
    #[inline(always)]
    fn as_x64v4(self) -> Option<X64V4Token> {
        None
    }

    /// Returns the proof if this token is exactly [`X64V4xToken`].
    #[inline(always)]
    fn as_x64v4x(self) -> Option<X64V4xToken> {
        None
    }

    /// Returns the proof if this token is exactly [`Avx512Fp16Token`].
    #[inline(always)]
    fn as_avx512_fp16(self) -> Option<Avx512Fp16Token> {
        None
    }

    /// Returns the proof if this token is exactly [`NeonToken`].
    #[inline(always)]
    fn as_neon(self) -> Option<NeonToken> {
        None
    }

    /// Returns the proof if this token is exactly [`NeonAesToken`].
    #[inline(always)]
    fn as_neon_aes(self) -> Option<NeonAesToken> {
        None
    }

    /// Returns the proof if this token is exactly [`NeonSha3Token`].
    #[inline(always)]
    fn as_neon_sha3(self) -> Option<NeonSha3Token> {
        None
    }

    /// Returns the proof if this token is exactly [`NeonCrcToken`].
    #[inline(always)]
    fn as_neon_crc(self) -> Option<NeonCrcToken> {
        None
    }

    /// Returns the proof if this token is exactly [`Arm64V2Token`].
    #[inline(always)]
    fn as_arm_v2(self) -> Option<Arm64V2Token> {
        None
    }

    /// Returns the proof if this token is exactly [`Arm64V3Token`].
    #[inline(always)]
    fn as_arm_v3(self) -> Option<Arm64V3Token> {
        None
    }

    /// Returns the proof if this token is exactly [`Wasm128Token`].
    #[inline(always)]
    fn as_wasm128(self) -> Option<Wasm128Token> {
        None
    }

    /// Returns the proof if this token is exactly [`Wasm128RelaxedToken`].
    #[inline(always)]
    fn as_wasm128_relaxed(self) -> Option<Wasm128RelaxedToken> {
        None
    }

    /// Returns the proof if this token is exactly [`ScalarToken`].
    #[inline(always)]
    fn as_scalar(self) -> Option<ScalarToken> {
        None
    }

    /// Returns a [`X64V1Token`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_x64v1`] to check for the exact type.
    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        None
    }

    /// Returns a [`X64V2Token`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_x64v2`] to check for the exact type.
    #[inline(always)]
    fn try_v2(self) -> Option<X64V2Token> {
        None
    }

    /// Returns a [`X64CryptoToken`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_x64_crypto`] to check for the exact type.
    #[inline(always)]
    fn try_x64_crypto(self) -> Option<X64CryptoToken> {
        None
    }

    /// Returns a [`X64V3Token`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_x64v3`] to check for the exact type.
    #[inline(always)]
    fn try_v3(self) -> Option<X64V3Token> {
        None
    }

    /// Returns a [`X64V3CryptoToken`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_x64v3_crypto`] to check for the exact type.
    #[inline(always)]
    fn try_v3_crypto(self) -> Option<X64V3CryptoToken> {
        None
    }

    /// Returns a [`X64V3GfniCryptoToken`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_x64v3_gfni_crypto`] to check for the exact type.
    #[inline(always)]
    fn try_v3_gfni_crypto(self) -> Option<X64V3GfniCryptoToken> {
        None
    }

    /// Returns a [`X64V3VnniToken`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_x64v3_vnni`] to check for the exact type.
    #[inline(always)]
    fn try_v3_vnni(self) -> Option<X64V3VnniToken> {
        None
    }

    /// Returns a [`X64V4Token`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_x64v4`] to check for the exact type.
    #[inline(always)]
    fn try_v4(self) -> Option<X64V4Token> {
        None
    }

    /// Returns a [`X64V4xToken`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_x64v4x`] to check for the exact type.
    #[inline(always)]
    fn try_v4x(self) -> Option<X64V4xToken> {
        None
    }

    /// Returns a [`Avx512Fp16Token`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_avx512_fp16`] to check for the exact type.
    #[inline(always)]
    fn try_avx512_fp16(self) -> Option<Avx512Fp16Token> {
        None
    }

    /// Returns a [`NeonToken`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_neon`] to check for the exact type.
    #[inline(always)]
    fn try_neon(self) -> Option<NeonToken> {
        None
    }

    /// Returns a [`NeonAesToken`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_neon_aes`] to check for the exact type.
    #[inline(always)]
    fn try_neon_aes(self) -> Option<NeonAesToken> {
        None
    }

    /// Returns a [`NeonSha3Token`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_neon_sha3`] to check for the exact type.
    #[inline(always)]
    fn try_neon_sha3(self) -> Option<NeonSha3Token> {
        None
    }

    /// Returns a [`NeonCrcToken`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_neon_crc`] to check for the exact type.
    #[inline(always)]
    fn try_neon_crc(self) -> Option<NeonCrcToken> {
        None
    }

    /// Returns a [`Arm64V2Token`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_arm_v2`] to check for the exact type.
    #[inline(always)]
    fn try_arm_v2(self) -> Option<Arm64V2Token> {
        None
    }

    /// Returns a [`Arm64V3Token`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_arm_v3`] to check for the exact type.
    #[inline(always)]
    fn try_arm_v3(self) -> Option<Arm64V3Token> {
        None
    }

    /// Returns a [`Wasm128Token`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_wasm128`] to check for the exact type.
    #[inline(always)]
    fn try_wasm128(self) -> Option<Wasm128Token> {
        None
    }

    /// Returns a [`Wasm128RelaxedToken`] proof from that token or a descendant in the
    /// registry's parent hierarchy. Returns `None` for other tokens.
    /// Use [`Self::as_wasm128_relaxed`] to check for the exact type.
    #[inline(always)]
    fn try_wasm128_relaxed(self) -> Option<Wasm128RelaxedToken> {
        None
    }
}

impl IntoConcreteToken for ScalarToken {
    #[inline(always)]
    fn as_scalar(self) -> Option<ScalarToken> {
        Some(self)
    }
}

impl IntoConcreteToken for X64V1Token {
    #[inline(always)]
    fn as_x64v1(self) -> Option<X64V1Token> {
        Some(self)
    }

    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        Some(self)
    }
}

impl IntoConcreteToken for X64V2Token {
    #[inline(always)]
    fn as_x64v2(self) -> Option<X64V2Token> {
        Some(self)
    }

    #[inline(always)]
    fn try_v2(self) -> Option<X64V2Token> {
        Some(self)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        Some(self.v1())
    }
}

impl IntoConcreteToken for X64CryptoToken {
    #[inline(always)]
    fn as_x64_crypto(self) -> Option<X64CryptoToken> {
        Some(self)
    }

    #[inline(always)]
    fn try_x64_crypto(self) -> Option<X64CryptoToken> {
        Some(self)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v2(self) -> Option<X64V2Token> {
        Some(self.v2())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        Some(self.v1())
    }
}

impl IntoConcreteToken for X64V3Token {
    #[inline(always)]
    fn as_x64v3(self) -> Option<X64V3Token> {
        Some(self)
    }

    #[inline(always)]
    fn try_v3(self) -> Option<X64V3Token> {
        Some(self)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v2(self) -> Option<X64V2Token> {
        Some(self.v2())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        Some(self.v1())
    }
}

impl IntoConcreteToken for X64V3CryptoToken {
    #[inline(always)]
    fn as_x64v3_crypto(self) -> Option<X64V3CryptoToken> {
        Some(self)
    }

    #[inline(always)]
    fn try_v3_crypto(self) -> Option<X64V3CryptoToken> {
        Some(self)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3(self) -> Option<X64V3Token> {
        Some(self.v3())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_x64_crypto(self) -> Option<X64CryptoToken> {
        Some(self.x64_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v2(self) -> Option<X64V2Token> {
        Some(self.v2())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        Some(self.v1())
    }
}

impl IntoConcreteToken for X64V3GfniCryptoToken {
    #[inline(always)]
    fn as_x64v3_gfni_crypto(self) -> Option<X64V3GfniCryptoToken> {
        Some(self)
    }

    #[inline(always)]
    fn try_v3_gfni_crypto(self) -> Option<X64V3GfniCryptoToken> {
        Some(self)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3_crypto(self) -> Option<X64V3CryptoToken> {
        Some(self.v3_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3(self) -> Option<X64V3Token> {
        Some(self.v3())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_x64_crypto(self) -> Option<X64CryptoToken> {
        Some(self.x64_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v2(self) -> Option<X64V2Token> {
        Some(self.v2())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        Some(self.v1())
    }
}

impl IntoConcreteToken for X64V3VnniToken {
    #[inline(always)]
    fn as_x64v3_vnni(self) -> Option<X64V3VnniToken> {
        Some(self)
    }

    #[inline(always)]
    fn try_v3_vnni(self) -> Option<X64V3VnniToken> {
        Some(self)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3_gfni_crypto(self) -> Option<X64V3GfniCryptoToken> {
        Some(self.v3_gfni_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3_crypto(self) -> Option<X64V3CryptoToken> {
        Some(self.v3_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3(self) -> Option<X64V3Token> {
        Some(self.v3())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_x64_crypto(self) -> Option<X64CryptoToken> {
        Some(self.x64_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v2(self) -> Option<X64V2Token> {
        Some(self.v2())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        Some(self.v1())
    }
}

impl IntoConcreteToken for X64V4Token {
    #[inline(always)]
    fn as_x64v4(self) -> Option<X64V4Token> {
        Some(self)
    }

    #[inline(always)]
    fn try_v4(self) -> Option<X64V4Token> {
        Some(self)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3(self) -> Option<X64V3Token> {
        Some(self.v3())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_x64_crypto(self) -> Option<X64CryptoToken> {
        Some(self.x64_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v2(self) -> Option<X64V2Token> {
        Some(self.v2())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        Some(self.v1())
    }
}

impl IntoConcreteToken for X64V4xToken {
    #[inline(always)]
    fn as_x64v4x(self) -> Option<X64V4xToken> {
        Some(self)
    }

    #[inline(always)]
    fn try_v4x(self) -> Option<X64V4xToken> {
        Some(self)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v4(self) -> Option<X64V4Token> {
        Some(self.v4())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3_gfni_crypto(self) -> Option<X64V3GfniCryptoToken> {
        Some(self.v3_gfni_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3(self) -> Option<X64V3Token> {
        Some(self.v3())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_x64_crypto(self) -> Option<X64CryptoToken> {
        Some(self.x64_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3_crypto(self) -> Option<X64V3CryptoToken> {
        Some(self.v3_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v2(self) -> Option<X64V2Token> {
        Some(self.v2())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        Some(self.v1())
    }
}

impl IntoConcreteToken for Avx512Fp16Token {
    #[inline(always)]
    fn as_avx512_fp16(self) -> Option<Avx512Fp16Token> {
        Some(self)
    }

    #[inline(always)]
    fn try_avx512_fp16(self) -> Option<Avx512Fp16Token> {
        Some(self)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v4(self) -> Option<X64V4Token> {
        Some(self.v4())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v3(self) -> Option<X64V3Token> {
        Some(self.v3())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_x64_crypto(self) -> Option<X64CryptoToken> {
        Some(self.x64_crypto())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v2(self) -> Option<X64V2Token> {
        Some(self.v2())
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[inline(always)]
    fn try_v1(self) -> Option<X64V1Token> {
        Some(self.v1())
    }
}

impl IntoConcreteToken for NeonToken {
    #[inline(always)]
    fn as_neon(self) -> Option<NeonToken> {
        Some(self)
    }

    #[inline(always)]
    fn try_neon(self) -> Option<NeonToken> {
        Some(self)
    }
}

impl IntoConcreteToken for NeonAesToken {
    #[inline(always)]
    fn as_neon_aes(self) -> Option<NeonAesToken> {
        Some(self)
    }

    #[inline(always)]
    fn try_neon_aes(self) -> Option<NeonAesToken> {
        Some(self)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn try_neon(self) -> Option<NeonToken> {
        Some(self.neon())
    }
}

impl IntoConcreteToken for NeonSha3Token {
    #[inline(always)]
    fn as_neon_sha3(self) -> Option<NeonSha3Token> {
        Some(self)
    }

    #[inline(always)]
    fn try_neon_sha3(self) -> Option<NeonSha3Token> {
        Some(self)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn try_neon(self) -> Option<NeonToken> {
        Some(self.neon())
    }
}

impl IntoConcreteToken for NeonCrcToken {
    #[inline(always)]
    fn as_neon_crc(self) -> Option<NeonCrcToken> {
        Some(self)
    }

    #[inline(always)]
    fn try_neon_crc(self) -> Option<NeonCrcToken> {
        Some(self)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn try_neon(self) -> Option<NeonToken> {
        Some(self.neon())
    }
}

impl IntoConcreteToken for Arm64V2Token {
    #[inline(always)]
    fn as_arm_v2(self) -> Option<Arm64V2Token> {
        Some(self)
    }

    #[inline(always)]
    fn try_arm_v2(self) -> Option<Arm64V2Token> {
        Some(self)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn try_neon(self) -> Option<NeonToken> {
        Some(self.neon())
    }
}

impl IntoConcreteToken for Arm64V3Token {
    #[inline(always)]
    fn as_arm_v3(self) -> Option<Arm64V3Token> {
        Some(self)
    }

    #[inline(always)]
    fn try_arm_v3(self) -> Option<Arm64V3Token> {
        Some(self)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn try_arm_v2(self) -> Option<Arm64V2Token> {
        Some(self.arm_v2())
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn try_neon(self) -> Option<NeonToken> {
        Some(self.neon())
    }
}

impl IntoConcreteToken for Wasm128Token {
    #[inline(always)]
    fn as_wasm128(self) -> Option<Wasm128Token> {
        Some(self)
    }

    #[inline(always)]
    fn try_wasm128(self) -> Option<Wasm128Token> {
        Some(self)
    }
}

impl IntoConcreteToken for Wasm128RelaxedToken {
    #[inline(always)]
    fn as_wasm128_relaxed(self) -> Option<Wasm128RelaxedToken> {
        Some(self)
    }

    #[inline(always)]
    fn try_wasm128_relaxed(self) -> Option<Wasm128RelaxedToken> {
        Some(self)
    }

    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn try_wasm128(self) -> Option<Wasm128Token> {
        Some(self.wasm128())
    }
}
