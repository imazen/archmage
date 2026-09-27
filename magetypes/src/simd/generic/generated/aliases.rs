// Generated fixed-policy aliases. Do not edit.
/// f32x4 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type f32x4<T> = core_types::f32x4<T, Explicit>;
/// f64x2 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type f64x2<T> = core_types::f64x2<T, Explicit>;
/// i8x16 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i8x16<T> = core_types::i8x16<T, Explicit>;
/// u8x16 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u8x16<T> = core_types::u8x16<T, Explicit>;
/// i16x8 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i16x8<T> = core_types::i16x8<T, Explicit>;
/// u16x8 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u16x8<T> = core_types::u16x8<T, Explicit>;
/// i32x4 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i32x4<T> = core_types::i32x4<T, Explicit>;
/// u32x4 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u32x4<T> = core_types::u32x4<T, Explicit>;
/// i64x2 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i64x2<T> = core_types::i64x2<T, Explicit>;
/// u64x2 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u64x2<T> = core_types::u64x2<T, Explicit>;
/// f32x8 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type f32x8<T> = core_types::f32x8<T, Explicit>;
/// f64x4 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type f64x4<T> = core_types::f64x4<T, Explicit>;
/// i8x32 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i8x32<T> = core_types::i8x32<T, Explicit>;
/// u8x32 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u8x32<T> = core_types::u8x32<T, Explicit>;
/// i16x16 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i16x16<T> = core_types::i16x16<T, Explicit>;
/// u16x16 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u16x16<T> = core_types::u16x16<T, Explicit>;
/// i32x8 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i32x8<T> = core_types::i32x8<T, Explicit>;
/// u32x8 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u32x8<T> = core_types::u32x8<T, Explicit>;
/// i64x4 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i64x4<T> = core_types::i64x4<T, Explicit>;
/// u64x4 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u64x4<T> = core_types::u64x4<T, Explicit>;
#[cfg(feature = "w512")]
/// f32x16 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type f32x16<T> = core_types::f32x16<T, Explicit>;
#[cfg(feature = "w512")]
/// f64x8 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type f64x8<T> = core_types::f64x8<T, Explicit>;
#[cfg(feature = "w512")]
/// i8x64 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i8x64<T> = core_types::i8x64<T, Explicit>;
#[cfg(feature = "w512")]
/// u8x64 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u8x64<T> = core_types::u8x64<T, Explicit>;
#[cfg(feature = "w512")]
/// i16x32 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i16x32<T> = core_types::i16x32<T, Explicit>;
#[cfg(feature = "w512")]
/// u16x32 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u16x32<T> = core_types::u16x32<T, Explicit>;
#[cfg(feature = "w512")]
/// i32x16 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i32x16<T> = core_types::i32x16<T, Explicit>;
#[cfg(feature = "w512")]
/// u32x16 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u32x16<T> = core_types::u32x16<T, Explicit>;
#[cfg(feature = "w512")]
/// i64x8 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type i64x8<T> = core_types::i64x8<T, Explicit>;
#[cfg(feature = "w512")]
/// u64x8 vector constructed with an explicit CPU capability token.
#[allow(non_camel_case_types)]
pub type u64x8<T> = core_types::u64x8<T, Explicit>;
/// Vectors constructed in a matching target-feature context.
pub mod local {
    /// f32x4 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type f32x4<T> = super::core_types::f32x4<T, super::Context>;
    /// f64x2 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type f64x2<T> = super::core_types::f64x2<T, super::Context>;
    /// i8x16 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i8x16<T> = super::core_types::i8x16<T, super::Context>;
    /// u8x16 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u8x16<T> = super::core_types::u8x16<T, super::Context>;
    /// i16x8 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i16x8<T> = super::core_types::i16x8<T, super::Context>;
    /// u16x8 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u16x8<T> = super::core_types::u16x8<T, super::Context>;
    /// i32x4 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i32x4<T> = super::core_types::i32x4<T, super::Context>;
    /// u32x4 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u32x4<T> = super::core_types::u32x4<T, super::Context>;
    /// i64x2 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i64x2<T> = super::core_types::i64x2<T, super::Context>;
    /// u64x2 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u64x2<T> = super::core_types::u64x2<T, super::Context>;
    /// f32x8 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type f32x8<T> = super::core_types::f32x8<T, super::Context>;
    /// f64x4 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type f64x4<T> = super::core_types::f64x4<T, super::Context>;
    /// i8x32 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i8x32<T> = super::core_types::i8x32<T, super::Context>;
    /// u8x32 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u8x32<T> = super::core_types::u8x32<T, super::Context>;
    /// i16x16 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i16x16<T> = super::core_types::i16x16<T, super::Context>;
    /// u16x16 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u16x16<T> = super::core_types::u16x16<T, super::Context>;
    /// i32x8 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i32x8<T> = super::core_types::i32x8<T, super::Context>;
    /// u32x8 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u32x8<T> = super::core_types::u32x8<T, super::Context>;
    /// i64x4 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i64x4<T> = super::core_types::i64x4<T, super::Context>;
    /// u64x4 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u64x4<T> = super::core_types::u64x4<T, super::Context>;
    #[cfg(feature = "w512")]
    /// f32x16 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type f32x16<T> = super::core_types::f32x16<T, super::Context>;
    #[cfg(feature = "w512")]
    /// f64x8 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type f64x8<T> = super::core_types::f64x8<T, super::Context>;
    #[cfg(feature = "w512")]
    /// i8x64 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i8x64<T> = super::core_types::i8x64<T, super::Context>;
    #[cfg(feature = "w512")]
    /// u8x64 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u8x64<T> = super::core_types::u8x64<T, super::Context>;
    #[cfg(feature = "w512")]
    /// i16x32 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i16x32<T> = super::core_types::i16x32<T, super::Context>;
    #[cfg(feature = "w512")]
    /// u16x32 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u16x32<T> = super::core_types::u16x32<T, super::Context>;
    #[cfg(feature = "w512")]
    /// i32x16 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i32x16<T> = super::core_types::i32x16<T, super::Context>;
    #[cfg(feature = "w512")]
    /// u32x16 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u32x16<T> = super::core_types::u32x16<T, super::Context>;
    #[cfg(feature = "w512")]
    /// i64x8 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type i64x8<T> = super::core_types::i64x8<T, super::Context>;
    #[cfg(feature = "w512")]
    /// u64x8 vector constructed in a matching target-feature context.
    #[allow(non_camel_case_types)]
    pub type u64x8<T> = super::core_types::u64x8<T, super::Context>;
}
