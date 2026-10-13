#![feature(prelude_import)]
//! Small, explicit SIMD boundaries. Experimental; see README for limitations.
#![no_std]
#![forbid(unsafe_code)]
extern crate core;
#[prelude_import]
use core::prelude::rust_2024::*;
extern crate std;
pub use artisanal_dispatch_macros::{bridge, dispatch, tier};
pub struct Scalar(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for Scalar {}
#[automatically_derived]
impl ::core::clone::Clone for Scalar {
    #[inline]
    fn clone(&self) -> Scalar {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for Scalar {}
#[automatically_derived]
impl ::core::fmt::Debug for Scalar {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "Scalar", &&self.0)
    }
}
impl Scalar {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if true {
            return Some(Self(()));
        }
        None
    }
    /// Construct in a context whose features rustc checks.
    #[inline]
    pub fn from_context() -> Self {
        Self(())
    }
}
pub struct V1(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for V1 {}
#[automatically_derived]
impl ::core::clone::Clone for V1 {
    #[inline]
    fn clone(&self) -> V1 {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for V1 {}
#[automatically_derived]
impl ::core::fmt::Debug for V1 {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "V1", &&self.0)
    }
}
impl V1 {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if true {
            return Some(Self(()));
        }
        {
            if true && (true || ::std_detect::detect::__is_feature_detected::sse())
                && (true || ::std_detect::detect::__is_feature_detected::sse2())
            {
                return Some(Self(()));
            }
        }
        None
    }
    /// Construct in a context whose features rustc checks.
    #[target_feature(enable = "sse,sse2")]
    #[inline]
    pub fn from_context() -> Self {
        Self(())
    }
}
impl From<V1> for Scalar {
    #[inline(always)]
    fn from(_: V1) -> Self {
        Self(())
    }
}
pub struct V2(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for V2 {}
#[automatically_derived]
impl ::core::clone::Clone for V2 {
    #[inline]
    fn clone(&self) -> V2 {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for V2 {}
#[automatically_derived]
impl ::core::fmt::Debug for V2 {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "V2", &&self.0)
    }
}
impl V2 {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if false {
            return Some(Self(()));
        }
        {
            if true && (true || ::std_detect::detect::__is_feature_detected::sse())
                && (true || ::std_detect::detect::__is_feature_detected::sse2())
                && (false || ::std_detect::detect::__is_feature_detected::sse3())
                && (false || ::std_detect::detect::__is_feature_detected::ssse3())
                && (false || ::std_detect::detect::__is_feature_detected::sse4_1())
                && (false || ::std_detect::detect::__is_feature_detected::sse4_2())
                && (false || ::std_detect::detect::__is_feature_detected::popcnt())
                && (false || ::std_detect::detect::__is_feature_detected::cmpxchg16b())
            {
                return Some(Self(()));
            }
        }
        None
    }
    /// Construct in a context whose features rustc checks.
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
    #[inline]
    pub fn from_context() -> Self {
        Self(())
    }
}
impl From<V2> for Scalar {
    #[inline(always)]
    fn from(_: V2) -> Self {
        Self(())
    }
}
impl From<V2> for V1 {
    #[inline(always)]
    fn from(_: V2) -> Self {
        Self(())
    }
}
pub struct V3(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for V3 {}
#[automatically_derived]
impl ::core::clone::Clone for V3 {
    #[inline]
    fn clone(&self) -> V3 {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for V3 {}
#[automatically_derived]
impl ::core::fmt::Debug for V3 {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "V3", &&self.0)
    }
}
impl V3 {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if false {
            return Some(Self(()));
        }
        {
            if true && (true || ::std_detect::detect::__is_feature_detected::sse())
                && (true || ::std_detect::detect::__is_feature_detected::sse2())
                && (false || ::std_detect::detect::__is_feature_detected::sse3())
                && (false || ::std_detect::detect::__is_feature_detected::ssse3())
                && (false || ::std_detect::detect::__is_feature_detected::sse4_1())
                && (false || ::std_detect::detect::__is_feature_detected::sse4_2())
                && (false || ::std_detect::detect::__is_feature_detected::popcnt())
                && (false || ::std_detect::detect::__is_feature_detected::cmpxchg16b())
                && (false || ::std_detect::detect::__is_feature_detected::avx())
                && (false || ::std_detect::detect::__is_feature_detected::avx2())
                && (false || ::std_detect::detect::__is_feature_detected::fma())
                && (false || ::std_detect::detect::__is_feature_detected::bmi1())
                && (false || ::std_detect::detect::__is_feature_detected::bmi2())
                && (false || ::std_detect::detect::__is_feature_detected::f16c())
                && (false || ::std_detect::detect::__is_feature_detected::lzcnt())
                && (false || ::std_detect::detect::__is_feature_detected::movbe())
            {
                return Some(Self(()));
            }
        }
        None
    }
    /// Construct in a context whose features rustc checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_context() -> Self {
        Self(())
    }
}
impl From<V3> for Scalar {
    #[inline(always)]
    fn from(_: V3) -> Self {
        Self(())
    }
}
impl From<V3> for V1 {
    #[inline(always)]
    fn from(_: V3) -> Self {
        Self(())
    }
}
impl From<V3> for V2 {
    #[inline(always)]
    fn from(_: V3) -> Self {
        Self(())
    }
}
pub struct V4(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for V4 {}
#[automatically_derived]
impl ::core::clone::Clone for V4 {
    #[inline]
    fn clone(&self) -> V4 {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for V4 {}
#[automatically_derived]
impl ::core::fmt::Debug for V4 {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "V4", &&self.0)
    }
}
impl V4 {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if false {
            return Some(Self(()));
        }
        {
            if true && (true || ::std_detect::detect::__is_feature_detected::sse())
                && (true || ::std_detect::detect::__is_feature_detected::sse2())
                && (false || ::std_detect::detect::__is_feature_detected::sse3())
                && (false || ::std_detect::detect::__is_feature_detected::ssse3())
                && (false || ::std_detect::detect::__is_feature_detected::sse4_1())
                && (false || ::std_detect::detect::__is_feature_detected::sse4_2())
                && (false || ::std_detect::detect::__is_feature_detected::popcnt())
                && (false || ::std_detect::detect::__is_feature_detected::cmpxchg16b())
                && (false || ::std_detect::detect::__is_feature_detected::avx())
                && (false || ::std_detect::detect::__is_feature_detected::avx2())
                && (false || ::std_detect::detect::__is_feature_detected::fma())
                && (false || ::std_detect::detect::__is_feature_detected::bmi1())
                && (false || ::std_detect::detect::__is_feature_detected::bmi2())
                && (false || ::std_detect::detect::__is_feature_detected::f16c())
                && (false || ::std_detect::detect::__is_feature_detected::lzcnt())
                && (false || ::std_detect::detect::__is_feature_detected::movbe())
                && (false || ::std_detect::detect::__is_feature_detected::pclmulqdq())
                && (false || ::std_detect::detect::__is_feature_detected::aes())
                && (false || ::std_detect::detect::__is_feature_detected::avx512f())
                && (false || ::std_detect::detect::__is_feature_detected::avx512bw())
                && (false || ::std_detect::detect::__is_feature_detected::avx512cd())
                && (false || ::std_detect::detect::__is_feature_detected::avx512dq())
                && (false || ::std_detect::detect::__is_feature_detected::avx512vl())
            {
                return Some(Self(()));
            }
        }
        None
    }
    /// Construct in a context whose features rustc checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_context() -> Self {
        Self(())
    }
}
impl From<V4> for Scalar {
    #[inline(always)]
    fn from(_: V4) -> Self {
        Self(())
    }
}
impl From<V4> for V1 {
    #[inline(always)]
    fn from(_: V4) -> Self {
        Self(())
    }
}
impl From<V4> for V2 {
    #[inline(always)]
    fn from(_: V4) -> Self {
        Self(())
    }
}
impl From<V4> for V3 {
    #[inline(always)]
    fn from(_: V4) -> Self {
        Self(())
    }
}
pub struct V4x(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for V4x {}
#[automatically_derived]
impl ::core::clone::Clone for V4x {
    #[inline]
    fn clone(&self) -> V4x {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for V4x {}
#[automatically_derived]
impl ::core::fmt::Debug for V4x {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "V4x", &&self.0)
    }
}
impl V4x {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if false {
            return Some(Self(()));
        }
        {
            if true && (true || ::std_detect::detect::__is_feature_detected::sse())
                && (true || ::std_detect::detect::__is_feature_detected::sse2())
                && (false || ::std_detect::detect::__is_feature_detected::sse3())
                && (false || ::std_detect::detect::__is_feature_detected::ssse3())
                && (false || ::std_detect::detect::__is_feature_detected::sse4_1())
                && (false || ::std_detect::detect::__is_feature_detected::sse4_2())
                && (false || ::std_detect::detect::__is_feature_detected::popcnt())
                && (false || ::std_detect::detect::__is_feature_detected::cmpxchg16b())
                && (false || ::std_detect::detect::__is_feature_detected::avx())
                && (false || ::std_detect::detect::__is_feature_detected::avx2())
                && (false || ::std_detect::detect::__is_feature_detected::fma())
                && (false || ::std_detect::detect::__is_feature_detected::bmi1())
                && (false || ::std_detect::detect::__is_feature_detected::bmi2())
                && (false || ::std_detect::detect::__is_feature_detected::f16c())
                && (false || ::std_detect::detect::__is_feature_detected::lzcnt())
                && (false || ::std_detect::detect::__is_feature_detected::movbe())
                && (false || ::std_detect::detect::__is_feature_detected::pclmulqdq())
                && (false || ::std_detect::detect::__is_feature_detected::aes())
                && (false || ::std_detect::detect::__is_feature_detected::avx512f())
                && (false || ::std_detect::detect::__is_feature_detected::avx512bw())
                && (false || ::std_detect::detect::__is_feature_detected::avx512cd())
                && (false || ::std_detect::detect::__is_feature_detected::avx512dq())
                && (false || ::std_detect::detect::__is_feature_detected::avx512vl())
                && (false
                    || ::std_detect::detect::__is_feature_detected::avx512vpopcntdq())
                && (false || ::std_detect::detect::__is_feature_detected::avx512ifma())
                && (false || ::std_detect::detect::__is_feature_detected::avx512vbmi())
                && (false || ::std_detect::detect::__is_feature_detected::avx512vbmi2())
                && (false || ::std_detect::detect::__is_feature_detected::avx512bitalg())
                && (false || ::std_detect::detect::__is_feature_detected::avx512vnni())
                && (false || ::std_detect::detect::__is_feature_detected::vpclmulqdq())
                && (false || ::std_detect::detect::__is_feature_detected::gfni())
                && (false || ::std_detect::detect::__is_feature_detected::vaes())
            {
                return Some(Self(()));
            }
        }
        None
    }
    /// Construct in a context whose features rustc checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_context() -> Self {
        Self(())
    }
}
impl From<V4x> for Scalar {
    #[inline(always)]
    fn from(_: V4x) -> Self {
        Self(())
    }
}
impl From<V4x> for V1 {
    #[inline(always)]
    fn from(_: V4x) -> Self {
        Self(())
    }
}
impl From<V4x> for V2 {
    #[inline(always)]
    fn from(_: V4x) -> Self {
        Self(())
    }
}
impl From<V4x> for V3 {
    #[inline(always)]
    fn from(_: V4x) -> Self {
        Self(())
    }
}
impl From<V4x> for V4 {
    #[inline(always)]
    fn from(_: V4x) -> Self {
        Self(())
    }
}
pub struct Neon(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for Neon {}
#[automatically_derived]
impl ::core::clone::Clone for Neon {
    #[inline]
    fn clone(&self) -> Neon {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for Neon {}
#[automatically_derived]
impl ::core::fmt::Debug for Neon {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "Neon", &&self.0)
    }
}
impl Neon {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if false {
            return Some(Self(()));
        }
        None
    }
}
impl From<Neon> for Scalar {
    #[inline(always)]
    fn from(_: Neon) -> Self {
        Self(())
    }
}
pub struct ArmV2(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for ArmV2 {}
#[automatically_derived]
impl ::core::clone::Clone for ArmV2 {
    #[inline]
    fn clone(&self) -> ArmV2 {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for ArmV2 {}
#[automatically_derived]
impl ::core::fmt::Debug for ArmV2 {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "ArmV2", &&self.0)
    }
}
impl ArmV2 {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if false {
            return Some(Self(()));
        }
        None
    }
}
impl From<ArmV2> for Scalar {
    #[inline(always)]
    fn from(_: ArmV2) -> Self {
        Self(())
    }
}
impl From<ArmV2> for Neon {
    #[inline(always)]
    fn from(_: ArmV2) -> Self {
        Self(())
    }
}
pub struct ArmV3(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for ArmV3 {}
#[automatically_derived]
impl ::core::clone::Clone for ArmV3 {
    #[inline]
    fn clone(&self) -> ArmV3 {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for ArmV3 {}
#[automatically_derived]
impl ::core::fmt::Debug for ArmV3 {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "ArmV3", &&self.0)
    }
}
impl ArmV3 {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if false {
            return Some(Self(()));
        }
        None
    }
}
impl From<ArmV3> for Scalar {
    #[inline(always)]
    fn from(_: ArmV3) -> Self {
        Self(())
    }
}
impl From<ArmV3> for Neon {
    #[inline(always)]
    fn from(_: ArmV3) -> Self {
        Self(())
    }
}
impl From<ArmV3> for ArmV2 {
    #[inline(always)]
    fn from(_: ArmV3) -> Self {
        Self(())
    }
}
pub struct Wasm128(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for Wasm128 {}
#[automatically_derived]
impl ::core::clone::Clone for Wasm128 {
    #[inline]
    fn clone(&self) -> Wasm128 {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for Wasm128 {}
#[automatically_derived]
impl ::core::fmt::Debug for Wasm128 {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "Wasm128", &&self.0)
    }
}
impl Wasm128 {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if false {
            return Some(Self(()));
        }
        None
    }
}
impl From<Wasm128> for Scalar {
    #[inline(always)]
    fn from(_: Wasm128) -> Self {
        Self(())
    }
}
pub struct Wasm128Relaxed(());
#[automatically_derived]
#[doc(hidden)]
unsafe impl ::core::clone::TrivialClone for Wasm128Relaxed {}
#[automatically_derived]
impl ::core::clone::Clone for Wasm128Relaxed {
    #[inline]
    fn clone(&self) -> Wasm128Relaxed {
        let _: ::core::clone::AssertParamIsClone<()>;
        *self
    }
}
#[automatically_derived]
impl ::core::marker::Copy for Wasm128Relaxed {}
#[automatically_derived]
impl ::core::fmt::Debug for Wasm128Relaxed {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_tuple_field1_finish(f, "Wasm128Relaxed", &&self.0)
    }
}
impl Wasm128Relaxed {
    /// Detect once at a boundary. no_std uses only build-time guarantees.
    #[inline]
    pub fn summon() -> Option<Self> {
        if false {
            return Some(Self(()));
        }
        None
    }
}
impl From<Wasm128Relaxed> for Scalar {
    #[inline(always)]
    fn from(_: Wasm128Relaxed) -> Self {
        Self(())
    }
}
impl From<Wasm128Relaxed> for Wasm128 {
    #[inline(always)]
    fn from(_: Wasm128Relaxed) -> Self {
        Self(())
    }
}
