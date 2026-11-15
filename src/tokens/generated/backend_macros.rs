//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Declarative target-feature boundary macro for generated SIMD backend trait
//! implementations in `magetypes`, avoiding ~1,700 proc-macro invocations
//! while preserving identical `#[target_feature]` checking and `$crate` token
//! type authentication.

/// Internal declarative boundary macro for generated concrete-token trait methods.
#[doc(hidden)]
#[macro_export]
macro_rules! __arcane_fn {
    ($tok:ident => fn $name:ident ( self $(, $arg:ident : $ty:ty)* $(,)? ) -> $ret:ty $body:block) => {
        #[inline(always)]
        fn $name(self $(, $arg: $ty)*) -> $ret {
            $crate::__arcane_fn!(@inner $tok => () ($($arg : $ty),*) -> $ret $body);
            // SAFETY: `__simd_inner` takes `$crate::$tok`, which rustc type-checks
            // against `self` (`$crate` cannot be shadowed or aliased), proving every
            // feature in this token's `#[target_feature]` list is available.
            #[allow(unsafe_code)]
            unsafe { __simd_inner(self $(, $arg)*) }
        }
    };
    ($tok:ident, $self_id:ident => fn $name:ident ( self $(, $arg:ident : $ty:ty)* $(,)? ) -> $ret:ty $body:block) => {
        #[inline(always)]
        fn $name(self $(, $arg: $ty)*) -> $ret {
            $crate::__arcane_fn!(@inner $tok, $self_id => () ($($arg : $ty),*) -> $ret $body);
            // SAFETY: `__simd_inner` takes `$crate::$tok`, which rustc type-checks
            // against `self` (`$crate` cannot be shadowed or aliased), proving every
            // feature in this token's `#[target_feature]` list is available.
            #[allow(unsafe_code)]
            unsafe { __simd_inner(self $(, $arg)*) }
        }
    };
    ($tok:ident => fn $name:ident < const $c:ident : $cty:ty > ( self $(, $arg:ident : $ty:ty)* $(,)? ) -> $ret:ty $body:block) => {
        #[inline(always)]
        fn $name<const $c: $cty>(self $(, $arg: $ty)*) -> $ret {
            $crate::__arcane_fn!(@inner $tok => (<const $c: $cty>) ($($arg : $ty),*) -> $ret $body);
            // SAFETY: `__simd_inner` takes `$crate::$tok`, which rustc type-checks
            // against `self` (`$crate` cannot be shadowed or aliased), proving every
            // feature in this token's `#[target_feature]` list is available.
            #[allow(unsafe_code)]
            unsafe { __simd_inner::<$c>(self $(, $arg)*) }
        }
    };
    ($tok:ident, $self_id:ident => fn $name:ident < const $c:ident : $cty:ty > ( self $(, $arg:ident : $ty:ty)* $(,)? ) -> $ret:ty $body:block) => {
        #[inline(always)]
        fn $name<const $c: $cty>(self $(, $arg: $ty)*) -> $ret {
            $crate::__arcane_fn!(@inner $tok, $self_id => (<const $c: $cty>) ($($arg : $ty),*) -> $ret $body);
            // SAFETY: `__simd_inner` takes `$crate::$tok`, which rustc type-checks
            // against `self` (`$crate` cannot be shadowed or aliased), proving every
            // feature in this token's `#[target_feature]` list is available.
            #[allow(unsafe_code)]
            unsafe { __simd_inner::<$c>(self $(, $arg)*) }
        }
    };
    (@sse2 fn $name:ident ( self $(, $arg:ident : $ty:ty)* $(,)? ) -> $ret:ty $body:block) => {
        #[inline(always)]
        fn $name(self $(, $arg: $ty)*) -> $ret {
            $crate::__arcane_fn!(@inner X64V1Token => () ($($arg : $ty),*) -> $ret $body);
            // SAFETY: `self.v1()` yields a `$crate::X64V1Token` proof (SSE + SSE2)
            // checked against `__simd_inner`'s parameter type.
            #[allow(unsafe_code)]
            unsafe { __simd_inner(self.v1() $(, $arg)*) }
        }
    };
    (@inner X64V1Token => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::X64V1Token $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V1Token, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::X64V1Token $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V2Token => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::X64V2Token $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V2Token, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::X64V2Token $(, $arg: $ty)*) -> $ret $body
};
(@inner X64CryptoToken => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,pclmulqdq,aes")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::X64CryptoToken $(, $arg: $ty)*) -> $ret $body
};
(@inner X64CryptoToken, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,pclmulqdq,aes")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::X64CryptoToken $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V3Token => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::X64V3Token $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V3Token, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::X64V3Token $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V3CryptoToken => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,vpclmulqdq,vaes")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::X64V3CryptoToken $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V3CryptoToken, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,vpclmulqdq,vaes")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::X64V3CryptoToken $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V3GfniCryptoToken => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,vpclmulqdq,vaes,gfni")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::X64V3GfniCryptoToken $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V3GfniCryptoToken, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,vpclmulqdq,vaes,gfni")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::X64V3GfniCryptoToken $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V3VnniToken => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,vpclmulqdq,vaes,gfni,avxvnni")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::X64V3VnniToken $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V3VnniToken, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,vpclmulqdq,vaes,gfni,avxvnni")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::X64V3VnniToken $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V4Token => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::X64V4Token $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V4Token, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::X64V4Token $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V4xToken => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::X64V4xToken $(, $arg: $ty)*) -> $ret $body
};
(@inner X64V4xToken, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::X64V4xToken $(, $arg: $ty)*) -> $ret $body
};
(@inner Avx512Fp16Token => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512fp16")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::Avx512Fp16Token $(, $arg: $ty)*) -> $ret $body
};
(@inner Avx512Fp16Token, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512fp16")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::Avx512Fp16Token $(, $arg: $ty)*) -> $ret $body
};
(@inner NeonToken => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::NeonToken $(, $arg: $ty)*) -> $ret $body
};
(@inner NeonToken, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::NeonToken $(, $arg: $ty)*) -> $ret $body
};
(@inner NeonAesToken => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,aes")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::NeonAesToken $(, $arg: $ty)*) -> $ret $body
};
(@inner NeonAesToken, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,aes")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::NeonAesToken $(, $arg: $ty)*) -> $ret $body
};
(@inner NeonSha3Token => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,sha3")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::NeonSha3Token $(, $arg: $ty)*) -> $ret $body
};
(@inner NeonSha3Token, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,sha3")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::NeonSha3Token $(, $arg: $ty)*) -> $ret $body
};
(@inner NeonCrcToken => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,crc")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::NeonCrcToken $(, $arg: $ty)*) -> $ret $body
};
(@inner NeonCrcToken, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,crc")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::NeonCrcToken $(, $arg: $ty)*) -> $ret $body
};
(@inner Arm64V2Token => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,crc,rdm,dotprod,fp16,aes,sha2")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::Arm64V2Token $(, $arg: $ty)*) -> $ret $body
};
(@inner Arm64V2Token, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,crc,rdm,dotprod,fp16,aes,sha2")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::Arm64V2Token $(, $arg: $ty)*) -> $ret $body
};
(@inner Arm64V3Token => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,crc,rdm,dotprod,fp16,aes,sha2,fhm,fcma,sha3,i8mm,bf16")]
    #[inline]
    fn __simd_inner $($gen)* (_: $crate::Arm64V3Token $(, $arg: $ty)*) -> $ret $body
};
(@inner Arm64V3Token, $self_id:ident => ($($gen:tt)*) ($($arg:ident : $ty:ty),*) -> $ret:ty $body:block) => {
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,crc,rdm,dotprod,fp16,aes,sha2,fhm,fcma,sha3,i8mm,bf16")]
    #[inline]
    #[allow(clippy::used_underscore_binding)]
    fn __simd_inner $($gen)* ($self_id: $crate::Arm64V3Token $(, $arg: $ty)*) -> $ret $body
};

}
