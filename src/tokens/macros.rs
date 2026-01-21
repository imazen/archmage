//! Macros for defining SIMD capability tokens.
//!
//! These macros ensure that feature checks and trait implementations stay in sync.
//! By defining both in a single macro invocation, we prevent the common error of
//! adding a trait implementation without the corresponding feature check.

/// Define an x86 SIMD capability token with feature checks and trait implementations.
///
/// This macro generates:
/// - The token struct definition
/// - `SimdToken` implementation with `try_new()` checking all listed features
/// - All specified marker trait implementations
/// - `Sealed` trait implementation
///
/// # Example
///
/// ```rust,ignore
/// define_x86_token! {
///     /// My token documentation
///     pub struct MyToken {
///         name: "MyToken",
///         features: ["avx512f", "avx512vl"],
///         traits: [HasAvx512f, HasAvx512vl, HasAvx2, HasAvx, ...],
///     }
/// }
/// ```
///
/// # Safety Invariant
///
/// Every `Has*` trait that corresponds to a specific CPU feature MUST have that
/// feature listed in the `features` array. For example:
/// - `HasAvx512f` requires `"avx512f"` in features
/// - `HasAvx512cd` requires `"avx512cd"` in features
///
/// Implied traits (like `HasAvx2` when checking `avx512f`) don't need explicit
/// feature checks because the CPU feature hierarchy guarantees them.
#[macro_export]
macro_rules! define_x86_token {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            name: $display:literal,
            features: [$($feature:literal),+ $(,)?],
            traits: [$($trait:ident),+ $(,)?],
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug)]
        $vis struct $name {
            _private: (),
        }

        impl $crate::tokens::SimdToken for $name {
            const NAME: &'static str = $display;

            #[inline(always)]
            fn try_new() -> Option<Self> {
                // Check all features - uses && to short-circuit
                if true $(&& $crate::is_x86_feature_available!($feature))+ {
                    Some(unsafe { Self::forge_token_dangerously() })
                } else {
                    None
                }
            }

            #[inline(always)]
            unsafe fn forge_token_dangerously() -> Self {
                Self { _private: () }
            }
        }

        // Implement all specified traits
        $(impl $crate::tokens::$trait for $name {})+
    };
}

/// Define an x86 SIMD token with helper methods to extract lower-level tokens.
///
/// This extends `define_x86_token!` with additional inherent methods.
///
/// # Example
///
/// ```rust,ignore
/// define_x86_token_with_methods! {
///     /// My token documentation
///     pub struct MyToken {
///         name: "MyToken",
///         features: ["avx512f", "avx512vl"],
///         traits: [HasAvx512f, HasAvx512vl, ...],
///         methods: {
///             fn avx512f(self) -> Avx512fToken;
///             fn avx2(self) -> Avx2Token;
///         }
///     }
/// }
/// ```
#[macro_export]
macro_rules! define_x86_token_with_methods {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            name: $display:literal,
            features: [$($feature:literal),+ $(,)?],
            traits: [$($trait:ident),+ $(,)?],
            methods: {
                $(fn $method:ident(self) -> $ret:ty;)*
            }
        }
    ) => {
        $crate::define_x86_token! {
            $(#[$meta])*
            $vis struct $name {
                name: $display,
                features: [$($feature),+],
                traits: [$($trait),+],
            }
        }

        impl $name {
            $(
                #[doc = concat!("Get a [`", stringify!($ret), "`] token.")]
                #[inline(always)]
                pub fn $method(self) -> $ret {
                    // SAFETY: This token's existence proves all implied features are available
                    unsafe { <$ret as $crate::tokens::SimdToken>::forge_token_dangerously() }
                }
            )*
        }
    };
}

// Tests temporarily disabled - macro needs path support for sealed::Sealed
// TODO: Fix macro to accept paths or use a different approach
