use archmage::{arcane, incant, IntoConcreteToken, X64V3Token, NeonToken, ScalarToken};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_inner_v3(_token: X64V3Token, x: f32) -> f32 {
    x * 2.0
}
#[inline(always)]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 {
    let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_inner_v3(_token, x) }
}
fn inner_scalar(_token: ScalarToken, x: f32) -> f32 {
    x * 2.0
}
fn pass_through<T: IntoConcreteToken>(token: T, x: f32) -> f32 {
    '__incant: {
        use archmage::IntoConcreteToken;
        let __incant_token = token;
        {
            if let Some(__t) = __incant_token.as_x64v3() {
                break '__incant inner_v3(__t, x);
            }
        }
        if let Some(__t) = __incant_token.as_scalar() {
            break '__incant inner_scalar(__t, x);
        }
        {
            fn __incant_token_name<__T: archmage::SimdToken>(_: &__T) -> &'static str {
                __T::NAME
            }
            {
                ::core::panicking::panic_fmt(
                    format_args!(
                        "incant!(.. with token): the held token `{0}` matches none of [{1}]. `with token` dispatches on the token\'s exact type; add a `default` arm, or dispatch with the token of a listed tier",
                        __incant_token_name(& __incant_token),
                        "v4, v3, neon, wasm128, scalar",
                    ),
                );
            }
        }
    }
}
fn main() {}
