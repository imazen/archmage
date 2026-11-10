//! Relaxed min/max only when both semantic policies permit native results.
//! Compile-time module support, plus testable token suppression; no hardware probe.
use archmage::{SimdToken, Wasm128RelaxedToken, Wasm128Token};
use core::arch::wasm32 as w;

macro_rules! kernels {
    ($name:ident,$float:ident,$width:literal,$make:ident,$extract:ident,$min:ident,$max:ident) => {
        #[inline(always)]
        pub(crate) fn $name<T: SimdToken, const MAX: bool, const N: usize>(
            token: T,
            a: [$float; N],
            b: [$float; N],
        ) -> Option<[$float; N]> {
            let t = (&token as &dyn core::any::Any).downcast_ref::<Wasm128Token>()?;
            Wasm128RelaxedToken::summon()?;
            Some(kernel::$name::<MAX, N>(*t, a, b))
        }
    };
}
// The feature function works in native chunks; exact array sizes are constants.
macro_rules! feature_kernel {
    ($name:ident,$float:ident,$width:literal,$splat:ident,$replace:ident,$extract:ident,$min:ident,$max:ident,[$($lane:literal),+]) => {
        #[archmage::arcane]
        pub(super) fn $name<const MAX:bool,const N:usize>(_t:Wasm128Token,a:[$float;N],b:[$float;N])->[$float;N] {
            let mut out=[0.0;N];
            for base in (0..N).step_by($width) {
                let mut va=w::$splat(0.0); let mut vb=w::$splat(0.0);
                $(va=w::$replace::<$lane>(va,a[base+$lane]);vb=w::$replace::<$lane>(vb,b[base+$lane]);)+
                let r=if MAX {w::$max(va,vb)} else {w::$min(va,vb)};
                $(out[base+$lane]=w::$extract::<$lane>(r);)+
            }
            out
        }
    };
}
kernels!(
    f32,
    f32,
    4,
    f32x4,
    f32x4_extract_lane,
    f32x4_relaxed_min,
    f32x4_relaxed_max
);
kernels!(
    f64,
    f64,
    2,
    f64x2,
    f64x2_extract_lane,
    f64x2_relaxed_min,
    f64x2_relaxed_max
);
mod kernel {
    use super::*;
    feature_kernel!(
        f32,
        f32,
        4,
        f32x4_splat,
        f32x4_replace_lane,
        f32x4_extract_lane,
        f32x4_relaxed_min,
        f32x4_relaxed_max,
        [0, 1, 2, 3]
    );
    feature_kernel!(
        f64,
        f64,
        2,
        f64x2_splat,
        f64x2_replace_lane,
        f64x2_extract_lane,
        f64x2_relaxed_min,
        f64x2_relaxed_max,
        [0, 1]
    );
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    #[test]
    fn testable_relaxed_selection() {
        let _guard = archmage::testing::lock_token_testing();
        let t = Wasm128Token::summon().expect("SIMD module");
        assert!(super::f32::<_, false, 4>(t, [1.; 4], [2.; 4]).is_some());
        if Wasm128RelaxedToken::compiled_with().is_none() {
            Wasm128RelaxedToken::dangerously_disable_token_process_wide(true).unwrap();
            assert!(super::f32::<_, false, 4>(t, [1.; 4], [2.; 4]).is_none());
            Wasm128RelaxedToken::dangerously_disable_token_process_wide(false).unwrap();
            assert!(super::f32::<_, false, 4>(t, [1.; 4], [2.; 4]).is_some());
        }
    }
}
