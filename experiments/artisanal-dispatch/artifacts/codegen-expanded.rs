#![feature(prelude_import)]
#![forbid(unsafe_code)]
extern crate std;
#[prelude_import]
use std::prelude::rust_2024::*;
mod kernels {
    use artisanal_dispatch::{V3, bridge, tier};
    const _: () = ();
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn leaf_v3(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
    #[inline(always)]
    fn leaf_v3_t(__proof: ::artisanal_dispatch::V3, a: f32, b: f32, c: f32) -> f32 {
        unsafe { leaf_v3(a, b, c) }
    }
    const _: () = ();
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline(never)]
    fn matching(a: f32, b: f32, c: f32) -> f32 {
        leaf_v3(a, b, c)
    }
    #[inline(never)]
    fn proof_boundary(t: V3, a: f32, b: f32, c: f32) -> f32 {
        leaf_v3_t(t, a, b, c)
    }
    const _: () = ();
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline(never)]
    fn enter_v3(a: f32, b: f32, c: f32) -> f32 {
        matching(a, b, c)
    }
    #[inline(always)]
    fn enter_v3_t(__proof: ::artisanal_dispatch::V3, a: f32, b: f32, c: f32) -> f32 {
        unsafe { enter_v3(a, b, c) }
    }
    pub fn run() {
        let t = V3::summon().expect("run this example with std on a V3 CPU");
        let x = std::hint::black_box(1.25);
        {
            ::std::io::_print(
                format_args!(
                    "{0} {1}\n",
                    proof_boundary(t, x, x, x),
                    enter_v3_t(t, x, x, x),
                ),
            );
        };
    }
}
fn main() {
    kernels::run();
}
