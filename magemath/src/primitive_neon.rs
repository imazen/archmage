//! Propagating IEEE min/max are one NEON instruction, including zero ordering.
use archmage::{NeonToken, SimdToken};
use core::arch::aarch64 as n;
macro_rules! entry {
    ($name:ident,$float:ident) => {
        #[inline(always)]
        pub(crate) fn $name<T: SimdToken, const MAX: bool, const N: usize>(
            t: T,
            a: [$float; N],
            b: [$float; N],
        ) -> Option<[$float; N]> {
            let proof = (&t as &dyn core::any::Any).downcast_ref::<NeonToken>()?;
            Some(kernel::$name::<MAX, N>(*proof, a, b))
        }
    };
}
macro_rules! kernel {
 ($name:ident,$float:ident,$width:literal,$splat:ident,$replace:ident,$extract:ident,$min:ident,$max:ident,[$($lane:literal),+])=>{
  #[archmage::arcane]
  pub(super) fn $name<const MAX:bool,const N:usize>(_t:NeonToken,a:[$float;N],b:[$float;N])->[$float;N]{
   let mut out=[0.0;N];
   for base in (0..N).step_by($width){
    let mut va=n::$splat(0.0);let mut vb=n::$splat(0.0);
    $(va=n::$replace::<$lane>(a[base+$lane],va);vb=n::$replace::<$lane>(b[base+$lane],vb);)+
    let r=if MAX{n::$max(va,vb)}else{n::$min(va,vb)};
    $(out[base+$lane]=n::$extract::<$lane>(r);)+
   }out
  }
 };
}
entry!(f32, f32);
entry!(f64, f64);
#[inline(always)]
pub(crate) fn estimate<T: SimdToken, const RSQRT: bool, const N: usize>(
    t: T,
    a: [f32; N],
) -> Option<[f32; N]> {
    let proof = (&t as &dyn core::any::Any).downcast_ref::<NeonToken>()?;
    Some(kernel::estimate::<RSQRT, N>(*proof, a))
}
mod kernel {
    #[archmage::arcane]
    pub(super) fn estimate<const RSQRT: bool, const N: usize>(
        _t: NeonToken,
        a: [f32; N],
    ) -> [f32; N] {
        let mut out = [0.0; N];
        for base in (0..N).step_by(4) {
            let mut v = n::vdupq_n_f32(0.0);
            v = n::vsetq_lane_f32::<0>(a[base], v);
            v = n::vsetq_lane_f32::<1>(a[base + 1], v);
            v = n::vsetq_lane_f32::<2>(a[base + 2], v);
            v = n::vsetq_lane_f32::<3>(a[base + 3], v);
            let r = if RSQRT {
                n::vrsqrteq_f32(v)
            } else {
                n::vrecpeq_f32(v)
            };
            out[base] = n::vgetq_lane_f32::<0>(r);
            out[base + 1] = n::vgetq_lane_f32::<1>(r);
            out[base + 2] = n::vgetq_lane_f32::<2>(r);
            out[base + 3] = n::vgetq_lane_f32::<3>(r);
        }
        out
    }

    use super::*;
    kernel!(
        f32,
        f32,
        4,
        vdupq_n_f32,
        vsetq_lane_f32,
        vgetq_lane_f32,
        vminq_f32,
        vmaxq_f32,
        [0, 1, 2, 3]
    );
    kernel!(
        f64,
        f64,
        2,
        vdupq_n_f64,
        vsetq_lane_f64,
        vgetq_lane_f64,
        vminq_f64,
        vmaxq_f64,
        [0, 1]
    );
}
