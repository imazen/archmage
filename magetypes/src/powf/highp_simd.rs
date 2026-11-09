//! Packed binary64 evaluation of the canonical HIGHP graph.
//!
//! Only native-fused plans enter here. Table reduction is integer/scalar data
//! preparation; the nine fused nodes and all separately rounded nodes retain
//! the scalar graph's order. Exceptional packs use that same scalar graph.
use super::{
    graph,
    highp_data::{A, C, LOG, SHIFT, TABLE},
    highp_scalar,
};
use crate::simd::backends::F64x2Backend;

macro_rules! packed {
    ($name:ident, $vector:ident, $backend:ident, $lanes:literal) => {
        #[inline(always)]
        pub(super) fn $name<
            T: crate::simd::backends::$backend + F64x2Backend,
            const OPTIONS: u32,
            const N: usize,
        >(
            t: T,
            x: [f32; N],
            n: f32,
        ) -> [f32; N] {
            use crate::simd::generic::$vector as V;
            if OPTIONS & 3 == 2 && n == 0. {
                return [1.; N];
            }
            let mut out = [0.; N];
            let mut offset = 0;
            while offset + $lanes <= N {
                let mut normal = OPTIONS & 3 == 2 || (n.is_finite() && n != 0.);
                if OPTIONS & 3 != 2 {
                    for lane in 0..$lanes {
                        normal &=
                            x[offset + lane].to_bits().wrapping_sub(0x0080_0000) < 0x7f00_0000;
                    }
                }
                if OPTIONS & 3 == 2 {
                    // Zero is admitted, but uncommon. One pack predicate keeps
                    // zero restoration out of the ordinary arithmetic graph.
                    for lane in 0..$lanes {
                        normal &= x[offset + lane] != 0.;
                    }
                }
                if normal {
                    let mut zs = [0.; $lanes];
                    let mut invcs = [0.; $lanes];
                    let mut y0s = [0.; $lanes];
                    for lane in 0..$lanes {
                        let ix = x[offset + lane].to_bits();
                        let tmp = ix.wrapping_sub(0x3f33_0000);
                        let index = ((tmp >> 19) & 15) as usize;
                        let top = tmp & 0xff80_0000;
                        zs[lane] = f32::from_bits(ix.wrapping_sub(top)) as f64;
                        invcs[lane] = LOG[index].0;
                        y0s[lane] = LOG[index].1 + f64::from((top as i32) >> 23);
                    }
                    let r = V::from_array_t(t, zs)
                        .mul_add_relaxed(V::from_array_t(t, invcs), V::splat_t(t, -1.));
                    let r2 = r * r;
                    let h = V::splat_t(t, A[0]).mul_add_relaxed(r, V::splat_t(t, A[1]));
                    let p = V::splat_t(t, A[2]).mul_add_relaxed(r, V::splat_t(t, A[3]));
                    let r4 = r2 * r2;
                    let q = V::splat_t(t, A[4]).mul_add_relaxed(r, V::from_array_t(t, y0s));
                    let q = p.mul_add_relaxed(r2, q);
                    let logx = h.mul_add_relaxed(r4, q);
                    let ylogx = V::splat_t(t, f64::from(n)) * logx;
                    if OPTIONS & 3 != 2 {
                        // Keep the complete scalar under/overflow policy for the
                        // uncommon edge region, including signed final rounding.
                        for value in ylogx.to_array() {
                            normal &= value.abs() < 126.;
                        }
                    }
                    if normal {
                        let kd = ylogx + V::splat_t(t, SHIFT);
                        let kis = kd.to_array();
                        let r = ylogx - (kd - V::splat_t(t, SHIFT));
                        let mut scales = [0.; $lanes];
                        for lane in 0..$lanes {
                            let ki = kis[lane].to_bits();
                            scales[lane] =
                                f64::from_bits(TABLE[(ki & 31) as usize].wrapping_add(ki << 47));
                        }
                        let z = V::splat_t(t, C[0]).mul_add_relaxed(r, V::splat_t(t, C[1]));
                        let r2 = r * r;
                        let tail = V::splat_t(t, C[2]).mul_add_relaxed(r, V::splat_t(t, 1.));
                        let result =
                            (z.mul_add_relaxed(r2, tail) * V::from_array_t(t, scales)).to_array();
                        for lane in 0..$lanes {
                            out[offset + lane] = result[lane] as f32;
                        }
                        offset += $lanes;
                        continue;
                    }
                }
                for lane in 0..$lanes {
                    out[offset + lane] = if OPTIONS & 3 == 2 {
                        highp_scalar::pow::<T, { graph::NATIVE }, true>(t, x[offset + lane], n)
                    } else {
                        highp_scalar::pow::<T, { graph::NATIVE }, false>(t, x[offset + lane], n)
                    };
                }
                offset += $lanes;
            }
            while offset < N {
                out[offset] = if OPTIONS & 3 == 2 {
                    highp_scalar::pow::<T, { graph::NATIVE }, true>(t, x[offset], n)
                } else {
                    highp_scalar::pow::<T, { graph::NATIVE }, false>(t, x[offset], n)
                };
                offset += 1;
            }
            out
        }
    };
}
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
packed!(packed2, f64x2, F64x2Backend, 2);
#[cfg(target_arch = "x86_64")]
packed!(packed4, f64x4, F64x4Backend, 4);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
packed!(packed8, f64x8, F64x8Backend, 8);
