/* Copyright (c) 2015, Intel Corporation Redistribution and use in source and
 binary forms, with or without modification, are permitted provided that the
 following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Intel Corporation nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */
//! Normal-domain integer emulation of Intel's published RCP14S algorithm.
//! Source URL and original C notice are in reference/ORIGIN.md.
//! Prototype only; all supported inputs have 2^-125 <= abs(x) <= 2^125.
#![allow(dead_code)] // Qualification and timing tools instantiate different widths.
const COEFFICIENTS: [i32; 64] = [
    266362865, 262296529, 258352053, 254524313, 250808189, 247199589, 243693387, 240284469,
    236970781, 233745161, 230607603, 227554015, 224578253, 221681339, 218857129, 216103575,
    213419655, 210800247, 208245353, 205751897, 203316811, 200939069, 198615601, 196345379,
    194126359, 191957515, 189836801, 187761141, 185730539, 183746015, 181801429, 179897803,
    178033091, 176207289, 174418353, 172665255, 170946975, 169262487, 167610767, 165991815,
    164403585, 162844025, 161316209, 159815019, 158342501, 156896605, 155476311, 154081617,
    152712523, 151366981, 150044991, 148746555, 147468597, 146214191, 144980267, 143766821,
    142573857, 141399325, 140246295, 139113747, 137997583, 136897803, 135814407, 134747395,
];
#[inline(always)]
pub(crate) fn scalar(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = bits & 0x80000000;
    let exp = (bits >> 23) & 255;
    let mant = bits & 0x7fffff;
    if mant == 0 {
        return f32::from_bits(sign | ((254 - exp) << 23));
    }
    let c = COEFFICIENTS[(mant >> 17) as usize];
    let a = c >> 10;
    let b = c & 1023;
    let d = ((mant >> 7) & 1023) as i32 - 512;
    let s = ((a << 8) - b * d) as u32;
    f32::from_bits(sign | ((253 - exp) << 23) | ((s >> 2) & 0x7fff80))
}
macro_rules! vector {
    ($name:ident,$v:ident,$i:ident,$bound:ident,$n:literal) => {
        #[inline(always)]
        pub(crate) fn $name<T: magetypes::simd::backends::$bound>(
            t: T,
            x: magetypes::simd::generic::$v<T>,
        ) -> magetypes::simd::generic::$v<T> {
            use magetypes::simd::generic::$i as I;
            let bits = x.bitcast_to_i32();
            let sp = |x| I::splat_t(t, x);
            let coefficients = I::from_array_t(
                t,
                x.to_array()
                    .map(|v| COEFFICIENTS[((v.to_bits() >> 17) & 63) as usize]),
            );
            let a = coefficients.shr_arithmetic_const::<10>();
            let b = coefficients & sp(1023);
            let d = (bits.shr_logical_const::<7>() & sp(1023)) - sp(512);
            let s = a.shl_const::<8>() - b * d;
            let exp = bits.shr_logical_const::<23>() & sp(255);
            let sign = bits & sp(i32::MIN);
            let normal = sign
                | (sp(253) - exp).shl_const::<23>()
                | (s.shr_logical_const::<2>() & sp(0x7fff80));
            let power = sign | (sp(254) - exp).shl_const::<23>();
            I::blend((bits & sp(0x7fffff)).simd_eq(sp(0)), power, normal).bitcast_to_f32()
        }
    };
}
vector!(emulate4, f32x4, i32x4, F32x4Convert, 4);
vector!(emulate8, f32x8, i32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(emulate16, f32x16, i32x16, F32x16Convert, 16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod native {
    use archmage::X64V4Token;
    use core::arch::x86_64 as x;
    macro_rules! entry {
        ($name:ident,$b:ident,$intr:ident,$n:literal) => {
            #[archmage::arcane]
            pub(crate) fn $name(t: X64V4Token, input: [f32; $n]) -> [f32; $n] {
                use magetypes::simd::backends::$b as B;
                B::to_array(t, x::$intr(B::from_array(t, input)))
            }
        };
    }
    entry!(native4, F32x4Backend, _mm_rcp14_ps, 4);
    entry!(native8, F32x8Backend, _mm256_rcp14_ps, 8);
    entry!(native16, F32x16Backend, _mm512_rcp14_ps, 16);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) use native::{native4, native8, native16};
