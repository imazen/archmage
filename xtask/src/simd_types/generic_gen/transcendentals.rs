//! Transcendentals generator and mod.rs generator.

use indoc::formatdoc;

use crate::simd_types::types::{SimdType, SimdWidth};

pub(super) fn gen_transcendentals(
    float_type: &str,
    int_type: &str,
    lanes: usize,
    float_backend: &str,
    convert_trait: &str,
    int_backend: &str,
) -> String {
    formatdoc! {r#"
        //! Transcendental math functions for `{float_type}<T>`.
        //!
        //! Generic implementations using IEEE 754 bit manipulation and polynomial
        //! approximation. Available when `T: {convert_trait}` (float↔int conversion).
        //!
        //! Two precision tiers:
        //! - **lowp** (~1% max error): Fast, suitable for perceptual/audio work
        //! - **midp** (~3 ULP): Accurate, suitable for most numerical work
        //!
        //! Variant suffixes:
        //! - `_unchecked`: No edge case handling (fastest, undefined for ≤0/NaN/Inf)
        //! - (normal): Basic edge case handling (0→-inf, negative→NaN for log)
        //! - `_precise`: Full handling including denormals

        use crate::simd::backends::{{{float_backend}, {convert_trait}, {int_backend}}};
        use crate::simd::generic::{{{float_type}, {int_type}}};

        /// Splat an i32 into {int_type} (disambiguates from f32 splat).
        #[inline(always)]
        fn splat_i32<T: {convert_trait}>(v: i32) -> {int_type}<T> {{
            {int_type}::from_repr_unchecked(<T as {int_backend}>::splat(v))
        }}

        /// Splat an f32 into {float_type}.
        #[inline(always)]
        fn splat_f32<T: {convert_trait}>(v: f32) -> {float_type}<T> {{
            {float_type}::from_repr_unchecked(<T as {float_backend}>::splat(v))
        }}

        impl<T: {convert_trait}> {float_type}<T> {{
            // ====== Low-Precision Transcendentals (~1% error) ======

            /// Low-precision base-2 logarithm (~1% max error).
            ///
            /// Uses rational polynomial approximation on the mantissa.
            /// Result is undefined for x <= 0.
            #[inline(always)]
            pub fn log2_lowp(self) -> Self {{
                const P0: f32 = -1.850_383_3e-6;
                const P1: f32 = 1.428_716_1;
                const P2: f32 = 0.742_458_7;
                const Q0: f32 = 0.990_328_14;
                const Q1: f32 = 1.009_671_8;
                const Q2: f32 = 0.174_093_43;

                let x_bits = self.bitcast_to_i32();
                let offset = splat_i32::<T>(0x3f2a_aaab_u32 as i32);
                let exp_bits = x_bits - offset;
                let exp_shifted = exp_bits.shr_arithmetic_const::<23>();
                let mantissa_bits = x_bits - exp_shifted.shl_const::<23>();
                let mantissa = mantissa_bits.bitcast_to_f32();
                let exp_val = exp_shifted.to_f32();

                let m = mantissa - splat_f32::<T>(1.0);

                let yp = splat_f32::<T>(P2).mul_add(m, splat_f32::<T>(P1));
                let yp = yp.mul_add(m, splat_f32::<T>(P0));

                let yq = splat_f32::<T>(Q2).mul_add(m, splat_f32::<T>(Q1));
                let yq = yq.mul_add(m, splat_f32::<T>(Q0));

                yp / yq + exp_val
            }}

            /// Low-precision base-2 logarithm, no edge case handling.
            #[inline(always)]
            pub fn log2_lowp_unchecked(self) -> Self {{
                self.log2_lowp()
            }}

            /// Low-precision base-2 exponential (~1% max error).
            #[inline(always)]
            pub fn exp2_lowp(self) -> Self {{
                const C0: f32 = 1.0;
                const C1: f32 = core::f32::consts::LN_2;
                const C2: f32 = 0.240_226_5;
                const C3: f32 = 0.055_504_11;

                let x = self.max(splat_f32::<T>(-126.0)).min(splat_f32::<T>(126.0));
                let xi = x.floor();
                let xf = x - xi;

                let poly = splat_f32::<T>(C3).mul_add(xf, splat_f32::<T>(C2));
                let poly = poly.mul_add(xf, splat_f32::<T>(C1));
                let poly = poly.mul_add(xf, splat_f32::<T>(C0));

                let xi_i32 = xi.to_i32_round();
                let scale_bits = (xi_i32 + splat_i32::<T>(127)).shl_const::<23>();
                poly * scale_bits.bitcast_to_f32()
            }}

            /// Low-precision base-2 exponential, no edge case handling.
            #[inline(always)]
            pub fn exp2_lowp_unchecked(self) -> Self {{
                self.exp2_lowp()
            }}

            /// Low-precision natural logarithm.
            #[inline(always)]
            pub fn ln_lowp(self) -> Self {{
                self.log2_lowp() * splat_f32::<T>(core::f32::consts::LN_2)
            }}

            /// Low-precision natural logarithm, no edge case handling.
            #[inline(always)]
            pub fn ln_lowp_unchecked(self) -> Self {{
                self.ln_lowp()
            }}

            /// Low-precision natural exponential.
            #[inline(always)]
            pub fn exp_lowp(self) -> Self {{
                (self * splat_f32::<T>(core::f32::consts::LOG2_E)).exp2_lowp()
            }}

            /// Low-precision natural exponential, no edge case handling.
            #[inline(always)]
            pub fn exp_lowp_unchecked(self) -> Self {{
                self.exp_lowp()
            }}

            /// Low-precision base-10 logarithm.
            #[inline(always)]
            pub fn log10_lowp(self) -> Self {{
                self.log2_lowp() * splat_f32::<T>(core::f32::consts::LN_2 / core::f32::consts::LN_10)
            }}

            /// Low-precision base-10 logarithm, no edge case handling.
            #[inline(always)]
            pub fn log10_lowp_unchecked(self) -> Self {{
                self.log10_lowp()
            }}

            /// Low-precision power function: `self^n`. Returns 0 for zero input.
            #[inline(always)]
            pub fn pow_lowp(self, n: f32) -> Self {{
                let result = (self.log2_lowp() * splat_f32::<T>(n)).exp2_lowp();
                // Zero masking: pow(0, n) = 0 for n > 0
                let is_zero = self.simd_eq(splat_f32::<T>(0.0));
                Self::blend(is_zero, splat_f32::<T>(0.0), result)
            }}

            /// Low-precision power function, no edge case handling.
            #[inline(always)]
            pub fn pow_lowp_unchecked(self, n: f32) -> Self {{
                self.pow_lowp(n)
            }}

            // ====== Mid-Precision Transcendentals (~3 ULP) ======

            /// Mid-precision base-2 logarithm (~3 ULP).
            ///
            /// Uses (a-1)/(a+1) transform with odd polynomial evaluation.
            /// Result is undefined for x <= 0.
            #[inline(always)]
            pub fn log2_midp_unchecked(self) -> Self {{
                const SQRT2_OVER_2: u32 = 0x3f35_04f3;
                const ONE_BITS: u32 = 0x3f80_0000;
                const MANTISSA_MASK: i32 = 0x007f_ffff_u32 as i32;

                const C0: f32 = 2.885_39;
                const C1: f32 = 0.961_800_76;
                const C2: f32 = 0.576_974_45;
                const C3: f32 = 0.434_411_97;

                let x_bits = self.bitcast_to_i32();

                let offset = splat_i32::<T>((ONE_BITS - SQRT2_OVER_2) as i32);
                let adjusted = x_bits + offset;

                let exp_raw = adjusted.shr_arithmetic_const::<23>();
                let n = (exp_raw - splat_i32::<T>(127)).to_f32();

                let mantissa_bits = adjusted & splat_i32::<T>(MANTISSA_MASK);
                let a = (mantissa_bits + splat_i32::<T>(SQRT2_OVER_2 as i32)).bitcast_to_f32();

                let one = splat_f32::<T>(1.0);
                let y = (a - one) / (a + one);
                let y2 = y * y;

                let poly = splat_f32::<T>(C3).mul_add(y2, splat_f32::<T>(C2));
                let poly = poly.mul_add(y2, splat_f32::<T>(C1));
                let poly = poly.mul_add(y2, splat_f32::<T>(C0));

                poly.mul_add(y, n)
            }}

            /// Mid-precision base-2 logarithm with edge case handling.
            ///
            /// Returns -inf for 0, NaN for negative values.
            #[inline(always)]
            pub fn log2_midp(self) -> Self {{
                let result = self.log2_midp_unchecked();
                let zero = splat_f32::<T>(0.0);
                let result = Self::blend(
                    self.simd_eq(zero),
                    splat_f32::<T>(f32::NEG_INFINITY),
                    result,
                );
                Self::blend(self.simd_lt(zero), splat_f32::<T>(f32::NAN), result)
            }}

            /// Mid-precision base-2 logarithm with denormal handling.
            #[inline(always)]
            pub fn log2_midp_precise(self) -> Self {{
                self.log2_midp()
            }}

            /// Mid-precision base-2 exponential (~1 ULP). Undefined for extreme inputs.
            ///
            /// Uses round-to-nearest splitting to keep |frac| <= 0.5, giving
            /// ~1000x less polynomial truncation error than floor-based splitting.
            #[inline(always)]
            pub fn exp2_midp_unchecked(self) -> Self {{
                const C0: f32 = 1.0;
                const C1: f32 = core::f32::consts::LN_2;
                const C2: f32 = 0.240_226_46;
                const C3: f32 = 0.055_504_545;
                const C4: f32 = 0.009_618_055;
                const C5: f32 = 0.001_333_37;
                const C6: f32 = 0.000_154_47;

                // Round-to-nearest keeps |frac| <= 0.5 (vs floor's [0,1))
                // Clamp xi to 127 so the bit trick (n+127)<<23 doesn't overflow
                let xi = self.round().min(splat_f32::<T>(127.0));
                let xf = self - xi;

                let poly = splat_f32::<T>(C6).mul_add(xf, splat_f32::<T>(C5));
                let poly = poly.mul_add(xf, splat_f32::<T>(C4));
                let poly = poly.mul_add(xf, splat_f32::<T>(C3));
                let poly = poly.mul_add(xf, splat_f32::<T>(C2));
                let poly = poly.mul_add(xf, splat_f32::<T>(C1));
                let poly = poly.mul_add(xf, splat_f32::<T>(C0));

                let xi_i32 = xi.to_i32_round();
                let scale_bits = (xi_i32 + splat_i32::<T>(127)).shl_const::<23>();
                poly * scale_bits.bitcast_to_f32()
            }}

            /// Mid-precision base-2 exponential with clamping.
            ///
            /// Returns 0 for x < -126 (denormal results can't be constructed),
            /// inf for x >= 128.
            #[inline(always)]
            pub fn exp2_midp(self) -> Self {{
                let underflow_limit = splat_f32::<T>(-126.0);
                let overflow_limit = splat_f32::<T>(128.0);

                // Clamp to prevent overflow in intermediate calculations
                let clamped = self.max(underflow_limit).min(overflow_limit);
                let result = clamped.exp2_midp_unchecked();

                // Handle edge cases: large negative → 0, large positive → inf
                // 2^128 > f32::MAX, so >= 128 must return inf
                let is_underflow = self.simd_lt(underflow_limit);
                let is_overflow = self.simd_ge(overflow_limit);
                let zero = splat_f32::<T>(0.0);
                let inf = splat_f32::<T>(f32::INFINITY);
                let result = Self::blend(is_underflow, zero, result);
                Self::blend(is_overflow, inf, result)
            }}

            /// Mid-precision base-2 exponential with full edge case handling.
            #[inline(always)]
            pub fn exp2_midp_precise(self) -> Self {{
                self.exp2_midp()
            }}

            /// Mid-precision natural logarithm.
            #[inline(always)]
            pub fn ln_midp(self) -> Self {{
                self.log2_midp() * splat_f32::<T>(core::f32::consts::LN_2)
            }}

            /// Mid-precision natural logarithm, no edge case handling.
            #[inline(always)]
            pub fn ln_midp_unchecked(self) -> Self {{
                self.log2_midp_unchecked() * splat_f32::<T>(core::f32::consts::LN_2)
            }}

            /// Mid-precision natural logarithm with denormal handling.
            #[inline(always)]
            pub fn ln_midp_precise(self) -> Self {{
                self.ln_midp()
            }}

            /// Mid-precision natural exponential.
            #[inline(always)]
            pub fn exp_midp(self) -> Self {{
                (self * splat_f32::<T>(core::f32::consts::LOG2_E)).exp2_midp()
            }}

            /// Mid-precision natural exponential, no edge case handling.
            #[inline(always)]
            pub fn exp_midp_unchecked(self) -> Self {{
                (self * splat_f32::<T>(core::f32::consts::LOG2_E)).exp2_midp_unchecked()
            }}

            /// Mid-precision natural exponential with full edge case handling.
            #[inline(always)]
            pub fn exp_midp_precise(self) -> Self {{
                self.exp_midp()
            }}

            /// Mid-precision base-10 logarithm.
            #[inline(always)]
            pub fn log10_midp(self) -> Self {{
                self.log2_midp() * splat_f32::<T>(core::f32::consts::LN_2 / core::f32::consts::LN_10)
            }}

            /// Mid-precision base-10 logarithm, no edge case handling.
            #[inline(always)]
            pub fn log10_midp_unchecked(self) -> Self {{
                self.log2_midp_unchecked()
                    * splat_f32::<T>(core::f32::consts::LN_2 / core::f32::consts::LN_10)
            }}

            /// Mid-precision base-10 logarithm with denormal handling.
            #[inline(always)]
            pub fn log10_midp_precise(self) -> Self {{
                self.log10_midp()
            }}

            /// Mid-precision power function: `self^n`.
            #[inline(always)]
            pub fn pow_midp(self, n: f32) -> Self {{
                (self.log2_midp() * splat_f32::<T>(n)).exp2_midp()
            }}

            /// Mid-precision power function, no edge case handling.
            #[inline(always)]
            pub fn pow_midp_unchecked(self, n: f32) -> Self {{
                (self.log2_midp_unchecked() * splat_f32::<T>(n)).exp2_midp_unchecked()
            }}

            /// Mid-precision power function with full edge case handling.
            #[inline(always)]
            pub fn pow_midp_precise(self, n: f32) -> Self {{
                self.pow_midp(n)
            }}

            /// Low-precision cube root (~15 bits, ~4.5 decimal digits).
            ///
            /// Uses Kahan's bit-hack initial approximation followed by 1 Halley
            /// iteration. Max error ~259 ULP vs `std::f32::cbrt`, uniform across
            /// all magnitudes (1e-38..1e38). Average ~58 ULP, median ~40 ULP.
            ///
            /// Fastest cbrt variant — 1.8x faster than `cbrt_midp` (1 division
            /// vs 2). Suitable for perceptual color (Oklab/XYB) targeting 8-bit
            /// output, or any context where ~4.5 decimal digits suffice.
            ///
            /// Returns ±0 for ±0 input. Does not handle denormals or infinity — use
            /// `cbrt_midp_precise` for those.
            #[inline(always)]
            pub fn cbrt_lowp(self) -> Self {{
                const MAGIC: u32 = 0x2a50_8c2d;

                let sign_mask = splat_f32::<T>(-0.0);
                let sign = self & sign_mask;
                let abs_x = self.abs();

                let abs_arr = abs_x.to_array();
                let approx_arr: [f32; {lanes}] =
                    core::array::from_fn(|i| f32::from_bits((abs_arr[i].to_bits() / 3) + MAGIC));
                let mut y = {float_type}::from_repr_unchecked(<T as {float_backend}>::from_array(approx_arr));

                // Halley iteration: y *= (y³ + 2x) / (2y³ + x)
                // Compute ratio first to avoid intermediate overflow.
                // Triples bits of precision: ~5 → ~15
                let two = splat_f32::<T>(2.0);
                let y3 = y * y * y;
                y *= (y3 + two * abs_x) / (two * y3 + abs_x);

                let result = y | sign;

                // Zero masking: cbrt(±0) = ±0 (bit hack gives garbage for zero)
                let is_zero = self.simd_eq(splat_f32::<T>(0.0));
                Self::blend(is_zero, self, result)
            }}

            /// Mid-precision cube root (max 3 ULP vs `std::f32::cbrt`).
            ///
            /// Uses Kahan's bit-hack initial approximation followed by 2 Halley
            /// iterations. Each Halley step triples precision: ~5 → ~15 → ~45
            /// bits, saturating f32's 24-bit mantissa. Error is uniform across
            /// all magnitudes (1e-38..1e38): max 3 ULP, average 0.47 ULP.
            ///
            /// Uses 2 divisions (vs 3 for Newton-Raphson at equivalent accuracy),
            /// making it ~35% faster at equal or better precision.
            ///
            /// Returns ±0 for ±0 input. Does not handle denormals or infinity — use
            /// `cbrt_midp_precise` for those.
            #[inline(always)]
            pub fn cbrt_midp(self) -> Self {{
                const MAGIC: u32 = 0x2a50_8c2d;

                let sign_mask = splat_f32::<T>(-0.0);
                let sign = self & sign_mask;
                let abs_x = self.abs();

                let abs_arr = abs_x.to_array();
                let approx_arr: [f32; {lanes}] =
                    core::array::from_fn(|i| f32::from_bits((abs_arr[i].to_bits() / 3) + MAGIC));
                let mut y = {float_type}::from_repr_unchecked(<T as {float_backend}>::from_array(approx_arr));

                // 2 Halley iterations: y *= (y³ + 2x) / (2y³ + x)
                // Compute ratio first to avoid intermediate overflow.
                let two = splat_f32::<T>(2.0);
                for _ in 0..2 {{
                    let y3 = y * y * y;
                    y *= (y3 + two * abs_x) / (two * y3 + abs_x);
                }}

                let result = y | sign;

                // Zero masking: cbrt(±0) = ±0 (bit hack gives garbage for zero)
                let is_zero = self.simd_eq(splat_f32::<T>(0.0));
                Self::blend(is_zero, self, result)
            }}

            /// Mid-precision cube root with denormal and zero handling (max 3 ULP).
            ///
            /// Wraps `cbrt_midp()` with denormal scaling and zero masking.
            /// Handles all edge cases including denormals, zeros, and negative values.
            #[inline(always)]
            pub fn cbrt_midp_precise(self) -> Self {{
                let zero = splat_f32::<T>(0.0);
                let is_zero = self.simd_eq(zero);

                let abs_x = self.abs();
                let is_denorm = abs_x.simd_lt(splat_f32::<T>(1.175_494_4e-38));

                let scaled = self * splat_f32::<T>(16_777_216.0);
                let x_for_cbrt = Self::blend(is_denorm, scaled, self);

                let result = x_for_cbrt.cbrt_midp();

                let scaled_result = result * splat_f32::<T>(1.0 / 256.0);
                let result = Self::blend(is_denorm, scaled_result, result);

                Self::blend(is_zero, self, result)
            }}
        }}
    "#}
}

/// Generate `generic/mod.rs` from the list of all SIMD types.
///
/// Produces the module declarations, transcendentals modules, and pub use
/// re-exports in the exact order matching the existing handwritten file.
pub(super) fn gen_mod_rs(all_types: &[SimdType]) -> String {
    let mut code = String::new();

    // File header
    code.push_str(&formatdoc! {r#"
        //! Generic SIMD types parameterized by backend token.
        //!
        //! These types are the strategy-pattern wrappers: `f32x8<T>` where `T`
        //! determines the platform implementation. Write one generic function,
        //! get monomorphized per backend at dispatch time.
        //!
        //! # Example
        //!
        //! ```ignore
        //! use magetypes::simd::backends::F32x8Backend;
        //! use magetypes::simd::generic::f32x8;
        //!
        //! fn dot<T: F32x8Backend>(token: T, a: &[f32; 8], b: &[f32; 8]) -> f32 {{
        //!     let va = f32x8::<T>::load(token, a);
        //!     let vb = f32x8::<T>::load(token, b);
        //!     (va * vb).reduce_add()
        //! }}
        //! ```

        #![allow(non_camel_case_types)]

    "#});

    // Collect W128+W256 and W512 types
    let w128_w256: Vec<&SimdType> = all_types
        .iter()
        .filter(|t| t.width != SimdWidth::W512)
        .collect();
    let w512: Vec<&SimdType> = all_types
        .iter()
        .filter(|t| t.width == SimdWidth::W512)
        .collect();

    // Block ops modules (hardcoded list, alphabetical)
    let block_ops_types = [
        "f32x4", "f32x8", "f64x2", "f64x4", "i32x4", "i32x8", "i8x16", "u32x4",
    ];
    for name in &block_ops_types {
        code.push_str(&format!("mod block_ops_{name};\n"));
    }

    // W128+W256 type mod declarations + transcendentals, all sorted alphabetically.
    // The transcendentals modules sort between i8x32_impl and u16x16_impl naturally.
    let mut w128_w256_names: Vec<String> = w128_w256.iter().map(|t| t.name()).collect();
    w128_w256_names.sort();

    let mut all_mods: Vec<String> = w128_w256_names
        .iter()
        .map(|name| format!("{name}_impl"))
        .collect();
    all_mods.push("transcendentals_f32x4".to_string());
    all_mods.push("transcendentals_f32x8".to_string());
    all_mods.sort();

    for mod_name in &all_mods {
        code.push_str(&format!("mod {mod_name};\n"));
    }

    // W128+W256 pub use declarations (specific non-alphabetical order:
    // floats by width, then signed ints by element size, then unsigned ints by element size)
    let pub_use_order = [
        "f32x4", "f32x8", "f64x2", "f64x4", "i8x16", "i8x32", "i16x8", "i16x16", "i32x4", "i32x8",
        "i64x2", "i64x4", "u8x16", "u8x32", "u16x8", "u16x16", "u32x4", "u32x8", "u64x2", "u64x4",
    ];
    code.push('\n');
    for name in &pub_use_order {
        if w128_w256_names.contains(&name.to_string()) {
            code.push_str(&format!("pub use {name}_impl::{name};\n"));
        }
    }

    // 512-bit section
    code.push_str("\n// 512-bit generic wrapper types\n");

    // W512 mod declarations (alphabetical, plus transcendentals)
    let mut w512_names: Vec<String> = w512.iter().map(|t| t.name()).collect();
    w512_names.sort();
    let mut w512_mods: Vec<String> = w512_names
        .iter()
        .map(|name| format!("{name}_impl"))
        .collect();
    w512_mods.push("transcendentals_f32x16".to_string());
    w512_mods.sort();
    for mod_name in &w512_mods {
        code.push_str(&format!("mod {mod_name};\n"));
    }

    // W512 pub use declarations (specific order:
    // floats, then signed ints by element size, then unsigned ints by element size)
    let w512_pub_use_order = [
        "f32x16", "f64x8", "i8x64", "i16x32", "i32x16", "i64x8", "u8x64", "u16x32", "u32x16",
        "u64x8",
    ];
    code.push('\n');
    for name in &w512_pub_use_order {
        if w512_names.contains(&name.to_string()) {
            code.push_str(&format!("pub use {name}_impl::{name};\n"));
        }
    }

    code
}
