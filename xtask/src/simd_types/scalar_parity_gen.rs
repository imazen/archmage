//! Scalar-vs-native backend parity test generator.
//!
//! Generates tests that compare ScalarToken backend output against the native
//! hardware backend (X64V3Token, NeonToken, Wasm128Token) for every SIMD math
//! operation on edge-case inputs. This catches bugs like the scalar rounding
//! divergence (issue #20) where scalar `round()` used ties-away-from-zero
//! while all hardware used ties-to-even.

use indoc::formatdoc;

/// Generate the scalar_parity.rs test file.
pub fn generate_scalar_parity() -> String {
    let mut code = String::with_capacity(64 * 1024);

    code.push_str(&file_header());
    code.push_str(&comparison_helpers());
    code.push_str(&edge_case_constants());
    code.push_str(&scalar_vs_native_macro());
    code.push_str(&cfg_gated_invocations());

    code
}

fn file_header() -> String {
    formatdoc! {r#"
        //! Scalar-vs-native backend parity tests.
        //!
        //! **Auto-generated** by `cargo xtask generate` — do not edit manually.
        //!
        //! For every SIMD math operation, this tests that the ScalarToken backend
        //! produces identical results to the native hardware backend on edge-case
        //! inputs. Catches divergences like ties-away-from-zero vs ties-to-even
        //! (issue #20).

        #![allow(unused_imports)]
        #![allow(clippy::approx_constant)]
        #![allow(clippy::excessive_precision)]
        #![allow(clippy::float_cmp)]

    "#}
}

fn comparison_helpers() -> String {
    formatdoc! {r#"
        // ============================================================================
        // Comparison helpers
        // ============================================================================

        /// Bit-exact f32 comparison, treating all NaN values as equal.
        fn assert_f32_exact(scalar: &[f32], native: &[f32], op: &str, input: &[f32]) {{
            assert_eq!(scalar.len(), native.len(), "{{op}}: length mismatch");
            for i in 0..scalar.len() {{
                let s = scalar[i];
                let n = native[i];
                if s.is_nan() && n.is_nan() {{ continue; }}
                if s.to_bits() == n.to_bits() {{ continue; }}
                panic!(
                    "{{op}} divergence at lane {{i}}: scalar={{s}} (0x{{:08x}}) native={{n}} (0x{{:08x}}) input={{input:?}}",
                    s.to_bits(), n.to_bits()
                );
            }}
        }}

        /// Bit-exact f64 comparison, treating all NaN values as equal.
        fn assert_f64_exact(scalar: &[f64], native: &[f64], op: &str, input: &[f64]) {{
            assert_eq!(scalar.len(), native.len(), "{{op}}: length mismatch");
            for i in 0..scalar.len() {{
                let s = scalar[i];
                let n = native[i];
                if s.is_nan() && n.is_nan() {{ continue; }}
                if s.to_bits() == n.to_bits() {{ continue; }}
                panic!(
                    "{{op}} divergence at lane {{i}}: scalar={{s}} (0x{{:016x}}) native={{n}} (0x{{:016x}}) input={{input:?}}",
                    s.to_bits(), n.to_bits()
                );
            }}
        }}

        /// f32 comparison with ULP tolerance, treating NaN==NaN and ±0 as equal.
        fn assert_f32_ulps(scalar: &[f32], native: &[f32], op: &str, input: &[f32], max_ulps: u32) {{
            assert_eq!(scalar.len(), native.len(), "{{op}}: length mismatch");
            for i in 0..scalar.len() {{
                let s = scalar[i];
                let n = native[i];
                if s.is_nan() && n.is_nan() {{ continue; }}
                if s.to_bits() == n.to_bits() {{ continue; }}
                // Allow ±0 difference (FMA vs mul+add can produce different zero signs)
                if s == 0.0 && n == 0.0 {{ continue; }}
                // Both must be finite and same sign for ULP comparison to make sense
                if s.is_nan() || n.is_nan() || s.is_infinite() || n.is_infinite() {{
                    panic!(
                        "{{op}} divergence at lane {{i}}: scalar={{s}} native={{n}} (NaN/Inf mismatch) input={{input:?}}"
                    );
                }}
                let ulps = (s.to_bits() as i64 - n.to_bits() as i64).unsigned_abs() as u32;
                if ulps > max_ulps {{
                    panic!(
                        "{{op}} divergence at lane {{i}}: scalar={{s}} native={{n}} ({{ulps}} ulps, max={{max_ulps}}) input={{input:?}}"
                    );
                }}
            }}
        }}

        /// f64 comparison with ULP tolerance, treating all NaN values as equal.
        fn assert_f64_ulps(scalar: &[f64], native: &[f64], op: &str, input: &[f64], max_ulps: u64) {{
            assert_eq!(scalar.len(), native.len(), "{{op}}: length mismatch");
            for i in 0..scalar.len() {{
                let s = scalar[i];
                let n = native[i];
                if s.is_nan() && n.is_nan() {{ continue; }}
                if s.to_bits() == n.to_bits() {{ continue; }}
                if s.is_nan() || n.is_nan() || s.is_infinite() || n.is_infinite() {{
                    panic!(
                        "{{op}} divergence at lane {{i}}: scalar={{s}} native={{n}} (NaN/Inf mismatch) input={{input:?}}"
                    );
                }}
                let ulps = (s.to_bits() as i128 - n.to_bits() as i128).unsigned_abs() as u64;
                if ulps > max_ulps {{
                    panic!(
                        "{{op}} divergence at lane {{i}}: scalar={{s}} native={{n}} ({{ulps}} ulps, max={{max_ulps}}) input={{input:?}}"
                    );
                }}
            }}
        }}

        /// f32 comparison with relative tolerance (for approximate operations).
        fn assert_f32_approx(scalar: &[f32], native: &[f32], op: &str, input: &[f32], rel_tol: f32) {{
            assert_eq!(scalar.len(), native.len(), "{{op}}: length mismatch");
            for i in 0..scalar.len() {{
                let s = scalar[i];
                let n = native[i];
                if s.is_nan() && n.is_nan() {{ continue; }}
                if s.to_bits() == n.to_bits() {{ continue; }}
                if s.is_nan() || n.is_nan() {{
                    panic!("{{op}} NaN mismatch at lane {{i}}: scalar={{s}} native={{n}} input={{input:?}}");
                }}
                if s.is_infinite() && n.is_infinite() && s.signum() == n.signum() {{ continue; }}
                let denom = s.abs().max(n.abs()).max(f32::MIN_POSITIVE);
                let rel_err = (s - n).abs() / denom;
                if rel_err > rel_tol {{
                    panic!(
                        "{{op}} divergence at lane {{i}}: scalar={{s}} native={{n}} (rel_err={{rel_err}}, max={{rel_tol}}) input={{input:?}}"
                    );
                }}
            }}
        }}

        /// f32 comparison that treats ±0 as equal (for ops where signed zero differs).
        fn assert_f32_signed_zero_tolerant(scalar: &[f32], native: &[f32], op: &str, input: &[f32]) {{
            assert_eq!(scalar.len(), native.len(), "{{op}}: length mismatch");
            for i in 0..scalar.len() {{
                let s = scalar[i];
                let n = native[i];
                if s.is_nan() && n.is_nan() {{ continue; }}
                if s.to_bits() == n.to_bits() {{ continue; }}
                // Allow ±0 difference
                if s == 0.0 && n == 0.0 {{ continue; }}
                panic!(
                    "{{op}} divergence at lane {{i}}: scalar={{s}} (0x{{:08x}}) native={{n}} (0x{{:08x}}) input={{input:?}}",
                    s.to_bits(), n.to_bits()
                );
            }}
        }}

        /// f64 comparison that treats ±0 as equal.
        fn assert_f64_signed_zero_tolerant(scalar: &[f64], native: &[f64], op: &str, input: &[f64]) {{
            assert_eq!(scalar.len(), native.len(), "{{op}}: length mismatch");
            for i in 0..scalar.len() {{
                let s = scalar[i];
                let n = native[i];
                if s.is_nan() && n.is_nan() {{ continue; }}
                if s.to_bits() == n.to_bits() {{ continue; }}
                if s == 0.0 && n == 0.0 {{ continue; }}
                panic!(
                    "{{op}} divergence at lane {{i}}: scalar={{s}} (0x{{:016x}}) native={{n}} (0x{{:016x}}) input={{input:?}}",
                    s.to_bits(), n.to_bits()
                );
            }}
        }}

        /// f32 FMA comparison: allows both relative and absolute tolerance.
        /// FMA (one rounding) vs separate mul+add (two roundings) can produce large relative
        /// errors near zero due to catastrophic cancellation, but the absolute error is small.
        fn assert_f32_fma(scalar: &[f32], native: &[f32], op: &str, input: &[f32]) {{
            assert_eq!(scalar.len(), native.len(), "{{op}}: length mismatch");
            for i in 0..scalar.len() {{
                let s = scalar[i];
                let n = native[i];
                if s.is_nan() && n.is_nan() {{ continue; }}
                if s.to_bits() == n.to_bits() {{ continue; }}
                if s == 0.0 && n == 0.0 {{ continue; }} // ±0
                if s.is_nan() || n.is_nan() {{
                    panic!("{{op}} NaN mismatch at lane {{i}}: scalar={{s}} native={{n}} input={{input:?}}");
                }}
                if s.is_infinite() && n.is_infinite() && s.signum() == n.signum() {{ continue; }}
                let abs_err = (s - n).abs();
                // Allow absolute error up to 1e-6 (handles near-zero cancellation)
                if abs_err < 1e-6 {{ continue; }}
                // Allow relative error up to 1e-4 for larger values
                let denom = s.abs().max(n.abs());
                let rel_err = abs_err / denom;
                if rel_err > 1e-4 {{
                    panic!(
                        "{{op}} divergence at lane {{i}}: scalar={{s}} native={{n}} (abs_err={{abs_err}}, rel_err={{rel_err}}) input={{input:?}}"
                    );
                }}
            }}
        }}

        /// f64 FMA comparison.
        fn assert_f64_fma(scalar: &[f64], native: &[f64], op: &str, input: &[f64]) {{
            assert_eq!(scalar.len(), native.len(), "{{op}}: length mismatch");
            for i in 0..scalar.len() {{
                let s = scalar[i];
                let n = native[i];
                if s.is_nan() && n.is_nan() {{ continue; }}
                if s.to_bits() == n.to_bits() {{ continue; }}
                if s == 0.0 && n == 0.0 {{ continue; }}
                if s.is_nan() || n.is_nan() {{
                    panic!("{{op}} NaN mismatch at lane {{i}}: scalar={{s}} native={{n}} input={{input:?}}");
                }}
                if s.is_infinite() && n.is_infinite() && s.signum() == n.signum() {{ continue; }}
                let abs_err = (s - n).abs();
                if abs_err < 1e-12 {{ continue; }}
                let denom = s.abs().max(n.abs());
                let rel_err = abs_err / denom;
                if rel_err > 1e-10 {{
                    panic!(
                        "{{op}} divergence at lane {{i}}: scalar={{s}} native={{n}} (abs_err={{abs_err}}, rel_err={{rel_err}}) input={{input:?}}"
                    );
                }}
            }}
        }}

        /// Check if a f32 slice contains NaN or infinity.
        fn has_nan_or_inf_f32(data: &[f32]) -> bool {{
            data.iter().any(|x| x.is_nan() || x.is_infinite())
        }}

        /// Check if a f64 slice contains NaN or infinity.
        fn has_nan_or_inf_f64(data: &[f64]) -> bool {{
            data.iter().any(|x| x.is_nan() || x.is_infinite())
        }}

        /// Bit-exact i32 comparison.
        fn assert_i32_exact(scalar: &[i32], native: &[i32], op: &str, input: &[i32]) {{
            assert_eq!(scalar, native, "{{op}} divergence: scalar={{scalar:?}} native={{native:?}} input={{input:?}}");
        }}

        /// Bit-exact u32 comparison.
        fn assert_u32_exact(scalar: &[u32], native: &[u32], op: &str, input: &[u32]) {{
            assert_eq!(scalar, native, "{{op}} divergence: scalar={{scalar:?}} native={{native:?}} input={{input:?}}");
        }}

        /// Bit-exact i16 comparison.
        fn assert_i16_exact(scalar: &[i16], native: &[i16], op: &str, input: &[i16]) {{
            assert_eq!(scalar, native, "{{op}} divergence: scalar={{scalar:?}} native={{native:?}} input={{input:?}}");
        }}

        /// Bit-exact u16 comparison.
        fn assert_u16_exact(scalar: &[u16], native: &[u16], op: &str, input: &[u16]) {{
            assert_eq!(scalar, native, "{{op}} divergence: scalar={{scalar:?}} native={{native:?}} input={{input:?}}");
        }}

        /// Bit-exact i8 comparison.
        fn assert_i8_exact(scalar: &[i8], native: &[i8], op: &str, input: &[i8]) {{
            assert_eq!(scalar, native, "{{op}} divergence: scalar={{scalar:?}} native={{native:?}} input={{input:?}}");
        }}

        /// Bit-exact u8 comparison.
        fn assert_u8_exact(scalar: &[u8], native: &[u8], op: &str, input: &[u8]) {{
            assert_eq!(scalar, native, "{{op}} divergence: scalar={{scalar:?}} native={{native:?}} input={{input:?}}");
        }}

    "#}
}

fn edge_case_constants() -> String {
    formatdoc! {r#"
        // ============================================================================
        // Edge-case input constants
        // ============================================================================

        // Designed to catch rounding, NaN propagation, sign, overflow, and boundary errors.
        // Length 32 is divisible by 4, 8, and 16 for clean chunking across all vector widths.

        const F32_EDGE_A: [f32; 32] = [
            // Rounding ties (the exact bug class from issue #20)
            0.5, -0.5, 1.5, -1.5,
            2.5, -2.5, 3.5, -3.5,
            // Zeros and signs
            0.0, -0.0, 1.0, -1.0,
            // Special values
            f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::EPSILON,
            // Extremes
            f32::MIN, f32::MAX, f32::MIN_POSITIVE, -f32::MIN_POSITIVE,
            // Near rounding boundary (2^23 = 8388608)
            8388607.5, 8388608.5, -8388607.5, -8388608.5,
            // Near i32 overflow boundary
            2147483520.0, -2147483520.0, 2147483648.0, -2147483648.0,
            // Miscellaneous
            0.1, 0.9, 100.0, -100.0,
        ];

        const F32_EDGE_B: [f32; 32] = [
            // Second operand for binary operations
            1.0, -1.0, 2.0, -2.0,
            0.5, -0.5, 0.25, -0.25,
            3.0, -3.0, 0.0, -0.0,
            f32::INFINITY, f32::NAN, 1.0, f32::NEG_INFINITY,
            f32::EPSILON, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, f32::EPSILON,
            1.0, -1.0, 1.0, -1.0,
            1.0, -1.0, 1.0, -1.0,
            42.0, -42.0, 0.001, -0.001,
        ];

        // Third operand for ternary ops (mul_add, mul_sub)
        const F32_EDGE_C: [f32; 32] = [
            0.5, -0.5, 0.5, -0.5,
            1.0, -1.0, 1.0, -1.0,
            0.0, -0.0, 0.0, -0.0,
            f32::NAN, 1.0, f32::INFINITY, f32::NEG_INFINITY,
            0.0, 0.0, 0.0, 0.0,
            100.0, -100.0, 100.0, -100.0,
            0.0, 0.0, 0.0, 0.0,
            0.1, -0.1, 0.1, -0.1,
        ];

        const F64_EDGE_A: [f64; 16] = [
            // Rounding ties
            0.5, -0.5, 1.5, -1.5,
            2.5, -2.5, 3.5, -3.5,
            // Special values
            0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY,
            f64::NAN, f64::MAX, f64::MIN, f64::MIN_POSITIVE,
        ];

        const F64_EDGE_B: [f64; 16] = [
            1.0, -1.0, 2.0, -2.0,
            0.5, -0.5, 0.25, -0.25,
            3.0, -3.0, f64::NAN, 1.0,
            f64::INFINITY, f64::EPSILON, -f64::MIN_POSITIVE, 42.0,
        ];

        const F64_EDGE_C: [f64; 16] = [
            0.5, -0.5, 0.5, -0.5,
            1.0, -1.0, 1.0, -1.0,
            0.0, -0.0, 1.0, f64::NAN,
            0.0, 100.0, -100.0, 0.1,
        ];

        const I32_EDGE_A: [i32; 32] = [
            0, 1, -1, 2,
            -2, 42, -42, 127,
            -128, 255, -256, 1000,
            -1000, i32::MAX, i32::MIN, i32::MAX - 1,
            i32::MIN + 1, 0x7F, 0xFF, 0x7FFF,
            0xFFFF_u32 as i32, 0x7FFF_FFFF, -0x7FFF_FFFF, 100,
            -100, 1024, -1024, 0,
            0, 0, 0, 0,
        ];

        const I32_EDGE_B: [i32; 32] = [
            1, -1, 2, -2,
            3, -3, 1, -1,
            42, -42, 0, 0,
            1, -1, 1, -1,
            1, -1, 0, 0,
            1, -1, 1, -1,
            1, 1, 1, 1,
            0, 0, 0, 0,
        ];

        const I16_EDGE_A: [i16; 32] = [
            0, 1, -1, 2,
            -2, 42, -42, 127,
            -128, 255, -256, 1000,
            -1000, i16::MAX, i16::MIN, i16::MAX - 1,
            i16::MIN + 1, 0x7F, -0x7F, 0x7FFF,
            -0x7FFF, 100, -100, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ];

        const I16_EDGE_B: [i16; 32] = [
            1, -1, 2, -2,
            3, -3, 1, -1,
            42, -42, 0, 0,
            1, -1, 1, -1,
            1, -1, 0, 0,
            1, -1, 1, -1,
            1, 1, 1, 1,
            0, 0, 0, 0,
        ];

    "#}
}

/// Generates the macro that creates test modules for a given native token type.
fn scalar_vs_native_macro() -> String {
    let mut code = String::with_capacity(32 * 1024);

    code.push_str(
        "// ============================================================================\n",
    );
    code.push_str("// Test suite macro\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    code.push_str("macro_rules! scalar_vs_native {\n");
    code.push_str("    ($native_token:ty) => {\n");
    code.push_str(
        "        // Re-export file-level items so nested modules can reach them via super::\n",
    );
    code.push_str("        use super::*;\n");
    code.push_str("        use archmage::{ScalarToken, SimdToken};\n");
    code.push_str("        use magetypes::simd::generic;\n\n");

    // Generate f32x4 tests
    gen_float_type_tests(&mut code, "f32", 4);
    gen_float_type_tests(&mut code, "f32", 8);
    gen_float_type_tests(&mut code, "f64", 2);
    gen_float_type_tests(&mut code, "f64", 4);

    // Generate int type tests
    gen_int_type_tests(&mut code, "i32", 4, true);
    gen_int_type_tests(&mut code, "i32", 8, true);
    gen_int_type_tests(&mut code, "i16", 8, true);
    gen_int_type_tests(&mut code, "i16", 16, true);
    gen_int_type_tests(&mut code, "u32", 4, false);
    gen_int_type_tests(&mut code, "u32", 8, false);

    // Generate f32↔i32 conversion tests
    gen_conversion_tests(&mut code, 4);
    gen_conversion_tests(&mut code, 8);

    code.push_str("    };\n"); // close macro arm
    code.push_str("}\n\n");

    code
}

fn gen_float_type_tests(code: &mut String, elem: &str, lanes: usize) {
    let type_name = format!("{elem}x{lanes}");
    let edge_a = if elem == "f32" {
        "F32_EDGE_A"
    } else {
        "F64_EDGE_A"
    };
    let edge_b = if elem == "f32" {
        "F32_EDGE_B"
    } else {
        "F64_EDGE_B"
    };
    let edge_c = if elem == "f32" {
        "F32_EDGE_C"
    } else {
        "F64_EDGE_C"
    };
    let assert_exact = if elem == "f32" {
        "assert_f32_exact"
    } else {
        "assert_f64_exact"
    };
    let assert_ulps = if elem == "f32" {
        "assert_f32_ulps"
    } else {
        "assert_f64_ulps"
    };
    let assert_approx = if elem == "f32" {
        "assert_f32_approx"
    } else {
        "assert_f32_approx"
    }; // f64 approx reuses f32 pattern logic
    let ulp_1 = if elem == "f32" { "1u32" } else { "1u64" };
    let fma_assert = if elem == "f32" {
        "assert_f32_fma"
    } else {
        "assert_f64_fma"
    };

    code.push_str(&formatdoc! {r#"
        mod {type_name}_parity {{
            use super::*;

            fn run_unary(
                op: &str,
                f: impl Fn(generic::{type_name}<ScalarToken>, generic::{type_name}<$native_token>) -> (
                    [{elem}; {lanes}], [{elem}; {lanes}]
                ),
                cmp: impl Fn(&[{elem}], &[{elem}], &str, &[{elem}]),
            ) {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    for chunk in super::{edge_a}.chunks_exact({lanes}) {{
                        let input: [{elem}; {lanes}] = chunk.try_into().unwrap();
                        let vs = generic::{type_name}::<ScalarToken>::from_array(token_s, input);
                        let vn = generic::{type_name}::<$native_token>::from_array(token_n, input);
                        let (s, n) = f(vs, vn);
                        cmp(&s, &n, op, &input);
                    }}
                }}
            }}

            /// Like run_unary but skips chunks containing NaN or infinity.
            fn run_unary_finite(
                op: &str,
                f: impl Fn(generic::{type_name}<ScalarToken>, generic::{type_name}<$native_token>) -> (
                    [{elem}; {lanes}], [{elem}; {lanes}]
                ),
                cmp: impl Fn(&[{elem}], &[{elem}], &str, &[{elem}]),
            ) {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    for chunk in super::{edge_a}.chunks_exact({lanes}) {{
                        let input: [{elem}; {lanes}] = chunk.try_into().unwrap();
                        if super::has_nan_or_inf_{elem}(&input) {{ continue; }}
                        let vs = generic::{type_name}::<ScalarToken>::from_array(token_s, input);
                        let vn = generic::{type_name}::<$native_token>::from_array(token_n, input);
                        let (s, n) = f(vs, vn);
                        cmp(&s, &n, op, &input);
                    }}
                }}
            }}

            fn run_binary(
                op: &str,
                f: impl Fn(
                    generic::{type_name}<ScalarToken>, generic::{type_name}<ScalarToken>,
                    generic::{type_name}<$native_token>, generic::{type_name}<$native_token>,
                ) -> ([{elem}; {lanes}], [{elem}; {lanes}]),
                cmp: impl Fn(&[{elem}], &[{elem}], &str, &[{elem}]),
            ) {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    for (ca, cb) in super::{edge_a}.chunks_exact({lanes}).zip(super::{edge_b}.chunks_exact({lanes})) {{
                        let a: [{elem}; {lanes}] = ca.try_into().unwrap();
                        let b: [{elem}; {lanes}] = cb.try_into().unwrap();
                        let as_ = generic::{type_name}::<ScalarToken>::from_array(token_s, a);
                        let bs = generic::{type_name}::<ScalarToken>::from_array(token_s, b);
                        let an = generic::{type_name}::<$native_token>::from_array(token_n, a);
                        let bn = generic::{type_name}::<$native_token>::from_array(token_n, b);
                        let (s, n) = f(as_, bs, an, bn);
                        cmp(&s, &n, op, &a);
                    }}
                }}
            }}

            /// Like run_binary but skips chunks where either input contains NaN or infinity.
            fn run_binary_finite(
                op: &str,
                f: impl Fn(
                    generic::{type_name}<ScalarToken>, generic::{type_name}<ScalarToken>,
                    generic::{type_name}<$native_token>, generic::{type_name}<$native_token>,
                ) -> ([{elem}; {lanes}], [{elem}; {lanes}]),
                cmp: impl Fn(&[{elem}], &[{elem}], &str, &[{elem}]),
            ) {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    for (ca, cb) in super::{edge_a}.chunks_exact({lanes}).zip(super::{edge_b}.chunks_exact({lanes})) {{
                        let a: [{elem}; {lanes}] = ca.try_into().unwrap();
                        let b: [{elem}; {lanes}] = cb.try_into().unwrap();
                        if super::has_nan_or_inf_{elem}(&a) || super::has_nan_or_inf_{elem}(&b) {{ continue; }}
                        let as_ = generic::{type_name}::<ScalarToken>::from_array(token_s, a);
                        let bs = generic::{type_name}::<ScalarToken>::from_array(token_s, b);
                        let an = generic::{type_name}::<$native_token>::from_array(token_n, a);
                        let bn = generic::{type_name}::<$native_token>::from_array(token_n, b);
                        let (s, n) = f(as_, bs, an, bn);
                        cmp(&s, &n, op, &a);
                    }}
                }}
            }}

    "#});

    // Unary exact ops
    for op in &["round", "floor", "ceil", "abs", "sqrt", "not"] {
        let call = match *op {
            "neg" => "(-v).to_array()".to_string(),
            "not" => "v.not().to_array()".to_string(),
            _ => format!("v.{op}().to_array()"),
        };
        code.push_str(&formatdoc! {r#"
            #[test]
            fn {op}() {{
                run_unary("{type_name}::{op}", |vs, vn| {{
                    ({call_s}, {call_n})
                }}, super::{assert_exact});
            }}

        "#, call_s = call.replace("v.", "vs.").replace("(-v)", "(-vs)"),
            call_n = call.replace("v.", "vn.").replace("(-v)", "(-vn)")
        });
    }

    // neg — signed-zero tolerant (x86 backend uses sub(0,x) which loses -0)
    {
        let assert_sz = if elem == "f32" {
            "assert_f32_signed_zero_tolerant"
        } else {
            "assert_f64_signed_zero_tolerant"
        };
        code.push_str(&formatdoc! {r#"
            #[test]
            fn neg() {{
                run_unary("{type_name}::neg", |vs, vn| {{
                    ((-vs).to_array(), (-vn).to_array())
                }}, super::{assert_sz});
            }}

        "#});
    }

    // Binary exact ops (NaN-safe — these propagate NaN consistently)
    for op in &["add", "sub", "mul", "div", "bitand", "bitor", "bitxor"] {
        let call = match *op {
            "add" => "(a + b).to_array()".to_string(),
            "sub" => "(a - b).to_array()".to_string(),
            "mul" => "(a * b).to_array()".to_string(),
            "div" => "(a / b).to_array()".to_string(),
            "bitand" => "(a & b).to_array()".to_string(),
            "bitor" => "(a | b).to_array()".to_string(),
            "bitxor" => "(a ^ b).to_array()".to_string(),
            _ => format!("a.{op}(b).to_array()"),
        };
        code.push_str(&formatdoc! {r#"
            #[test]
            fn {op}() {{
                run_binary("{type_name}::{op}", |as_, bs, an, bn| {{
                    ({call_s}, {call_n})
                }}, super::{assert_exact});
            }}

        "#, call_s = call.replace("(a ", "(as_ ").replace("a.", "as_.").replace(" b)", " bs)").replace("b)", "bs)"),
            call_n = call.replace("(a ", "(an ").replace("a.", "an.").replace(" b)", " bn)").replace("b)", "bn)")
        });
    }

    // min, max — skip NaN inputs (NaN propagation differs: SSE min/max returns
    // second operand when first is NaN, scalar f32::min/max returns non-NaN)
    for op in &["min", "max"] {
        code.push_str(&formatdoc! {r#"
            #[test]
            fn {op}() {{
                run_binary_finite("{type_name}::{op}", |as_, bs, an, bn| {{
                    (as_.{op}(bs).to_array(), an.{op}(bn).to_array())
                }}, super::{assert_exact});
            }}

        "#});
    }

    // mul_add, mul_sub — relative tolerance (scalar does a*b+c with two roundings,
    // hardware FMA does one fused rounding, which can differ by many ULPs)
    for op in &["mul_add", "mul_sub"] {
        code.push_str(&formatdoc! {r#"
            #[test]
            fn {op}() {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    for ((ca, cb), cc) in super::{edge_a}.chunks_exact({lanes})
                        .zip(super::{edge_b}.chunks_exact({lanes}))
                        .zip(super::{edge_c}.chunks_exact({lanes}))
                    {{
                        let a: [{elem}; {lanes}] = ca.try_into().unwrap();
                        let b: [{elem}; {lanes}] = cb.try_into().unwrap();
                        let c: [{elem}; {lanes}] = cc.try_into().unwrap();
                        // Skip chunks with NaN/Inf/extreme values
                        if [a.as_slice(), b.as_slice(), c.as_slice()].iter()
                            .any(|arr| arr.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e30))
                        {{ continue; }}
                        let as_ = generic::{type_name}::<ScalarToken>::from_array(token_s, a);
                        let bs = generic::{type_name}::<ScalarToken>::from_array(token_s, b);
                        let cs = generic::{type_name}::<ScalarToken>::from_array(token_s, c);
                        let an = generic::{type_name}::<$native_token>::from_array(token_n, a);
                        let bn = generic::{type_name}::<$native_token>::from_array(token_n, b);
                        let cn = generic::{type_name}::<$native_token>::from_array(token_n, c);
                        let s = as_.{op}(bs, cs).to_array();
                        let n = an.{op}(bn, cn).to_array();
                        // FMA vs mul+add can differ by many ULPs with catastrophic cancellation.
                        // Just check that NaN/Inf agreement is maintained and finite values
                        // are in the same ballpark (1e-4 relative tolerance).
                        // FMA vs mul+add can differ significantly with catastrophic cancellation.
                        // Use signed-zero-tolerant comparison (allows ±0 and NaN agreement).
                        super::{fma_assert}(&s, &n, "{type_name}::{op}", &a);
                    }}
                }}
            }}

        "#});
    }

    // Reductions — reduce_add gets tolerance, reduce_min/max are exact
    code.push_str(&formatdoc! {r#"
            #[test]
            fn reduce_add() {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    for chunk in super::{edge_a}.chunks_exact({lanes}) {{
                        let input: [{elem}; {lanes}] = chunk.try_into().unwrap();
                        // Skip chunks with NaN/Inf or extreme magnitudes (catastrophic cancellation
                        // from different FP associativity between tree and left-fold reduction)
                        if input.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e9) {{ continue; }}
                        let s = generic::{type_name}::<ScalarToken>::from_array(token_s, input).reduce_add();
                        let n = generic::{type_name}::<$native_token>::from_array(token_n, input).reduce_add();
                        if s.is_nan() && n.is_nan() {{ continue; }}
                        if s.to_bits() != n.to_bits() {{
                            // Allow relative tolerance for FP associativity
                            let denom = s.abs().max(n.abs()).max({elem}::MIN_POSITIVE);
                            let rel_err = (s - n).abs() / denom;
                            assert!(rel_err < 1e-6,
                                "{type_name}::reduce_add divergence: scalar={{s}} native={{n}} (rel_err={{rel_err}}) input={{input:?}}");
                        }}
                    }}
                }}
            }}

            #[test]
            fn reduce_min() {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    for chunk in super::{edge_a}.chunks_exact({lanes}) {{
                        let input: [{elem}; {lanes}] = chunk.try_into().unwrap();
                        if super::has_nan_or_inf_{elem}(&input) {{ continue; }}
                        let s = generic::{type_name}::<ScalarToken>::from_array(token_s, input).reduce_min();
                        let n = generic::{type_name}::<$native_token>::from_array(token_n, input).reduce_min();
                        // Allow ±0 difference (hardware min may return different zero sign)
                        if s == 0.0 && n == 0.0 {{ continue; }}
                        assert_eq!(s.to_bits(), n.to_bits(),
                            "{type_name}::reduce_min divergence: scalar={{s}} native={{n}} input={{input:?}}");
                    }}
                }}
            }}

            #[test]
            fn reduce_max() {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    for chunk in super::{edge_a}.chunks_exact({lanes}) {{
                        let input: [{elem}; {lanes}] = chunk.try_into().unwrap();
                        if super::has_nan_or_inf_{elem}(&input) {{ continue; }}
                        let s = generic::{type_name}::<ScalarToken>::from_array(token_s, input).reduce_max();
                        let n = generic::{type_name}::<$native_token>::from_array(token_n, input).reduce_max();
                        if s == 0.0 && n == 0.0 {{ continue; }}
                        assert_eq!(s.to_bits(), n.to_bits(),
                            "{type_name}::reduce_max divergence: scalar={{s}} native={{n}} input={{input:?}}");
                    }}
                }}
            }}

    "#});

    // Comparisons — NaN-safe ops use all inputs, NaN-problematic ops use finite only.
    // simd_eq, simd_lt, simd_le, simd_gt, simd_ge: these are "ordered" comparisons,
    // both scalar and hardware return false (all-zeros) for NaN operands.
    for op in &["simd_eq", "simd_lt", "simd_le", "simd_gt", "simd_ge"] {
        code.push_str(&formatdoc! {r#"
            #[test]
            fn {op}() {{
                run_binary("{type_name}::{op}", |as_, bs, an, bn| {{
                    (as_.{op}(bs).to_array(), an.{op}(bn).to_array())
                }}, super::{assert_exact});
            }}

        "#});
    }

    // simd_ne — skip NaN inputs (ordered vs unordered comparison semantics differ
    // between scalar != operator and hardware cmpneqps)
    code.push_str(&formatdoc! {r#"
            #[test]
            fn simd_ne() {{
                run_binary_finite("{type_name}::simd_ne", |as_, bs, an, bn| {{
                    (as_.simd_ne(bs).to_array(), an.simd_ne(bn).to_array())
                }}, super::{assert_exact});
            }}

    "#});

    // Approximation ops — relative tolerance.
    //
    // `_approx` variants: tolerance matches the tightest hardware spec we
    // honor. ARMv8 specifies `vrecpeq_f32` / `vrsqrteq_f32` with max
    // relative error ≤ 2⁻⁸ ≈ 3.91e-3 (x86 `_mm_rcp_ps` is tighter at
    // 1.5 × 2⁻¹² ≈ 3.66e-4, and real ARM silicon typically delivers
    // closer to that too). QEMU-user emulates NEON at the spec worst
    // case, so 4e-3 is the correct universal bound. Anything tighter
    // just encodes x86 precision as the contract.
    //
    // `recip` / `rsqrt`: Newton-refined. On x86 one Newton step over a
    // 12-bit estimate reaches full f32 precision (~24-bit). On NEON
    // we do two Newton steps over the 8-bit estimate to hit the same
    // precision, so 1e-5 holds on all platforms.
    if elem == "f32" {
        for (op, tol) in &[
            ("rcp_approx", "4e-3"),
            ("rsqrt_approx", "4e-3"),
            ("recip", "1e-5"),
            ("rsqrt", "1e-5"),
        ] {
            code.push_str(&formatdoc! {r#"
            #[test]
            fn {op}() {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    // Use only positive, non-special values for reciprocal/rsqrt
                    let safe_inputs: Vec<{elem}> = super::{edge_a}.iter()
                        .filter(|&&x| x.is_finite() && x > 0.01 && x < 1e30)
                        .copied()
                        .collect();
                    for chunk in safe_inputs.chunks_exact({lanes}) {{
                        let input: [{elem}; {lanes}] = chunk.try_into().unwrap();
                        let s = generic::{type_name}::<ScalarToken>::from_array(token_s, input).{op}().to_array();
                        let n = generic::{type_name}::<$native_token>::from_array(token_n, input).{op}().to_array();
                        super::assert_f32_approx(&s, &n, "{type_name}::{op}", &input, {tol});
                    }}
                }}
            }}

            "#});
        }
    }

    code.push_str("        }\n\n"); // close module
}

fn gen_int_type_tests(code: &mut String, elem: &str, lanes: usize, signed: bool) {
    let type_name = format!("{elem}x{lanes}");
    let (edge_a, edge_b) = match elem {
        "i32" | "u32" => ("I32_EDGE_A", "I32_EDGE_B"),
        "i16" | "u16" => ("I16_EDGE_A", "I16_EDGE_B"),
        _ => ("I32_EDGE_A", "I32_EDGE_B"), // fallback
    };

    // For unsigned types, we cast from the signed edge arrays
    let cast_a = if elem.starts_with('u') {
        format!("super::{edge_a}.map(|x| x as {elem})")
    } else {
        format!("super::{edge_a}")
    };
    let cast_b = if elem.starts_with('u') {
        format!("super::{edge_b}.map(|x| x as {elem})")
    } else {
        format!("super::{edge_b}")
    };

    let assert_fn = match elem {
        "i32" => "assert_i32_exact",
        "u32" => "assert_u32_exact",
        "i16" => "assert_i16_exact",
        "u16" => "assert_u16_exact",
        "i8" => "assert_i8_exact",
        "u8" => "assert_u8_exact",
        _ => "assert_i32_exact",
    };

    code.push_str(&formatdoc! {r#"
        mod {type_name}_parity {{
            use super::*;

    "#});

    // add, sub
    for op in &["add", "sub"] {
        let expr = match *op {
            "add" => "(a + b).to_array()",
            "sub" => "(a - b).to_array()",
            _ => unreachable!(),
        };
        code.push_str(&formatdoc! {r#"
            #[test]
            fn {op}() {{
                let token_s = ScalarToken;
                let edge_a = {cast_a};
                let edge_b = {cast_b};
                if let Some(token_n) = <$native_token>::summon() {{
                    for (ca, cb) in edge_a.chunks_exact({lanes}).zip(edge_b.chunks_exact({lanes})) {{
                        let a: [{elem}; {lanes}] = ca.try_into().unwrap();
                        let b: [{elem}; {lanes}] = cb.try_into().unwrap();
                        let as_ = generic::{type_name}::<ScalarToken>::from_array(token_s, a);
                        let bs = generic::{type_name}::<ScalarToken>::from_array(token_s, b);
                        let an = generic::{type_name}::<$native_token>::from_array(token_n, a);
                        let bn = generic::{type_name}::<$native_token>::from_array(token_n, b);
                        let s = {expr_s};
                        let n = {expr_n};
                        super::{assert_fn}(&s, &n, "{type_name}::{op}", &a);
                    }}
                }}
            }}

        "#, expr_s = expr.replace("(a ", "(as_ ").replace(" b)", " bs)"),
            expr_n = expr.replace("(a ", "(an ").replace(" b)", " bn)")
        });
    }

    // min, max
    for op in &["min", "max"] {
        code.push_str(&formatdoc! {r#"
            #[test]
            fn {op}() {{
                let token_s = ScalarToken;
                let edge_a = {cast_a};
                let edge_b = {cast_b};
                if let Some(token_n) = <$native_token>::summon() {{
                    for (ca, cb) in edge_a.chunks_exact({lanes}).zip(edge_b.chunks_exact({lanes})) {{
                        let a: [{elem}; {lanes}] = ca.try_into().unwrap();
                        let b: [{elem}; {lanes}] = cb.try_into().unwrap();
                        let s = generic::{type_name}::<ScalarToken>::from_array(token_s, a).{op}(
                            generic::{type_name}::<ScalarToken>::from_array(token_s, b)
                        ).to_array();
                        let n = generic::{type_name}::<$native_token>::from_array(token_n, a).{op}(
                            generic::{type_name}::<$native_token>::from_array(token_n, b)
                        ).to_array();
                        super::{assert_fn}(&s, &n, "{type_name}::{op}", &a);
                    }}
                }}
            }}

        "#});
    }

    // abs, neg (signed only)
    if signed {
        for op in &["abs", "neg"] {
            let call = match *op {
                "neg" => "(-v).to_array()",
                _ => "v.abs().to_array()",
            };
            code.push_str(&formatdoc! {r#"
            #[test]
            fn {op}() {{
                let token_s = ScalarToken;
                let edge_a = {cast_a};
                if let Some(token_n) = <$native_token>::summon() {{
                    for chunk in edge_a.chunks_exact({lanes}) {{
                        let input: [{elem}; {lanes}] = chunk.try_into().unwrap();
                        let s = {call_s};
                        let n = {call_n};
                        super::{assert_fn}(&s, &n, "{type_name}::{op}", &input);
                    }}
                }}
            }}

            "#,
                call_s = call.replace("v.", &format!("generic::{type_name}::<ScalarToken>::from_array(token_s, input).")).replace("(-v)", &format!("(-generic::{type_name}::<ScalarToken>::from_array(token_s, input))")),
                call_n = call.replace("v.", &format!("generic::{type_name}::<$native_token>::from_array(token_n, input).")).replace("(-v)", &format!("(-generic::{type_name}::<$native_token>::from_array(token_n, input))"))
            });
        }
    }

    // bitwise: not, and, or, xor
    code.push_str(&formatdoc! {r#"
            #[test]
            fn not() {{
                let token_s = ScalarToken;
                let edge_a = {cast_a};
                if let Some(token_n) = <$native_token>::summon() {{
                    for chunk in edge_a.chunks_exact({lanes}) {{
                        let input: [{elem}; {lanes}] = chunk.try_into().unwrap();
                        let s = generic::{type_name}::<ScalarToken>::from_array(token_s, input).not().to_array();
                        let n = generic::{type_name}::<$native_token>::from_array(token_n, input).not().to_array();
                        super::{assert_fn}(&s, &n, "{type_name}::not", &input);
                    }}
                }}
            }}

    "#});

    code.push_str("        }\n\n"); // close module
}

fn gen_conversion_tests(code: &mut String, lanes: usize) {
    let f_type = format!("f32x{lanes}");
    let i_type = format!("i32x{lanes}");

    code.push_str(&formatdoc! {r#"
        mod convert_{f_type}_parity {{
            use super::*;

            /// The exact bug from issue #20: to_i32_round must use ties-to-even.
            #[test]
            fn to_i32_round() {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    // Focus on values that are safe to convert to i32
                    let safe_f32: Vec<f32> = super::F32_EDGE_A.iter()
                        .filter(|&&x| x.is_finite() && x.abs() < 2147483520.0)
                        .copied()
                        .collect();
                    for chunk in safe_f32.chunks_exact({lanes}) {{
                        let input: [f32; {lanes}] = chunk.try_into().unwrap();
                        let s = generic::{f_type}::<ScalarToken>::from_array(token_s, input)
                            .to_i32_round().to_array();
                        let n = generic::{f_type}::<$native_token>::from_array(token_n, input)
                            .to_i32_round().to_array();
                        super::assert_i32_exact(&s, &n, "{f_type}::to_i32_round", &input.map(|x| x as i32));
                    }}
                }}
            }}

            #[test]
            fn to_i32() {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    let safe_f32: Vec<f32> = super::F32_EDGE_A.iter()
                        .filter(|&&x| x.is_finite() && x.abs() < 2147483520.0)
                        .copied()
                        .collect();
                    for chunk in safe_f32.chunks_exact({lanes}) {{
                        let input: [f32; {lanes}] = chunk.try_into().unwrap();
                        let s = generic::{f_type}::<ScalarToken>::from_array(token_s, input)
                            .to_i32().to_array();
                        let n = generic::{f_type}::<$native_token>::from_array(token_n, input)
                            .to_i32().to_array();
                        super::assert_i32_exact(&s, &n, "{f_type}::to_i32", &input.map(|x| x as i32));
                    }}
                }}
            }}

            #[test]
            fn from_i32_roundtrip() {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    for chunk in super::I32_EDGE_A.chunks_exact({lanes}) {{
                        let input: [i32; {lanes}] = chunk.try_into().unwrap();
                        let s = generic::{f_type}::<ScalarToken>::from_i32(
                            token_s,
                            generic::{i_type}::<ScalarToken>::from_array(token_s, input)
                        ).to_array();
                        let n = generic::{f_type}::<$native_token>::from_i32(
                            token_n,
                            generic::{i_type}::<$native_token>::from_array(token_n, input)
                        ).to_array();
                        super::assert_f32_exact(&s, &n, "{f_type}::from_i32", &s);
                    }}
                }}
            }}

            #[test]
            fn bitcast_roundtrip() {{
                let token_s = ScalarToken;
                if let Some(token_n) = <$native_token>::summon() {{
                    for chunk in super::F32_EDGE_A.chunks_exact({lanes}) {{
                        let input: [f32; {lanes}] = chunk.try_into().unwrap();
                        // f32 → i32 bitcast → f32 bitcast should be identity
                        let s = generic::{f_type}::<ScalarToken>::from_i32_bitcast(
                            token_s,
                            generic::{f_type}::<ScalarToken>::from_array(token_s, input).bitcast_to_i32()
                        ).to_array();
                        let n = generic::{f_type}::<$native_token>::from_i32_bitcast(
                            token_n,
                            generic::{f_type}::<$native_token>::from_array(token_n, input).bitcast_to_i32()
                        ).to_array();
                        // Compare bit patterns (NaN payload must survive roundtrip)
                        for i in 0..{lanes} {{
                            assert_eq!(s[i].to_bits(), n[i].to_bits(),
                                "{f_type}::bitcast_roundtrip divergence at lane {{i}}: scalar={{}} native={{}}",
                                s[i], n[i]);
                        }}
                    }}
                }}
            }}
        }}

    "#});
}

fn cfg_gated_invocations() -> String {
    formatdoc! {r#"
        // ============================================================================
        // Architecture-specific invocations
        // ============================================================================

        #[cfg(target_arch = "x86_64")]
        mod x86 {{
            scalar_vs_native!(archmage::X64V3Token);
        }}

        #[cfg(target_arch = "aarch64")]
        mod arm {{
            scalar_vs_native!(archmage::NeonToken);
        }}

        #[cfg(target_arch = "wasm32")]
        mod wasm {{
            scalar_vs_native!(archmage::Wasm128Token);
        }}
    "#}
}
