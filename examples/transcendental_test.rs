//! Quick test for transcendental functions (lowp variants)
//!
//! Run with: `cargo run --example transcendental_test --release`

// x86-only example - stub main for other platforms
#[cfg(not(target_arch = "x86_64"))]
fn main() {}

#[cfg(target_arch = "x86_64")]
mod x86_impl {
    use archmage::{SimdToken, X64V3Token, arcane};
    use magetypes::simd::f32x8;

    #[arcane]
    pub fn test_log2(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
        f32x8::load(token, input).log2_lowp().to_array()
    }

    #[arcane]
    pub fn test_exp2(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
        f32x8::load(token, input).exp2_lowp().to_array()
    }

    #[arcane]
    pub fn test_pow(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
        f32x8::load(token, input).pow_lowp(n).to_array()
    }

    #[arcane]
    pub fn test_ln(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
        f32x8::load(token, input).ln_lowp().to_array()
    }

    #[arcane]
    pub fn test_exp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
        f32x8::load(token, input).exp_lowp().to_array()
    }

    #[arcane]
    pub fn test_log10(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
        f32x8::load(token, input).log10_lowp().to_array()
    }

    #[arcane]
    pub fn test_cbrt_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
        f32x8::load(token, input).cbrt_midp().to_array()
    }

    pub fn main() {
        let Some(token) = X64V3Token::try_new() else {
            eprintln!("AVX2+FMA not available");
            return;
        };

        // Test log2_lowp
        let input = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
        let lg_arr = test_log2(token, &input);
        println!("log2_lowp([0.5, 1, 2, 4, 8, 16, 32, 64]) = {:?}", lg_arr);

        // Expected: [-1, 0, 1, 2, 3, 4, 5, 6]
        let expected = [-1.0f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for (i, (&got, &exp)) in lg_arr.iter().zip(expected.iter()).enumerate() {
            let err = (got - exp).abs();
            assert!(
                err < 0.001,
                "log2_lowp error at {}: got {}, expected {}, err {}",
                i,
                got,
                exp,
                err
            );
        }
        println!("log2_lowp: PASS");

        // Test exp2_lowp
        let input = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let e_arr = test_exp2(token, &input);
        println!("exp2_lowp([-2, -1, 0, 1, 2, 3, 4, 5]) = {:?}", e_arr);

        // Expected: [0.25, 0.5, 1, 2, 4, 8, 16, 32]
        let expected = [0.25f32, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        for (i, (&got, &exp)) in e_arr.iter().zip(expected.iter()).enumerate() {
            let err = (got - exp).abs() / exp.abs().max(0.001);
            assert!(
                err < 0.01,
                "exp2_lowp error at {}: got {}, expected {}, rel_err {}",
                i,
                got,
                exp,
                err
            );
        }
        println!("exp2_lowp: PASS");

        // Test pow_lowp(x, 2.4) - sRGB gamma
        let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let p_arr = test_pow(token, &input, 2.4);
        println!("pow_lowp([0.1..0.8], 2.4) = {:?}", p_arr);

        for (i, (&got, &inp)) in p_arr.iter().zip(input.iter()).enumerate() {
            let exp = inp.powf(2.4);
            let err = (got - exp).abs() / exp.abs().max(0.001);
            assert!(
                err < 0.01,
                "pow_lowp error at {}: got {}, expected {}, rel_err {}",
                i,
                got,
                exp,
                err
            );
        }
        println!("pow_lowp: PASS");

        // Test ln_lowp
        let input = [
            1.0,
            core::f32::consts::E,
            7.389056,
            20.0855,
            54.598,
            148.41,
            403.4,
            1096.6,
        ];
        let ln_arr = test_ln(token, &input);
        println!("ln_lowp([e^0, e^1, e^2, ...]) = {:?}", ln_arr);

        // Expected: approximately [0, 1, 2, 3, 4, 5, 6, 7]
        let expected = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        for (i, (&got, &exp)) in ln_arr.iter().zip(expected.iter()).enumerate() {
            let err = (got - exp).abs();
            assert!(
                err < 0.01,
                "ln_lowp error at {}: got {}, expected {}, err {}",
                i,
                got,
                exp,
                err
            );
        }
        println!("ln_lowp: PASS");

        // Test exp_lowp
        let input = [0.0, 1.0, 2.0, -1.0, -2.0, 0.5, -0.5, 3.0];
        let e_arr = test_exp(token, &input);
        println!("exp_lowp([0, 1, 2, -1, -2, 0.5, -0.5, 3]) = {:?}", e_arr);

        for (i, (&got, &inp)) in e_arr.iter().zip(input.iter()).enumerate() {
            let exp = inp.exp();
            let err = (got - exp).abs() / exp.abs().max(0.001);
            assert!(
                err < 0.01,
                "exp_lowp error at {}: got {}, expected {}, rel_err {}",
                i,
                got,
                exp,
                err
            );
        }
        println!("exp_lowp: PASS");

        // Test log10_lowp
        let input = [0.1, 1.0, 10.0, 100.0, 1000.0, 0.01, 0.001, 10000.0];
        let lg_arr = test_log10(token, &input);
        println!(
            "log10_lowp([0.1, 1, 10, 100, 1000, 0.01, 0.001, 10000]) = {:?}",
            lg_arr
        );

        // Expected: [-1, 0, 1, 2, 3, -2, -3, 4]
        let expected = [-1.0f32, 0.0, 1.0, 2.0, 3.0, -2.0, -3.0, 4.0];
        for (i, (&got, &exp)) in lg_arr.iter().zip(expected.iter()).enumerate() {
            let err = (got - exp).abs();
            assert!(
                err < 0.01,
                "log10_lowp error at {}: got {}, expected {}, err {}",
                i,
                got,
                exp,
                err
            );
        }
        println!("log10_lowp: PASS");

        // Test cbrt_midp
        let input = [1.0, 8.0, 27.0, 64.0, 125.0, 0.125, 0.001, 1000.0];
        let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 0.5, 0.1, 10.0];
        let cbrt_arr = test_cbrt_midp(token, &input);
        println!(
            "cbrt_midp([1, 8, 27, 64, 125, 0.125, 0.001, 1000]) = {:?}",
            cbrt_arr
        );

        for (i, (&got, &exp)) in cbrt_arr.iter().zip(expected.iter()).enumerate() {
            let err = (got - exp).abs() / exp.abs().max(0.001);
            assert!(
                err < 0.0001,
                "cbrt_midp error at {}: got {}, expected {}, rel_err {}",
                i,
                got,
                exp,
                err
            );
        }
        println!("cbrt_midp: PASS");

        // Test cbrt_midp with negative values
        let input = [-1.0, -8.0, -27.0, 1.0, 8.0, 27.0, -64.0, 64.0];
        let cbrt_arr = test_cbrt_midp(token, &input);
        println!(
            "cbrt_midp([-1, -8, -27, 1, 8, 27, -64, 64]) = {:?}",
            cbrt_arr
        );

        let expected = [-1.0f32, -2.0, -3.0, 1.0, 2.0, 3.0, -4.0, 4.0];
        for (i, (&got, &exp)) in cbrt_arr.iter().zip(expected.iter()).enumerate() {
            let err = (got - exp).abs() / exp.abs().max(0.001);
            assert!(
                err < 0.0001,
                "cbrt_midp negative error at {}: got {}, expected {}, rel_err {}",
                i,
                got,
                exp,
                err
            );
        }
        println!("cbrt_midp (negative): PASS");

        println!("\nAll transcendental tests PASSED!");
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    x86_impl::main();
}
