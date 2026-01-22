//! Quick test for transcendental functions

use archmage::SimdToken;
use archmage::simd::*;

fn main() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    // Test log2
    let v = f32x8::load(token, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]);
    let lg = v.log2();
    let lg_arr = lg.to_array();
    println!("log2([0.5, 1, 2, 4, 8, 16, 32, 64]) = {:?}", lg_arr);

    // Expected: [-1, 0, 1, 2, 3, 4, 5, 6]
    let expected = [-1.0f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for (i, (&got, &exp)) in lg_arr.iter().zip(expected.iter()).enumerate() {
        let err = (got - exp).abs();
        assert!(err < 0.001, "log2 error at {}: got {}, expected {}, err {}", i, got, exp, err);
    }
    println!("log2: PASS");

    // Test exp2
    let v = f32x8::load(token, &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let e = v.exp2();
    let e_arr = e.to_array();
    println!("exp2([-2, -1, 0, 1, 2, 3, 4, 5]) = {:?}", e_arr);

    // Expected: [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    let expected = [0.25f32, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    for (i, (&got, &exp)) in e_arr.iter().zip(expected.iter()).enumerate() {
        let err = (got - exp).abs() / exp.abs().max(0.001);
        assert!(err < 0.01, "exp2 error at {}: got {}, expected {}, rel_err {}", i, got, exp, err);
    }
    println!("exp2: PASS");

    // Test pow(x, 2.4) - sRGB gamma
    let v = f32x8::load(token, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let p = v.pow(2.4);
    let p_arr = p.to_array();
    let v_arr = v.to_array();
    println!("pow([0.1..0.8], 2.4) = {:?}", p_arr);

    for (i, (&got, &inp)) in p_arr.iter().zip(v_arr.iter()).enumerate() {
        let exp = inp.powf(2.4);
        let err = (got - exp).abs() / exp.abs().max(0.001);
        assert!(err < 0.01, "pow error at {}: got {}, expected {}, rel_err {}", i, got, exp, err);
    }
    println!("pow: PASS");

    // Test ln
    let v = f32x8::load(token, &[1.0, 2.718281828, 7.389056, 20.0855, 54.598, 148.41, 403.4, 1096.6]);
    let ln = v.ln();
    let ln_arr = ln.to_array();
    println!("ln([e^0, e^1, e^2, ...]) = {:?}", ln_arr);

    // Expected: approximately [0, 1, 2, 3, 4, 5, 6, 7]
    let expected = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    for (i, (&got, &exp)) in ln_arr.iter().zip(expected.iter()).enumerate() {
        let err = (got - exp).abs();
        assert!(err < 0.01, "ln error at {}: got {}, expected {}, err {}", i, got, exp, err);
    }
    println!("ln: PASS");

    // Test exp
    let v = f32x8::load(token, &[0.0, 1.0, 2.0, -1.0, -2.0, 0.5, -0.5, 3.0]);
    let e = v.exp();
    let e_arr = e.to_array();
    let v_arr = v.to_array();
    println!("exp([0, 1, 2, -1, -2, 0.5, -0.5, 3]) = {:?}", e_arr);

    for (i, (&got, &inp)) in e_arr.iter().zip(v_arr.iter()).enumerate() {
        let exp = inp.exp();
        let err = (got - exp).abs() / exp.abs().max(0.001);
        assert!(err < 0.01, "exp error at {}: got {}, expected {}, rel_err {}", i, got, exp, err);
    }
    println!("exp: PASS");

    // Test log10
    let v = f32x8::load(token, &[0.1, 1.0, 10.0, 100.0, 1000.0, 0.01, 0.001, 10000.0]);
    let lg = v.log10();
    let lg_arr = lg.to_array();
    println!("log10([0.1, 1, 10, 100, 1000, 0.01, 0.001, 10000]) = {:?}", lg_arr);

    // Expected: [-1, 0, 1, 2, 3, -2, -3, 4]
    let expected = [-1.0f32, 0.0, 1.0, 2.0, 3.0, -2.0, -3.0, 4.0];
    for (i, (&got, &exp)) in lg_arr.iter().zip(expected.iter()).enumerate() {
        let err = (got - exp).abs();
        assert!(err < 0.01, "log10 error at {}: got {}, expected {}, err {}", i, got, exp, err);
    }
    println!("log10: PASS");

    println!("\nAll transcendental tests PASSED!");
}
