//! Cross-architecture SIMD operation exercise.
//!
//! Runs all SIMD math operations on edge-case inputs and writes results
//! as JSON Lines to stdout. Used in CI to compare outputs across x86_64,
//! aarch64, and wasm32 architectures.
//!
//! Run with: cargo run --example arch_exercise --features "std avx512"
//!
//! Output format (one JSON object per line):
//! {"type":"f32x4","op":"round","input":[0.5,-0.5,1.5,-1.5],"output":[0.0,-0.0,2.0,-2.0]}

use archmage::SimdToken;
use magetypes::simd::generic;

// ============================================================================
// Edge-case inputs (same as scalar_parity.rs)
// ============================================================================

const F32_EDGE: [f32; 32] = [
    0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 0.0, -0.0, 1.0, -1.0,
    f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::EPSILON, f32::MIN, f32::MAX,
    f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 8388607.5, 8388608.5, -8388607.5,
    -8388608.5, 2147483520.0, -2147483520.0, 2147483648.0, -2147483648.0, 0.1, 0.9,
    100.0, -100.0,
];

const F32_EDGE_B: [f32; 32] = [
    1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25, 3.0, -3.0, 0.0, -0.0,
    f32::INFINITY, f32::NAN, 1.0, f32::NEG_INFINITY, f32::EPSILON, f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE, f32::EPSILON, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    42.0, -42.0, 0.001, -0.001,
];

const I32_EDGE: [i32; 32] = [
    0, 1, -1, 2, -2, 42, -42, 127, -128, 255, -256, 1000, -1000, i32::MAX, i32::MIN,
    i32::MAX - 1, i32::MIN + 1, 0x7F, 0xFF, 0x7FFF, 0xFFFF_u32 as i32, 0x7FFF_FFFF,
    -0x7FFF_FFFF, 100, -100, 1024, -1024, 0, 0, 0, 0, 0,
];

// ============================================================================
// Output helpers
// ============================================================================

fn fmt_f32_array(arr: &[f32]) -> String {
    let items: Vec<String> = arr
        .iter()
        .map(|x| {
            if x.is_nan() {
                "\"NaN\"".to_string()
            } else if x.is_infinite() {
                if *x > 0.0 {
                    "\"Inf\"".to_string()
                } else {
                    "\"-Inf\"".to_string()
                }
            } else {
                // Use hex bits for exact representation
                format!("\"0x{:08x}\"", x.to_bits())
            }
        })
        .collect();
    format!("[{}]", items.join(","))
}

fn fmt_i32_array(arr: &[i32]) -> String {
    let items: Vec<String> = arr.iter().map(|x| x.to_string()).collect();
    format!("[{}]", items.join(","))
}

fn emit(ty: &str, op: &str, input: &str, output: &str) {
    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "wasm32") {
        "wasm32"
    } else {
        "unknown"
    };
    println!(
        "{{\"arch\":\"{arch}\",\"type\":\"{ty}\",\"op\":\"{op}\",\"input\":{input},\"output\":{output}}}"
    );
}

// ============================================================================
// Exercise functions (generic over token type)
// ============================================================================

fn exercise_f32x4<T: magetypes::simd::backends::F32x4Backend + magetypes::simd::backends::F32x4Convert>(token: T) {
    for chunk in F32_EDGE.chunks_exact(4) {
        let input: [f32; 4] = chunk.try_into().unwrap();
        let input_s = fmt_f32_array(&input);
        let v = generic::f32x4::<T>::from_array(token, input);

        emit("f32x4", "round", &input_s, &fmt_f32_array(&v.round().to_array()));
        emit("f32x4", "floor", &input_s, &fmt_f32_array(&v.floor().to_array()));
        emit("f32x4", "ceil", &input_s, &fmt_f32_array(&v.ceil().to_array()));
        emit("f32x4", "sqrt", &input_s, &fmt_f32_array(&v.sqrt().to_array()));
        emit("f32x4", "abs", &input_s, &fmt_f32_array(&v.abs().to_array()));
        emit("f32x4", "neg", &input_s, &fmt_f32_array(&(-v).to_array()));
    }

    // Binary ops
    for (ca, cb) in F32_EDGE.chunks_exact(4).zip(F32_EDGE_B.chunks_exact(4)) {
        let a: [f32; 4] = ca.try_into().unwrap();
        let b: [f32; 4] = cb.try_into().unwrap();
        let input_s = fmt_f32_array(&a);
        let va = generic::f32x4::<T>::from_array(token, a);
        let vb = generic::f32x4::<T>::from_array(token, b);

        emit("f32x4", "add", &input_s, &fmt_f32_array(&(va + vb).to_array()));
        emit("f32x4", "sub", &input_s, &fmt_f32_array(&(va - vb).to_array()));
        emit("f32x4", "mul", &input_s, &fmt_f32_array(&(va * vb).to_array()));
        emit("f32x4", "div", &input_s, &fmt_f32_array(&(va / vb).to_array()));
        emit("f32x4", "min", &input_s, &fmt_f32_array(&va.min(vb).to_array()));
        emit("f32x4", "max", &input_s, &fmt_f32_array(&va.max(vb).to_array()));
    }

    // Conversions (safe values only)
    let safe: Vec<f32> = F32_EDGE
        .iter()
        .filter(|&&x| x.is_finite() && x.abs() < 2147483520.0)
        .copied()
        .collect();
    for chunk in safe.chunks_exact(4) {
        let input: [f32; 4] = chunk.try_into().unwrap();
        let input_s = fmt_f32_array(&input);
        let v = generic::f32x4::<T>::from_array(token, input);
        emit(
            "f32x4",
            "to_i32_round",
            &input_s,
            &fmt_i32_array(&v.to_i32_round().to_array()),
        );
        emit(
            "f32x4",
            "to_i32",
            &input_s,
            &fmt_i32_array(&v.to_i32().to_array()),
        );
    }
}

fn exercise_f32x8<T: magetypes::simd::backends::F32x8Backend + magetypes::simd::backends::F32x8Convert>(token: T) {
    for chunk in F32_EDGE.chunks_exact(8) {
        let input: [f32; 8] = chunk.try_into().unwrap();
        let input_s = fmt_f32_array(&input);
        let v = generic::f32x8::<T>::from_array(token, input);

        emit("f32x8", "round", &input_s, &fmt_f32_array(&v.round().to_array()));
        emit("f32x8", "floor", &input_s, &fmt_f32_array(&v.floor().to_array()));
        emit("f32x8", "ceil", &input_s, &fmt_f32_array(&v.ceil().to_array()));
        emit("f32x8", "abs", &input_s, &fmt_f32_array(&v.abs().to_array()));
        emit("f32x8", "neg", &input_s, &fmt_f32_array(&(-v).to_array()));
    }

    for (ca, cb) in F32_EDGE.chunks_exact(8).zip(F32_EDGE_B.chunks_exact(8)) {
        let a: [f32; 8] = ca.try_into().unwrap();
        let b: [f32; 8] = cb.try_into().unwrap();
        let input_s = fmt_f32_array(&a);
        let va = generic::f32x8::<T>::from_array(token, a);
        let vb = generic::f32x8::<T>::from_array(token, b);

        emit("f32x8", "add", &input_s, &fmt_f32_array(&(va + vb).to_array()));
        emit("f32x8", "sub", &input_s, &fmt_f32_array(&(va - vb).to_array()));
        emit("f32x8", "mul", &input_s, &fmt_f32_array(&(va * vb).to_array()));
        emit("f32x8", "div", &input_s, &fmt_f32_array(&(va / vb).to_array()));
        emit("f32x8", "min", &input_s, &fmt_f32_array(&va.min(vb).to_array()));
        emit("f32x8", "max", &input_s, &fmt_f32_array(&va.max(vb).to_array()));
    }

    let safe: Vec<f32> = F32_EDGE
        .iter()
        .filter(|&&x| x.is_finite() && x.abs() < 2147483520.0)
        .copied()
        .collect();
    for chunk in safe.chunks_exact(8) {
        let input: [f32; 8] = chunk.try_into().unwrap();
        let input_s = fmt_f32_array(&input);
        let v = generic::f32x8::<T>::from_array(token, input);
        emit(
            "f32x8",
            "to_i32_round",
            &input_s,
            &fmt_i32_array(&v.to_i32_round().to_array()),
        );
    }
}

fn exercise_i32x4<T: magetypes::simd::backends::I32x4Backend>(token: T) {
    for chunk in I32_EDGE.chunks_exact(4) {
        let input: [i32; 4] = chunk.try_into().unwrap();
        let input_s = fmt_i32_array(&input);
        let v = generic::i32x4::<T>::from_array(token, input);

        emit("i32x4", "abs", &input_s, &fmt_i32_array(&v.abs().to_array()));
        emit("i32x4", "neg", &input_s, &fmt_i32_array(&(-v).to_array()));
        emit("i32x4", "not", &input_s, &fmt_i32_array(&v.not().to_array()));
    }

    for (ca, cb) in I32_EDGE.chunks_exact(4).zip(I32_EDGE.chunks_exact(4).skip(1)) {
        let a: [i32; 4] = ca.try_into().unwrap();
        let b: [i32; 4] = cb.try_into().unwrap();
        let input_s = fmt_i32_array(&a);
        let va = generic::i32x4::<T>::from_array(token, a);
        let vb = generic::i32x4::<T>::from_array(token, b);

        emit("i32x4", "add", &input_s, &fmt_i32_array(&(va + vb).to_array()));
        emit("i32x4", "sub", &input_s, &fmt_i32_array(&(va - vb).to_array()));
        emit("i32x4", "min", &input_s, &fmt_i32_array(&va.min(vb).to_array()));
        emit("i32x4", "max", &input_s, &fmt_i32_array(&va.max(vb).to_array()));
    }
}

fn main() {
    // Dispatch to available hardware token, or fall back to scalar
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            exercise_f32x4(token);
            exercise_f32x8(token);
            exercise_i32x4(token);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let Some(token) = archmage::NeonToken::summon() {
            exercise_f32x4(token);
            exercise_i32x4(token);
            // f32x8 on ARM uses polyfill (2x NEON)
            exercise_f32x8(token);
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(token) = archmage::Wasm128Token::summon() {
            exercise_f32x4(token);
            exercise_i32x4(token);
            exercise_f32x8(token);
            return;
        }
    }

    // Fallback: scalar
    let token = archmage::ScalarToken;
    exercise_f32x4(token);
    exercise_f32x8(token);
    exercise_i32x4(token);
}
