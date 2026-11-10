//! Diagnostic comparison of std FMA with the independent rational fixtures.
//! Run explicitly: this investigates std, not Magetypes' implementation.
#![forbid(unsafe_code)]
#[path = "../tests/common/fma_vectors.rs"]
mod fixtures;

fn main() {
    let mut failures = 0;
    for (i, row) in fixtures::F32.iter().enumerate() {
        let [a, b, c, expected] = row.map(f32::from_bits);
        let actual =
            std::hint::black_box(a).mul_add(std::hint::black_box(b), std::hint::black_box(c));
        if !(actual.is_nan() && expected.is_nan()) && actual.to_bits() != expected.to_bits() {
            eprintln!(
                "f32 row {i}: input={row:08x?} actual={:08x} expected={:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
            failures += 1;
        }
    }
    for (i, row) in fixtures::F64.iter().enumerate() {
        let [a, b, c, expected] = row.map(f64::from_bits);
        let actual =
            std::hint::black_box(a).mul_add(std::hint::black_box(b), std::hint::black_box(c));
        if !(actual.is_nan() && expected.is_nan()) && actual.to_bits() != expected.to_bits() {
            eprintln!(
                "f64 row {i}: input={row:016x?} actual={:016x} expected={:016x}",
                actual.to_bits(),
                expected.to_bits()
            );
            failures += 1;
        }
    }
    assert_eq!(
        failures, 0,
        "std FMA mismatches; Magetypes has separate contract tests"
    );
}
