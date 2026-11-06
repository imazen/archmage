//! Check externally generated MPFR records. Missing inputs are a hard error.
#![forbid(unsafe_code)]
use zenmath::scalar;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: core_math_check FUNCTION RECORDS");
    let function: fn(f32) -> f32 = match args[1].as_str() {
        "exp2f" => scalar::exp2f,
        "expf" => scalar::expf,
        "log2f" => scalar::log2f,
        "logf" => scalar::logf,
        "cbrtf" => scalar::cbrtf,
        _ => panic!("unknown function"),
    };
    let data = std::fs::read(&args[2]).expect("MPFR records required");
    assert!(!data.is_empty() && data.len() % 8 == 0);
    let mut failed = 0;
    for record in data.chunks_exact(8) {
        let input = u32::from_le_bytes(record[..4].try_into().unwrap());
        let expected = u32::from_le_bytes(record[4..].try_into().unwrap());
        let actual = function(f32::from_bits(input));
        if actual.to_bits() != expected && !(actual.is_nan() && f32::from_bits(expected).is_nan()) {
            if failed < 10 {
                eprintln!(
                    "{} input={input:08x} expected={expected:08x} actual={:08x}",
                    args[1],
                    actual.to_bits()
                );
            }
            failed += 1;
        }
    }
    assert_eq!(
        failed,
        0,
        "{}: {failed} failures / {}",
        args[1],
        data.len() / 8
    );
    println!("{}: {} MPFR records passed", args[1], data.len() / 8);
}
