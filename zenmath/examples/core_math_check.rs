//! Check externally generated MPFR records. Missing inputs are a hard error.
#![forbid(unsafe_code)]
use zenmath::scalar;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: core_math_check FUNCTION RECORDS");
    let function: fn(f32, f32) -> f32 = match args[1].as_str() {
        "exp2f" => |x, _| scalar::exp2f(x),
        "expf" => |x, _| scalar::expf(x),
        "log2f" => |x, _| scalar::log2f(x),
        "logf" => |x, _| scalar::logf(x),
        "cbrtf" => |x, _| scalar::cbrtf(x),
        "powf" => scalar::powf,
        "tanhf" => |x, _| scalar::tanhf(x),
        "atanf" => |x, _| scalar::atanf(x),
        "log10f" => |x, _| scalar::log10f(x),
        _ => panic!("unknown function"),
    };
    let data = std::fs::read(&args[2]).expect("MPFR records required");
    assert!(!data.is_empty() && data.len() % 12 == 0);
    let mut failed = 0;
    for record in data.chunks_exact(12) {
        let input = u32::from_le_bytes(record[..4].try_into().unwrap());
        let expected = u32::from_le_bytes(record[8..].try_into().unwrap());
        let y = u32::from_le_bytes(record[4..8].try_into().unwrap());
        let actual = function(f32::from_bits(input), f32::from_bits(y));
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
        data.len() / 12
    );
    println!("{}: {} MPFR records passed", args[1], data.len() / 12);
}
