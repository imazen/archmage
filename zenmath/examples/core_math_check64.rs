//! Check externally generated MPFR records. Missing inputs are a hard error.
#![forbid(unsafe_code)]
use zenmath::scalar;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: core_math_check64 FUNCTION RECORDS");
    let function: fn(f64, f64) -> f64 = match args[1].as_str() {
        "exp" => |x, _| scalar::exp(x),
        "exp2" => |x, _| scalar::exp2(x),
        "cbrt" => |x, _| scalar::cbrt(x),
        _ => panic!("unknown function"),
    };
    let data = std::fs::read(&args[2]).expect("MPFR records required");
    assert!(!data.is_empty() && data.len() % 24 == 0);
    let mut failed = 0;
    for record in data.as_chunks::<24>().0 {
        let input = u64::from_le_bytes(record[..8].try_into().unwrap());
        let expected = u64::from_le_bytes(record[16..].try_into().unwrap());
        let y = u64::from_le_bytes(record[8..16].try_into().unwrap());
        let actual = function(f64::from_bits(input), f64::from_bits(y));
        if actual.to_bits() != expected && !(actual.is_nan() && f64::from_bits(expected).is_nan()) {
            if failed < 10 {
                eprintln!(
                    "{} input={input:016x} expected={expected:016x} actual={:016x}",
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
        data.len() / 24
    );
    println!("{}: {} MPFR records passed", args[1], data.len() / 24);
}
