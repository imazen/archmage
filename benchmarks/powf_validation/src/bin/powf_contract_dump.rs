//! Read `base_bits exponent_bits domain` rows; emit actual bounded powf results.
#![allow(deprecated)] // Deliberately validates the retained Magetypes compatibility implementation.
#![forbid(unsafe_code)]
use magetypes::powf::{PowfDomain, PowfExecution, PowfOptions, try_powf_runtime};
use std::io::{self, BufRead};
fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 3);
        let base = u32::from_str_radix(fields[0], 16).unwrap();
        let exponent = u32::from_str_radix(fields[1], 16).unwrap();
        let domain = match fields[2] {
            "0" => PowfDomain::PositiveNormal,
            "1" => PowfDomain::NonNegativeNormal,
            "2" => PowfDomain::Gamma,
            _ => panic!("unknown domain"),
        };
        let options = PowfOptions {
            domain,
            execution: PowfExecution::Caller,
            ..PowfOptions::new()
        };
        let b = f32::from_bits(base);
        let n = f32::from_bits(exponent);
        match try_powf_runtime(b, n, options) {
            Ok(result) => println!(
                "{base:08x} {exponent:08x} {} {:08x} {:08x}",
                fields[2],
                result.to_bits(),
                b.powf(n).to_bits()
            ),
            Err(error) => println!(
                "{base:08x} {exponent:08x} {} {:?} {}",
                fields[2], error.kind, error.lane_mask
            ),
        }
    }
}
