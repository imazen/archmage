//! Apply a checked power curve to four samples and one scalar.
//!
//! Run: cargo run -p magetypes --example bounded_powf
#![forbid(unsafe_code)]

use magetypes::powf::{PowfError, PowfOptions, try_powf_array_with, try_powf_with};

// Gamma accepts +0 or bases in [2^-16, 2^16]. The exponent must be zero
// or a normal floating-point value in [0, 4]. This preset checks those bounds
// and selects an available backend automatically.
// Portable mode gives matching result bits; MIDP is an approximation.
const CURVE: u32 = PowfOptions::PORTABLE_GAMMA_MIDP.bits();

fn main() -> Result<(), PowfError> {
    let samples = [0.0, 0.125, 0.5, 1.0];
    let exponent = 2.2;

    // The array API evaluates a complete pack and returns no partial result
    // if any lane violates the selected domain.
    let curved = try_powf_array_with::<CURVE, 4>(samples, exponent)?;
    for (sample, result) in samples.into_iter().zip(curved) {
        println!("{sample:.3}^{exponent:.1} = {result:.6}");
    }

    // Use the same portable preset for a single sample, such as a slice tail.
    let remaining = try_powf_with::<CURVE>(0.75, exponent)?;
    println!("Remaining sample: {remaining:.6}");

    // A negative base is outside this preset's domain. The error identifies
    // lane 2; the other three lanes are not returned as partial results.
    let invalid = [0.0, 0.125, -0.5, 1.0];
    if let Err(error) = try_powf_array_with::<CURVE, 4>(invalid, exponent) {
        println!("Rejected input: {error}");
    }
    Ok(())
}
