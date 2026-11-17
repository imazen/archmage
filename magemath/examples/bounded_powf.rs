//! Apply a checked power curve to four samples, one scalar, and a slice.
//!
//! Run: cargo run -p magemath --features std --example bounded_powf
#![forbid(unsafe_code)]

use magemath::powf::{PowfError, presets::GammaMidp, try_powf_array_with, try_powf_with};

fn main() -> Result<(), PowfError> {
    let samples = [0.0, 0.125, 0.5, 1.0];
    let exponent = 2.2;

    // GammaMidp accepts +0 or bases in [2^-16, 2^16] and exponents in [0, 4].
    // The array API evaluates a complete pack and returns no partial result
    // if any lane violates the selected domain.
    let curved = try_powf_array_with::<GammaMidp, 4>(samples, exponent)?;
    for (sample, result) in samples.into_iter().zip(curved) {
        println!("{sample:.3}^{exponent:.1} = {result:.6}");
    }

    // Use the same profile for a single sample, such as a slice tail.
    let remaining = try_powf_with::<GammaMidp>(0.75, exponent)?;
    println!("Remaining sample: {remaining:.6}");

    // A negative base is outside this profile's domain. The error identifies
    // lane 2; the other three lanes are not returned as partial results.
    let invalid = [0.0, 0.125, -0.5, 1.0];
    if let Err(error) = try_powf_array_with::<GammaMidp, 4>(invalid, exponent) {
        println!("Rejected input: {error}");
    }
    Ok(())
}
