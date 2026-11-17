//! Apply a checked power curve to four samples, one scalar, and a slice.
//!
//! Run: cargo run -p magemath --features std --example bounded_powf
#![forbid(unsafe_code)]

use magemath::powf::{Powf, PowfError, PowfOptions};

const GAMMA: u128 = PowfOptions::new().gamma().midp().bits();

fn main() -> Result<(), PowfError> {
    let samples = [0.0, 0.125, 0.5, 1.0];
    let exponent = 2.2;

    // `.gamma()` accepts +0 or bases in [2^-16, 2^16] and exponents in [0, 4].
    // The array API evaluates a complete pack and returns no partial result
    // if any lane violates the selected domain.
    let curved = Powf::<GAMMA>::array(samples, exponent)?;
    for (sample, result) in samples.into_iter().zip(curved) {
        println!("{sample:.3}^{exponent:.1} = {result:.6}");
    }

    // Use the same options for a single sample or a whole slice.
    let remaining = Powf::<GAMMA>::scalar(0.75, exponent)?;
    println!("Remaining sample: {remaining:.6}");

    let mut slice_out = [0.0; 4];
    if let Err(err) = Powf::<GAMMA>::slice(&samples, &mut slice_out, exponent) {
        println!("Unexpected slice rejection: {err}");
    }

    // A negative base is outside this domain. The error identifies
    // lane 2; the other three lanes are not returned as partial results.
    let invalid = [0.0, 0.125, -0.5, 1.0];
    if let Err(error) = Powf::<GAMMA>::array(invalid, exponent) {
        println!("Rejected input: {error}");
    }
    Ok(())
}
