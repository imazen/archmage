//! Broad external oracle/default and shared scalar/packed policy parity.
#![forbid(unsafe_code)]
use magemath::{
    PowBehavior as B, PowConsistency, PowInfinity, PowNaN, PowOutput, PowRange, PowSign,
    SubnormalInputs, SubnormalOutputs,
};
use std::{env, fs, path::Path};
fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()
}
fn check<const P: u64>(label: &str, records: &[[f32; 3]], out: &Path) {
    let mut output = Vec::with_capacity(records.len() * 4);
    for (n, chunk) in records.chunks(8).enumerate() {
        let x = core::array::from_fn(|i| chunk.get(i).map_or(1.0, |r| r[0]));
        let y = core::array::from_fn(|i| chunk.get(i).map_or(0.0, |r| r[1]));
        let got = magemath::portable::array::pow_f32x8_with::<P>(x, y);
        for (i, r) in chunk.iter().enumerate() {
            let want = zenmath::scalar::powf_with::<P>(r[0], r[1]);
            assert!(
                same(got[i], want),
                "{label} scalar/packed index{} x{:08x} y{:08x} got{:08x} want{:08x}",
                n * 8 + i,
                r[0].to_bits(),
                r[1].to_bits(),
                got[i].to_bits(),
                want.to_bits()
            );
            if P == 0 {
                assert!(
                    same(got[i], r[2]),
                    "independent MPFR index{} got{:08x} want{:08x}",
                    n * 8 + i,
                    got[i].to_bits(),
                    r[2].to_bits()
                );
            }
            output.extend(got[i].to_bits().to_le_bytes());
        }
    }
    fs::write(out.join(format!("{label}.bin")), output).unwrap();
    println!(
        "{label}: {} records scalar/packed passed{}",
        records.len(),
        if P == 0 { " and independent MPFR" } else { "" }
    );
}
fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(), 3, "RECORDS OUTPUT_DIR");
    let data = fs::read(&args[1]).expect("external oracle records required");
    assert!(!data.is_empty() && data.len() % 12 == 0);
    let records: Vec<[f32; 3]> = data
        .as_chunks::<12>()
        .0
        .iter()
        .map(|r| {
            core::array::from_fn(|i| {
                f32::from_bits(u32::from_le_bytes(r[i * 4..i * 4 + 4].try_into().unwrap()))
            })
        })
        .collect();
    let out = Path::new(&args[2]);
    fs::create_dir_all(out).unwrap();
    check::<0>("correct", &records, out);
    check::<{ B::MIDP.to_u64() }>("midp", &records, out);
    check::<{ B::LOWP.to_u64() }>("lowp", &records, out);
    check::<
        {
            B::MIDP
                .consistency(PowConsistency::PortableSeparate)
                .to_u64()
        },
    >("midp_separate", &records, out);
    check::<
        {
            B::LOWP
                .consistency(PowConsistency::PortableSeparate)
                .to_u64()
        },
    >("lowp_separate", &records, out);
    check::<{ B::CORRECT.sign(PowSign::Magnitude).to_u64() }>("magnitude", &records, out);
    check::<{ B::CORRECT.sign(PowSign::SignedMagnitude).to_u64() }>(
        "signed_magnitude",
        &records,
        out,
    );
    check::<
        {
            B::CORRECT
                .subnormal_inputs(SubnormalInputs::FlushSignedZero)
                .to_u64()
        },
    >("flush_inputs", &records, out);
    check::<
        {
            B::CORRECT
                .subnormal_outputs(SubnormalOutputs::FlushSignedZero)
                .to_u64()
        },
    >("flush_outputs", &records, out);
    check::<{ B::CORRECT.nan(PowNaN::CanonicalResult).to_u64() }>("canonical", &records, out);
    check::<{ B::CORRECT.infinity(PowInfinity::RejectInputsToNaN).to_u64() }>(
        "reject_infinite",
        &records,
        out,
    );
    check::<{ B::CORRECT.range(PowRange::ClampUnitInterval).to_u64() }>(
        "clamp_inputs",
        &records,
        out,
    );
    check::<{ B::CORRECT.output(PowOutput::SaturateFinite).to_u64() }>(
        "finite_outputs",
        &records,
        out,
    );
    check::<{ B::CORRECT.output(PowOutput::ClampUnitInterval).to_u64() }>(
        "unit_outputs",
        &records,
        out,
    );
    check::<{ B::MIDP.range(PowRange::PositiveNormalPrecondition).to_u64() }>(
        "pre_midp", &records, out,
    );
    check::<{ B::LOWP.range(PowRange::PositiveNormalPrecondition).to_u64() }>(
        "pre_lowp", &records, out,
    );
    check::<
        {
            B::MIDP
                .range(PowRange::PositiveNormalPrecondition)
                .consistency(PowConsistency::PortableSeparate)
                .to_u64()
        },
    >("pre_midp_separate", &records, out);
    check::<
        {
            B::LOWP
                .range(PowRange::PositiveNormalPrecondition)
                .consistency(PowConsistency::PortableSeparate)
                .to_u64()
        },
    >("pre_lowp_separate", &records, out);
}
