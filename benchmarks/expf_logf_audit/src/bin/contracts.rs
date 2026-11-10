//! Review protocol, deliberately outside human-facing examples and default tests.
use magetypes::{expf::*, logf::*};
use std::panic::{UnwindSafe, catch_unwind};

fn probe<T: std::fmt::Debug>(name: &str, f: impl FnOnce() -> T + UnwindSafe) {
    match catch_unwind(f) {
        Ok(v) => println!("{name}\t{v:?}"),
        Err(_) => println!("{name}\tPANIC"),
    }
}

fn policy<const O: u32>() {
    let eo = ExpfOptions::new()
        .with_domain(if O & 1 == 0 {
            ExpfDomain::Bounded
        } else {
            ExpfDomain::Full
        })
        .with_execution(if O & 4 == 0 {
            ExpfExecution::Automatic
        } else {
            ExpfExecution::Caller
        })
        .with_precision(match O & 72 {
            0 => ExpfPrecision::Midp,
            8 => ExpfPrecision::Lowp,
            64 => ExpfPrecision::LegacyMidp,
            _ => ExpfPrecision::LegacyLowp,
        })
        .with_portability(if O & 16 == 0 {
            ExpfPortability::BitExact
        } else {
            ExpfPortability::BackendDependent
        })
        .with_validation(if O & 32 == 0 {
            ExpfValidation::Checked
        } else {
            ExpfValidation::AssumeValid
        });
    let lo = LogfOptions::new()
        .with_domain(if O & 1 == 0 {
            LogfDomain::Positive
        } else {
            LogfDomain::Specials
        })
        .with_execution(if O & 4 == 0 {
            LogfExecution::Automatic
        } else {
            LogfExecution::Caller
        })
        .with_precision(match O & 72 {
            0 => LogfPrecision::Midp,
            8 => LogfPrecision::Lowp,
            64 => LogfPrecision::LegacyMidp,
            _ => LogfPrecision::LegacyLowp,
        })
        .with_portability(if O & 16 == 0 {
            LogfPortability::BitExact
        } else {
            LogfPortability::BackendDependent
        })
        .with_validation(if O & 32 == 0 {
            LogfValidation::Checked
        } else {
            LogfValidation::AssumeValid
        });
    assert_eq!(eo.bits(), O);
    assert_eq!(lo.bits(), O);
    probe(&format!("exp.const.{O}"), || {
        try_exp2_with::<O>(0.375).map(f32::to_bits)
    });
    probe(&format!("exp.runtime.{O}"), || {
        try_exp2_runtime(0.375, eo).map(f32::to_bits)
    });
    probe(&format!("log.const.{O}"), || {
        try_log2_with::<O>(1.375).map(f32::to_bits)
    });
    probe(&format!("log.runtime.{O}"), || {
        try_log2_runtime(1.375, lo).map(f32::to_bits)
    });
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    macro_rules! policies { ($($o:literal),*) => { $(policy::<$o>();)* }; }
    policies!(
        0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45,
        48, 49, 52, 53, 56, 57, 60, 61, 64, 65, 68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92,
        93, 96, 97, 100, 101, 104, 105, 108, 109, 112, 113, 116, 117, 120, 121, 124, 125
    );
    for (label, x) in [
        ("negzero", -0.0),
        ("poszero", 0.0),
        ("negone", -1.0),
        ("tiny", f32::from_bits(1)),
        ("negative-tiny", -f32::from_bits(1)),
        ("max", f32::MAX),
    ] {
        probe(&format!("log.specials.{label}"), || {
            try_log2_with::<1>(x).map(f32::to_bits)
        });
        probe(&format!("log1p.{label}"), || {
            try_log1p_with::<0>(x).map(f32::to_bits)
        });
        probe(&format!("exp.{label}"), || {
            try_exp_with::<0>(x).map(f32::to_bits)
        });
    }
    probe("log1p.token.error-mask", || {
        try_log1p_with_token::<0, _>(archmage::ScalarToken, -2.0)
    });
    probe("log.token.error-mask", || {
        try_log2_with_token::<0, _>(archmage::ScalarToken, -2.0)
    });
    for bits in [
        0xbe80_0000,
        0xbf00_0000,
        0xbf80_0000,
        0x3e80_0000,
        0x3f00_0000,
        0x3f80_0000,
        0x42ff_0000,
    ] {
        probe(&format!("exp.lowp.{bits:08x}"), || {
            try_exp2_with::<12>(f32::from_bits(bits)).map(f32::to_bits)
        });
    }
}
