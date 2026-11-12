//! Enumerate the actual policy builders, including private validity predicates.
//! The source modules are compiled unchanged; no second flag encoder is written.
#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../../../../magemath/src/expf/options.rs"]
mod expf;
#[allow(dead_code)]
#[path = "../../../../magemath/src/logf/options.rs"]
mod logf;
#[allow(dead_code)]
#[path = "../../../../magemath/src/minmax/options.rs"]
mod minmax;
#[allow(dead_code)]
#[path = "../../../../magemath/src/powf/options.rs"]
mod powf;
#[allow(dead_code)]
#[path = "../../../../magemath/src/recip/options.rs"]
mod recip;

fn row(family: &str, axes: [&dyn core::fmt::Debug; 6], bits: u32, supported: bool) {
    println!(
        "{}\t{}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\t{}\t{}",
        family,
        cfg!(feature = "std"),
        axes[0],
        axes[1],
        axes[2],
        axes[3],
        axes[4],
        axes[5],
        bits,
        supported
    );
}
macro_rules! transcendental {
    ($module:ident,$options:ident,$domain:ident,[$($d:ident),+],$precision:ident,[$($p:ident),+],$port:ident,$validation:ident,$execution:ident) => {{
        use $module::*;
        for d in [$($domain::$d),+] {
            for p in [$($precision::$p),+] {
                for portability in [$port::BitExact, $port::BackendDependent] {
                    for validation in [$validation::Checked, $validation::AssumeValid] {
                        for execution in [$execution::Automatic, $execution::Caller] {
                            let o = $options::new().with_domain(d).with_precision(p)
                                .with_portability(portability).with_validation(validation)
                                .with_execution(execution);
                            row(stringify!($module), [&d, &p, &portability, &"-", &validation, &execution],
                                o.bits(), $options::valid(o.bits()));
                        }
                    }
                }
            }
        }
    }};
}
fn main() {
    println!("family\tstd\tdomain\tprecision_or_nan\tportability_or_zero\treserved\tvalidation\texecution\tbits\tsupported");
    transcendental!(powf, PowfOptions, PowfDomain,
        [PositiveNormal,NonNegativeNormal,Gamma,PositiveGamma,Full],
        PowfPrecision,[Lowp,Midp,Highp],PowfPortability,PowfValidation,PowfExecution);
    transcendental!(expf, ExpfOptions, ExpfDomain,[Bounded,Full],
        ExpfPrecision,[Lowp,Midp],ExpfPortability,ExpfValidation,ExpfExecution);
    transcendental!(logf, LogfOptions, LogfDomain,[Positive,Specials],
        LogfPrecision,[Lowp,Midp],LogfPortability,LogfValidation,LogfExecution);
    transcendental!(recip, RecipOptions, RecipDomain,[Full,Normal],
        RecipPrecision,[Lowp,Midp,Highp],RecipPortability,RecipValidation,RecipExecution);
    use minmax::*;
    for d in [MinMaxDomain::Full, MinMaxDomain::Finite, MinMaxDomain::Normal] {
        for nan in [MinMaxNaN::Number, MinMaxNaN::Propagate, MinMaxNaN::Backend] {
            for zero in [MinMaxZero::Ordered, MinMaxZero::Backend] {
                for validation in [MinMaxValidation::Checked, MinMaxValidation::AssumeValid] {
                    for execution in [MinMaxExecution::Automatic, MinMaxExecution::Caller] {
                        let o = MinMaxOptions::new().with_domain(d).with_nan(nan).with_zero(zero)
                            .with_validation(validation).with_execution(execution);
                        row("minmax", [&d, &nan, &zero, &"-", &validation, &execution],
                            o.bits(), MinMaxOptions::valid(o.bits()));
                    }
                }
            }
        }
    }
}
