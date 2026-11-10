//! Matched generic consumers for the crate-ownership study.
#![forbid(unsafe_code)]
use archmage::SimdToken;

#[archmage::magetypes(v3, neon, scalar)]
#[inline(never)]
fn run<const CALLER: bool>(token: Token, input: [f32; 8]) -> [[f32; 8]; 9] {
    use math::{expf as e, logf as l, powf as p};
    macro_rules! invoke {
        ($pe:expr,$ee:expr,$le:expr) => {
            [
                p::try_powf_array_with::<
                    {
                        p::PowfOptions::PORTABLE_GAMMA_MIDP
                            .with_execution($pe)
                            .bits()
                    },
                    8,
                >(input, 2.4)
                .unwrap(),
                p::try_powf_array_with::<
                    {
                        p::PowfOptions::PORTABLE_GAMMA_LOWP
                            .with_execution($pe)
                            .bits()
                    },
                    8,
                >(input, 2.4)
                .unwrap(),
                p::try_powf_array_with::<
                    {
                        p::PowfOptions::PORTABLE_GAMMA_HIGHP
                            .with_execution($pe)
                            .bits()
                    },
                    8,
                >(input, 2.4)
                .unwrap(),
                e::try_exp2_array_with::<
                    { e::ExpfOptions::PORTABLE_MIDP.with_execution($ee).bits() },
                    8,
                >(input)
                .unwrap(),
                e::try_exp2_array_with::<
                    { e::ExpfOptions::PORTABLE_LOWP.with_execution($ee).bits() },
                    8,
                >(input)
                .unwrap(),
                l::try_log2_array_with::<
                    { l::LogfOptions::PORTABLE_MIDP.with_execution($le).bits() },
                    8,
                >(input)
                .unwrap(),
                l::try_log2_array_with::<
                    { l::LogfOptions::PORTABLE_LOWP.with_execution($le).bits() },
                    8,
                >(input)
                .unwrap(),
                l::try_log1p_array_with::<
                    { l::LogfOptions::PORTABLE_MIDP.with_execution($le).bits() },
                    8,
                >(input)
                .unwrap(),
                input.map(|x| {
                    p::try_powf_with_token::<
                        {
                            p::PowfOptions::PORTABLE_GAMMA_MIDP
                                .with_execution($pe)
                                .bits()
                        },
                        _,
                    >(token, x, 2.4)
                    .unwrap()
                }),
            ]
        };
    }
    if CALLER {
        invoke!(
            p::PowfExecution::Caller,
            e::ExpfExecution::Caller,
            l::LogfExecution::Caller
        )
    } else {
        invoke!(
            p::PowfExecution::Automatic,
            e::ExpfExecution::Automatic,
            l::LogfExecution::Automatic
        )
    }
}
fn main() {
    for i in 0..32 {
        let input = std::hint::black_box(core::array::from_fn(|lane| {
            0.5 + ((i * 8 + lane) % 256) as f32 / 512.
        }));
        let mut rows = vec![
            run_scalar::<false>(archmage::ScalarToken, input),
            run_scalar::<true>(archmage::ScalarToken, input),
        ];
        #[cfg(target_arch = "x86_64")]
        {
            let t = archmage::X64V3Token::summon().expect("V3 required");
            rows.push(run_v3::<false>(t, input));
            rows.push(run_v3::<true>(t, input));
        }
        #[cfg(target_arch = "aarch64")]
        {
            let t = archmage::NeonToken::summon().expect("NEON required");
            rows.push(run_neon::<false>(t, input));
            rows.push(run_neon::<true>(t, input));
        }
        for row in rows {
            for values in row {
                for x in values {
                    print!("{:08x} ", x.to_bits());
                }
                println!();
            }
        }
    }
}
