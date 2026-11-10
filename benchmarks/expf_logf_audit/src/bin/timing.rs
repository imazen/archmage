//! All-retained-round zenbench comparison, with the pre-PR source in each group.
#![forbid(unsafe_code)]
use archmage::SimdToken;
#[path = "../frozen.rs"]
mod frozen;

fn normalize(x: f32) -> f32 {
    f32::from_bits(0x3f000000 | (x.to_bits() & 0x007fffff))
}

macro_rules! width {
    ($name:ident, $scalar:ident, $v3:ident, $neon:ident, $dispatch:ident, $v:ident, $frozen:ident, $n:literal) => {
        #[archmage::magetypes(v3, neon, scalar)]
        #[inline(never)]
        fn $name<const ARM: usize, const LOG: bool, const LOW: bool>(
            token: Token,
            input: &[f32],
            output: &mut [f32],
            chain: bool,
        ) {
            use frozen::$frozen::Frozen;
            use magetypes::simd::generic::$v as V;
            let mut acc = V::splat_t(token, 0.75);
            for (xs, ys) in input
                .as_chunks::<$n>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<$n>().0)
            {
                let x = if chain {
                    acc
                } else {
                    V::from_array_t(token, *xs)
                };
                let y = match ARM {
                    0 => {
                        if LOG {
                            if LOW {
                                x.old_log2_lowp(token)
                            } else {
                                x.old_log2_midp(token)
                            }
                        } else if LOW {
                            x.old_exp2_lowp(token)
                        } else {
                            x.old_exp2_midp(token)
                        }
                    }
                    1 => {
                        if LOG {
                            if LOW { x.log2_lowp() } else { x.log2_midp() }
                        } else if LOW {
                            x.exp2_lowp()
                        } else {
                            x.exp2_midp()
                        }
                    }
                    2 => {
                        let a = if LOG {
                            if LOW {
                                magetypes::logf::try_log2_array_with::<8, $n>(x.to_array()).unwrap()
                            } else {
                                magetypes::logf::try_log2_array_with::<0, $n>(x.to_array()).unwrap()
                            }
                        } else if LOW {
                            magetypes::expf::try_exp2_array_with::<8, $n>(x.to_array()).unwrap()
                        } else {
                            magetypes::expf::try_exp2_array_with::<0, $n>(x.to_array()).unwrap()
                        };
                        V::from_array_t(token, a)
                    }
                    3 => {
                        let a = if LOG {
                            if LOW {
                                magetypes::logf::try_log2_array_with::<40, $n>(x.to_array())
                                    .unwrap()
                            } else {
                                magetypes::logf::try_log2_array_with::<32, $n>(x.to_array())
                                    .unwrap()
                            }
                        } else if LOW {
                            magetypes::expf::try_exp2_array_with::<40, $n>(x.to_array()).unwrap()
                        } else {
                            magetypes::expf::try_exp2_array_with::<32, $n>(x.to_array()).unwrap()
                        };
                        V::from_array_t(token, a)
                    }
                    4 => {
                        let a = if LOG {
                            if LOW {
                                magetypes::logf::try_log2_array_with::<56, $n>(x.to_array())
                                    .unwrap()
                            } else {
                                magetypes::logf::try_log2_array_with::<48, $n>(x.to_array())
                                    .unwrap()
                            }
                        } else if LOW {
                            magetypes::expf::try_exp2_array_with::<56, $n>(x.to_array()).unwrap()
                        } else {
                            magetypes::expf::try_exp2_array_with::<48, $n>(x.to_array()).unwrap()
                        };
                        V::from_array_t(token, a)
                    }
                    5 => V::from_array_t(
                        token,
                        x.to_array().map(|v| if LOG { v.log2() } else { v.exp2() }),
                    ),
                    _ => unreachable!(),
                };
                *ys = y.to_array();
                if chain {
                    acc = V::from_array_t(token, y.to_array().map(normalize));
                }
            }
        }
        fn $dispatch<const ARM: usize, const LOG: bool, const LOW: bool>(
            context: &str,
            input: &[f32],
            output: &mut [f32],
            chain: bool,
        ) {
            match context {
                "scalar" => $scalar::<ARM, LOG, LOW>(archmage::ScalarToken, input, output, chain),
                #[cfg(target_arch = "x86_64")]
                "native" => $v3::<ARM, LOG, LOW>(
                    archmage::X64V3Token::summon().expect("V3 required"),
                    input,
                    output,
                    chain,
                ),
                #[cfg(target_arch = "aarch64")]
                "native" => $neon::<ARM, LOG, LOW>(
                    archmage::NeonToken::summon().expect("NEON required"),
                    input,
                    output,
                    chain,
                ),
                _ => panic!("unsupported context"),
            }
        }
    };
}
width!(
    loop4,
    loop4_scalar,
    loop4_v3,
    loop4_neon,
    dispatch4,
    f32x4,
    x4,
    4
);
width!(
    loop8,
    loop8_scalar,
    loop8_v3,
    loop8_neon,
    dispatch8,
    f32x8,
    x8,
    8
);
width!(
    loop16,
    loop16_scalar,
    loop16_v3,
    loop16_neon,
    dispatch16,
    f32x16,
    x16,
    16
);

fn execute<const ARM: usize, const LOG: bool, const LOW: bool>(
    width: usize,
    context: &str,
    input: &[f32],
    output: &mut [f32],
    chain: bool,
) {
    match width {
        4 => dispatch4::<ARM, LOG, LOW>(context, input, output, chain),
        8 => dispatch8::<ARM, LOG, LOW>(context, input, output, chain),
        16 => dispatch16::<ARM, LOG, LOW>(context, input, output, chain),
        _ => panic!("unsupported width"),
    }
}

fn register<const LOG: bool, const LOW: bool>(
    suite: &mut zenbench::Suite,
    name: &str,
    width: usize,
    context: &str,
    size: usize,
    chain: bool,
) {
    suite.group(name, |g| {
        g.config()
            .min_rounds(50)
            .max_rounds(50)
            .auto_rounds(false)
            .warmup_time(std::time::Duration::from_millis(10))
            .max_time(std::time::Duration::from_millis(100))
            .max_wall_time(std::time::Duration::from_secs(180));
        macro_rules! arm {
            ($id:literal,$label:literal) => {{
                let input: Vec<f32> = (0..size).map(|i| 0.5 + (i % 256) as f32 / 512.).collect();
                let context = context.to_owned();
                g.bench($label, move |b| {
                    let mut output = vec![0.; size];
                    execute::<$id, LOG, LOW>(width, &context, &input, &mut output, chain);
                    b.iter(|| {
                        execute::<$id, LOG, LOW>(
                            width,
                            &context,
                            std::hint::black_box(&input),
                            std::hint::black_box(&mut output),
                            chain,
                        )
                    });
                    std::hint::black_box(output);
                });
            }};
        }
        arm!(0, "main_91a13739_legacy");
        arm!(1, "PR148_legacy_method");
        arm!(2, "PR148_canonical_BitExact_Checked_Automatic");
        arm!(3, "PR148_canonical_BitExact_AssumeValid_Automatic");
        arm!(4, "PR148_canonical_BackendDependent_AssumeValid_Automatic");
        arm!(5, "std_f32_per_lane");
    });
}

fn main() {
    let filter = std::env::var("AUDIT_CELL").expect("frozen cell required");
    let p: Vec<_> = filter.split('/').collect();
    assert_eq!(p.len(), 6);
    let width = p[2].parse().unwrap();
    let size = p[4].parse().unwrap();
    let chain = p[5] == "chain";
    let output = std::env::var("AUDIT_RESULT").expect("new result path required");
    assert!(!std::path::Path::new(&output).exists());
    zenbench::run_gated(zenbench::GateConfig::strict(), |s| match (p[0], p[1]) {
        ("exp2", "midp") => register::<false, false>(s, &filter, width, p[3], size, chain),
        ("exp2", "lowp") => register::<false, true>(s, &filter, width, p[3], size, chain),
        ("log2", "midp") => register::<true, false>(s, &filter, width, p[3], size, chain),
        ("log2", "lowp") => register::<true, true>(s, &filter, width, p[3], size, chain),
        _ => panic!("unsupported operation"),
    })
    .save(std::path::Path::new(&output))
    .unwrap();
}
