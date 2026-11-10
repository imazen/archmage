//! Named options, no changes to the production policy implementation.
#![forbid(unsafe_code)]
use archmage::SimdToken;
fn normalize(x: f32) -> f32 {
    f32::from_bits(0x3f000000 | (x.to_bits() & 0x007fffff))
}
include!("../policies_generated.rs");

fn label(flags: u32) -> String {
    format!(
        "{}/{}/{}/{}/{}",
        if flags & 64 == 0 {
            "canonical"
        } else {
            "legacy-fit"
        },
        if flags & 16 == 0 {
            "BitExact"
        } else {
            "BackendDependent"
        },
        if flags & 1 != 0 {
            "Full-Specials"
        } else {
            "Finite-Positive"
        },
        if flags & 32 == 0 {
            "Checked"
        } else {
            "AssumeValid"
        },
        if flags & 4 == 0 {
            "Automatic-array"
        } else {
            "Caller-array-ScalarToken"
        }
    )
}
fn register<const LOG: bool>(
    suite: &mut zenbench::Suite,
    name: &str,
    width: usize,
    context: &str,
    size: usize,
    chain: bool,
    low: bool,
) {
    suite.group(name, |g| {
        g.config()
            .min_rounds(50)
            .max_rounds(50)
            .auto_rounds(false)
            .warmup_time(std::time::Duration::from_millis(10))
            .max_time(std::time::Duration::from_millis(100))
            .max_wall_time(std::time::Duration::from_secs(180));
        for flags in
            (0..128).filter(|x| x & 2 == 0 && ((x & 8 != 0) == low) && !(x & 1 != 0 && x & 32 != 0))
        {
            let input: Vec<f32> = (0..size).map(|i| 0.5 + (i % 256) as f32 / 512.).collect();
            let context = context.to_owned();
            g.bench(&format!("bits{flags:03}/{}", label(flags)), move |b| {
                let mut output = vec![0.; size];
                b.iter(|| {
                    execute::<LOG>(
                        width,
                        flags,
                        &context,
                        std::hint::black_box(&input),
                        std::hint::black_box(&mut output),
                        chain,
                        false,
                    )
                });
                std::hint::black_box(output);
            });
        }
        let input: Vec<f32> = (0..size).map(|i| 0.5 + (i % 256) as f32 / 512.).collect();
        let context = context.to_owned();
        let flags = if low { 44 } else { 36 };
        g.bench(
            "BitExact/Finite-Positive/AssumeValid/explicit-token-scalar-lane-map",
            move |b| {
                let mut output = vec![0.; size];
                b.iter(|| {
                    execute::<LOG>(
                        width,
                        flags,
                        &context,
                        std::hint::black_box(&input),
                        std::hint::black_box(&mut output),
                        chain,
                        true,
                    )
                });
                std::hint::black_box(output);
            },
        );
    });
}
fn main() {
    if let Ok(prefix) = std::env::var("POLICY_VERIFY_PREFIX") {
        verify(std::path::Path::new(&prefix));
        return;
    }
    let cell = std::env::var("AUDIT_CELL").expect("frozen cell required");
    let p: Vec<_> = cell.split('/').collect();
    assert_eq!(p.len(), 6);
    let width = p[2].parse().unwrap();
    let size = p[4].parse().unwrap();
    let chain = p[5] == "chain";
    let low = p[1] == "lowp";
    let output = std::env::var("AUDIT_RESULT").expect("new path");
    assert!(!std::path::Path::new(&output).exists());
    zenbench::run_gated(zenbench::GateConfig::strict(), |s| match p[0] {
        "log2" => register::<true>(s, &cell, width, p[3], size, chain, low),
        "exp2" => register::<false>(s, &cell, width, p[3], size, chain, low),
        _ => panic!("op"),
    })
    .save(std::path::Path::new(&output))
    .unwrap();
}

fn verify(prefix: &std::path::Path) {
    use std::io::Write;
    let data = prefix.with_extension("bin");
    let meta = prefix.with_extension("json");
    assert!(!data.exists() && !meta.exists());
    let mut writer = std::io::BufWriter::new(std::fs::File::create(data).unwrap());
    let mut segments = Vec::new();
    let mut offset = 0u64;
    for log in [false, true] {
        for low in [false, true] {
            let mut cases: Vec<_> = (0..128)
                .filter(|f| f & 2 == 0 && ((f & 8 != 0) == low) && !(f & 1 != 0 && f & 32 != 0))
                .map(|f| (f, false))
                .collect();
            cases.push((if low { 44 } else { 36 }, true));
            for (flags, token_lanes) in cases {
                for width in [4, 8] {
                    for context in ["scalar", "native"] {
                        for size in [64, 4096] {
                            for chain in [false, true] {
                                let input: Vec<f32> =
                                    (0..size).map(|i| 0.5 + (i % 256) as f32 / 512.).collect();
                                let mut output = vec![0.; size];
                                if log {
                                    execute::<true>(
                                        width,
                                        flags,
                                        context,
                                        &input,
                                        &mut output,
                                        chain,
                                        token_lanes,
                                    )
                                } else {
                                    execute::<false>(
                                        width,
                                        flags,
                                        context,
                                        &input,
                                        &mut output,
                                        chain,
                                        token_lanes,
                                    )
                                }
                                for value in output {
                                    writer.write_all(&value.to_bits().to_le_bytes()).unwrap();
                                }
                                segments.push(serde_json::json!({"log":log,"flags":flags,"token_lanes":token_lanes,"width":width,"context":context,"size":size,"chain":chain,"offset":offset}));
                                offset += size as u64 * 4;
                            }
                        }
                    }
                }
            }
        }
    }
    writer.flush().unwrap();
    std::fs::write(meta, serde_json::to_vec_pretty(&segments).unwrap()).unwrap();
    println!("{} segments, {} output bytes", segments.len(), offset);
}
