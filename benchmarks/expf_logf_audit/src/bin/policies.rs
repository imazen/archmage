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
