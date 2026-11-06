#![forbid(unsafe_code)]
use portable_f32_study::{CANDIDATES, eval};
use std::{env, fs, path::Path};
fn main() {
    let args: Vec<_> = env::args().collect();
    assert!(
        (4..=5).contains(&args.len()),
        "results OP RECORDS OUTPUT_DIRECTORY [CANDIDATE]"
    );
    if let Some(candidate) = args.get(4) {
        assert!(
            CANDIDATES.contains(&candidate.as_str()),
            "unknown candidate"
        );
        assert!(
            portable_f32_study::supports(candidate, &args[1]),
            "candidate does not support operation"
        );
    }
    let data = fs::read(&args[2]).expect("records required");
    assert!(!data.is_empty() && data.len() % 12 == 0);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for r in data.as_chunks::<12>().0 {
        x.push(f32::from_bits(u32::from_le_bytes(
            r[..4].try_into().unwrap(),
        )));
        y.push(f32::from_bits(u32::from_le_bytes(
            r[4..8].try_into().unwrap(),
        )));
    }
    fs::create_dir_all(&args[3]).unwrap();
    for candidate in CANDIDATES {
        if args.get(4).is_some_and(|c| c != candidate) {
            continue;
        }
        if !portable_f32_study::supports(candidate, &args[1]) {
            continue;
        }
        let mut out = x.clone();
        eval(candidate, &args[1], &mut out, &y);
        let bits: Vec<u8> = out.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        fs::write(
            Path::new(&args[3]).join(format!("{candidate}-{}.bin", args[1])),
            bits,
        )
        .unwrap();
        println!("{candidate}/{}: {} outputs", args[1], out.len());
    }
}
