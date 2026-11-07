#![forbid(unsafe_code)]
#[cfg(feature = "proof")]
use archmage::prelude::*;
use std::{fs::OpenOptions, io::Write, path::Path};
use zenhash::Xxh3;
const MIN: [u32; 2] = [0x3db71074, 0x3b475041];
const MAX: u32 = 0x3f800000;
const Y: [u32; 2] = [0x4019999a, 0x3ed55555];
const EXPECTED: [u128; 2] = [
    0x01ac8deaf0e5dfa2d0d75a859383ce568,
    0x1c229cfcfabcb6d54aa44974481b4db6,
];
#[cfg(feature = "proof")]
#[attune(_scalar_t, _v3_t, _neon_t, _wasm128_t, _v4x_t(cfg(avx512)))]
fn evaluate(path: &str, x: &[f32], out: &mut [f32], y: f32) {
    let t = Token::from_context();
    assert_eq!(x.len(), out.len());
    match path {
        "ordinary" => {
            for (&x, out) in x.iter().zip(out) {
                *out = zenmath::scalar::powf(x, y);
            }
        }
        "context" => {
            for (&x, out) in x.iter().zip(out) {
                *out = attuned!(
                    zenmath::simd::scalar::powf(x, y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
            }
        }
        "array4" => {
            for (xs, ys) in x.chunks(4).zip(out.chunks_mut(4)) {
                let mut a = [1.0; 4];
                a[..xs.len()].copy_from_slice(xs);
                let b = attuned!(
                    zenmath::simd::array::pow_f32x4(a, [y; 4]),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                ys.copy_from_slice(&b[..xs.len()]);
            }
        }
        "array8" => {
            // Non-V4x eight-input adapter is two four-input packed blocks,
            // matching the reviewed V3 prototype. True V4x8 is selected below.
            for (xs, ys) in x.chunks(4).zip(out.chunks_mut(4)) {
                let mut a = [1.0; 4];
                a[..xs.len()].copy_from_slice(xs);
                let b = attuned!(
                    zenmath::simd::array::pow_f32x4(a, [y; 4]),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                ys.copy_from_slice(&b[..xs.len()]);
            }
        }
        _ => panic!("unknown path {path}"),
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune(_v4x_t)]
fn wide(x: &[f32], out: &mut [f32], y: f32) {
    let t = Token::from_context();
    for (xs, ys) in x.chunks(8).zip(out.chunks_mut(8)) {
        let mut a = [1.0; 8];
        a[..xs.len()].copy_from_slice(xs);
        let b = zenmath::simd::array::pow_f32x8_v4x_t(t, a, [y; 8]);
        ys.copy_from_slice(&b[..xs.len()]);
    }
}
#[cfg(feature = "proof")]
fn call(tier: &str, path: &str, x: &[f32], out: &mut [f32], y: f32) {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if tier == "v4x" && path == "array8" {
        wide_v4x_t(X64V4xToken::summon().expect("required V4x"), x, out, y);
        return;
    }
    match tier {
        "scalar" => evaluate_scalar_t(ScalarToken, path, x, out, y),
        #[cfg(target_arch = "x86_64")]
        "v3" => evaluate_v3_t(X64V3Token::summon().expect("required V3"), path, x, out, y),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4x" => evaluate_v4x_t(
            X64V4xToken::summon().expect("required V4x"),
            path,
            x,
            out,
            y,
        ),
        #[cfg(target_arch = "aarch64")]
        "neon" => evaluate_neon_t(NeonToken::summon().expect("required NEON"), path, x, out, y),
        #[cfg(target_arch = "wasm32")]
        "wasm128" => evaluate_wasm128_t(
            Wasm128Token::summon().expect("required SIMD128"),
            path,
            x,
            out,
            y,
        ),
        _ => panic!("unsupported required tier {tier}"),
    }
}
#[cfg(not(feature = "proof"))]
fn call(tier: &str, path: &str, x: &[f32], out: &mut [f32], y: f32) {
    assert_eq!(tier, "scalar");
    assert_eq!(path, "ordinary");
    for (&x, out) in x.iter().zip(out) {
        *out = zenmath::scalar::powf(x, y);
    }
}
fn run(root: &Path, tier: &str, path: &str, k: usize, full: bool) {
    let count = if full {
        u64::from(MAX - MIN[k]) + 1
    } else {
        65536
    };
    let mut log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(format!("{tier}-{path}-{k}.tsv")))
        .unwrap();
    writeln!(log,"# gamma-validation-v1; zenhash4c4a993b XXH3-128 seed0; LE-output-u32; no NaNs in admitted domain; signed zeros unchanged; aggregate is ordered chunk hashes, not raw stream; full={full}").unwrap();
    let mut raw = Xxh3::new();
    let mut aggregate = Xxh3::new();
    aggregate.update(b"gamma-production-ordered-chunks-v1");
    let mut completed = 0u64;
    let mut chunk = 0u64;
    while completed < count {
        let n = (count - completed).min(1 << 20) as usize;
        let start = u64::from(MIN[k]) + completed;
        let input: Vec<f32> = (start..start + n as u64)
            .map(|b| f32::from_bits(b as u32))
            .collect();
        let mut out = vec![0.0; n];
        call(tier, path, &input, &mut out, f32::from_bits(Y[k]));
        let mut bytes = Vec::with_capacity(n * 4);
        for (&x, &actual) in input.iter().zip(&out) {
            let want = zenmath::scalar::reference::powf(x, f32::from_bits(Y[k]));
            assert_eq!(
                actual.to_bits(),
                want.to_bits(),
                "MISMATCH {tier}/{path}/{k} x={:08x} y={:08x}",
                x.to_bits(),
                Y[k]
            );
            bytes.extend_from_slice(&actual.to_bits().to_le_bytes());
        }
        raw.update(&bytes);
        let digest = zenhash::xxh3_128(&bytes);
        for value in [chunk, start, n as u64] {
            aggregate.update(&value.to_le_bytes());
        }
        aggregate.update(&digest.to_le_bytes());
        writeln!(log, "{chunk}\t{start}\t{n}\t{digest:032x}").unwrap();
        log.flush().unwrap();
        completed += n as u64;
        chunk += 1;
    }
    let raw = raw.digest128();
    if full {
        assert_eq!(raw, EXPECTED[k], "frozen raw output digest differs");
    }
    writeln!(
        log,
        "COMPLETE\t{completed}\t{raw:032x}\t{:032x}",
        aggregate.digest128()
    )
    .unwrap();
    println!("PASS {tier}/{path}/{k} completed={completed} raw={raw:032x}");
}
fn main() {
    let a: Vec<_> = std::env::args().collect();
    assert_eq!(a.len(), 5, "output-dir tier path full|smoke");
    assert!(a[4] == "full" || a[4] == "smoke");
    let root = Path::new(&a[1]);
    std::fs::create_dir_all(root).unwrap();
    for k in 0..2 {
        run(root, &a[2], &a[3], k, a[4] == "full");
    }
}
