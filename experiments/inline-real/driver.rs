//! End-to-end in-memory consumer probe; generated Cargo manifests select policy.
#![forbid(unsafe_code)]
use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::{hint::black_box, path::Path, sync::Arc, time::Duration};
use svtav1_encoder::{
    pipeline::EncodePipeline,
    rate_control::{RcConfig, RcMode},
};

fn encoder(width: usize, qp: u8, preset: u8) -> EncodePipeline {
    let rc = RcConfig {
        mode: RcMode::Cqp,
        qp,
        ..RcConfig::default()
    };
    EncodePipeline::new(width as u32, width as u32, preset, rc, 0, 1)
        .with_bit_depth(8)
        .with_tile_rows_log2(0)
        .with_tile_cols_log2(0)
        .with_sb_size(None)
        .with_chroma_420(true)
}
fn decoder() -> Decoder {
    let mut settings = Settings::default();
    settings.threads = 1;
    settings.max_frame_delay = 1;
    Decoder::with_settings(settings).expect("decoder setup")
}
fn pixels(frame: &Frame) -> Vec<u8> {
    let mut bytes = Vec::new();
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                bytes.extend_from_slice(row);
            }
            for p in [planes.u(), planes.v()].into_iter().flatten() {
                for row in p.rows() {
                    bytes.extend_from_slice(row);
                }
            }
        }
        Planes::Depth16(planes) => {
            for row in planes.y().rows() {
                for v in row {
                    bytes.extend(v.to_le_bytes());
                }
            }
            for p in [planes.u(), planes.v()].into_iter().flatten() {
                for row in p.rows() {
                    for v in row {
                        bytes.extend(v.to_le_bytes());
                    }
                }
            }
        }
    }
    bytes
}
fn config(g: &mut zenbench::BenchGroup) {
    let c = g.config();
    c.auto_rounds = false;
    c.min_rounds = 20;
    c.max_rounds = 20;
    c.min_iterations = 1;
    c.max_iterations = 1;
    c.warmup_time = Duration::from_millis(100);
    c.max_time = Duration::from_secs(60);
    c.max_wall_time = Duration::from_secs(90);
    c.stack_jitter = false;
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 6, "driver input.yuv width qp preset outdir");
    let data = Arc::new(std::fs::read(&args[1]).expect("input"));
    let width: usize = args[2].parse().unwrap();
    let qp: u8 = args[3].parse().unwrap();
    let preset: u8 = args[4].parse().unwrap();
    let out = Path::new(&args[5]);
    std::fs::create_dir(out).expect("fresh output directory");
    let n = width * width;
    assert_eq!(data.len(), n * 3 / 2);
    let encode = |p: &mut EncodePipeline, d: &[u8]| {
        p.try_encode_frame_420(&d[..n], &d[n..n + n / 4], &d[n + n / 4..], width)
            .expect("encode")
    };
    let obu = Arc::new(encode(&mut encoder(width, qp, preset), &data));
    assert!(!obu.is_empty());
    let frame = decoder()
        .decode(&obu)
        .expect("decode")
        .expect("one still frame");
    assert_eq!(
        (frame.width() as usize, frame.height() as usize),
        (width, width)
    );
    std::fs::write(out.join("encoded.obu"), &*obu).unwrap();
    std::fs::write(out.join("decoded.yuv"), pixels(&frame)).unwrap();
    let result = zenbench::run(|suite| {
        suite.group("encode", |g| {
            config(g);
            let data = data.clone();
            g.bench("zenav1-svt", move |b| {
                let data = data.clone();
                b.with_input(move || encoder(width, qp, preset))
                    .run(|mut p| {
                        let bytes = p
                            .try_encode_frame_420(
                                &data[..n],
                                &data[n..n + n / 4],
                                &data[n + n / 4..],
                                width,
                            )
                            .unwrap();
                        // Return the owner too: its destruction is outside the timer.
                        (p, black_box(bytes))
                    });
            });
        });
        suite.group("decode", |g| {
            config(g);
            let obu = obu.clone();
            g.bench("rav1d-safe", move |b| {
                let obu = obu.clone();
                b.with_input(decoder).run(|mut d| {
                    let frame = d.decode(&obu).unwrap().expect("one still frame");
                    (d, black_box(frame))
                });
            });
        });
    });
    println!("RELIABLE\t{}\t{}", !result.unreliable, result.gate_waits);
    for group in result.comparisons {
        for b in group.benchmarks {
            println!(
                "SUMMARY\t{}\t{}\t{}\t{}\t{}",
                group.group_name, b.summary.n, b.summary.mean, b.summary.median, b.summary.mad
            );
        }
        for (round, sample) in group.samples.iter().enumerate() {
            println!(
                "SAMPLE\t{}\t{}\t{}\t{}",
                group.group_name, round, sample.iterations, sample.elapsed_ns[0]
            );
        }
    }
}
