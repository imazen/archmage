#![forbid(unsafe_code)]
use std::hint::black_box;
use zenmath_pow_opt_study::*;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let d = Domain::parse(&args[1]);
    let label = &args[2];
    let passes: usize = args[3].parse().unwrap();
    let mut uniform = false;
    let mut distribution = Distribution::Uniform;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--uniform" => {
                uniform = true;
                i += 1;
            }
            "--distribution" => {
                distribution = match args.get(i + 1).expect("distribution value").as_str() {
                    "historical" => Distribution::Historical,
                    "uniform" => Distribution::Uniform,
                    "log" => Distribution::Log,
                    "boundary" => Distribution::Boundary,
                    _ => panic!("unknown distribution"),
                };
                i += 2;
            }
            _ => panic!("unknown profiling argument"),
        }
    }

    let c = domain_cases(d, uniform)
        .into_iter()
        .find(|c| &c.name == label)
        .expect("missing required case");
    let (x, y) = distributed_inputs(d, distribution, 65536);
    let mut out = vec![0.0; x.len()];
    for _ in 0..passes {
        (c.run)(black_box(&x), black_box(&y), black_box(&mut out));
        black_box(&out);
    }
    println!(
        "CHECKSUM {}",
        out.iter().fold(0u64, |h, x| h.wrapping_mul(0x100000001b3)
            ^ u64::from(x.to_bits()))
    );
}
