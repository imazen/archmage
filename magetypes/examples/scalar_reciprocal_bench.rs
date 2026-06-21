//! Scalar (no-SIMD fallback) f32 reciprocal throughput: the bit-hack
//! `rsqrt_approx` (seed + 2 Newton steps) vs the exact sqrt+div it replaces, plus
//! `rcp_approx` (exact division) for reference. Runs natively.
//!
//! ```sh
//! cargo run --release -p magetypes --example scalar_reciprocal_bench --features std
//! ```

#[cfg(feature = "std")]
fn main() {
    use archmage::{ScalarToken, SimdToken};
    use magetypes::simd::generic::f32x8;
    use std::hint::black_box;
    use std::time::Instant;

    const N: usize = 2048;
    const REPS: usize = 20000;

    fn run<F>(t: ScalarToken, inp: &[f32], out: &mut [f32], op: F)
    where
        F: Fn(f32x8<ScalarToken>) -> f32x8<ScalarToken>,
    {
        for (ci, co) in inp.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
            op(f32x8::from_array(t, ci.try_into().unwrap())).store(co.try_into().unwrap());
        }
    }

    let token = ScalarToken::summon().expect("scalar");
    let input: Vec<f32> = (0..N).map(|i| 0.1 + (i % 997) as f32 * 0.1).collect();
    let mut out = vec![0.0f32; N];

    macro_rules! bench {
        ($name:expr, $op:expr) => {{
            for _ in 0..500 {
                run(token, &input, &mut out, $op);
            }
            let t0 = Instant::now();
            for _ in 0..REPS {
                run(token, black_box(&input), &mut out, $op);
                black_box(&out);
            }
            let per = t0.elapsed().as_nanos() as f64 / (REPS as f64 * N as f64);
            println!("{:<30} {:.4} ns/elem", $name, per);
            per
        }};
    }

    println!("Scalar f32 reciprocal throughput (N={N}, reps={REPS})\n");
    // rcp_approx is exact division (a bit-hack measured ~1.8x SLOWER on scalar).
    let rcp_approx = bench!("rcp_approx (division)", |v: f32x8<ScalarToken>| v
        .rcp_approx());
    let recip_div = bench!("recip (division)", |v: f32x8<ScalarToken>| v.recip());
    // rsqrt_approx is the bit-hack; rsqrt is the exact sqrt+div it replaces.
    let rsqrt_approx = bench!("rsqrt_approx (bit-hack x2)", |v: f32x8<ScalarToken>| v
        .rsqrt_approx());
    let rsqrt_full = bench!("rsqrt (sqrt + division)", |v: f32x8<ScalarToken>| v.rsqrt());

    println!();
    println!(
        "rcp_approx vs recip (both division): {:.2}x",
        recip_div / rcp_approx
    );
    println!(
        "rsqrt_approx (bit-hack) vs sqrt+div: {:.2}x faster",
        rsqrt_full / rsqrt_approx
    );
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("scalar_reciprocal_bench: build with --features std");
}
