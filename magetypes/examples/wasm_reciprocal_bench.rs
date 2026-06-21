//! WASM f32 reciprocal throughput: the bit-hack `rsqrt_approx` (seed + 2 Newton
//! steps) vs the exact sqrt+div it replaces, plus `rcp_approx` (exact division)
//! for reference. WASM SIMD has no hardware reciprocal estimate, so this is the
//! comparison that justifies the `_approx` choices.
//!
//! zenbench has no WASM backend, so timing uses `std::time::Instant` (WASI
//! clocks work under wasmtime). Run:
//!
//! ```sh
//! CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime \
//! RUSTFLAGS="-C target-feature=+simd128" \
//! cargo run --release -p magetypes --example wasm_reciprocal_bench \
//!   --target wasm32-wasip1 --features std
//! ```

#[cfg(all(target_arch = "wasm32", feature = "std"))]
fn main() {
    use archmage::{SimdToken, Wasm128Token};
    use magetypes::simd::generic::f32x4;
    use std::hint::black_box;
    use std::time::Instant;

    const N: usize = 2048; // 8 KB → L1-resident, compute-bound
    const REPS: usize = 6000;

    fn run<F>(t: Wasm128Token, inp: &[f32], out: &mut [f32], op: F)
    where
        F: Fn(f32x4<Wasm128Token>) -> f32x4<Wasm128Token>,
    {
        for (ci, co) in inp.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
            op(f32x4::from_array(t, ci.try_into().unwrap())).store(co.try_into().unwrap());
        }
    }

    let token = Wasm128Token::summon().expect("wasm simd128");
    let input: Vec<f32> = (0..N).map(|i| 0.1 + (i % 997) as f32 * 0.1).collect();
    let mut out = vec![0.0f32; N];

    macro_rules! bench {
        ($name:expr, $op:expr) => {{
            for _ in 0..200 {
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

    println!("WASM f32 reciprocal throughput (N={N}, reps={REPS})\n");
    // rcp_approx is exact division (a bit-hack measured no faster than div).
    let rcp_approx = bench!("rcp_approx (division)", |v: f32x4<Wasm128Token>| v
        .rcp_approx());
    let recip_div = bench!("recip (f32x4 division)", |v: f32x4<Wasm128Token>| v.recip());
    // rsqrt_approx is the bit-hack; rsqrt is the exact sqrt+div it replaces.
    let rsqrt_approx = bench!("rsqrt_approx (bit-hack x2)", |v: f32x4<Wasm128Token>| v
        .rsqrt_approx());
    let rsqrt_full = bench!("rsqrt (sqrt + division)", |v: f32x4<Wasm128Token>| v
        .rsqrt());

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

#[cfg(not(all(target_arch = "wasm32", feature = "std")))]
fn main() {
    eprintln!("wasm_reciprocal_bench: build for --target wasm32-wasip1 with --features std");
}
