//! WASM f32 transcendental throughput: `log2`, `exp2` and `pow(x, 2.4)` at midp
//! against their `_portable` forms, which fuse each multiply-add in software on
//! WASM (relaxed SIMD may round twice), so they give the same bits as every
//! other backend. Each form must first give the scalar backend's portable bits.
//!
//! zenbench has no WASM backend, so timing uses `std::time::Instant` (WASI
//! clocks work under wasmtime); the forms run in a rotating order over several
//! rounds and the median is printed. Run:
//!
//! ```sh
//! CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime \
//! RUSTFLAGS="-C target-feature=+simd128" \
//! cargo run --release -p magetypes --example wasm_transcendental_bench \
//!   --target wasm32-wasip1 --features std
//! ```

#[cfg(all(target_arch = "wasm32", feature = "std"))]
fn main() {
    use archmage::{ScalarToken, SimdToken, Wasm128Token};
    use magetypes::simd::generic::f32x4;
    use std::hint::black_box;
    use std::time::Instant;

    const N: usize = 2048; // 8 KB: L1-resident, compute-bound
    const REPS: usize = 400;
    const ROUNDS: usize = 9;

    type Op = fn(f32x4<Wasm128Token>) -> f32x4<Wasm128Token>;
    type ScalarOp = fn(f32x4<ScalarToken>) -> f32x4<ScalarToken>;

    fn run(t: Wasm128Token, inp: &[f32], out: &mut [f32], op: Op) {
        for (ci, co) in inp.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
            op(f32x4::from_array_t(t, ci.try_into().unwrap())).store(co.try_into().unwrap());
        }
    }

    let token = Wasm128Token::summon().expect("wasm simd128");
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let logs: Vec<f32> = (0..N).map(|_| 2f32.powf(next() * 40.0 - 20.0)).collect();
    let exps: Vec<f32> = (0..N).map(|_| next() * 60.0 - 30.0).collect();

    let forms: [(&str, bool, Op, Option<ScalarOp>); 6] = [
        ("log2_midp", false, |v| v.log2_midp(), None),
        (
            "log2_midp_portable",
            false,
            |v| v.log2_midp_portable(),
            Some(|v| v.log2_midp_portable()),
        ),
        ("exp2_midp", true, |v| v.exp2_midp(), None),
        (
            "exp2_midp_portable",
            true,
            |v| v.exp2_midp_portable(),
            Some(|v| v.exp2_midp_portable()),
        ),
        ("pow_midp(2.4)", false, |v| v.pow_midp(2.4), None),
        (
            "pow_midp_portable(2.4)",
            false,
            |v| v.pow_midp_portable(2.4),
            Some(|v| v.pow_midp_portable(2.4)),
        ),
    ];

    // the portable forms must give the scalar backend's bits
    for (name, exp, op, scalar) in &forms {
        let Some(scalar) = scalar else { continue };
        let inp = if *exp { &exps } else { &logs };
        let mut got = vec![0f32; N];
        run(token, inp, &mut got, *op);
        for (ci, g) in inp.chunks_exact(4).zip(got.chunks_exact(4)) {
            let want = scalar(f32x4::from_array_t(ScalarToken, ci.try_into().unwrap())).to_array();
            for (w, x) in want.iter().zip(g.iter()) {
                assert_eq!(
                    w.to_bits(),
                    x.to_bits(),
                    "{name} differs from the scalar backend"
                );
            }
        }
    }

    let mut out = vec![0f32; N];
    let mut times = vec![Vec::new(); forms.len()];
    for round in 0..ROUNDS {
        for k in 0..forms.len() {
            let i = (k + round) % forms.len();
            let (_, exp, op, _) = forms[i];
            let inp = if exp { &exps } else { &logs };
            for _ in 0..20 {
                run(token, inp, &mut out, op);
            }
            let t0 = Instant::now();
            for _ in 0..REPS {
                run(token, black_box(inp), &mut out, op);
                black_box(&out);
            }
            times[i].push(t0.elapsed().as_nanos() as f64 / (REPS * N) as f64);
        }
    }
    println!("WASM f32x4 transcendentals (N={N}, reps={REPS}, median of {ROUNDS} rounds)\n");
    for (i, (name, ..)) in forms.iter().enumerate() {
        let mut t = times[i].clone();
        t.sort_by(f64::total_cmp);
        println!("{name:<26} {:.3} ns/elem", t[ROUNDS / 2]);
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "std")))]
fn main() {
    println!(
        "wasm_transcendental_bench: build for wasm32-wasip1 with +simd128 (see the module docs)"
    );
}
