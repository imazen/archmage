//! Full-domain vector/scalar equality; run explicitly through just.
use magemath::portable::slice;
use std::sync::atomic::{AtomicU64, Ordering};
#[test]
#[ignore = "all 2^32 f32 bit patterns; run just magemath-core-exhaustive"]
fn every_binary32_input() {
    let jobs: usize = std::env::var("RAYON_NUM_THREADS")
        .expect("run-heavy must set RAYON_NUM_THREADS")
        .parse()
        .unwrap();
    assert!(jobs > 0);
    type Slice = fn(&mut [f32]);
    type Scalar = fn(f32) -> f32;
    for (name, vector, scalar) in [
        (
            "expf",
            slice::exp_f32 as Slice,
            zenmath::scalar::expf as Scalar,
        ),
        ("logf", slice::ln_f32, zenmath::scalar::logf),
        ("log2f", slice::log2_f32, zenmath::scalar::log2f),
        (
            "powf(2.4)",
            |a: &mut [f32]| slice::pow_f32(a, &[2.4; 256]),
            |a| zenmath::scalar::powf(a, 2.4),
        ),
    ] {
        let next = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..jobs {
                s.spawn(|| {
                    loop {
                        let start = next.fetch_add(65536, Ordering::Relaxed);
                        if start >= 1u64 << 32 {
                            break;
                        }
                        for base in (start..start + 65536).step_by(256) {
                            let mut a = core::array::from_fn::<_, 256, _>(|i| {
                                f32::from_bits((base + i as u64) as u32)
                            });
                            vector(&mut a);
                            for (i, got) in a.into_iter().enumerate() {
                                let x = f32::from_bits((base + i as u64) as u32);
                                let want = scalar(x);
                                assert!(
                                    (got.is_nan() && want.is_nan())
                                        || got.to_bits() == want.to_bits(),
                                    "{name} input {:08x}: {:08x} != {:08x}",
                                    x.to_bits(),
                                    got.to_bits(),
                                    want.to_bits()
                                );
                            }
                        }
                    }
                });
            }
        });
        println!("{name}: all 4294967296 input patterns passed");
    }
}
