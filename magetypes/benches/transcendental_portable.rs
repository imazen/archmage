//! The `*_midp_portable` transcendentals against their plain `*_midp` forms,
//! per backend.
//!
//! Where the CPU has FMA the portable forms run the same instructions as the
//! plain ones. The scalar backend fuses each multiply-add with the FMA
//! instruction when the CPU has one and in software when it does not; with
//! `TRANSCENDENTAL_PORTABLE_NO_FMA` set, this bench switches x86-64-v3 off
//! before timing, so the scalar group times the software path (x86-64 only).
//! zenbench does not build for wasm32.
//!
//! One shape: `out[i] = f(x[i])` over 1,024 L1-resident vectors inside one
//! `#[arcane]` region, for `log2`, `exp2` and `pow(x, 2.4)`. Before timing, each
//! group's portable kernels must give the scalar backend's portable bits.
//!
//! Run (no `-Ctarget-cpu=native`; bench what users get):
//!   cargo bench -p magetypes --bench transcendental_portable --features avx512   # x86_64
//!   TRANSCENDENTAL_PORTABLE_NO_FMA=1 cargo bench -p magetypes --bench transcendental_portable --features avx512
//!   cargo bench -p magetypes --bench transcendental_portable                     # aarch64

/// Vectors per pass.
const N: usize = 1024;

/// Defines one stream kernel `out[i] = op(x[i])`.
macro_rules! stream {
    ($name:ident, $vec:ident, $token:ident, |$v:ident| $op:expr) => {
        #[arcane]
        pub fn $name(t: $token, x: &[Lanes], out: &mut [Lanes]) {
            for (x, o) in x.iter().zip(out.iter_mut()) {
                let $v = $vec::<$token>::load_t(t, x);
                *o = ($op).to_array();
            }
        }
    };
}

/// The six kernels for one vector type on one token.
macro_rules! kernels {
    ($module:ident, $vec:ident, $token:ident, $lanes:literal) => {
        pub mod $module {
            use archmage::{arcane, $token};
            use magetypes::simd::generic::$vec;

            pub type Lanes = [f32; $lanes];
            pub type Kernel = fn($token, &[Lanes], &mut [Lanes]);

            stream!(log2_plain, $vec, $token, |v| v.log2_midp());
            stream!(log2_portable, $vec, $token, |v| v.log2_midp_portable());
            stream!(exp2_plain, $vec, $token, |v| v.exp2_midp());
            stream!(exp2_portable, $vec, $token, |v| v.exp2_midp_portable());
            stream!(pow_plain, $vec, $token, |v| v.pow_midp(2.4));
            stream!(pow_portable, $vec, $token, |v| v.pow_midp_portable(2.4));

            /// (name, plain or portable, uses the exp inputs, kernel)
            pub const ALL: [(&str, bool, bool, Kernel); 6] = [
                ("log2_midp", false, false, log2_plain),
                ("log2_midp_portable", true, false, log2_portable),
                ("exp2_midp", false, true, exp2_plain),
                ("exp2_midp_portable", true, true, exp2_portable),
                ("pow_midp(2.4)", false, false, pow_plain),
                ("pow_midp_portable(2.4)", true, false, pow_portable),
            ];
        }
    };
}

#[cfg(target_arch = "x86_64")]
kernels!(v3_f32x8, f32x8, X64V3Token, 8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
kernels!(v4_f32x16, f32x16, X64V4Token, 16);
#[cfg(target_arch = "aarch64")]
kernels!(neon_f32x8, f32x8, NeonToken, 8);
kernels!(scalar_f32x8, f32x8, ScalarToken, 8);

/// `N` vectors of 16 lanes' worth of inputs, flat: positive values over 40
/// octaves for the logarithms and `pow`, and [-30, 30] for `exp2`.
fn inputs() -> (Vec<f32>, Vec<f32>) {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let logs = (0..N * 16)
        .map(|_| 2f32.powf(next() * 40.0 - 20.0))
        .collect();
    let exps = (0..N * 16).map(|_| next() * 60.0 - 30.0).collect();
    (logs, exps)
}

fn lanes<const L: usize>(flat: &[f32]) -> Vec<[f32; L]> {
    flat.as_chunks::<L>().0.iter().take(N).copied().collect()
}

/// The scalar backend's portable outputs for `flat`, as bits.
fn scalar_bits(flat: &[f32], exp: bool, name: &str) -> Vec<u32> {
    let x = flat.as_chunks::<8>().0.to_vec();
    let k = scalar_f32x8::ALL
        .iter()
        .find(|(n, portable, e, _)| *portable && *e == exp && *n == name)
        .unwrap()
        .3;
    let mut out = vec![[0f32; 8]; x.len()];
    k(archmage::ScalarToken, &x, &mut out);
    out.iter().flatten().map(|f| f.to_bits()).collect()
}

/// Checks a group's portable kernels against the scalar backend, then
/// registers the group.
macro_rules! group {
    ($suite:expr, $label:expr, $module:ident, $token:expr, $lanes:literal) => {{
        let t = $token;
        let (logs, exps) = inputs();
        let (logs_l, exps_l) = (lanes::<$lanes>(&logs), lanes::<$lanes>(&exps));
        for &(name, portable, exp, k) in &$module::ALL {
            if !portable {
                continue;
            }
            let x = if exp { &exps_l } else { &logs_l };
            let mut out = vec![[0f32; $lanes]; x.len()];
            k(t, x, &mut out);
            let got: Vec<u32> = out.iter().flatten().map(|f| f.to_bits()).collect();
            let flat = if exp { &exps } else { &logs };
            let want = scalar_bits(&flat[..got.len()], exp, name);
            assert_eq!(
                got, want,
                "{}: {name} differs from the scalar backend",
                $label
            );
        }
        $suite.group($label, |gr| {
            for &(name, _, exp, k) in &$module::ALL {
                let data = if exp { exps_l.clone() } else { logs_l.clone() };
                let mut out = vec![[0f32; $lanes]; data.len()];
                gr.bench(name, move |bench| {
                    bench.iter(|| {
                        k(t, &data, &mut out);
                        out[N - 1][0]
                    })
                });
            }
        });
    }};
}

fn bench_transcendentals(suite: &mut zenbench::prelude::Suite) {
    use archmage::SimdToken;
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::X64V3Token;
        if std::env::var_os("TRANSCENDENTAL_PORTABLE_NO_FMA").is_some() {
            X64V3Token::dangerously_disable_token_process_wide(true)
                .expect("x86-64-v3 is compile-time guaranteed in this build; drop -Ctarget-cpu");
            eprintln!("x86-64-v3 switched off: the scalar backend fuses in software");
        }
        match X64V3Token::summon() {
            Some(t) => group!(suite, "f32x8, AVX2 (V3)", v3_f32x8, t, 8),
            None => eprintln!("skipped the AVX2 group: x86-64-v3 unavailable or switched off"),
        }
        #[cfg(feature = "avx512")]
        match archmage::X64V4Token::summon() {
            Some(t) => group!(suite, "f32x16, AVX-512 (V4)", v4_f32x16, t, 16),
            None => eprintln!("skipped the AVX-512 group: x86-64-v4 unavailable or switched off"),
        }
        let fma = if X64V3Token::summon().is_some() {
            "f32x8, scalar backend (FMA instruction)"
        } else {
            "f32x8, scalar backend (software FMA)"
        };
        group!(suite, fma, scalar_f32x8, archmage::ScalarToken, 8);
    }
    #[cfg(target_arch = "aarch64")]
    {
        match archmage::NeonToken::summon() {
            Some(t) => group!(suite, "f32x8, NEON", neon_f32x8, t, 8),
            None => eprintln!("skipped the NEON group: NEON not detected"),
        }
        group!(
            suite,
            "f32x8, scalar backend (FMA instruction)",
            scalar_f32x8,
            archmage::ScalarToken,
            8
        );
    }
}

zenbench::main!(bench_transcendentals);
