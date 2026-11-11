//! Fixed independent-process qualification matrix. All results are retained.
use archmage::{ScalarToken, SimdToken};
use magemath::recip::{RecipExecution, RecipOptions, RecipValidation};
use zenbench::{GateConfig, black_box};
const MID: u32 = RecipOptions::PORTABLE_MIDP
    .with_execution(RecipExecution::Caller)
    .with_validation(RecipValidation::AssumeValid)
    .bits();
const HIGH: u32 = RecipOptions::HIGHP
    .with_execution(RecipExecution::Caller)
    .bits();
macro_rules! kernels {
    ($op:ident,$stream:ident,$chain:ident,$v:ident,$bound:ident,$w:literal,$parallel:ident,$emulate:ident,$native14:ident,$current:ident) => {
        #[inline(always)]
        fn $op<T: magetypes::simd::backends::$bound, const MODE: u8>(
            t: T,
            x: magetypes::simd::generic::$v<T>,
        ) -> magetypes::simd::generic::$v<T> {
            if MODE == 0 {
                return x.recip();
            }
            if MODE == 1 {
                return crate::seed_graphs::$parallel(t, x);
            }
            if MODE == 2 {
                return magemath::recip::$current::<MID, _>(t, x).unwrap();
            }
            if MODE == 3 {
                return magemath::recip::$current::<HIGH, _>(t, x).unwrap();
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            {
                let any = &t as &dyn core::any::Any;
                if let Some(proof) = any.downcast_ref::<archmage::X64V4Token>() {
                    return magetypes::simd::generic::$v::from_array_t(
                        t,
                        crate::rcp14::native::$native14(*proof, x.to_array()),
                    );
                }
                if let Some(proof) = any.downcast_ref::<archmage::X64V4xToken>() {
                    return magetypes::simd::generic::$v::from_array_t(
                        t,
                        crate::rcp14::native::$native14(proof.v4(), x.to_array()),
                    );
                }
            }
            crate::rcp14::$emulate(t, x)
        }
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, scalar)]
        fn $stream<const MODE: u8, const N: usize>(t: Token, input: &[f32; N], out: &mut [f32; N]) {
            use magetypes::simd::generic::$v as V;
            for (a, b) in input
                .as_chunks::<$w>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<$w>().0.iter_mut())
            {
                *b = $op::<_, MODE>(t, V::from_array_t(t, *a)).to_array();
            }
        }
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, scalar)]
        fn $chain<const MODE: u8>(t: Token, initial: f32) -> [f32; $w] {
            use magetypes::simd::generic::$v as V;
            let mut x = V::splat_t(t, initial);
            for _ in 0..64 {
                x = $op::<_, MODE>(t, x) + V::splat_t(t, 0.5);
            }
            x.to_array()
        }
    };
}
kernels!(
    op4,
    stream4,
    chain4,
    f32x4,
    F32x4Convert,
    4,
    recip4,
    emulate4,
    native4,
    try_recip_f32x4_with
);
kernels!(
    op8,
    stream8,
    chain8,
    f32x8,
    F32x8Convert,
    8,
    recip8,
    emulate8,
    native8,
    try_recip_f32x8_with
);
#[cfg(feature = "w512")]
kernels!(
    op16,
    stream16,
    chain16,
    f32x16,
    F32x16Convert,
    16,
    recip16,
    emulate16,
    native16,
    try_recip_f32x16_with
);
macro_rules! group {
    ($suite:ident,$name:expr,$t:ident,$stream:ident,$chain:ident,$n:literal,$dependent:literal) => {{
        $suite.group($name, |g| {
            g.config().min_rounds(50).max_rounds(50).auto_rounds(false);
            let input: [f32; $n] = core::array::from_fn(|i| {
                let x = 0.25 + ((i % 64 + 1) as f32) * 0.03125;
                if i % 3 == 1 { -x } else { x }
            });
            macro_rules! arm {
                ($label:literal,$mode:literal) => {
                    g.bench($label, move |b| {
                        let mut out = [0.0; $n];
                        b.iter(|| {
                            if $dependent {
                                black_box($chain::<$mode>($t, black_box(0.75)));
                            } else {
                                $stream::<$mode, $n>($t, black_box(&input), black_box(&mut out));
                                black_box(&out);
                            }
                        });
                    });
                };
            }
            arm!("magetypes recip", 0);
            arm!("portable parallel MIDP", 1);
            arm!("portable serial MIDP", 2);
            arm!("portable HIGHP division", 3);
            arm!("canonical RCP14", 4);
        });
    }};
}
macro_rules! width {
    ($suite:ident,$tier:ident,$t:ident,$stream:ident,$chain:ident,$w:literal) => {{
        group!(
            $suite,
            format!("{}/f32x{}/stream64", $tier, $w),
            $t,
            $stream,
            $chain,
            64,
            false
        );
        group!(
            $suite,
            format!("{}/f32x{}/stream4096", $tier, $w),
            $t,
            $stream,
            $chain,
            4096,
            false
        );
        group!(
            $suite,
            format!("{}/f32x{}/chain64", $tier, $w),
            $t,
            $stream,
            $chain,
            64,
            true
        );
    }};
}
macro_rules! matrix {
    ($suite:ident,$tier:ident,$token:expr,$s4:ident,$c4:ident,$s8:ident,$c8:ident,$s16:ident,$c16:ident) => {{
        let t = $token;
        width!($suite, $tier, t, $s4, $c4, 4);
        width!($suite, $tier, t, $s8, $c8, 8);
        #[cfg(feature = "w512")]
        width!($suite, $tier, t, $s16, $c16, 16);
    }};
}
pub(crate) fn run() {
    let args: Vec<_> = std::env::args().collect();
    let tier = args
        .iter()
        .position(|a| a == "--qual-tier")
        .and_then(|i| args.get(i + 1))
        .expect("--qual-tier required")
        .clone();
    let exploratory = args.iter().any(|a| a == "--exploratory-macos");
    assert!(
        !exploratory || cfg!(target_os = "macos"),
        "exploratory mode is only for the explicitly unconfined macOS study"
    );
    let mut gate = GateConfig::strict();
    // macOS cannot enforce heterogeneous core classes. Retain all monitoring and
    // dirty rounds, but classify these user-authorized timings as exploratory.
    if exploratory {
        gate.strict = false;
    }
    let result = zenbench::run_gated(gate, |suite| {
        let plan = suite.plan();
        plan.mode = if exploratory {
            zenbench::BenchmarkMode::Iterate
        } else {
            zenbench::BenchmarkMode::Confirm
        };
        plan.processes = if exploratory { 1 } else { 5 };
        plan.rounds = Some(50);
        plan.margin_pct = 2.0;
        plan.confidence = 0.95;
        plan.workload_id =
            Some("recip-normal-signed-period64-stream64-stream4096-positive-chain64-v1".into());
        plan.scope = Some(format!("recip-qualification-{tier}"));
        match tier.as_str() {
            "scalar" => matrix!(
                suite,
                tier,
                ScalarToken,
                stream4_scalar,
                chain4_scalar,
                stream8_scalar,
                chain8_scalar,
                stream16_scalar,
                chain16_scalar
            ),
            #[cfg(target_arch = "x86_64")]
            "v3" => matrix!(
                suite,
                tier,
                archmage::X64V3Token::summon().expect("V3 required"),
                stream4_v3,
                chain4_v3,
                stream8_v3,
                chain8_v3,
                stream16_v3,
                chain16_v3
            ),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            "v4" => matrix!(
                suite,
                tier,
                archmage::X64V4Token::summon().expect("V4 required"),
                stream4_v4,
                chain4_v4,
                stream8_v4,
                chain8_v4,
                stream16_v4,
                chain16_v4
            ),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            "v4x" => matrix!(
                suite,
                tier,
                archmage::X64V4xToken::summon().expect("V4x required"),
                stream4_v4x,
                chain4_v4x,
                stream8_v4x,
                chain8_v4x,
                stream16_v4x,
                chain16_v4x
            ),
            #[cfg(target_arch = "aarch64")]
            "neon" => matrix!(
                suite,
                tier,
                archmage::NeonToken::summon().expect("NEON required"),
                stream4_neon,
                chain4_neon,
                stream8_neon,
                chain8_neon,
                stream16_neon,
                chain16_neon
            ),
            _ => panic!("qualification tier unavailable"),
        }
    });
    if std::env::var("ZENBENCH_SUBPROCESS").as_deref() == Ok("1") {
        return;
    }
    result
        .save(std::env::var("ARCHMAGE_PRIMITIVE_RESULTS").expect("durable result path"))
        .unwrap();
    result.print_report();
    if exploratory {
        eprintln!("EXPLORATORY: macOS scheduler chooses core class; excluded from promotion gate");
        return;
    }
    let issues = result.confirmation_issues();
    assert!(
        issues.is_empty(),
        "qualification measurement invalid: {issues:?}"
    );
}
