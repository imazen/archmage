//! Shared scalar/vector/tail verifier. Cases are split by numerical family
//! so debug WASM compilation does not materialize all 64 graphs in one crate.
#![allow(dead_code)]
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
use magetypes::powf::{PowfDomain, PowfExecution, PowfOptions, PowfPortability, try_powf_with};
const CALLER: u32 = PowfOptions::new()
    .with_execution(PowfExecution::Caller)
    .bits();
const BACKEND: u32 = PowfOptions::new()
    .with_portability(PowfPortability::BackendDependent)
    .bits();
const POSITIVE: u32 = PowfOptions::new()
    .with_domain(PowfDomain::PositiveGamma)
    .bits()
    & !3;
// A feature-context tail must use the same graph as the packed body. In
// particular BackendDependent Caller must retain its token rather than switch
// to ScalarToken (which intentionally uses separate arithmetic).
#[archmage::magetypes(v4x(cfg(avx512)), v3, neon, wasm128, scalar)]
fn bounded_widths_and_tails(token: Token) {
    #[cfg(target_arch = "wasm32")]
    {
        let a = magetypes::simd::generic::f32x4::splat_t(token, f32::from_bits(0x3f80_0001));
        let c = magetypes::simd::generic::f32x4::splat_t(token, f32::from_bits(0xbf80_0002));
        let bits = a.mul_add_relaxed(a, c).to_array()[0].to_bits();
        assert!(bits == 0 || bits == 0x2880_0000);
        std::println!(
            "powf-tail-token={} relaxed-f32-madd-bits={bits:08x}",
            <Token as archmage::SimdToken>::NAME
        );
    }
    use magetypes::simd::generic::{f32x4, f32x8};
    // Keep each graph in one test frame. The production API intentionally
    // forces inlining; duplicating it inside every verifier loop costs gigabytes
    // of WASM debug compiler memory without adding coverage.
    #[inline(never)]
    fn eval4<const FLAGS: u32>(token: Token, x: [f32; 4], n: f32) -> [f32; 4] {
        f32x4::from_array_t(token, x)
            .try_powf_with::<FLAGS>(n)
            .unwrap()
            .to_array()
    }
    #[inline(never)]
    fn eval8<const FLAGS: u32>(token: Token, x: [f32; 8], n: f32) -> [f32; 8] {
        f32x8::from_array_t(token, x)
            .try_powf_with::<FLAGS>(n)
            .unwrap()
            .to_array()
    }
    #[cfg(feature = "w512")]
    #[inline(never)]
    fn eval16<const FLAGS: u32>(token: Token, x: [f32; 16], n: f32) -> [f32; 16] {
        magetypes::simd::generic::f32x16::from_array_t(token, x)
            .try_powf_with::<FLAGS>(n)
            .unwrap()
            .to_array()
    }
    #[inline(never)]
    fn eval_scalar<const FLAGS: u32>(x: f32, n: f32) -> f32 {
        try_powf_with::<FLAGS>(x, n).unwrap()
    }
    #[inline(never)]
    fn eval_tail<const FLAGS: u32>(token: Token, x: f32, n: f32) -> f32 {
        magetypes::powf::try_powf_with_token::<FLAGS, _>(token, x, n).unwrap()
    }
    fn scalar_bits<const FLAGS: u32>(token: Token, x: f32, n: f32) -> u32 {
        let result = eval_tail::<FLAGS>(token, x, n);
        assert_eq!(
            result.to_bits(),
            eval4::<FLAGS>(token, [x; 4], n)[0].to_bits()
        );
        if FLAGS & CALLER == 0 || FLAGS & BACKEND == 0 {
            assert_eq!(eval_scalar::<FLAGS>(x, n).to_bits(), result.to_bits());
        }
        result.to_bits()
    }
    fn check<const FLAGS: u32>(token: Token) {
        let mut state = 0x9e37_79b9_u32;
        for n in [0.0_f32, -0.0, 0.5, 2.4, 4.0] {
            let input: [f32; 513] = core::array::from_fn(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if FLAGS & 3 != 0 && FLAGS & POSITIVE == 0 && i % 17 == 0 {
                    0.0
                } else {
                    f32::from_bits(0x3780_0000 + state % 0x1000_0001)
                }
            });
            macro_rules! width {
                ($eval:ident, $lanes:literal) => {
                    for length in (0..=33).chain([64, 65, 127, 255, 511, 512, 513]) {
                        let mut chunks = input[..length].chunks_exact($lanes);
                        let mut offset = 0;
                        for chunk in chunks.by_ref() {
                            let result = $eval::<FLAGS>(token, chunk.try_into().unwrap(), n)
                                .map(f32::to_bits);
                            // Compare each pack immediately; retain no full
                            // reference-output buffer for this policy/width.
                            for (&x, actual) in chunk.iter().zip(result) {
                                assert_eq!(actual, scalar_bits::<FLAGS>(token, x, n));
                            }
                            offset += $lanes;
                        }
                        for &x in chunks.remainder() {
                            let result = eval_tail::<FLAGS>(token, x, n);
                            assert_eq!(result.to_bits(), scalar_bits::<FLAGS>(token, x, n));
                            offset += 1;
                        }
                        assert_eq!(offset, length);
                    }
                };
            }
            width!(eval4, 4);
            width!(eval8, 8);
            #[cfg(feature = "w512")]
            {
                width!(eval16, 16);
            }
        }
    }
    cases!(check, token);
}

#[test]
fn single_vector_and_all_tail_remainders_match() {
    bounded_widths_and_tails_scalar(archmage::ScalarToken);
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = archmage::X64V3Token::summon() {
            bounded_widths_and_tails_v3(t);
        }
        #[cfg(feature = "avx512")]
        if let Some(t) = archmage::X64V4xToken::summon() {
            bounded_widths_and_tails_v4x(t);
        }
    }
    #[cfg(target_arch = "aarch64")]
    bounded_widths_and_tails_neon(archmage::NeonToken::summon().unwrap());
    #[cfg(target_arch = "wasm32")]
    bounded_widths_and_tails_wasm128(archmage::Wasm128Token::summon().unwrap());
}
