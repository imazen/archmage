//! On x86_64 with std, portable f64 FMA checks archmage's cached V3 token at
//! run time and uses the FMA instruction when the CPU has it, so on CI hosts
//! the software path is not reached through the public API. This binary
//! disables the V3 token process-wide (its own process, so no other test sees
//! it) and checks the software path against std's correctly rounded `mul_add`.
#![cfg(target_arch = "x86_64")]
#![forbid(unsafe_code)]
#[path = "common/fma_expected.rs"]
#[allow(dead_code)] // shared oracle; this binary uses only `fused_expected`
mod fma_expected;
use fma_expected::FmaExpected;

fn same64(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{actual:?} != {expected:?}"
        );
    }
}

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Random sign and significand, binary exponent uniform in `-span..=span`.
fn normal64(state: &mut u64, span: u64) -> f64 {
    let r = random(state);
    let exponent = 1023 + (r % (2 * span + 1)) - span;
    f64::from_bits(exponent << 52 | random(state) >> 12 | r & 1 << 63)
}

#[test]
fn software_f64_path_with_v3_disabled() {
    use archmage::{ScalarToken, SimdToken, X64V3Token};
    use magetypes::simd::generic::f64x4;
    if X64V3Token::dangerously_disable_token_process_wide(true).is_err() {
        // The build guarantees V3 (`-Ctarget-cpu`), so the hardware path is
        // compile-time and the software path is unreachable here by design.
        assert_eq!(X64V3Token::compiled_with(), Some(true));
        return;
    }
    assert!(X64V3Token::summon().is_none());
    let mut state = 0x2545_f491_4f6c_dd1d;
    let n = if cfg!(miri) { 200 } else { 200_000 };
    for _ in 0..n {
        let bits: [[f64; 3]; 4] = core::array::from_fn(|i| match i {
            0 => core::array::from_fn(|_| f64::from_bits(random(&mut state))),
            _ => {
                let (a, b) = (normal64(&mut state, 60), normal64(&mut state, 60));
                [
                    a,
                    b,
                    if i == 2 {
                        -(a * b)
                    } else {
                        normal64(&mut state, 130)
                    },
                ]
            }
        });
        for [a, b, c] in bits {
            same64(magetypes::nostd_math::fma(a, b, c), a.fused_expected(b, c));
        }
        let t = ScalarToken;
        let lanes = |k: usize| core::array::from_fn::<f64, 4, _>(|i| bits[i][k]);
        let got = f64x4::from_array_t(t, lanes(0))
            .mul_add_portable(
                f64x4::from_array_t(t, lanes(1)),
                f64x4::from_array_t(t, lanes(2)),
            )
            .to_array();
        for i in 0..4 {
            same64(got[i], bits[i][0].fused_expected(bits[i][1], bits[i][2]));
        }
    }
}
