//! Keep the process-wide token changes in a dedicated test binary.
#![forbid(unsafe_code)]
#![cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]

use archmage::{SimdToken, Wasm128RelaxedToken, Wasm128Token};
use magemath::cbrtf::{Cbrtf, CbrtfOptions};

#[test]
fn bit_exact_slices_with_relaxed_token_disabled() {
    const OPTS: CbrtfOptions = CbrtfOptions::new().caller();
    const EXPECTED: u32 = 0x3f89_e393;
    let token = Wasm128Token::summon().expect("SIMD128 is enabled");
    let input = [f32::from_bits(0x3fa0_0497); 17];

    for disabled in [false, true, false] {
        Wasm128RelaxedToken::dangerously_disable_token_process_wide(disabled).unwrap();
        assert_eq!(Wasm128RelaxedToken::summon().is_none(), disabled);
        assert!(Wasm128Token::summon().is_some());

        for len in [2, 4, 17] {
            let src = &input[..len];
            let mut const_out = [0.0; 17];
            let mut runtime_out = [0.0; 17];
            Cbrtf::<{ OPTS.bits() }>::slice_with_token(token, src, &mut const_out[..len]).unwrap();
            OPTS.runtime()
                .slice_with_token(token, src, &mut runtime_out[..len])
                .unwrap();

            for (entry, out) in [("const", const_out), ("runtime", runtime_out)] {
                for (lane, &value) in out[..len].iter().enumerate() {
                    assert_eq!(
                        value.to_bits(),
                        EXPECTED,
                        "{entry} len={len} lane={lane} relaxed_disabled={disabled}"
                    );
                }
            }
        }
    }
}
