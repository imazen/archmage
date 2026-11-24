//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Test token slots and transitive cascade descendant indices derived from
//! the token registry DAG.

use crate::testing::TokenSlot;

pub(crate) fn build_token_slots() -> Vec<TokenSlot> {
    #[allow(unused_mut)]
    let mut slots = Vec::new();

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        slots.push(TokenSlot::new::<crate::X64V1Token>(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9],
            crate::X64V1Token::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::X64V2Token>(
            &[2, 3, 4, 5, 6, 7, 8, 9],
            crate::X64V2Token::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::X64CryptoToken>(
            &[4, 5, 6, 7, 8, 9],
            crate::X64CryptoToken::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::X64V3Token>(
            &[4, 5, 6, 7, 8, 9],
            crate::X64V3Token::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::X64V3CryptoToken>(
            &[5, 6, 8],
            crate::X64V3CryptoToken::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::X64V3GfniCryptoToken>(
            &[6, 8],
            crate::X64V3GfniCryptoToken::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::X64V3VnniToken>(
            &[],
            crate::X64V3VnniToken::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::X64V4Token>(
            &[8, 9],
            crate::X64V4Token::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::X64V4xToken>(
            &[],
            crate::X64V4xToken::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::Avx512Fp16Token>(
            &[],
            crate::Avx512Fp16Token::dangerously_disable_token_process_wide,
        ));
    }

    #[cfg(target_arch = "aarch64")]
    {
        slots.push(TokenSlot::new::<crate::NeonToken>(
            &[1, 2, 3, 4, 5],
            crate::NeonToken::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::NeonAesToken>(
            &[],
            crate::NeonAesToken::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::NeonSha3Token>(
            &[],
            crate::NeonSha3Token::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::NeonCrcToken>(
            &[],
            crate::NeonCrcToken::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::Arm64V2Token>(
            &[5],
            crate::Arm64V2Token::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::Arm64V3Token>(
            &[],
            crate::Arm64V3Token::dangerously_disable_token_process_wide,
        ));
    }

    #[cfg(target_arch = "wasm32")]
    {
        slots.push(TokenSlot::new::<crate::Wasm128Token>(
            &[1],
            crate::Wasm128Token::dangerously_disable_token_process_wide,
        ));
        slots.push(TokenSlot::new::<crate::Wasm128RelaxedToken>(
            &[],
            crate::Wasm128RelaxedToken::dangerously_disable_token_process_wide,
        ));
    }

    // ScalarToken is always available and does not contribute to permutations.
    slots
}
