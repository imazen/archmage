//! # Why `forge_token_dangerously()` is deprecated and dangerous
//!
//! This example demonstrates the UB that archmage's token system prevents.
//!
//! ## The safe path
//!
//! `summon()` verifies CPU features at runtime before creating a token.
//! If the CPU lacks the required features, it returns `None`. Safe, correct.
//!
//! ## The dangerous path
//!
//! `forge_token_dangerously()` creates the token unconditionally. No feature
//! check. If the CPU doesn't have the required features, any `#[arcane]`
//! function using those intrinsics will execute illegal instructions.
//!
//! ## Consequences
//!
//! On real hardware without AVX2: **SIGILL** (illegal instruction signal,
//! immediate process termination).
//!
//! Under Miri: the `#[target_feature]` attribute tells the interpreter the
//! features are available, so Miri won't catch this specific class of UB.
//! The counterexample documents this limitation.
//!
//! ## What `#[arcane]` generates
//!
//! ```rust,ignore
//! // You write:
//! #[arcane]
//! fn my_kernel(token: X64V3Token, data: &[f32; 8]) -> f32 { ... }
//!
//! // Macro generates:
//! fn my_kernel(token: X64V3Token, data: &[f32; 8]) -> f32 {
//!     #[target_feature(enable = "avx2,fma,bmi1,bmi2,f16c,lzcnt,...")]
//!     fn inner(data: &[f32; 8]) -> f32 { ... }
//!     // SAFETY: Token proves CPU support was verified via summon().
//!     unsafe { inner(data) }
//! }
//! ```
//!
//! The token's existence is the ONLY proof that the `unsafe` call is sound.
//! Forge the token → the proof is fake → unsound.

use archmage::{SimdToken, X64V3Token};

fn main() {
    println!("=== archmage safety counterexample ===\n");

    // === SAFE: summon() verifies features ===
    println!("1. Safe path: X64V3Token::summon()");
    match X64V3Token::summon() {
        Some(_token) => {
            println!("   AVX2+FMA confirmed by runtime detection.");
            println!("   Safe to call any #[arcane] function with this token.\n");
        }
        None => {
            println!("   CPU lacks AVX2+FMA. summon() returned None.");
            println!("   Graceful fallback — no crash, no UB.\n");
        }
    }

    // === DANGEROUS: forge bypasses all checks ===
    println!("2. Dangerous path: forge_token_dangerously()");
    println!("   This creates a token WITHOUT checking CPU features.");
    println!("   The token is a lie — it claims AVX2+FMA support");
    println!("   even if the CPU has neither.\n");

    #[allow(deprecated, unused_variables)]
    let forged = unsafe { X64V3Token::forge_token_dangerously() };
    // `forged` exists even if CPU has no AVX2!

    println!("   Token forged. On a CPU without AVX2, calling any");
    println!("   #[arcane] function with this token would cause:");
    println!("   - SIGILL (illegal instruction) on real hardware");
    println!("   - Silent undefined behavior in the general case\n");

    // We do NOT actually call an #[arcane] function with the forged token.
    // That would be UB on CPUs without AVX2.

    println!("3. What #[arcane] does under the hood:");
    println!("   #[target_feature(enable = \"avx2,fma,...\")]");
    println!("   fn inner(...) {{ /* AVX2 intrinsics here */ }}");
    println!("   unsafe {{ inner(...) }}  // <-- THIS is the unsafe call");
    println!();
    println!("   The token proves the unsafe call is sound.");
    println!("   summon() → real proof. forge() → fake proof.\n");

    println!("4. Feature information from the token:");
    println!("   NAME:              {}", X64V3Token::NAME);
    println!("   TARGET_FEATURES:   {}", X64V3Token::TARGET_FEATURES);
    println!(
        "   ENABLE_FLAGS:      {}",
        X64V3Token::ENABLE_TARGET_FEATURES
    );
    println!(
        "   DISABLE_FLAGS:     {}",
        X64V3Token::DISABLE_TARGET_FEATURES
    );
    println!();

    println!("=== End of counterexample ===");
    println!("In real code: always use summon(), never forge.");
}
