//! Detect and report all SIMD features available on the current CPU.
//!
//! Run with: cargo run --example detect_features
//!
//! Compares three detection methods on AArch64:
//! 1. `cfg!(target_feature)` — compile-time, affected by -Ctarget-cpu=native
//! 2. `std::arch::is_aarch64_feature_detected!` — runtime CPUID
//! 3. `archmage::is_aarch64_feature_available!` — archmage's combined macro
//!
//! Discrepancies between methods indicate bugs in archmage's detection.

use archmage::SimdToken;

fn main() {
    println!("=== CPU Feature Detection ===");
    println!("arch: {}", std::env::consts::ARCH);
    println!("os: {}", std::env::consts::OS);
    println!("testable_dispatch: {}", cfg!(feature = "testable_dispatch"));
    println!();

    #[cfg(target_arch = "aarch64")]
    detect_aarch64();

    #[cfg(target_arch = "x86_64")]
    detect_x86_64();

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    println!("(no feature detection for this architecture)");
}

#[cfg(target_arch = "aarch64")]
fn detect_aarch64() {
    use archmage::is_aarch64_feature_available;
    use std::arch::is_aarch64_feature_detected;

    // Compare all three detection methods for Arm64V2 features
    println!("--- Arm64V2 Features: Detection Method Comparison ---");
    println!(
        "  {:10} {:>10} {:>10} {:>10}",
        "feature", "cfg!", "std", "archmage"
    );

    macro_rules! compare_feature {
        ($name:literal) => {
            let cfg_val = cfg!(target_feature = $name);
            let std_val = is_aarch64_feature_detected!($name);
            let am_val = is_aarch64_feature_available!($name);
            let mismatch = if cfg_val != std_val || std_val != am_val {
                " *** MISMATCH ***"
            } else {
                ""
            };
            println!(
                "  {:10} {:>10} {:>10} {:>10}{}",
                $name,
                yn(cfg_val),
                yn(std_val),
                yn(am_val),
                mismatch
            );
        };
    }

    compare_feature!("neon");
    compare_feature!("crc");
    compare_feature!("rdm");
    compare_feature!("dotprod");
    compare_feature!("fp16");
    compare_feature!("aes");
    compare_feature!("sha2");

    // Additional V3 features
    println!();
    println!("--- Arm64V3 Additional Features ---");
    println!(
        "  {:10} {:>10} {:>10} {:>10}",
        "feature", "cfg!", "std", "archmage"
    );
    compare_feature!("fhm");
    compare_feature!("fcma");
    compare_feature!("sha3");
    compare_feature!("i8mm");
    compare_feature!("bf16");

    // Token compiled_with status
    println!();
    println!("--- Token compiled_with() ---");
    println!("  NeonToken:    {:?}", archmage::NeonToken::compiled_with());
    println!(
        "  Arm64V2Token: {:?}",
        archmage::Arm64V2Token::compiled_with()
    );
    println!(
        "  Arm64V3Token: {:?}",
        archmage::Arm64V3Token::compiled_with()
    );

    // Token summon results
    println!();
    println!("--- Archmage Token Availability ---");
    report_token("NeonToken", archmage::NeonToken::summon().is_some());
    report_token("NeonAesToken", archmage::NeonAesToken::summon().is_some());
    report_token("NeonSha3Token", archmage::NeonSha3Token::summon().is_some());
    report_token("NeonCrcToken", archmage::NeonCrcToken::summon().is_some());
    report_token("Arm64V2Token", archmage::Arm64V2Token::summon().is_some());
    report_token("Arm64V3Token", archmage::Arm64V3Token::summon().is_some());

    // Additional raw feature detection
    println!();
    println!("--- Other Features (std detection) ---");
    report("sve", is_aarch64_feature_detected!("sve"));
    report("sve2", is_aarch64_feature_detected!("sve2"));
    report("lse", is_aarch64_feature_detected!("lse"));
    report("sm4", is_aarch64_feature_detected!("sm4"));
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_64() {
    use std::arch::is_x86_feature_detected;

    println!("--- x86_64 SIMD Features ---");
    report("sse2", is_x86_feature_detected!("sse2"));
    report("sse4.2", is_x86_feature_detected!("sse4.2"));
    report("popcnt", is_x86_feature_detected!("popcnt"));
    report("avx2", is_x86_feature_detected!("avx2"));
    report("fma", is_x86_feature_detected!("fma"));
    report("avx512f", is_x86_feature_detected!("avx512f"));
    report("avx512bw", is_x86_feature_detected!("avx512bw"));
    report("avx512vl", is_x86_feature_detected!("avx512vl"));

    println!();
    println!("--- Archmage Token Availability ---");
    report_token("X64V1Token", archmage::X64V1Token::summon().is_some());
    report_token("X64V2Token", archmage::X64V2Token::summon().is_some());
    report_token("X64V3Token", archmage::X64V3Token::summon().is_some());
}

#[cfg(target_arch = "aarch64")]
fn yn(b: bool) -> &'static str {
    if b { "YES" } else { "no" }
}

fn report(name: &str, detected: bool) {
    let icon = if detected { "[+]" } else { "[ ]" };
    println!("  {icon} {name}");
}

fn report_token(name: &str, available: bool) {
    let icon = if available { "[+]" } else { "[ ]" };
    println!("  {icon} {name}::summon()");
}
