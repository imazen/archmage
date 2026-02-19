//! Detect and report all AArch64 SIMD features available on the current CPU.
//!
//! Run with: cargo run --example detect_features
//!
//! On x86_64, reports x86 features instead.

use archmage::SimdToken;

fn main() {
    println!("=== CPU Feature Detection ===");
    println!("arch: {}", std::env::consts::ARCH);
    println!("os: {}", std::env::consts::OS);
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
    use std::arch::is_aarch64_feature_detected;

    println!("--- AArch64 SIMD Features ---");

    // Baseline
    report("neon", is_aarch64_feature_detected!("neon"));

    // Compute extensions (candidates for archmage tokens)
    println!();
    println!("--- Compute Extensions ---");
    report("dotprod", is_aarch64_feature_detected!("dotprod"));
    report("rdm", is_aarch64_feature_detected!("rdm"));
    report("fp16", is_aarch64_feature_detected!("fp16"));
    report("fhm", is_aarch64_feature_detected!("fhm"));
    report("fcma", is_aarch64_feature_detected!("fcma"));
    report("i8mm", is_aarch64_feature_detected!("i8mm"));
    report("bf16", is_aarch64_feature_detected!("bf16"));

    // Crypto
    println!();
    println!("--- Crypto Extensions ---");
    report("aes", is_aarch64_feature_detected!("aes"));
    report("sha2", is_aarch64_feature_detected!("sha2"));
    report("sha3", is_aarch64_feature_detected!("sha3"));
    report("sm4", is_aarch64_feature_detected!("sm4"));
    report("crc", is_aarch64_feature_detected!("crc"));

    // SVE (not usable for intrinsics on stable, but detect anyway)
    println!();
    println!("--- Scalable Vector Extensions ---");
    report("sve", is_aarch64_feature_detected!("sve"));
    report("sve2", is_aarch64_feature_detected!("sve2"));

    // System features
    println!();
    println!("--- System Features ---");
    report("lse", is_aarch64_feature_detected!("lse"));
    report("rcpc", is_aarch64_feature_detected!("rcpc"));
    report("rcpc2", is_aarch64_feature_detected!("rcpc2"));
    report("dpb", is_aarch64_feature_detected!("dpb"));
    report("dpb2", is_aarch64_feature_detected!("dpb2"));
    report("dit", is_aarch64_feature_detected!("dit"));
    report("flagm", is_aarch64_feature_detected!("flagm"));
    report("ssbs", is_aarch64_feature_detected!("ssbs"));
    report("bti", is_aarch64_feature_detected!("bti"));
    report("frintts", is_aarch64_feature_detected!("frintts"));
    report("jsconv", is_aarch64_feature_detected!("jsconv"));
    report("paca", is_aarch64_feature_detected!("paca"));
    report("pacg", is_aarch64_feature_detected!("pacg"));
    report("mte", is_aarch64_feature_detected!("mte"));
    report("rand", is_aarch64_feature_detected!("rand"));

    // Archmage tokens
    println!();
    println!("--- Archmage Token Availability ---");
    report_token("NeonToken", archmage::NeonToken::summon().is_some());
    report_token("Arm64V2Token", archmage::Arm64V2Token::summon().is_some());
    report_token("Arm64V3Token", archmage::Arm64V3Token::summon().is_some());
    report_token("NeonAesToken", archmage::NeonAesToken::summon().is_some());
    report_token("NeonSha3Token", archmage::NeonSha3Token::summon().is_some());
    report_token("NeonCrcToken", archmage::NeonCrcToken::summon().is_some());

    // Summary line for CI parsing
    println!();
    println!("--- Summary (tab-separated for CI) ---");
    let features = [
        ("neon", is_aarch64_feature_detected!("neon")),
        ("dotprod", is_aarch64_feature_detected!("dotprod")),
        ("rdm", is_aarch64_feature_detected!("rdm")),
        ("fp16", is_aarch64_feature_detected!("fp16")),
        ("fhm", is_aarch64_feature_detected!("fhm")),
        ("fcma", is_aarch64_feature_detected!("fcma")),
        ("i8mm", is_aarch64_feature_detected!("i8mm")),
        ("bf16", is_aarch64_feature_detected!("bf16")),
        ("aes", is_aarch64_feature_detected!("aes")),
        ("sha2", is_aarch64_feature_detected!("sha2")),
        ("sha3", is_aarch64_feature_detected!("sha3")),
        ("sm4", is_aarch64_feature_detected!("sm4")),
        ("crc", is_aarch64_feature_detected!("crc")),
        ("sve", is_aarch64_feature_detected!("sve")),
        ("sve2", is_aarch64_feature_detected!("sve2")),
        ("lse", is_aarch64_feature_detected!("lse")),
    ];
    print!("FEATURES\t");
    for (name, _) in &features {
        print!("{name}\t");
    }
    println!();
    print!("DETECTED\t");
    for (_, detected) in &features {
        print!("{}\t", if *detected { "YES" } else { "no" });
    }
    println!();
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

fn report(name: &str, detected: bool) {
    let icon = if detected { "[+]" } else { "[ ]" };
    println!("  {icon} {name}");
}

fn report_token(name: &str, available: bool) {
    let icon = if available { "[+]" } else { "[ ]" };
    println!("  {icon} {name}::summon()");
}
