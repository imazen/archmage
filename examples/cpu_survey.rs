//! Comprehensive CPU feature survey.
//!
//! Reports everything about the current CPU: model name, architecture, OS,
//! archmage token availability, and full feature detection.
//!
//! Run with: cargo run --example cpu_survey --features "avx512"
//!
//! Unlike `detect_features.rs` (which compares detection methods for mismatch
//! debugging), this is a pure survey tool — it reports what's available on a
//! given machine without comparing or diagnosing.

use archmage::SimdToken;

fn main() {
    println!("╔══════════════════════════════════════════╗");
    println!("║         ARCHMAGE CPU SURVEY              ║");
    println!("╚══════════════════════════════════════════╝");
    println!();

    // ── CPU Identification ──────────────────────────────────────────────
    println!("── CPU Identification ──");
    println!("  arch:     {}", std::env::consts::ARCH);
    println!("  os:       {}", std::env::consts::OS);
    println!("  cpu:      {}", cpu_model_name());
    println!();

    // ── Archmage Token Availability ─────────────────────────────────────
    println!("── Archmage Token Availability ──");
    println!();

    println!("  x86 tokens:");
    report_token::<archmage::X64V1Token>("X64V1Token", "Sse2Token");
    report_token::<archmage::X64V2Token>("X64V2Token", "");
    report_token::<archmage::X64CryptoToken>("X64CryptoToken", "");
    report_token::<archmage::X64V3Token>("X64V3Token", "Desktop64");
    report_token::<archmage::X64V3CryptoToken>("X64V3CryptoToken", "");
    report_token::<archmage::X64V4Token>("X64V4Token", "Avx512Token, Server64");
    report_token::<archmage::X64V4xToken>("X64V4xToken", "");
    report_token::<archmage::Avx512Fp16Token>("Avx512Fp16Token", "");
    println!();

    println!("  ARM tokens:");
    report_token::<archmage::NeonToken>("NeonToken", "Arm64");
    report_token::<archmage::NeonAesToken>("NeonAesToken", "");
    report_token::<archmage::NeonSha3Token>("NeonSha3Token", "");
    report_token::<archmage::NeonCrcToken>("NeonCrcToken", "");
    report_token::<archmage::Arm64V2Token>("Arm64V2Token", "");
    report_token::<archmage::Arm64V3Token>("Arm64V3Token", "");
    println!();

    println!("  WASM tokens:");
    report_token::<archmage::Wasm128Token>("Wasm128Token", "");
    report_token::<archmage::Wasm128RelaxedToken>("Wasm128RelaxedToken", "");
    println!();

    println!("  Universal:");
    report_token::<archmage::ScalarToken>("ScalarToken", "");
    println!();

    // ── Full Feature Detection ──────────────────────────────────────────
    #[cfg(target_arch = "x86_64")]
    survey_x86_64();

    #[cfg(target_arch = "aarch64")]
    survey_aarch64();

    #[cfg(target_arch = "wasm32")]
    println!("  (WASM feature detection is compile-time only)");

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    println!("  (no feature detection for this architecture)");
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU model name detection
// ═══════════════════════════════════════════════════════════════════════════

fn cpu_model_name() -> String {
    #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "x86")))]
    {
        if let Some(name) = read_proc_cpuinfo_model() {
            return name;
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        if let Some(name) = read_proc_cpuinfo_arm() {
            return name;
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(name) = sysctl_cpu_name() {
            return name;
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(name) = windows_cpu_name() {
            return name;
        }
    }

    "Unknown".to_string()
}

#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "x86")))]
fn read_proc_cpuinfo_model() -> Option<String> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in content.lines() {
        if let Some(value) = line.strip_prefix("model name") {
            let value = value.trim_start_matches([' ', '\t', ':']);
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
fn read_proc_cpuinfo_arm() -> Option<String> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut implementer = None;
    let mut part = None;
    for line in content.lines() {
        if let Some(value) = line.strip_prefix("CPU implementer") {
            let value = value.trim_start_matches(|c: char| c == ' ' || c == '\t' || c == ':');
            implementer = Some(value.to_string());
        }
        if let Some(value) = line.strip_prefix("CPU part") {
            let value = value.trim_start_matches(|c: char| c == ' ' || c == '\t' || c == ':');
            part = Some(value.to_string());
        }
        // Return after first core's info
        if implementer.is_some() && part.is_some() {
            break;
        }
    }

    match (implementer, part) {
        (Some(imp), Some(prt)) => {
            let vendor = match imp.as_str() {
                "0x41" => "ARM",
                "0x42" => "Broadcom",
                "0x43" => "Cavium",
                "0x44" => "DEC",
                "0x46" => "Fujitsu",
                "0x48" => "HiSilicon",
                "0x4e" => "NVIDIA",
                "0x50" => "APM",
                "0x51" => "Qualcomm",
                "0x53" => "Samsung",
                "0x56" => "Marvell",
                "0x61" => "Apple",
                "0x63" => "Intel (Arm)",
                "0x6d" => "Microsoft",
                "0xc0" => "Ampere",
                _ => &imp,
            };
            Some(format!("{vendor} (part {prt})"))
        }
        _ => {
            // Fallback: try "Hardware" or "model name" lines
            let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
            for line in content.lines() {
                for prefix in &["Hardware", "model name"] {
                    if let Some(value) = line.strip_prefix(prefix) {
                        let value =
                            value.trim_start_matches(|c: char| c == ' ' || c == '\t' || c == ':');
                        if !value.is_empty() {
                            return Some(value.to_string());
                        }
                    }
                }
            }
            None
        }
    }
}

#[cfg(target_os = "macos")]
fn sysctl_cpu_name() -> Option<String> {
    // Try Apple Silicon chip name first
    if let Ok(output) = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
    {
        let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !name.is_empty() {
            return Some(name);
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn windows_cpu_name() -> Option<String> {
    // Try PROCESSOR_IDENTIFIER environment variable
    if let Ok(name) = std::env::var("PROCESSOR_IDENTIFIER") {
        if !name.is_empty() {
            return Some(name);
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Token reporting
// ═══════════════════════════════════════════════════════════════════════════

fn report_token<T: SimdToken>(name: &str, aliases: &str) {
    let available = T::summon().is_some();
    let icon = if available { "[+]" } else { "[ ]" };
    let display = T::NAME;
    if aliases.is_empty() {
        println!("    {icon} {name:<22} ({display})");
    } else {
        println!("    {icon} {name:<22} ({display}) — aka {aliases}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// x86_64 full feature survey
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
fn survey_x86_64() {
    use std::arch::is_x86_feature_detected;

    macro_rules! feat {
        ($name:tt) => {
            let detected = is_x86_feature_detected!($name);
            let icon = if detected { "[+]" } else { "[ ]" };
            println!("    {icon} {}", $name);
        };
    }

    println!("── x86_64 Feature Detection ──");
    println!();

    println!("  Baseline (v1):");
    feat!("sse");
    feat!("sse2");
    println!();

    println!("  Tier v2 (Nehalem 2008+):");
    feat!("sse3");
    feat!("ssse3");
    feat!("sse4.1");
    feat!("sse4.2");
    feat!("popcnt");
    feat!("cmpxchg16b");
    println!();

    println!("  Tier v3 (Haswell 2013+):");
    feat!("avx");
    feat!("avx2");
    feat!("fma");
    feat!("bmi1");
    feat!("bmi2");
    feat!("f16c");
    feat!("lzcnt");
    feat!("movbe");
    println!();

    println!("  Tier v4 — AVX-512 (Skylake-X 2017+):");
    feat!("avx512f");
    feat!("avx512bw");
    feat!("avx512cd");
    feat!("avx512dq");
    feat!("avx512vl");
    println!();

    println!("  AVX-512 extensions (v4x, Ice Lake+):");
    feat!("avx512vpopcntdq");
    feat!("avx512ifma");
    feat!("avx512vbmi");
    feat!("avx512vbmi2");
    feat!("avx512bitalg");
    feat!("avx512vnni");
    println!();

    println!("  AVX-512 other:");
    feat!("avx512bf16");
    feat!("avx512fp16");
    feat!("avx512vp2intersect");
    println!();

    println!("  Crypto:");
    feat!("aes");
    feat!("pclmulqdq");
    feat!("sha");
    feat!("sha512");
    feat!("sm3");
    feat!("sm4");
    feat!("vaes");
    feat!("vpclmulqdq");
    feat!("gfni");
    feat!("kl");
    feat!("widekl");
    println!();

    println!("  VEX ML extensions (Alder Lake+):");
    feat!("avxvnni");
    feat!("avxifma");
    feat!("avxneconvert");
    feat!("avxvnniint8");
    feat!("avxvnniint16");
    println!();

    println!("  Bit manipulation:");
    feat!("adx");
    println!();

    println!("  RNG:");
    feat!("rdrand");
    feat!("rdseed");
    println!();

    println!("  State save:");
    feat!("fxsr");
    feat!("xsave");
    feat!("xsavec");
    feat!("xsaveopt");
    feat!("xsaves");
    println!();

    println!("  AMD legacy:");
    feat!("sse4a");
    feat!("tbm");
}

// ═══════════════════════════════════════════════════════════════════════════
// AArch64 full feature survey
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
fn survey_aarch64() {
    macro_rules! feat {
        ($name:tt) => {
            let detected = std::arch::is_aarch64_feature_detected!($name);
            let icon = if detected { "[+]" } else { "[ ]" };
            println!("    {icon} {}", $name);
        };
    }

    println!("── AArch64 Feature Detection ──");
    println!();

    println!("  SIMD / compute:");
    feat!("neon");
    feat!("crc");
    feat!("rdm");
    feat!("dotprod");
    feat!("fp16");
    feat!("fhm");
    feat!("fcma");
    feat!("bf16");
    feat!("i8mm");
    feat!("jsconv");
    feat!("frintts");
    println!();

    println!("  Crypto:");
    feat!("aes");
    feat!("sha2");
    feat!("sha3");
    feat!("sm4");
    println!();

    println!("  Atomics / memory:");
    feat!("lse");
    feat!("rcpc");
    feat!("rcpc2");
    println!();

    println!("  SVE (not used by archmage):");
    feat!("sve");
    feat!("sve2");
    feat!("sve2-aes");
    feat!("sve2-bitperm");
    feat!("sve2-sha3");
    feat!("sve2-sm4");
    feat!("f32mm");
    feat!("f64mm");
    println!();

    println!("  Security:");
    feat!("bti");
    feat!("mte");
    feat!("dit");
    feat!("sb");
    feat!("ssbs");
    feat!("paca");
    feat!("pacg");
    println!();

    println!("  System:");
    feat!("dpb");
    feat!("dpb2");
    feat!("rand");
    feat!("flagm");
    feat!("tme");
}
