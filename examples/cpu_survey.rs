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
    report_token::<archmage::X64V3Token>("X64V3Token", "Avx2FmaToken");
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
    // x86 CPUID brand string — works on all x86 OSes, no OS API needed
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if let Some(name) = cpuid_brand_string() {
            return name;
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        if let Some(name) = read_proc_cpuinfo_arm() {
            return name;
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if let Some(name) = sysctl_apple_chip() {
            return name;
        }
    }

    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        if let Some(name) = windows_arm_cpu_name() {
            return name;
        }
    }

    "Unknown".to_string()
}

// ── x86 CPUID brand string (all platforms) ──────────────────────────────

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn cpuid_brand_string() -> Option<String> {
    // Check if extended CPUID leaves 0x80000002..0x80000004 are supported
    let max_ext = unsafe { core::arch::x86_64::__cpuid(0x80000000) }.eax;
    if max_ext < 0x80000004 {
        return None;
    }

    let mut brand = [0u8; 48];
    for (i, leaf) in (0x80000002u32..=0x80000004).enumerate() {
        let result = unsafe { core::arch::x86_64::__cpuid(leaf) };
        let offset = i * 16;
        brand[offset..offset + 4].copy_from_slice(&result.eax.to_le_bytes());
        brand[offset + 4..offset + 8].copy_from_slice(&result.ebx.to_le_bytes());
        brand[offset + 8..offset + 12].copy_from_slice(&result.ecx.to_le_bytes());
        brand[offset + 12..offset + 16].copy_from_slice(&result.edx.to_le_bytes());
    }

    let name = String::from_utf8_lossy(&brand);
    let name = name.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

// ── Linux AArch64: /proc/cpuinfo + part number lookup ───────────────────

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
        if implementer.is_some() && part.is_some() {
            break;
        }
    }

    match (implementer, part) {
        (Some(imp), Some(prt)) => Some(arm_chip_name(&imp, &prt)),
        _ => {
            // Fallback: try "Hardware" or "model name" lines
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

/// Resolve ARM implementer + part to a marketing name.
#[cfg(any(
    all(target_os = "linux", target_arch = "aarch64"),
    all(target_os = "windows", target_arch = "aarch64")
))]
fn arm_chip_name(implementer: &str, part: &str) -> String {
    let name = match (implementer, part) {
        // ARM Ltd (0x41)
        ("0x41", "0xd03") => "ARM Cortex-A53",
        ("0x41", "0xd04") => "ARM Cortex-A35",
        ("0x41", "0xd05") => "ARM Cortex-A55",
        ("0x41", "0xd06") => "ARM Cortex-A65",
        ("0x41", "0xd07") => "ARM Cortex-A57",
        ("0x41", "0xd08") => "ARM Cortex-A72",
        ("0x41", "0xd09") => "ARM Cortex-A73",
        ("0x41", "0xd0a") => "ARM Cortex-A75",
        ("0x41", "0xd0b") => "ARM Cortex-A76",
        ("0x41", "0xd0c") => "ARM Neoverse N1",
        ("0x41", "0xd0d") => "ARM Cortex-A77",
        ("0x41", "0xd0e") => "ARM Cortex-A76AE",
        ("0x41", "0xd40") => "ARM Neoverse V1",
        ("0x41", "0xd41") => "ARM Cortex-A78",
        ("0x41", "0xd42") => "ARM Cortex-A78AE",
        ("0x41", "0xd43") => "ARM Cortex-A65AE",
        ("0x41", "0xd44") => "ARM Cortex-X1",
        ("0x41", "0xd46") => "ARM Cortex-A510",
        ("0x41", "0xd47") => "ARM Cortex-A710",
        ("0x41", "0xd48") => "ARM Cortex-X2",
        ("0x41", "0xd49") => "ARM Neoverse N2 (Cobalt 100)",
        ("0x41", "0xd4a") => "ARM Neoverse E1",
        ("0x41", "0xd4b") => "ARM Cortex-A78C",
        ("0x41", "0xd4c") => "ARM Cortex-X1C",
        ("0x41", "0xd4d") => "ARM Cortex-A715",
        ("0x41", "0xd4e") => "ARM Cortex-X3",
        ("0x41", "0xd4f") => "ARM Neoverse V2",
        ("0x41", "0xd80") => "ARM Cortex-A520",
        ("0x41", "0xd81") => "ARM Cortex-A720",
        ("0x41", "0xd82") => "ARM Cortex-X4",
        ("0x41", "0xd84") => "ARM Neoverse V3",
        ("0x41", "0xd85") => "ARM Cortex-X925",
        ("0x41", "0xd87") => "ARM Cortex-A725",
        // Qualcomm (0x51)
        ("0x51", "0x800") => "Qualcomm Kryo 260 / 280",
        ("0x51", "0x802") => "Qualcomm Kryo 385 Gold",
        ("0x51", "0x803") => "Qualcomm Kryo 385 Silver",
        ("0x51", "0x804") => "Qualcomm Kryo 485 Gold",
        ("0x51", "0x805") => "Qualcomm Kryo 485 Silver",
        ("0x51", "0xc00") => "Qualcomm Falkor",
        ("0x51", "0x001") => "Qualcomm Oryon (Snapdragon X)",
        // Apple (0x61) — usually from Asahi Linux
        ("0x61", "0x022") => "Apple M1 Icestorm (E)",
        ("0x61", "0x023") => "Apple M1 Firestorm (P)",
        ("0x61", "0x024") => "Apple M1 Pro/Max Icestorm (E)",
        ("0x61", "0x025") => "Apple M1 Pro/Max Firestorm (P)",
        ("0x61", "0x028") => "Apple M1 Ultra Icestorm (E)",
        ("0x61", "0x029") => "Apple M1 Ultra Firestorm (P)",
        ("0x61", "0x030") => "Apple M2 Blizzard (E)",
        ("0x61", "0x031") => "Apple M2 Avalanche (P)",
        ("0x61", "0x032") => "Apple M2 Pro/Max Blizzard (E)",
        ("0x61", "0x033") => "Apple M2 Pro/Max Avalanche (P)",
        ("0x61", "0x034") => "Apple M2 Ultra Blizzard (E)",
        ("0x61", "0x035") => "Apple M2 Ultra Avalanche (P)",
        ("0x61", "0x036") => "Apple M3 Sawtooth (E)",
        ("0x61", "0x037") => "Apple M3 Everest (P)",
        ("0x61", "0x038") => "Apple M3 Pro/Max Sawtooth (E)",
        ("0x61", "0x039") => "Apple M3 Pro/Max Everest (P)",
        ("0x61", "0x049") => "Apple M4 (E)",
        ("0x61", "0x048") => "Apple M4 (P)",
        // Ampere (0xc0)
        ("0xc0", "0xac3") => "Ampere Altra",
        ("0xc0", "0xac4") => "Ampere Altra Max",
        // NVIDIA (0x4e)
        ("0x4e", "0x004") => "NVIDIA Denver 2",
        ("0x4e", "0x003") => "NVIDIA Carmel",
        // Samsung (0x53)
        ("0x53", "0x001") => "Samsung Exynos M1/M2",
        // Fujitsu (0x46)
        ("0x46", "0x001") => "Fujitsu A64FX",
        // Microsoft (0x6d)
        ("0x6d", _) => "Microsoft Cobalt",
        _ => {
            let vendor = match implementer {
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
                _ => implementer,
            };
            return format!("{vendor} (part {part})");
        }
    };
    name.to_string()
}

// ── macOS Apple Silicon ─────────────────────────────────────────────────

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn sysctl_apple_chip() -> Option<String> {
    // hw.chip_name gives "Apple M1" etc. on Apple Silicon
    for key in &["machdep.cpu.brand_string", "hw.chip_name"] {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", key])
            .output()
        {
            let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !name.is_empty() && name != "0" {
                return Some(name);
            }
        }
    }
    None
}

// ── Windows ARM64 ───────────────────────────────────────────────────────

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
fn windows_arm_cpu_name() -> Option<String> {
    // PROCESSOR_IDENTIFIER gives "ARMv8 (64-bit) Family 8 Model D49 Revision 0"
    // Parse out model to get part number for lookup
    if let Ok(ident) = std::env::var("PROCESSOR_IDENTIFIER") {
        // Extract model number: "Model D49" -> 0xd49
        if let Some(model_pos) = ident.find("Model ") {
            let model_str = &ident[model_pos + 6..];
            let model_end = model_str
                .find(|c: char| c == ' ' || c == ',')
                .unwrap_or(model_str.len());
            let model_hex = &model_str[..model_end];
            // Windows gives uppercase hex without 0x prefix
            if let Ok(model_num) = u32::from_str_radix(model_hex.trim_start_matches("0x"), 16) {
                let part = format!("0x{model_num:03x}");
                // Windows doesn't expose implementer; guess from context
                // GitHub ARM runners use Cobalt 100 (Microsoft 0x6d) or
                // Snapdragon X (Qualcomm 0x51). Check both.
                for implementer in &["0x41", "0x6d", "0x51"] {
                    let name = arm_chip_name(implementer, &part);
                    if !name.contains("(part ") {
                        return Some(name);
                    }
                }
                // Fallback: just format what we know
                return Some(format!("ARM (part {part}, from Windows Model {model_hex})"));
            }
        }
        // If parsing failed, return the raw identifier (still better than nothing)
        if !ident.is_empty() {
            return Some(ident);
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
        ($name:tt, $desc:expr) => {
            let detected = is_x86_feature_detected!($name);
            let icon = if detected { "[+]" } else { "[ ]" };
            println!("    {icon} {:<24} {}", $name, $desc);
        };
    }

    println!("── x86_64 Feature Detection ──");
    println!();

    println!("  Baseline (v1):");
    feat!("sse", "Streaming SIMD Extensions");
    feat!("sse2", "SSE2 (128-bit integer/double SIMD)");
    println!();

    println!("  Tier v2 (Nehalem 2008+):");
    feat!("sse3", "SSE3 (horizontal add, complex math)");
    feat!("ssse3", "Supplemental SSE3 (byte shuffle, abs)");
    feat!("sse4.1", "SSE4.1 (blend, dot product, round)");
    feat!("sse4.2", "SSE4.2 (string compare, CRC32)");
    feat!("popcnt", "Population count");
    feat!("cmpxchg16b", "128-bit compare-and-swap");
    println!();

    println!("  Tier v3 (Haswell 2013+):");
    feat!("avx", "Advanced Vector Extensions (256-bit float)");
    feat!("avx2", "AVX2 (256-bit integer SIMD)");
    feat!("fma", "Fused multiply-add (3 operand)");
    feat!("bmi1", "Bit Manipulation Instructions 1");
    feat!("bmi2", "Bit Manipulation Instructions 2 (PDEP/PEXT)");
    feat!("f16c", "Half-precision float conversion");
    feat!("lzcnt", "Leading zero count");
    feat!("movbe", "Move data after byte swap");
    println!();

    println!("  Tier v4 — AVX-512 (Skylake-X 2017+):");
    feat!("avx512f", "AVX-512 Foundation (512-bit SIMD)");
    feat!("avx512bw", "AVX-512 Byte/Word operations");
    feat!("avx512cd", "AVX-512 Conflict Detection");
    feat!("avx512dq", "AVX-512 Doubleword/Quadword");
    feat!("avx512vl", "AVX-512 Vector Length (128/256-bit)");
    println!();

    println!("  AVX-512 extensions (v4x, Ice Lake+):");
    feat!("avx512vpopcntdq", "AVX-512 Vector POPCNT DW/QW");
    feat!("avx512ifma", "AVX-512 Integer FMA (52-bit)");
    feat!("avx512vbmi", "AVX-512 Vector Byte Manipulation");
    feat!("avx512vbmi2", "AVX-512 VBMI2 (compress/expand byte)");
    feat!("avx512bitalg", "AVX-512 Bit Algorithms (POPCNT byte)");
    feat!("avx512vnni", "AVX-512 Vector Neural Network (INT8 dot)");
    println!();

    println!("  AVX-512 other:");
    feat!("avx512bf16", "AVX-512 BFloat16 conversion/dot");
    feat!("avx512fp16", "AVX-512 native Float16 arithmetic");
    feat!("avx512vp2intersect", "AVX-512 VP2INTERSECT");
    println!();

    println!("  Crypto:");
    feat!("aes", "AES-NI (128-bit AES rounds)");
    feat!("pclmulqdq", "Carry-less multiplication (GF(2))");
    feat!("sha", "SHA-1/SHA-256 acceleration");
    feat!("sha512", "SHA-512 acceleration");
    feat!("sm3", "ShangMi 3 hash");
    feat!("sm4", "ShangMi 4 cipher");
    feat!("vaes", "Vectorized AES (256/512-bit)");
    feat!("vpclmulqdq", "Vectorized CLMUL (256/512-bit)");
    feat!("gfni", "Galois Field arithmetic");
    feat!("kl", "Key Locker");
    feat!("widekl", "Wide Key Locker");
    println!();

    println!("  VEX ML extensions (Alder Lake+):");
    feat!("avxvnni", "AVX-VNNI (INT8 dot, VEX-encoded)");
    feat!("avxifma", "AVX-IFMA (52-bit integer FMA, VEX)");
    feat!("avxneconvert", "AVX-NE-CONVERT (BF16/FP16, no except)");
    feat!("avxvnniint8", "AVX-VNNI-INT8 (signed INT8 dot)");
    feat!("avxvnniint16", "AVX-VNNI-INT16 (INT16 dot product)");
    println!();

    println!("  Bit manipulation:");
    feat!("adx", "Multi-precision add-carry (ADCX/ADOX)");
    println!();

    println!("  RNG:");
    feat!("rdrand", "Hardware random number generator");
    feat!("rdseed", "Hardware random seed generator");
    println!();

    println!("  State save:");
    feat!("fxsr", "FXSAVE/FXRSTOR (legacy SSE state)");
    feat!("xsave", "XSAVE/XRSTOR (extended state)");
    feat!("xsavec", "XSAVE compacted format");
    feat!("xsaveopt", "XSAVEOPT (skip unchanged state)");
    feat!("xsaves", "XSAVES (supervisor state)");
    println!();

    println!("  AMD legacy:");
    feat!("sse4a", "SSE4a (AMD: EXTRQ/INSERTQ/MOVNTSS)");
    feat!("tbm", "Trailing Bit Manipulation (AMD Piledriver)");
}

// ═══════════════════════════════════════════════════════════════════════════
// AArch64 full feature survey
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
fn survey_aarch64() {
    macro_rules! feat {
        ($name:tt, $desc:expr) => {
            let detected = std::arch::is_aarch64_feature_detected!($name);
            let icon = if detected { "[+]" } else { "[ ]" };
            println!("    {icon} {:<24} {}", $name, $desc);
        };
    }

    println!("── AArch64 Feature Detection ──");
    println!();

    println!("  SIMD / compute:");
    feat!("neon", "NEON/AdvSIMD (128-bit SIMD, baseline)");
    feat!("crc", "CRC32 instructions");
    feat!("rdm", "Rounding Double Multiply Accumulate");
    feat!("dotprod", "Integer dot product (SDOT/UDOT)");
    feat!("fp16", "Half-precision float arithmetic");
    feat!("fhm", "FP16 multiply-accumulate to FP32");
    feat!("fcma", "Floating-point complex multiply-add");
    feat!("bf16", "BFloat16 (BFDOT/BFMMLA)");
    feat!("i8mm", "Int8 matrix multiply (SMMLA/UMMLA)");
    feat!("jsconv", "JavaScript FJCVTZS conversion");
    feat!("frintts", "FRINT32Z/FRINT64Z rounding");
    println!();

    println!("  Crypto:");
    feat!("aes", "AES encrypt/decrypt + MixColumns");
    feat!("sha2", "SHA-1/SHA-256 acceleration");
    feat!("sha3", "SHA-3/SHA-512 acceleration (EOR3/RAX1)");
    feat!("sm4", "ShangMi 4 cipher");
    println!();

    println!("  Atomics / memory:");
    feat!("lse", "Large System Extensions (atomic CAS/SWP)");
    feat!("rcpc", "Release-Consistent Processor Consistent");
    feat!("rcpc2", "RCPC2 (LDAPR seq-cst acquire)");
    println!();

    println!("  SVE (not used by archmage):");
    feat!("sve", "Scalable Vector Extension");
    feat!("sve2", "SVE2 (extended integer/crypto)");
    feat!("sve2-aes", "SVE2 AES acceleration");
    feat!("sve2-bitperm", "SVE2 bit permutation");
    feat!("sve2-sha3", "SVE2 SHA3 acceleration");
    feat!("sve2-sm4", "SVE2 SM4 cipher");
    feat!("f32mm", "SVE FP32 matrix multiply");
    feat!("f64mm", "SVE FP64 matrix multiply");
    println!();

    println!("  Security:");
    feat!("bti", "Branch Target Identification");
    feat!("mte", "Memory Tagging Extension");
    feat!("dit", "Data Independent Timing");
    feat!("sb", "Speculation Barrier");
    feat!("ssbs", "Speculative Store Bypass Safe");
    feat!("paca", "Pointer Authentication (address key)");
    feat!("pacg", "Pointer Authentication (generic key)");
    println!();

    println!("  System:");
    feat!("dpb", "Data Cache Clean to PoP (DC CVAP)");
    feat!("dpb2", "Data Cache Clean to PoDP (DC CVADP)");
    feat!("rand", "Hardware RNG (RNDR/RNDRRS)");
    feat!("flagm", "Condition flag manipulation (CFINV)");
    feat!("tme", "Transactional Memory Extension");
}
