//! Guards the Apple aarch64 detection fallback against silently widening
//! back to all `target_vendor = "apple"` targets.
//!
//! Background: `__impl_aarch64_apple_or_runtime_check!` returns `true`
//! unconditionally — no runtime check — for the ten features every Apple
//! Silicon (M1+) chip has. That is only sound where the executing hardware
//! is provably Apple Silicon: macOS, Mac Catalyst (`macabi`), and the
//! aarch64 simulators (`sim`), all of which run exclusively on M1+ Macs.
//!
//! Device iOS/tvOS/watchOS/visionOS targets must NOT get the fallback:
//! their target-spec baselines guarantee only `{aes, neon, sha2}`, and the
//! hardware they support includes A7–A12 cores lacking crc/rdm/dotprod/
//! fp16/fhm/fcma/sha3. An unconditional `true` there lets safe code summon
//! `Arm64V2Token`/`NeonSha3Token`/`NeonCrcToken` on CPUs without the
//! features — undefined behavior (in practice SIGILL).
//!
//! This exact bug shipped: the fallback was originally gated on
//! `all(target_vendor = "apple", target_arch = "aarch64")` alone, making
//! every device-iOS `summon()` a lie. These guards are pure source scans,
//! so they run on every platform and CI lane — the regression can't land
//! silently again. (We cannot execute-and-assert on real A-series devices
//! in CI, so source-level guarding is the strongest available check.)

use std::path::Path;

fn detect_src() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/detect.rs"))
        .expect("read src/detect.rs")
}

/// Extract the body of `__impl_aarch64_apple_or_runtime_check!` from
/// src/detect.rs.
fn apple_fallback_macro_body() -> String {
    let src = detect_src();
    let start = src
        .find("macro_rules! __impl_aarch64_apple_or_runtime_check")
        .expect("__impl_aarch64_apple_or_runtime_check must exist in src/detect.rs");
    // The next macro_rules! definition bounds the body.
    let rest = &src[start..];
    let end = rest[1..]
        .find("macro_rules!")
        .map(|i| i + 1)
        .unwrap_or(rest.len());
    rest[..end].to_string()
}

#[test]
fn apple_fallback_is_narrowed_to_guaranteed_apple_silicon_hosts() {
    let body = apple_fallback_macro_body();

    // The unconditional-true arm must carry the host-narrowing cfg.
    for marker in [
        r#"target_os = "macos""#,
        r#"target_abi = "macabi""#,
        r#"target_abi = "sim""#,
    ] {
        assert!(
            body.contains(marker),
            "__impl_aarch64_apple_or_runtime_check! lost the `{marker}` narrowing — \
             an unconditional `true` for plain target_vendor=apple hands out tokens \
             on A7–A12 iOS/tvOS/watchOS devices that lack the features (UB/SIGILL \
             from safe code). See this test's module docs."
        );
    }

    // And it must not degrade to vendor-only gating: every cfg(all(...))
    // that mentions the apple vendor must also mention the os/abi narrowing.
    for (i, cfg_site) in body.match_indices(r#"target_vendor = "apple""#) {
        // Look at a window around the match for the narrowing terms.
        let window_start = i.saturating_sub(200);
        let window_end = (i + cfg_site.len() + 300).min(body.len());
        let window = &body[window_start..window_end];
        assert!(
            window.contains(r#"target_os = "macos""#),
            "a `target_vendor = \"apple\"` cfg in \
             __impl_aarch64_apple_or_runtime_check! is not accompanied by the \
             os/abi narrowing:\n{window}"
        );
    }
}

#[test]
fn device_ios_paths_use_runtime_detection() {
    let body = apple_fallback_macro_body();
    assert!(
        body.contains("__impl_aarch64_runtime_only_check"),
        "the non-guaranteed arm of __impl_aarch64_apple_or_runtime_check! must \
         fall through to __impl_aarch64_runtime_only_check! (fails closed where \
         runtime detection is unavailable)"
    );
}

/// On a macOS aarch64 host (the only Apple platform CI can execute), the
/// fallback plus real detection must agree that the Apple Silicon baseline
/// tokens summon. This is an execution check of the sound direction.
#[cfg(all(target_vendor = "apple", target_arch = "aarch64", target_os = "macos"))]
#[test]
fn macos_apple_silicon_baseline_tokens_summon() {
    assert!(
        archmage::Arm64V2Token::summon().is_some(),
        "every Apple Silicon Mac has the full Arm64V2 feature set"
    );
    assert!(archmage::NeonCrcToken::summon().is_some());
    assert!(archmage::NeonSha3Token::summon().is_some());
}
