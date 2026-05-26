//! Guards the Windows-on-ARM detection bridge against silently reverting to
//! the IPFP-only `_fast!` path.
//!
//! Background: `winarm-cpufeatures` exposes two detection layers. The fast
//! layer (`is_aarch64_feature_detected_fast!`, `Features::current()`,
//! `query_fast`) reads only `IsProcessorFeaturePresent` and **never** the
//! registry — by design. The registry-aware layer (`Features::current_full`)
//! additionally decodes the `ID_AA64*_EL1` system registers, recovering the
//! ~30 feature names Windows can't expose otherwise (`fhm`, `fcma`, `sm4`,
//! `lse128`, `paca`, `mte`, `flagm`, `rcpc2`/`rcpc3`, `rand`, …).
//!
//! `Arm64V2Token` needs `rdm`/`fp16` and `Arm64V3Token` needs `fhm`/`fcma` —
//! all registry-classified on pre-26100 Windows. If the bridge uses the fast
//! path, those tokens' `summon()` returns `None` on hardware that has the
//! features (Cobalt 100, Snapdragon X, Graviton 3+), and the `registry`
//! Cargo feature becomes inert despite being enabled.
//!
//! This regression actually shipped in git history (`962fceb`, `e13907d`
//! used `is_aarch64_feature_detected_fast!`). It compiles cleanly and passes
//! every test except the `#[ignore]`'d `cobalt100_runner_*` hardware
//! assertion on the lone `windows-11-arm` CI runner. These guards are pure
//! source scans, so they run on **every** platform and CI lane — the
//! regression can't land silently again.

use std::path::Path;

fn detect_src() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/detect.rs"))
        .expect("read src/detect.rs")
}

/// The IPFP-only winarm entry points. Naming any of these in the bridge
/// means registry-classified features silently report `false`.
const FAST_PATH_MARKERS: &[&str] = &[
    "is_aarch64_feature_detected_fast",
    "query_fast",
    "snapshot_fast",
    "Features::current()",
];

#[test]
fn bridge_macro_routes_through_registry_aware_fn() {
    let src = detect_src();

    let macro_start = src
        .find("macro_rules! __winarm_cpufeatures_detected")
        .expect("Windows-ARM bridge macro `__winarm_cpufeatures_detected` must exist");
    // The `__priv_winarm` module is emitted immediately after the macro.
    let macro_end = src[macro_start..]
        .find("pub mod __priv_winarm")
        .map(|i| macro_start + i)
        .expect("`__priv_winarm` module must follow the bridge macro");
    let macro_body = &src[macro_start..macro_end];

    assert!(
        macro_body.contains("registry_aware_detected"),
        "the bridge macro must route through `registry_aware_detected()` — not name a \
         winarm detection API directly (so the registry-aware choice lives in exactly one place)"
    );
    for banned in FAST_PATH_MARKERS {
        assert!(
            !macro_body.contains(banned),
            "the bridge macro references the IPFP-only path `{banned}`, which skips the \
             registry decoder and drops ~30 features (fhm, fcma, sm4, …). Route through \
             `registry_aware_detected()` instead."
        );
    }
}

#[test]
fn registry_aware_detected_uses_current_full() {
    let src = detect_src();

    let fn_start = src
        .find("pub fn registry_aware_detected")
        .expect("`registry_aware_detected()` (the single registry-aware entry point) must exist");
    // Slice to the function's closing brace (module-level 4-space indent).
    // The doc comment sits *before* `fn_start`, so it's excluded from the body.
    let fn_end = src[fn_start..]
        .find("\n    }")
        .map(|i| fn_start + i)
        .expect("`registry_aware_detected()` must have a closing brace");
    let fn_body = &src[fn_start..fn_end];

    assert!(
        fn_body.contains("current_full"),
        "`registry_aware_detected()` must call `Features::current_full()` — the only \
         winarm API that consults the registry decoder"
    );
    for banned in FAST_PATH_MARKERS
        .iter()
        .chain(["current()", "is_detected"].iter())
    {
        assert!(
            !fn_body.contains(banned),
            "`registry_aware_detected()` uses the IPFP-only path `{banned}`. That silently \
             skips the registry and makes Arm64V2/V3 summon() return None on capable \
             Windows-ARM hardware."
        );
    }
}
