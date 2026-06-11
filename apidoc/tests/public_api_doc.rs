//! Public-API surface snapshots for the PARENT workspace (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
#[test]
fn public_api_surface_docs_are_current() {
    // Explicit crate list: archmage-macros surfaces through archmage's
    // re-exports; xtask + tests/no-features-crate are internal.
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .crates(["archmage", "magetypes"])
        .run();
}
