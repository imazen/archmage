# Public-API surface snapshots

Committed snapshots of the workspace's public API, **one directory per
compilation target** (issue #75): the surface is `target_arch`-gated
(`__m256` constructors exist only on x86_64, `int8x16_t` ones only on
aarch64, `v128` only on wasm32), so a single host-built snapshot was a
function of whoever last regenerated it. Per-target generation makes the
output byte-identical on any host.

| Directory  | Built with `--target`       |
|------------|-----------------------------|
| `x86_64/`  | `x86_64-unknown-linux-gnu`  |
| `aarch64/` | `aarch64-unknown-linux-gnu` |
| `wasm32/`  | `wasm32-wasip1`             |

Each directory holds three files per crate (`archmage`, `magetypes`), in the
[zenutils-apidoc](https://lib.rs/crates/zenutils-apidoc) format:
`<crate>.txt` (supported surface, default features), `<crate>.features.txt`
(additions from non-default features), `<crate>.internal.txt` (doc(hidden) +
excluded-feature surface).

- **Regenerate:** `just api-doc` (also chained into `just fmt`). Needs
  `rustup`; the nightly toolchain and target stdlibs are auto-installed.
  Commit the diff together with the code change that caused it.
- **Verify:** `just api-doc-check` — run by the `Public API Check` CI job,
  which fails when a committed snapshot is stale.
- **Runner:** `apidoc/tests/public_api_doc.rs` (a workspace-excluded
  package, so plain `cargo test` never builds it or runs rustdoc).

`ABLATION-*.md` are one-time hand-written audit reports (2026-06-11), not
generated files.
