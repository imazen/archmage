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

## 0.9.31-beta.2 compared with published 0.9.30

The compatibility baseline is the published `v0.9.30` tag, not an unpublished
constructor experiment or earlier beta draft. The snapshot diff contains:

- Archmage: `attune`, `attuned!`, and `reattune!` exports, at both the crate
  root and in the prelude, on all three targets.
- Magetypes: 90 additional x86 AVX-512 trait implementations for narrow
  vectors, conversions and bitcasts. They are feature-gated and delegate
  to V3. No existing function or constructor signature changes.
- No supported API removals. `_t(token, ...)` and the older deprecated
  constructor signatures were already published in 0.9.30.

Verified on 2026-10-12 against implementation commit `b9179436`: the committed
snapshots match fresh x86-64, AArch64 and WASM output. Both ordinary semver
checks pass; the precise Magetypes check compares 2,040 safe methods, including
40 carrying target features, with no failures. Reproduce with:

```sh
just api-doc-check
python3 -B xtask/check_semver.py archmage --baseline-version 0.9.30
python3 -B xtask/check_semver.py magetypes --baseline-version 0.9.30
git diff v0.9.30 -- docs/public-api/
```

Signature snapshots do not describe macro expansion semantics or numerical
behavior. See the [unified macro guide](../ATTUNE.md),
[release changelog](../../CHANGELOG.md), and
[FMA contract](../site/content/magetypes/isa-quirks.md#fused-arithmetic-and-reductions)
for those changes. In particular, portable software FMA now fixes subnormal
rounding without CPU probes; bare multiply-add keeps its backend-specific
rounding behavior. These are behavior changes, not signature changes.
