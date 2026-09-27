# Downstream public archmage type exposure, 2026-09-27

This source audit distinguishes re-exporting a type from accepting it in a
public signature. It excludes archmage/magetypes themselves, executable-only
bench probes, and APIs hidden by private parent modules. Architecture and Cargo
feature gates are retained. It is not an exhaustive rustdoc/semver audit of
every repository in the account; the counts below are confirmed findings.

Three downstream crates directly re-export archmage token types in ordinary
builds (one uses doc-hidden exports); zenjpeg adds a fourth through its
`__test-utils` feature. Broader public-signature exposure is confirmed in seven
crates: four ordinary library APIs and three development/test APIs.

| Crate | Exposure | Source |
|---|---|---|
| `linear-srgb` | normal public API; direct token re-exports | [source](https://github.com/imazen/linear-srgb/blob/c56e7940f4a123fb653bc60593376cf2eaed3e70/src/tokens/mod.rs#L54) |
| `jxl-encoder-simd` | normal public API; direct token/trait re-exports | [source](https://github.com/imazen/jxl-encoder/blob/cd9a7325f97f5e178863d867c81b694cd8b169aa/jxl-encoder-simd/src/lib.rs#L235) |
| `zenjxl-decoder-simd` | public, doc-hidden token/trait re-exports for macros; public descriptor token methods | [source](https://github.com/imazen/zenjxl-decoder/blob/426f179be58fbf7943e6030164c29b261eb290f1/zenjxl-decoder-simd/src/lib.rs#L46) |
| `zenav1-svt-dsp` | public me_sad functions accept tokens; no direct token re-export found | [source](https://github.com/imazen/zenav1-svt/blob/dd3e33709ef2d1d48b4e10af312ab5d164454bc1/rust/crates/svtav1-dsp/src/me_sad.rs#L64) |
| `zenjpeg` | __test-utils exposes encode/decode internals, including Desktop64 re-export and token parameters | [source](https://github.com/imazen/zenjpeg/blob/f325ca4fc8cfe5c07629f7b284fe8df9005b9788/zenjpeg/src/lib.rs#L259) |
| `zenavif` | _dev exposes SIMD modules whose public functions accept tokens | [source](https://github.com/imazen/zenavif/blob/dba8f5ee73ccd47b756a7a849a8d30a3e733108d/src/lib.rs#L120) |
| `zensim-validate` | unpublished development crate: public adam_simd functions accept X64V4Token | [source](https://github.com/imazen/zensim/blob/5a4d5fa3c21386b93e232d4777d456dc64d6d12d/zensim-validate/src/adam_simd.rs#L469) |

`linear-srgb` exposes archmage tokens, not magetypes vector types, in its checked
conversion signatures. Its public `tokens::{x4,x8,x16}` functions accept tokens
and use ordinary arrays/slices for pixel values.

`garb` has no observed archmage token or magetypes type in its public signatures
or re-exports. Its public feature-context helpers accept and return arrays;
internal `pub(super)` token helpers do not escape the crate. The experimental
`deinterleave` module exposes `rgb24_chunk8_to_planes_tokenless_v3`, among other
array-based helpers.

Sources: [linear-srgb conversion](https://github.com/imazen/linear-srgb/blob/c56e7940f4a123fb653bc60593376cf2eaed3e70/src/tokens/x8.rs#L32), [garb tokenless helper](https://github.com/imazen/garb/blob/53b2571a4e69061ae0c1b2a58e7d667e04f7e2e1/src/deinterleave.rs#L335), [garb re-export](https://github.com/imazen/garb/blob/53b2571a4e69061ae0c1b2a58e7d667e04f7e2e1/src/deinterleave.rs#L520).

Other checked examples: `zenblend`, `zenresize`, `zenpixels-convert`, and
`zenfilters` showed no matching public token/vector signatures or direct
re-exports in their fetched source. `zenwebp`'s token-taking YUV helpers are
inside `pub(crate) mod yuv`, so they are not external API. Likewise the matched
SVT encoder function is hidden by `pub(crate) mod inter_me`.

Evidence is preserved at
`/home/lilith/data/archmage/export-audit/2026-09-27/`: pinned remote source for
linear-srgb, garb, zenblend, zenpixels, zenjxl-decoder, zenresize, zenpipe; exact
local-snapshot commits and fetched missing module declarations; GitHub search
results and public-signature candidates. Other evidence reuses the earlier
[token audit](TOKEN-AUDIT-2026-09-27.pointer.md) source snapshots.

Discovery included GitHub manifest and source searches plus source inspection.
A broad archmage search hit GitHub's search rate limit; narrower searches and
core API source retrieval succeeded. Search snippets were only candidate
leads. The seven positives were verified in source with their public module
paths; negative findings remain limited to the inspected source and do not
claim whole-program macro/name resolution.
