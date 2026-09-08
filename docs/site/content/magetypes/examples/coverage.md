+++
title = "Production Coverage and Suspicious Gaps"
weight = 10
+++

The tutorial sequence follows actual uses in the local `zen/` collection. This
is an integration guide for writing SIMD kernels in those crates, not a guide
to implementing the codecs themselves. Source inspection establishes usage;
compilation, numerical tests, and assembly inspection establish different facts.

## What developers need to learn

| Production pattern | Concrete source | Where to learn it |
|---|---|---|
| Generated vector loop, chunks, and tail | zenfilters `scale_plane_simd` | [First kernel](@/archmage/getting-started/first-simd.md) |
| Reusable backend-generic helper | zenblend `blend_src_over_row<T>`; zenfilters FIR/stackblur | [SrcOver](@/magetypes/examples/pixel-blending.md), [generic gain](@/magetypes/examples/generic-kernels.md) |
| Type and const generics on the generated function itself | zenanalyze `accumulate_row_simd<..., R>`; zenavif `yuv420_strip_kernel<S, P>` | [Generic specialization](@/magetypes/dispatch/types-and-dispatch.md) |
| Const-mode specialization passed through dispatch | zenpixels-convert and zenpng `fused_cg_impl` | [Generic specialization](@/magetypes/dispatch/types-and-dispatch.md) |
| Generated SIMD plus a separate scalar algorithm | aom-rs `aom-txb/src/simd.rs` | [Separate scalar fallback](@/archmage/dispatch/incant.md) |
| Scalar loops under generated target features | zenpixels-convert swizzles; zenanalyze palette/grayscale | [Autoversion](@/archmage/dispatch/autoversion.md) |
| Matched intrinsic helper chains | zenfilters x86 scale; rav1d-safe NEON CfL helpers | [Rite](@/archmage/concepts/rite.md) |
| Mixed hand-tuned x86 and portable ARM/WASM | zenresize, zenblend, zenfilters | [Dispatch](@/archmage/dispatch/incant.md), [production walkthroughs](@/magetypes/examples/convolution.md) |
| f16 storage conversion | zenresize wide kernels | [Numeric conversion](@/magetypes/conversions/float-int.md) |
| Wider logical types plus native V4 | linear-srgb transfer curves | [Transcendentals](@/magetypes/math/transcendentals.md) |
| Bit-exact integer transforms | zenwebp add-green; aom-rs coefficient levels | [Byte transform](@/magetypes/examples/byte-transforms.md), [separate scalar fallback](@/archmage/dispatch/incant.md) |
| Pixel layout, strides, weights, padding, scratch ownership | zenresize streaming; zenfilters blur | [Memory](@/magetypes/memory/load-store.md), [production walkthroughs](@/magetypes/examples/convolution.md) |

The linked runnable examples explicitly identify adaptations. Some preserve a
production loop, others reduce a larger algorithm to demonstrate its integration
pattern. API exercises are labeled as such; they are not evidence of production
adoption or optimal codegen.

## Guidance not supported by the observed usage

A 2026-09-07 lexical search covered 4,387 Rust files across 90 top-level checkouts.
It excluded named duplicate `--` checkouts, hidden/scratch/retired trees, vendor,
target, and tests/benches/examples directories, and skipped comment-only lines.
It is a discovery scan, not a Rust AST census: macro-generated, aliased, or
unusually formatted calls can escape it. Zero matches means **not observed**,
not proof that no downstream user needs a feature.

| Area | Finding | Documentation treatment |
|---|---|---|
| `incant!(... with token)` | No production matches; implemented | Reference option. Checks held token type; does not upgrade or downcast it |
| `incant!(... without token)` | No production matches; implemented | Reference option for tokenless matching-tier callees, with compiled examples |
| `IntoConcreteToken` / exact-type branches | No production matches | Explain precisely; do not introduce into basic kernels |
| `from_context()` | No production matches at scan time | Document the safe context constructor; new tokenless-rite composition uses it |
| `stub` | No matches; parsers reject it | Removed stale claims that it is available |
| Nested expansion | Two matches in mozjpeg-rs entropy helpers | Keep a focused receiver/nested reference; do not claim every method needs it |
| Vector-reference `cast_slice` idiom | No matches for the scanned concrete/alias spellings | Reference-only. Ordinary pixel bytemuck casts are a different API |
| Tokenless `#[rite(tier)]` | Many matches, including rav1d-safe | Teach as an exercised internal-helper pattern |
| Pairwise widening / newer terminal byte operations | API tests do not establish adoption in older zen kernels | Explain contracts; require a concrete consumer before promoting a recipe |
| Checked portable gather/scatter; aligned/streaming methods | Not provided by the current generic API | Describe the gap; do not show fictitious methods |
| Baseline caller → bare generic SIMD loop | Older docs taught it; it lacks the needed feature context | Replace with complete generated entry/helper chains |
| Generic bounds inherently block inlining | Contradicted by Rust monomorphization and production helper patterns | Remove the claim |
| V4 automatically widens [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) | Contradicted by macro replacement and fixed lane count | Teach explicit logical shapes and supported backend lists |
| Universal speed/error multipliers | No general proof | Keep measurements scoped to CPU, operation, domain, and compiler |

There are suspicious comments in consumers too: zenpixels-convert's gamut
kernel comments claim [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) automatically becomes sixteen lanes for V4, while
its body explicitly fixes eight lanes. mozjpeg-rs comments describe a historical
method-sibling limitation; current archmage supports inherent method siblings.
Do not copy such comments into authoritative API guidance. Consumer feature
compatibility and performance need their own tests before changing those crates.

## Version and evidence scope

This site tracks the repository, whose current work is 0.9.29. API links point
to the latest **published** docs.rs version, which may lag it. `from_context()`,
the newer integer abstractions, and tokenless-rite token construction should not
be assumed available in 0.9.28. No crate publication is implied by this guide.

The search used local checkouts; pinned walkthroughs name their source revisions.
The additional inspected heads were zenanalyze `b102fa5`, zenavif `2ebca1b`,
zenpixels `e56f626`, zenpng `00d6deb`, aom-rs `028c5e1`, rav1d-safe `e73811f`,
and mozjpeg-rs `45d36a7`. Local edits can differ from those commits; the adaptation
notes describe the actual teaching extraction rather than claiming verbatim copies.

All website Rust fences run through `xtask/check_docs.py`, without implicit
imports or hidden helper implementations. Syntax-only notation uses text fences.
The numerical regression suite and ISA/codegen tests supplement these examples;
a passing doctest alone is not a performance proof.

## A shorter reading path

Installation and the first kernel are shared by both crates. The duplicate LLVM
boundary page points to one boundary explanation, and behavioral differences
point to the ISA tables. Four short source walkthroughs are combined into one
codec/filter page. Old URLs redirect to the canonical pages.

Read the first kernel, generic specialization, and memory chapters before
choosing the operations your algorithm needs. Keep rare macro options in their
reference pages instead of presenting a large option matrix at the beginning.
