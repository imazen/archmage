# Mixed token calls and generic consumers

Source review: 2026-09-27, archmage baseline `24857c15`; scalar dispatch fixed
in `3aef13f0`. Consumer links below
identify the pinned snapshots inspected, not a fresh fetch of downstream main.
The [token inventory](TOKEN-AUDIT-2026-09-27.pointer.md) preserves those sources;
zenfilters uses the separate export-audit snapshot at
`/home/lilith/data/archmage/export-audit/2026-09-27/zenpipe/`.
No consumer repository was modified or rebuilt for this review.

## Calls can mix independently of constructor style

Keep a token whenever it still supplies proof to another API. A function with
`token: Token` can use `use(f32x8)` and `f32x8::splat(value)` while passing
`token` to an older helper. Dropping constructor arguments does not require
dropping function parameters.

| Caller → callee | Current spelling and behavior |
|---|---|
| Ordinary caller → SIMD entry | Detect once with `incant!` over `#[magetypes]`/`#[arcane]` boundaries, or call an `#[autoversion]` dispatcher. |
| Tokenful tier body → tokenless variant family | `incant!(helper(args) without token)` calls the exact same suffix. No detection, no tier list. |
| Tokenless `#[rite(tiers...)]` → tokenful family | `incant!(helper(args), [tiers...])` selects a covered tier and emits its `from_context()` proof; no runtime upgrade. Include a scalar/default fallback when needed. |
| Covered tier body → known concrete tokenful helper | Pass the held token, its explicit downgrade, or `ConcreteToken::from_context()`. |
| Covered tier body → known tokenless helper | Call directly. Single-tier `#[rite(v3)]` keeps its original name. |
| Plain backend-generic helper → constructor | Keep `token: T`; call `Vector::splat_with_token(token, value)`. |

A tokenful `incant!` can still attempt runtime detection of stronger tiers
listed ahead of its covered fallback. Tokenless rite's rewrite only selects
covered tiers. These are intentionally different behaviors. `without token`
does not introspect the callee signature; both families must agree on suffixes.
The macros also do not propagate target features into nested `fn` items.

These calls, including turbofish type and const arguments, are exercised in
[`mixed_token_generics.rs`](../magetypes/tests/mixed_token_generics.rs):

```rust
#[magetypes(v3, neon, wasm128, scalar)]
fn entry<const ADD: bool, R: ChunkInput>(_token: Token, input: &[R; 8]) -> [i32; 8] {
    incant!(tokenless::<ADD, R>(input) without token)
}

#[rite(v3, neon, wasm128, scalar)]
fn tokenless<const ADD: bool, R: ChunkInput>(input: &[R; 8]) -> [i32; 8] {
    incant!(legacy::<ADD, R>(input), [v3, neon, wasm128, scalar])
}
```

**Resolved fallback gap (`3aef13f0`):** replacing the second attribute above
with `#[magetypes(rite, v3, neon, wasm128, scalar)]` now works too. Previously,
its scalar variant bypassed rite processing, leaving runtime dispatch that
attempted to call `legacy_v3` without the required context (E0133). Tokenless
rite scalar/default fallbacks now use the shared covered-tier rewrite; token
proof parameters retain their existing runtime boundary dispatch. Both forms
are exercised by the integration test. The vector-free regression in
[`magetypes_scalar_dispatch.rs`](../tests/magetypes_scalar_dispatch.rs) also
checks scalar/default fallbacks, the descriptive dispatch macro, and preserved
token-taking dispatch. No user unsafe is needed.

## Hard case 1: zenanalyze's pixel generic also selects a token-based trait

[`accumulate_row_simd`](https://github.com/imazen/zenanalyze/blob/b102fa5f3480f0c3911fa85a8a6ad23c64936f18/src/tier1.rs#L1277)
has three bool const parameters (`BT601`, `FULL`, `SKIN`) and `R: ChunkInput`.
Its caller selects eight const combinations. Inside the kernel,
[`R::load_chunk8(c, token)`](https://github.com/imazen/zenanalyze/blob/b102fa5f3480f0c3911fa85a8a6ad23c64936f18/src/tier1.rs#L1386)
uses a second generic, `Tok: DeinterleaveRgb24Chunk8`. The u8 implementation
routes through the token-selected garb loader; the f32 implementation gathers.

Recommended first rewrite: `define(f32x8)` → `use(f32x8)`, remove the token from
vector constructors, **retain `token: Token` and `R::load_chunk8(c, token)`**.
All four algorithm generics and existing `incant!` turbofish calls stay intact.
This preserves the trait's dispatch contract. Our integration test models that
two-axis generic interface and its owned Explicit→Context `.into()` boundary;
it does not port the production statistics or garb implementation.

Keep x8 here initially. The interface takes `[R; 24]` and returns three
`[f32; 8]` arrays; chunking, fixed reductions, and flush grouping also depend
on eight lanes. `use(f32xN)` requires a width-aware loader/reduction design and
numerical validation, not just replacing the type name.

## Hard case 2: zenfilters' backend-generic blur helpers

[`gaussian_blur_plane_dispatch_simd`](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/wide_simd.rs#L266)
calls `gaussian_blur_fir_generic<T: F32x8Backend + F32x8Convert + Copy>` or
`stackblur_plane_generic` according to the blur kernel. These helpers have
scalar slice interfaces and construct vectors inside their bodies.

Two valid migration choices:

- Keep the backend generic and token. Select `generic::local::f32x8<T>` and
  use `_with_token` constructors. This works without any new feature attribute
  and avoids changing the helper call graph.
- Replace backend genericity with tier-generated functions, keeping the
  algorithm single-source: `#[rite(neon, wasm128, use(f32x8))]`, no `T` or
  token parameter, short constructors. Inside the existing magetypes dispatcher,
  call `incant!(gaussian_blur_fir_generic(args) without token)` and likewise
  for stackblur. Preserve the actual consumer tier set; this file currently
  serves NEON/WASM and has separate x86 implementations.

Width adaptation is separate: FIR indexing uses `ci * 8`, `[f32; 8]` source
windows, and `chunks.len() * 8`. Stackblur has `Vec<[f32; 8]>` scratch, `w / 8`
tiles, and an explicit remaining-column path. Every one must follow the chosen
lane count. Preserve edge replication, tails, and row geometry when changing
that algorithm. No blur rewrite or performance comparison was run here.

## Hard case 3: zenav1's generic primitives and borrowed transform arrays

[`clampv`, `negv`, `widen16`](https://github.com/imazen/zenav1-aom/blob/66f0661e79590cee2a7055bfb028a216bb61c239/crates/aom-dsp/src/transform/simd/prims.rs#L74)
are plain `T: I32x8Backend` helpers. Generated kernels such as
[`av1_fdct8_impl`](https://github.com/imazen/zenav1-aom/blob/66f0661e79590cee2a7055bfb028a216bb61c239/crates/aom-dsp/src/transform/simd/txfm1d_v3_gen.rs#L20)
accept `&[I32x8<Token>]` and `&mut [I32x8<Token>]`, and call architecture-specific
raw primitives. Changing only `define(i32x8)` to `use(i32x8)` mixes constructor
modes between temporaries, parameters, and raw helpers.

A gradual bridge is **generic over both backend and constructor mode**:

```rust
use magetypes::simd::{
    backends::I32x8Backend,
    generic::{ConstructorMode, core_types::i32x8 as V},
};

fn clampv<T: I32x8Backend, M: ConstructorMode>(t: T, v: V<T, M>, bit: i8) -> V<T, M> {
    if bit <= 0 || bit >= 32 { return v; }
    let hi = ((1i64 << (bit - 1)) - 1) as i32;
    let lo = (-(1i64 << (bit - 1))) as i32;
    v.clamp(V::splat_with_token(t, lo), V::splat_with_token(t, hi))
}
```

Use `generic::core_types` for the two-parameter type: the backward-compatible
`generic::i32x8<T>` alias intentionally fixes Explicit mode. Inference supplies
`M` from input vectors, including `&mut [V<T, M>]`; no conversion or unsafe
slice cast is needed. A constructor-only helper such as `widen16` has no vector
input, so its result annotation or turbofish must select the mode.

Alternatively migrate a complete internal call chain to Context signatures,
spelling `generic::local::i32x8<Token>` explicitly. `use(i32x8)` is body-local;
it does not define signature types. Magetypes substitutes `Token` in explicit
signatures even in tokenless rite mode; plain rite does not substitute Token.
Owned values can cross modes with `.into()`; borrowed buffers cannot do so
automatically. Update the transform generator, not its generated kernels.
Keep x8: it is the algorithm's column batching and raw primitive contract.

The test preserves clamp's identity cases and checks boundary integers for
both modes through borrowed buffers. It also compiles/runs a tokenless
magetypes helper with explicit Context vector signatures. This is evidence for
those migration mechanisms, not a complete AV1 transform validation.

## Control case: zenavif's sample/output generics need no backend rewrite

[`yuv420_strip_kernel<S: YuvSample, P: StripPixel>`](https://github.com/imazen/zenavif/blob/dba8f5ee73ccd47b756a7a849a8d30a3e733108d/src/yuv_convert.rs#L538)
and [`rgbx_to_yuv420_kernel<P: ForwardPixel>`](https://github.com/imazen/zenavif/blob/dba8f5ee73ccd47b756a7a849a8d30a3e733108d/src/yuv_convert.rs#L1583)
use algorithm generics and ordinary loops. These inspected bodies do not
construct magetypes vectors; their token is unused. Keep the existing boundary
if existing callers dispatch to it, or move dispatch to autoversion / an outer
entry and use tokenless rite internally. Preserve `S`, `P`, strides, chroma
edge handling, and arithmetic unchanged. There is no reason to introduce
adaptive vectors solely to remove an unused parameter.

## Validation and scope

`just test-mixed-token-generics` runs the committed extraction under
`#![forbid(unsafe_code)]`. It covers mixed calls, type/const generics, both
constructor modes, owned conversion, borrowed buffers, explicit vector
signatures, and scalar plus runtime-selected dispatch. Downstream migrations,
assembly comparisons, and compile-time changes are not measured by this test.
The extraction passed on x86_64 with default features and with `avx512`, and
on AArch64 under QEMU. WASM compiled with `cargo check --test`; no WASM runtime
was available. Logs: `~/tmp/archmage-mixed-{test,platforms}.log` on the audit host.

After the fallback/gating fixes, both rite spellings passed on x86_64 and
AArch64/QEMU. The adaptive tests additionally executed forced V3/V4/V4x paths
under SDE, and WASM compiled. Reproduce with `just test-tier-gates`; audit-host
logs are `~/tmp/archmage-two-fixes-{checks,platforms}.log`.
