# Token arguments and feature-context migration

This analysis concerns the 42 public vector-method names listed in
[TOKEN-PARAMETERS.md](TOKEN-PARAMETERS.md), plus token-receiver traits.
It evaluates a possible API change; it does not implement that change.

## Source inventory, 2026-09-27

A Sol agent used ripgrep 15.2.0 and a Rust tree-sitter parser to inventory
17 primary `zen*` repositories available on this host and three related
checkouts containing zen crates (two additional origin repositories). Sixteen primary repositories were local
snapshots; `zenav1-aom` used remote commit
`66f0661e79590cee2a7055bfb028a216bb61c239` because its local checkout had
foreign work. The extended checkouts were `jxl-encoder`, `cavif-rs`, and
`ravif`; the latter two share an origin. Only `jxl-encoder` added matched
constructor calls. Linked workspaces
were excluded. This is a source snapshot inventory, not a downstream build or
an inventory of every remote repository. The parent independently recomputed
totals from the call records and inspected representative generic helpers.

| Scope | Explicit-token calls | Enclosing functions | Production calls |
|---|---:|---:|---:|
| Primary zen repositories | 3,245 | 463 | 3,170 |
| Additional jxl-encoder | 1,325 | 206 | 1,325 |
| Combined | 4,570 | 669 | 4,495 |

Every counted call would require an argument edit if its existing signature
lost the token. This includes 25 partition calls, which need no CPU proof.
For feature-context migration, primary repositories have **341 direct
constructor/load calls in 69 plain functions**, plus eight partition calls in
those functions. Six more constructor expressions occur in local macro
definitions inside plain functions. `jxl-encoder` adds two plain functions with
one constructor call each. These are candidates for context changes, not a
measured compiler-error count; no downstream migration was compiled.

| Enclosing context | Primary zen calls | Combined calls |
|---|---:|---:|
| `#[magetypes]` | 1,862 | 1,985 |
| `#[arcane]` | 897 | 1,830 |
| `#[rite]` | 88 | 355 |
| Direct `#[target_feature]` | 1 | 1 |
| No recognized feature attribute | 349 | 351 |
| Macro-definition expressions requiring expansion review | 48 | 48 |

An annotated function is not automatically compatible: the concrete token's
features must be covered, closures and nested items require review, and scalar
variants must remain valid. Counts are expressions written in source, not macro
expansion counts. Function totals use source enclosing functions; the ten
zentone expressions in a module-level macro have no enclosing function at their
definition site. Their expansion callers were traced separately.

### Methods used

| Method | Primary zen calls | Combined calls |
|---|---:|---:|
| `splat` | 1,403 | 2,033 |
| `from_array` | 807 | 852 |
| `zero` | 514 | 676 |
| `from_slice` | 114 | 543 |
| `load` | 251 | 255 |
| `from_repr` | 90 | 114 |
| `from_m256` | 0 | 29 |
| `from_m256i` | 27 | 27 |
| `partition_slice_mut` | 16 | 16 |
| `load_8x8` | 9 | 9 |
| `partition_slice` | 9 | 9 |
| `from_m128i` | 5 | 5 |
| `from_i32x4` | 0 | 2 |

No uses were found for the other names in the 42-method inventory. The separate
search for WidthDispatch/F16Convert token-receiver calls, backend UFCS calls,
and constructor function-value references found no confirmed magetypes uses;
this search is not whole-program trait resolution.

Full call records retain path, line, signature, enclosing attributes, token
expression, import/type evidence, and snapshot commit. See the
[audit artifact pointer](TOKEN-AUDIT-2026-09-27.pointer.md) for the records,
per-function inventory, source snapshots, and reproduction scripts.

## Distinguish the changes

Removing a token parameter from an existing method changes every direct call
using that signature, even if its caller already has sufficient target features.
Function-value references can also change type. Source-call counts do not equal
the number of compiled variants produced by `#[magetypes]` or other macros.

Requiring target features adds a separate restriction. A plain generic helper
with a token parameter can call today's constructors. Its caller's features and
`#[inline(always)]` do not give the helper a feature context. `#[rite]` requires
a concrete tier, concrete token, or recognized feature trait; a backend bound
alone does not name a tier. See [rite expansion](../archmage-macros/src/rite.rs)
and [generic vector constructors](../magetypes/src/simd/generic/generated/f32x8_impl.rs).

`#[magetypes]` generates concrete per-tier bodies, including scalar/default
fallbacks where requested. Native variants carry feature attributes; scalar
variants need constructors that require no SIMD features. An annotation's
presence alone does not establish that it covers every concrete token used in
the body. See [variant generation](../archmage-macros/src/magetypes.rs).

## Proof already present in the inputs

These methods can reuse the token stored in a vector argument instead of
requiring a caller feature context:

- `from_i32` and `from_i32x4`/`from_i32x8`/`from_i32x16`: the integer vector
  already carries the same `T`. The existing receiver alternative is `to_f32()`.
- `from_i32_bitcast`: the existing receiver alternative is `bitcast_to_f32()`.
- `from_halves`: both halves carry the same `T` used by the result; the
  `F32x8FromHalves`/`F32x16FromHalves` bounds determine which widths that token
  supports. A possible additive receiver form is `lo.concat(hi)`.

Sources: [conversions](../magetypes/src/simd/generic/generated/i32x4_impl.rs),
[cross-width construction](../magetypes/src/simd/generic/cross_width.rs).

`partition_slice` and `partition_slice_mut` do not use their token arguments.
They return scalar array chunks and a remainder via `as_chunks`/`as_chunks_mut`;
adding CPU-feature requirements would not follow from those operations.
See [partition implementation](../magetypes/src/simd/generic/generated/f32x4_impl.rs).

## Additive naming options

Preserving existing names/signatures and adding context-only methods causes no
source break for callers that retain the token-based API.

| Existing call | Recommended additive name | Shorter alternative |
|---|---|---|
| `V::splat(token, value)` | `V::splat_in_context(value)` | `V::splat_ctx(value)` |
| `V::zero(token)` | `V::zero_in_context()` | `V::zero_ctx()` |
| `V::load(token, data)` | `V::load_in_context(data)` | `V::load_ctx(data)` |
| `V::from_array(token, data)` | `V::from_array_in_context(data)` | `V::from_array_ctx(data)` |
| `V::from_slice(token, data)` | `V::from_slice_in_context(data)` | `V::from_slice_ctx(data)` |
| `V::from_repr(token, repr)` | `V::from_repr_in_context(repr)` | `V::from_repr_ctx(repr)` |
| Native `V::from_m256(token, raw)` etc. | `V::from_raw(raw)` (pending change) | — |

`_in_context` states the call-site requirement without suggesting unsafety or
runtime detection. `_ctx` is shorter but less explicit. `_unchecked` and
`_unsafe` would misdescribe a compiler-checked safe call. Renaming existing
methods to `_with_token` while reusing their old names for context-only methods
would still break existing source; it is not an additive migration.

These are naming proposals, not a claim that one generic `impl<T: Backend>` can
have target features vary with `T`. Context-only methods need concrete tier
implementations or macro specialization. The existing token-taking generic API
remains useful for genuinely backend-generic helpers.

An existing alternative needs no new vector API:

```rust,ignore
#[archmage::rite(v3)]
fn helper(value: f32) {
    use magetypes::WidthDispatch;
    let simd = archmage::X64V3Token::from_context();
    let vector = simd.f32x8_splat(value);
    // Further constructors can use the same token receiver.
    let _ = vector;
}
```

The [WidthDispatch trait](../magetypes/src/width.rs) already provides
`<vector>_splat`, `<vector>_zero`, and `<vector>_load`.

## Short naming and compilation structure

Between `c_zero()` and `zero_c()`, the proposed suffix form keeps the operation
first and groups each contextual variant with its existing name. These are
associated constructors (`V::zero_c()`), not vector receiver methods. `_ctx`
is clearer than `_c` when the extra two characters are acceptable.

The current `zero(token)` and other common constructors are inherent associated
functions on `impl<T: Backend> Vector<T>`, delegating to sealed backend trait
methods whose receiver is the token. Context-only constructors would instead
be inherent functions on concrete token instantiations, as in the pending
`from_raw` generator. They do not require adding another public trait.
Rust disallows `#[target_feature]` on safe trait methods; it cannot be used to
encode a backend-dependent feature requirement in an ordinary safe trait call.
See the [Rust Reference](https://doc.rust-lang.org/reference/attributes/codegen.html#attributes.codegen.target_feature.allowed-positions).

Compile-time impact has not been measured. Six constructor variants across
30 vector types would mean 180 additional function definitions per fully
supported tier before target/feature cfg filtering. This is a count of proposed
API definitions, not a timing prediction. Source parsing, macro expansion,
type checking and metadata still have costs; generated machine-code work
depends on the concrete functions and generic instantiations collected for
code generation. See [rustc monomorphization](https://rustc-dev-guide.rust-lang.org/backend/monomorph.html).
The implementation can keep these wrappers small and delegate to the existing
constructors rather than duplicate backend algorithms. Numeric build overhead
requires a controlled before/after check and release-build comparison.

For closure review, the current Rust Reference explicitly says closures within
a `target_feature` function inherit its feature attributes. The inventory's
closure flag is source-context metadata, not evidence that those calls need
new annotations. Nested function items are a separate case. See
[closure inheritance](https://doc.rust-lang.org/reference/attributes/codegen.html#attributes.codegen.target_feature.closures).

## Opt-in alias selection: `define(...)` and `local(...)`

`define(f32x8)` retains the explicit-token alias and `f32x8::zero(token)`.
`local(f32x8)` selects `magetypes::simd::generic::local::f32x8<Token>` and
`f32x8::zero()`. Both aliases fix a constructor policy on the same generic
core. The policy is a sealed, zero-sized marker; arithmetic, storage, casts,
transcendentals, and width conversions share their implementations.

```rust
#[archmage::magetypes(local(f32x8), v3, neon, wasm128, scalar)]
fn scale(_token: Token, input: &[f32; 8]) -> [f32; 8] {
    (f32x8::load(input) * f32x8::splat(2.0)).to_array()
}
```

The generator derives both public constructor signatures from one token-taking
implementation. Existing constructors keep their arguments. Context constructors
are concrete-token inherent functions with registry-derived `#[target_feature]`,
`#[inline]`, and `#[forbid(unsafe_code)]`; their bodies obtain the token through its checked
`from_context()`. Scalar constructors use `ScalarToken` directly. The available
tiers follow the existing backend implementations; a mode does not add backend
capabilities that a token previously lacked. Simple contextual value constructors
call the backend directly and retain `new_repr` as the shared storage constructor.
Multi-step loaders and memory views keep their shared token-taking helpers, as
do explicit constructors and shared operations. The generator writes each
algorithm once; its structural flattening pass rejects blocks and ambiguous
backend bounds. See the [optimization measurements](../benchmarks/constructor_codegen_compile_2026-09-27.md).

The fixed alias matters: adding only a defaulted mode parameter makes inferred
`Vector::zero(token)` ambiguous (E0034). Fixing `Explicit` in the existing public
alias preserves that call. This was verified in the standalone compiler matrix
and in `magetypes/tests/magetypes_local_flag.rs`. A second vector implementation
is not necessary. The core is exposed under `generic::core_types` for code that
intentionally abstracts over the sealed `ConstructorMode` parameter.

The two modes are distinct Rust types. Use `.into()` to cross an existing API
boundary; both directions move the representation and stored token without
unsafe code or CPU detection. Ordinary operations preserve mode, including
integer/float conversions, borrowed bitcasts, half-float conversions, and
cross-width operations. No `Deref` conversion is involved.

`local` describes constructor selection, not a lifetime: a constructed vector
may leave the function. It still carries the token that proves CPU support.
A safe trait method cannot itself have a target-feature requirement; `From<Raw>`
and generic trait construction cannot substitute for these inherent constructors.
Safe `From` conversions between the two already-proven vector modes are valid.
See [type aliases](https://doc.rust-lang.org/reference/items/type-aliases.html)
and [target-feature restrictions](https://doc.rust-lang.org/reference/attributes/codegen.html#attributes.codegen.target_feature.allowed-positions).

The regression suite checks missing/weaker feature contexts, unsafe function
pointer coercion, nested functions, trait wrappers, matching and superset
contexts, legacy inference, all 30 vector shapes, and both alias modes together.
Cold build measurements use `scripts/measure-local-mode-compile.py`; source
snapshots and full logs are recorded with the measurement results under
`benchmarks/`.

## Validation of the implemented modes

The implementation landed in `8ef7db6f`; documentation and test-source cleanup
landed in `751960ac`. `cargo run -p xtask -- ci` completed successfully on
2026-09-27, including generation reproducibility, soundness scans, default and
no_std tests, macro snapshots, bare-metal compilation, API snapshots on three
targets, and documentation. The script's optional Miri, Docker/cross, and WASM
runtime stages were unavailable on this host.

Separately, the new constructor and raw-interop tests passed on x86-64 and
under AArch64 QEMU. All-feature compilation passed for x86-64, AArch64,
wasm32-unknown-unknown, and i686. The six earlier standalone design-probe cases
also retained their expected results. The changed library and new tests pass
Clippy with warnings denied. Broader all-target Clippy has existing failures in
unchanged examples/tests, recorded in `CLAUDE.md`.

Cold-build results and retained evidence are in
[the measurement report](../benchmarks/local_mode_compile_2026-09-27.md).
The full CI log is retained at
`/home/lilith/data/archmage/local-mode-compile/2026-09-27/validation-ci.log`.
