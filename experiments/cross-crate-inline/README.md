# Cross-crate inline experiment — 2026-10-07

Draft investigation; do not merge this experiment into main. The proposed
`attune`/`attuned!` interfaces are not implemented by this harness.

On Rust 1.98.1, explicit `#[inline]` enabled small and medium target-feature
kernels to inline across crate boundaries. Larger kernels remained calls.
Thin token wrappers with `#[inline]` or `#[inline(always)]` produced the same
normalized caller assembly as direct calls when the caller covered the callee's
features. Ordinary callers still crossed a feature boundary.

## Reproduction and scope

Source baseline: `cf07592212e96294ef9ca5dca9a364fa8d15d8ad`.
Compiler: rustc 1.98.1 (48a229cea 2026-09-01), x86_64-unknown-linux-gnu.
Assembly tool: cargo-show-asm 0.2.62. CPU: Intel Core Ultra 7 265K.

Run `just run /absolute/new/output/path` in this directory, or run `run.py
--out /absolute/new/output/path` under the host's resource-limited build wrapper.
The runner refuses to overwrite previous output. Its generated workspace uses
opt-level 3, LTO **off**, one codegen unit, no incremental compilation, empty
RUSTFLAGS, and no native CPU targeting. Dependencies are the local archmage
checkout plus a dependency-free helper proc macro. The output preserves the
generated source, dependency lockfile, environment, full command logs and assembly.

The 252 caller cases vary:

- Kernel: 1, 16 or 128 explicitly unrolled four-lane multiply/add stages.
- Kernel attribute: absent, `#[inline]`, or `#[inline(never)]`.
- Caller: 1 or 128 explicitly unrolled integer checksum steps.
- Context: baseline calling a V3 token entry; V3 calling V3; V3 calling V2.
- Route: direct where legal, or token wrapper with absent, `inline`,
  `inline(always)`, or `inline(never)` attribute.

The helper proc macro runs after existing `rite`/`arcane`, replacing the inline
attribute on the annotated function. This is necessary because current `rite`
replaces user inline attributes. Kernels use `rite`; token entries use the real
current `arcane` expansion, including its feature-enabled helper. This measures
existing Rust and archmage behavior, not an implementation of the draft macros.

All consumer entry functions have `inline(never)` so their bodies remain
inspectable; that does not prohibit inlining callees into those bodies. Both
caller sizes are observed, but this experiment does **not** determine a general
caller-size threshold. The 128-stage callee crosses the observed inline heuristic
where the 16-stage callee does not. It is not a calibrated LLVM threshold.

## Results

Each cell below is whether a provider call remains. Results agree for both
caller sizes and for matching/superset feature contexts.

| Kernel attribute | Stages | Direct | Wrapper absent | Wrapper inline | Wrapper always | Wrapper never |
|---|---:|---|---|---|---|---|
| absent | 1, 16, 128 | yes | yes | yes | yes | yes |
| inline | 1, 16 | no | yes | no | no | yes |
| inline | 128 | yes | yes | yes | yes | yes |
| never | 1, 16, 128 | yes | yes | yes | yes | yes |

All 72 baseline callers retained a provider transfer, regardless of kernel or
wrapper attributes. An always-inline wrapper did not erase the feature boundary.
Of the 180 matching/superset callers, 24 had no provider transfer and 156 retained
one. These are source-case counts; LLVM can alias identical functions.

All **72/72** comparisons of direct versus inline/always token-wrapper routes
had identical normalized caller assembly. This includes large kernels that
remain calls and kernels explicitly marked never-inline. Results are in
[results.csv](results.csv) and [comparisons.csv](comparisons.csv).

The unannotated ordinary `plain_tiny(x) -> x.wrapping_add(17)` control **did**
inline across crates (`lea eax, [rdi + 17]; ret`). Thus `#[inline]` is not a
universal prerequisite for cross-crate optimization on this compiler. The
unannotated target-feature kernels in this matrix did not inline across crates.

## Visibility and proposed API policy

Three separate negative compilation checks failed with E0603, as required:
direct calls to a private provider function, calls to a `pub(crate)` provider
function, and an exported macro attempting to call the private provider function.
Inlining and macro expansion do not grant source-level access to private items.

| Source visibility | Generated requested outputs | Downstream named access |
|---|---|---|
| private | private | no |
| pub(crate) | pub(crate) | no |
| pub | pub | yes, subject to containing modules/re-exports |

Consequences for the draft selectors:

- `pub #[attune(make(_*))]` exposes public feature-context functions. Use
  `#[inline]` by default, preserving explicit user inline policy.
- `pub #[attune(make(_*_t))]` exposes public token entries. Private implementation
  functions can remain private: their bodies can still optimize through an
  inline public wrapper. Downstream source cannot name those private functions.
- `make(_*, _*_t)` exposes both spellings. A downstream `attuned!` expansion can
  name the direct variant when its feature context covers the variant.
- `make(_)` exposes only the dispatcher. Downstream callers cannot select a
  private direct implementation through a macro. Either the family must expose
  another entry, or the call must explicitly retain dispatch semantics.

An alternative for token-entry-only families is for the caller macro to derive
the covered token with `from_context()` and call the public `_t` entry. This
matrix shows the underlying wrapper route can match direct codegen. It would
be a design change to the strict rule that attuned callers always name direct
variants, and needs an explicit contract. No public direct symbols should be
silently added just to make the macro work.

Recommended defaults remain `inline` on direct implementations, `inline(always)`
on the thin token wrapper, and no forced inline on the runtime dispatcher.
The experiment found no caller-assembly difference between inline and always
for these wrappers; always is a policy to remove the thin forwarding layer,
not a measured speed advantage. Dispatcher codegen was not measured here.

For the future single attribute, route an explicit `inline(never)` to the
implementation body, not automatically to the thin wrapper as well. The matrix
shows an always-inline wrapper can disappear while the never-inline body remains
a call. Document this routing: an attribute on the operation and an attribute
on every generated item are different policies.

## Validation and limits

Runtime validation exercised every case on four inputs, comparing all four
float lanes bit-for-bit with scalar multiply/add and the checksum with a loop
reference. All passed. All three privacy failures had the expected error code.
No generated Rust warnings in the successful run.

Assembly parsing reuses `xtask/codegen.py`, including alias resolution. The raw
rustc `.s` artifact is parsed because cargo-show-asm's whole-file display
demangles `.type`/`.size` directives even with `--keep-mangled` in this version.
Call counts are symbolic calls/tail transfers into the provider; GOTPCREL
references are statically identified callees, not runtime dispatcher evidence.

Recorded run: 16 seconds; resource wrapper reported:
`peak-RSS 0.30GiB | min-avail 21794MiB | peak-load 0.55`.
That is experiment execution, **not** a cold-compile comparison or runtime
performance measurement. No throughput or latency claim is made.

Not measured: other compilers/architectures, LTO, generic monomorphizations,
trait calls, central dispatchers, or the eventual attune expansion. Private-body
optimization through a public wrapper is distinct from private-name visibility.
