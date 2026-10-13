# Audit of the self-contained prototype

## Scope and size

The supported contract is free functions: a direct feature body, a proof-taking
bridge, and explicitly ordered dispatch. It does not implement unified `attune`.

`wc -l src/lib.rs macros/src/lib.rs macros/src/tiers.rs` gives 7 + 324 + 91 = **422**
lines, including comments and blanks. At 55 lines per page that fits eight pages.
There are no other implementation files. Tests, tooling, docs and compiler output
are excluded from that source count and kept visibly separate.

This is self-contained relative to Archmage, **not dependency-free**. The compiler
side uses syn 3.0.9, quote 1.0.47 and proc-macro2 1.0.107. Their parsing, quoting and
span handling remain dependencies; their source is not counted in the eight pages.
The runtime is `no_std`, with no `alloc` dependency; `std` is an opt-in feature.

## Safety argument

Read [tiers.rs](macros/src/tiers.rs), then the three emitters in
[lib.rs](macros/src/lib.rs).

1. Each tier's features are its parent's features plus its additions. The same
   expanded list feeds target-feature attributes, build-time checks and runtime
   detection. The registry test rejects cycles, duplicate features and parents
   from another architecture. Eleven tiers are defined: scalar, v1, v2, v3, v4,
   v4x, neon, arm_v2, arm_v3, wasm128 and wasm128_relaxed.
2. Proof fields are private. `summon()` succeeds only with build-time guarantees
   or positive standard-library feature detection. `from_context()` has the full
   tier target-feature attribute. No unchecked public constructor exists.
3. `From` is generated only for same-architecture feature subsets, or scalar.
   These conversions cannot add a feature or change architecture.
4. Every non-scalar bridge takes the concrete proof from the canonical runtime
   crate and forwards to its original function. The only emitted unsafe expression
   is this call. Scalar bridges do not emit unsafe. The function's enabled features
   are precisely those proven by its parameter.
5. Arbitrary user unsafe contracts cannot cross that boundary: unsafe, async,
   const, external-ABI and variadic functions are rejected. Extra attributes,
   including extra target features and unexpanded cfg_attr, are rejected. Parameter
   names cannot shadow the original callee or the generated proof parameter.
6. Generated anonymous constants reject impl/trait scope, including receiverless
   functions. Otherwise unqualified callee lookup could select an unrelated module
   function. Explicit receivers also receive a tailored diagnostic. Nested free
   functions remain legal. Their normal Rust restrictions on capturing outer
   generics and locals still apply.
7. Direct calls use Rust's own feature checking. Dispatch contains safe proof
   acquisition and calls to bridges; it emits no unsafe. Arguments are evaluated
   once before selection. A hygienic match and labelled block preserve argument
   `?`, `return`, `break` and `continue` semantics.

The relevant compiler contracts are documented in the Rust reference on
[target features](https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute)
and [unsafe operations](https://doc.rust-lang.org/reference/unsafety.html).
Native detection delegates to Rust's
[x86 detection](https://doc.rust-lang.org/std/macro.is_x86_feature_detected.html) and
[AArch64 detection](https://doc.rust-lang.org/std/arch/macro.is_aarch64_feature_detected.html).
This audit does not re-prove rustc, LLVM or those standard-library detectors.

## Surface and limitations

* No methods, associated functions or trait generation. Move the kernel to a free
  function and call it from an ordinary method. The compiler diagnostic for a
  receiverless associated function mentions the generated anonymous constant.
* No arbitrary target-feature strings, user tier registry, rename dictionary,
  implicit family discovery, vector-width selection, cached central dispatcher,
  automatic tier suffix inference, or legacy-attribute compatibility.
* Bridge arguments must be identifiers; `mut` is accepted, destructuring and `ref`
  patterns are rejected. Functions using type/const generics, explicit lifetimes,
  where clauses, argument/return `impl Trait`, and explicit closure parameter types
  are tested. This does not claim coverage of every Rust signature extension.
* The crate's canonical name is embedded in expansions. If a dependency is renamed,
  the consumer must also provide the canonical crate alias.
* `cfg`, `doc`, `inline`, `track_caller` and lint attributes are accepted. Bodies
  default to inline; inline(never) is the opt-out. Inline(always) on a feature body
  is rejected. Wrapper visibility matches the body. `expect` is changed to `allow`
  only on the generated forwarding copy; the expectation stays on the original.
* Dispatch accepts a function path and an explicit ordered tier list ending in
  scalar. It does not discover what variants exist. Non-native branches are
  architecture-gated; all listed native functions must exist. The token expansion
  is linear in the number of branches, verified by an expansion test.
* `std` detection is at `summon()` / `dispatch!` boundaries. Repeated dispatch still
  does repeated cached feature tests; no custom per-tier cache or function-pointer
  cache is implemented. An existing proof bridge avoids detection.
* WASM is selected from build flags, not detected at runtime. A scalar dispatch arm
  does not make a module containing SIMD instructions loadable in an engine that
  lacks SIMD. Ship/select suitable modules externally. Rust's WASM feature-call
  rules differ from native unsafe feature calls.
* The tier lists originate from Archmage c7897905. They include its tier-specific
  extras, such as AES/PCLMULQDQ in v4. This prototype's proofs are separate types;
  they are not interchangeable with Archmage proofs or Magetypes constructors.

## Evidence

[tools/check.sh](tools/check.sh) is the reproducible validation entry point.
[tools/ui.py](tools/ui.py) verifies rejected programs against the freshly built
runtime crate, using both a failure exit status and a diagnostic. There are no
missing-tool or missing-file skips.

Validated on 2026-10-13 with rustc 1.99.0:

| Check | Result |
|---|---|
| Formatting and Clippy, warnings denied | Passed |
| Native std and no_std tests | Passed |
| Compile-fail proof, feature and unsupported-scope tests | Passed |
| Rust 1.89 workspace/all-features check | Passed |
| i686 all-targets/all-features check | Passed |
| QEMU x86 Nehalem, forced scalar expectation | Passed |
| QEMU AArch64 Cortex-A53, forced Neon expectation | Passed |
| Wasmtime WASI baseline, SIMD and relaxed-SIMD builds | Passed |
| Bare-metal AArch64 and Thumb no_std checks | Passed |
| x86 release assembly assertions | Passed |

The QEMU and Wasmtime results are functional checks, not native ISA benchmarks.
No Windows/macOS run, codec comparison, cold-build comparison or large-kernel
inlining study is claimed. Feature-specific vector arithmetic beyond the small
x86 FMA probe is not exhaustively validated here.

The assembly probe builds without target-cpu=native or global feature flags.
[codegen.txt](artifacts/codegen.txt) shows one `vfmadd213ss` and a return for both
leaf and matching-context caller. The baseline proof boundary contains one tail
jump to the feature body, with no CPU check or additional wrapper call. This
establishes those concrete examples, not a universal optimizer guarantee.

The committed expanded Rust files are inspection artifacts from cargo-expand;
they are not portable handwritten inputs and are not recompiled as standalone
crates. Regenerate them with `just codegen`.

The complete validation run used the shared resource limiter with a 16 GiB cap
and two build jobs. Its final resource report was:

```text
run-heavy: done rc=0 2s | peak-RSS 0.17GiB | min-avail 25791MiB | peak-load 0.38
```

This was an incremental validation run, not a cold-build measurement. Full output
and implementation hashes are in [validation.txt](artifacts/validation.txt).
