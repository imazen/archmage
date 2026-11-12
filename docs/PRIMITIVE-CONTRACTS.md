# Configurable min/max, reciprocal and reciprocal square root

Magemath owns the policy families. Magetypes keeps its existing primitive methods
and basic `sqrt`; this addition does not deprecate or change them. Min/max covers
f32 and f64. Reciprocal/rsqrt currently covers f32 only.

The [complete permutation reference](PRIMITIVE-PERMUTATIONS.md) inventories
supported options, equal-flag redirects, precision evidence and the canonical
timing matrix. The nested tables are also included in each family's rustdoc.

## Contracts

Every builder ends in `.bits()`, carried as a `u32` const through the public
entry, vector delegate and feature kernel. Full-domain defaults are checked in
the sense that every input is supported; they need no input validation branch.
Restricted Checked profiles reject the whole operation and return every invalid
lane in `PrimitiveError::lane_mask`. AssumeValid removes that check; violating
its precondition gives unspecified numerical output, never memory UB.

| Family / profile | Domain | Numerical contract |
|---|---|---|
| MinMax PORTABLE_NUMBER | Full | Prefer a number over either quiet/signaling NaN; two NaNs yield NaN. Min orders opposite zeros to -0, max to +0. |
| MinMax PORTABLE_PROPAGATE | Full | Either NaN yields NaN; ordered zeros. |
| MinMax RELAXED | Full | Backend NaN choice and zero sign. With one NaN either number or NaN is permitted; two NaNs yield NaN. |
| MinMax NORMAL / UNCHECKED_NORMAL | Finite normal operands | Checked / AssumeValid; one native comparison/min/max graph, no NaN/zero repairs. |
| Recip HIGHP | Full | Rounded `1/x`; rsqrt is rounded `1 / rounded_sqrt(x)`, **not** correctly rounded mathematical rsqrt. |
| Recip PORTABLE_LOWP | 2^-125 ≤ abs(x) ≤ 2^125; positive for rsqrt | Fixed integer seed, one canonically fused Newton step; relative error ≤ 6e-3. |
| Recip PORTABLE_MIDP | Same | Two canonically fused Newton steps; relative error ≤ 1e-4. |
| Recip BACKEND_LOWP / BACKEND_MIDP | Same | Same error bounds, ISA-dependent estimates/refinement and bits. |

`MinMaxDomain::Finite` also permits subnormals and signed zeros and rejects
infinities/NaNs. Individual NaN and zero policies can be combined with it. All
restricted presets check by default except the explicitly named UNCHECKED_NORMAL.
For reciprocal, add `.with_validation(RecipValidation::AssumeValid)` explicitly.
NaN payload/sign/signaling encoding and floating exception flags are unspecified
throughout. Non-NaN bit portability assumes round-to-nearest and gradual underflow.

NaN choices collapse on finite domains; both NaN/zero choices collapse on normal
domains. Full-domain validation and HIGHP portability choices also canonicalize
away in `.bits()`, avoiding duplicate instantiations for equivalent policies.
These are semantic axes; min/max does not have a meaningless precision selector.

## Callers and tails

Typed vector entries retain the supplied vector backend and width. Native proof
types select their arithmetic at compile time; Scalar Automatic may use std FMA
without changing its vector representation.
Arrays select once for the whole operation unless Execution::Caller is selected.
Scalar approximate reciprocal entries select a provider once, including portable
fusion. Scalar min/max and HIGHP reciprocal need no CPU detection. `no_std` cannot dynamically
detect optional x86 features; supply a proof or compile-time feature support.

A native estimate may differ between logical widths even with the same token
(for example, an AVX-512 16-lane seed versus its 4-lane seed). Use the matching
width-specific tail entry:

```rust
use archmage::ScalarToken;
use magetypes::simd::generic::f32x8;
use magemath::recip::{RecipOptions, try_recip_f32x8_with, try_recip_f32x8_lane_with};
const FLAGS: u32 = RecipOptions::PORTABLE_MIDP.bits();
let t = ScalarToken;
let packed = try_recip_f32x8_with::<FLAGS, _>(t, f32x8::splat_t(t, 3.0)).unwrap();
let tail = try_recip_f32x8_lane_with::<FLAGS, _>(t, 3.0).unwrap();
assert_eq!(packed.to_array()[0].to_bits(), tail.to_bits());
```

Portable profiles agree across widths and scalar/vector entries. Native profiles
promise their matching-width tail, not equality with autonomous scalar dispatch.
Array widths are 1/4/8/(16 with w512) for f32, 1/2/4/(8 with w512) for f64.
Other widths return UnsupportedWidth. Slice scheduling belongs to the caller.

## ISA implementations

| ISA | Min/max | Backend LOWP → MIDP reciprocal / rsqrt | Validation evidence | Timing |
|---|---|---|---|---|
| x86 V3 | Normal aliases existing native min/max; portable repairs for Full | rcp/rsqrt estimate → one relaxed-FMA refinement | Native directed f32/f64, width/tail checks; x4 full mantissas [1,4) | Zen 5 and Intel E-core x4/x8/x16 studies; overall unqualified |
| x86 V4/V4x | Same semantic graph; native widths | Existing width's estimate → refinement | Native directed x4/x8/x16 and f64 x2/x4/x8 | Zen 5 x4/x8/x16 measured; overall unqualified |
| ARM NEON | Ordered Propagate uses fmin/fmax; Normal aliases existing fminnm/fmaxnm | Raw frecpe/frsqrte → one existing NEON refinement | Native M6 std/no_std directed and x4 mantissa scan | M6 exploratory x4/x8/x16; every round resource-flagged |
| WASM SIMD128 | Propagate uses strict min/max; Number uses repairs | Reciprocal division; rsqrt fixed two-step estimate for both grades | Wasmtime strict/relaxed std/no_std directed | Complete canonical std min/max/reciprocal/rsqrt x4/x8/x16 (f64 min/max x2/x4/x8); 18/66 groups fully clean, whole suite fails admission |
| WASM relaxed SIMD | Normal or explicitly permissive policies may use relaxed min/max | No standardized reciprocal/rsqrt estimate instruction | Compiled opcode probes; relaxed token disabling tested | Hosted fusing/nonfusing reciprocal measured; strict suites fail admission |
| Scalar | Comparison/select plus requested repairs | Division / two-step rsqrt estimate, both grades alias | Exact-rational HIGHP fixtures; directed and x4 mantissa scan | Zen 5 Caller x4/x8/x16; overall unqualified |

The earlier reciprocal research matrix measures current portable MIDP reciprocal and HIGHP
division, existing Magetypes `recip()`, and two research graphs. Backend LOWP/MIDP
profile timings remain limited to the earlier V3 x8 study; the table's timing
column points to that earlier family evidence. The new complete canonical-cell
matrix is tracked in [the permutation reference](PRIMITIVE-PERMUTATIONS.md);
its M6 min/max, reciprocal and rsqrt tables are complete and exploratory.

Relaxed WASM selection requires a module compiled with relaxed SIMD and an
internal `Wasm128RelaxedToken::summon()`; module validation establishes feature
availability. `testable_dispatch` can suppress that selection. This is an
explicitly disclosed internal token lookup, not runtime discovery of a feature
missing from a loaded module. It optimizes away in ordinary relaxed builds.

Scalar/WASM native LOWP and MIDP alias where no cheaper useful estimate exists.
HIGHP can beat approximations on a particular CPU; the grades specify numerical
bounds, not a universal timing order. NEON LOWP now uses the actual raw seed;
MIDP avoids the unnecessary second refinement found during codegen review.

## Evidence and costs

Portable LOWP/MIDP now use canonical fused steps: hardware where proved,
exact software otherwise. The relaxed-WASM provider reuses powf's cached probe
once per operation, only in compile-time relaxed builds, and honors token
suppression. See the [fusion implementation and study](../benchmarks/primitive_variants_2026-11-10/FUSION.md).
The [initial unfused study](../benchmarks/primitive_variants_2026-11-10/README.md)
is retained as historical evidence; it does not describe the current graph.

The clean Zen 5 x8 dependency chains improve with fusion: MIDP reciprocal
226.856 → 177.889 ns and MIDP rsqrt 321.171 → 274.920 ns per 64 eight-lane
steps. Existing native Magetypes methods take 159.488 ns and 218.321 ns in
those same comparisons. Portable MIDP therefore still has an 11.54% / 25.92%
cost against those native graphs. The whole run remains unqualified because
rsqrt throughput has only 22/50 clean rounds; no legacy replacement is promoted.

`just primitive-verify` runs exact fixture regeneration checks, API generation
checks, strict Clippy and fast std/no_std directed tests. `primitive-cross`
exercises strict/relaxed WASM and compiles ARM; `primitive-wasm-token-testing`
checks relaxed selection suppression. `primitive-codegen` emits compile-only
probes. `primitive-accuracy` is an explicit full-mantissa scan, outside ordinary
cargo test: fixed four-lane storage, all 16,777,216 f32 values in [1,4), both
exponent parity classes, against f64 references. Directed tests cover edge
exponents and signed reciprocal inputs. This is not an exhaustive enumeration
of the entire normal domain or a proof for every future native CPU estimate.

The full x4 scan on Zen 5 and M6 agrees on portable fused LOWP maxima
3.362326737260e-3 (reciprocal) / 1.752338672098e-3 (rsqrt), and MIDP
1.141549399374e-5 / 4.723778778297e-6. The earlier unfused maxima remain in
the historical study. V3 backend LOWP maxima were 3.152981953462e-4 /
2.586608049593e-4; MIDP 1.538581244089e-7 / 2.108307648628e-7.
The independent exact-rational fixtures distinguish mathematical rsqrt from
sqrt-then-divide; the public HIGHP contract intentionally names the latter.

On the native M6 x4 scan, raw NEON LOWP maxima are 2.853393554688e-3 /
3.276822880217e-3; one-refinement MIDP maxima are 8.228816000135e-6 /
1.618327600566e-5. These pass the same bounds and confirm that a second NEON
refinement is unnecessary for this MIDP contract on this CPU. No final-graph
NEON timing or Neoverse result is claimed.

The [portable seed/correction study](../benchmarks/primitive_variants_2026-11-10/PORTABLE-SEEDS.md)
records real masking/rounding counterexamples and two benchmark-only deterministic
MIDP alternatives. Their numerical scans pass, but every new timing group failed
admission; neither changes the public graph or qualifies a legacy replacement.


The later [reciprocal qualification](../benchmarks/primitive_variants_2026-11-10/RECIP-QUALIFICATION.md)
keeps both prototypes outside public profiles. Parallel correction matches an
independent streamed C fused graph in Scalar/x86/NEON/strict and relaxed WASM;
its full normalized x4 maximum relative error is 1.145700113618e-5. It improves
some clean Intel E-core and Zen 5 dependency cells, but regresses other streams.
Six of 36 strict groups have all 250 rounds clean; no whole strict suite passes.

A separate canonical RCP14 prototype uses one native AVX-512 instruction at
x4/x8/x16 and reproduces the same bits using a 256-byte integer coefficient
table elsewhere. Its measured maximum relative error, 5.438657899502e-5, meets
MIDP but is less accurate than the current portable graph. Native Zen 5 matches
Intel's published C reference; physical Intel AVX-512 remains unmeasured.
Emulation on AVX2/NEON is substantially slower, so it is an explicit asymmetric
tradeoff, not a newly selected public default. Existing HIGHP division remains
the simple bit-portable alternative and can be cheaper for dependency chains.
The linked report retains every width/workload time in ns and all admission
limits, including unconfined, resource-flagged M6 exploratory timings.

The [current patched-zenbench repeat](../benchmarks/primitive_variants_2026-11-10/RECIP-PATCHED-ZENBENCH.md)
pins observer-affinity and macOS-counter fixes at `73302a1c`, absent from the
registry release used above. Five of 36 strict groups have all 250 rounds clean;
every whole strict suite fails admission. A clean Zen 5 AVX2 x8 long stream
regresses from 154.399 to 189.541 ns with parallel correction, while clean Intel
E-core cells improve. M6 remains unconfined and 0/50 clean in every group.
The original evidence is retained; numerical tests and public graphs are unchanged.

The [bounded-widening and canonical-rounding study](../benchmarks/primitive_variants_2026-11-10/CERTIFIED-RECIP.md)
keeps two further reciprocal designs benchmark-only. Specialized f64 evaluation
matches the existing fused LOWP/MIDP bits and reduces software-only costs; retain
native FMA when available. Exact RN14/RN8 agrees across tested ISAs, using guarded
correction rather than merely masking an estimate. RN14 spends accuracy and its
correction costs can exceed HIGHP division. The study names every implementation,
provides all widths/workloads in ns, and retains failed admission. Neither design
changes the public graph or qualifies a faster-than-native portable replacement.
