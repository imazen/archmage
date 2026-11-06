# External algebraic methods prototype

**Yes: an external crate can provide ergonomic algebraic extension methods
without Magetypes changes.** This unpublished, draft-only experiment uses an
actual no_std library plus a separate consumer. Production APIs and MSRV stay
unchanged.
This experiment requires Rust 1.98; 1.98.1 was checked directly.

Import `algebraic_magetypes_prototype::AlgebraicOps` to write
`a.algebraic_mul(b)`. The trait is ours, so Rust's orphan rules allow its
implementation for foreign vector types; foreign inherent impls and replacing
foreign operators would not be available choices. Ordinary Add/Mul stay unchanged.

No public token() or AsMut/array-view accessor was found on generic vectors.
The preferred tokenless implementation mutates a copied self with IndexMut,
preserving its stored proof. A separate explicit-token trait reconstructs
through to_array/from_array_t. Both use public APIs only, no unsafe, no CPU
summon inside any operation, no allocation in the library. Consumer-local
lane maps isolate the crate boundary from the IndexMut-vs-reconstruction choice.

Implemented widths: f32x4/x8/x16, f64x2/x4/x8. No generic x1 wrapper was found.
All five binary methods plus a separately named horizontal algebraic addition
are implemented. Exact fixtures have distinct lanes across the full width and
varying positive, nonzero power-of-two denominators. Algebraic arithmetic has
no accuracy, evaluation-order or bit-reproducibility guarantee; finite mixed
observations do not create one.

The library's compatible dependency range is resolved by checked-in lockfiles.
Independent `published-beta/` and `published-030/` workspaces have **no patches**:
both compiled and ran against registry Magetypes/Archmage 0.9.31-beta.2 and
0.9.30 respectively. The measurement workspace patches its frozen local tree.
This proves unpublished production changes are unnecessary for these APIs;
it is not a promise that every future Magetypes version is compatible.

Focused workload matrix: f32x8/f64x4 polynomial, dot, division and remainder;
ordinary, external tokenless, external explicit-token, local-map and automatic
scalar algebraic loops, plus fused and horizontal-reduction controls where
meaningful. Remainder numerators span -1024..1022, with non-power-of-two
positive denominators; the workload exercises nonzero quotients. Other widths
have exact fixture coverage rather than a duplicated timing matrix.

All 644 timing cells completed across eight workloads, LTO off and thin, with
at least 30 paired rounds. [Full eight-workload tables](RESULTS.md) compare
ordinary, tokenless and local-map forms for V3, V4x, NEON and WASM128. The
[CSV tables](results/native-off-v3.csv) also retain explicit-token comparisons,
means, medians, MAD and paired uncertainty; `controls-*.csv` retain every form,
including automatic scalar algebraic loops and explicit fused controls.

The crate boundary disappears in the inspected polynomial consumers even
without LTO. V3 index/token/map loops each use six packed FMAs per 16 values;
NEON is also packed and strict WASM retains separate packed multiply/add.
There are no external calls or array stack traffic in these inspected hot
loops. See [linked assembly, symbols, hashes and IR](CODEGEN.md).

This is not a blanket speed win. For example, no-LTO NEON f32 dot measures
0.09489 ns/value tokenless versus 0.12985 ordinary, with a paired difference
95% interval [-0.04705, -0.02377]. No-LTO V3 f64 polynomial instead measures
0.16966 versus 0.12045, interval [0.02017, 0.08162]. Thin V4x f64 polynomial
also regresses: 0.19822 tokenless versus 0.14891 ordinary, interval
[0.01029, 0.09806]. Baseline scalar and automatic-loop controls matter;
explicit vectors do not own every algebraic gain.

Tokenless indexing is a usable ergonomic choice, not a guarantee of the same
optimizer decisions as reconstruction. Most index/map comparisons overlap
zero, but counterexamples exist: thin V4x f64 polynomial is slower than map
by [0.02184, 0.14924] ns/value. For cross-crate overhead specifically, compare
**explicit-token reconstruction with the same-form local map**, then inspect
codegen. Close timing means alone do not establish zero overhead. Binary layout
and increment placement differ even where hot-loop arithmetic matches.

Measurements use zenbench 0.1.11, 1,024 values, whole-loop feature contexts,
matching inputs and observed outputs, no target-cpu=native. Runtime tier
inventories record both supported and unavailable tiers; caller configuration
asserts expected tiers. Native x86 ran on Ryzen 9 7900X; NEON on Neoverse N1;
WASM used Wasmtime 49.0.5 on Ryzen 9 7900X. Zen 5, Windows, macOS, other buffer
sizes and dependency-latency workloads were not measured in this focused study.
Other vector widths have fixture coverage rather than throughput measurements.

Native allocates input/output buffers outside timed iterations. WASM
`ext_kernel` allocates the case vectors, inputs and output **per host invocation**,
amortized across 256 repeats. Within-platform comparisons are valid for these
wrappers; absolute native-versus-WASM values include different wrapper costs.
`ordinary` remainder is a local lane map of `%`, since Magetypes has no vector
Rem operator. Division and remainder timings therefore deserve their own
interpretation rather than a general packed arithmetic claim.

Intervals in RESULTS use all retained paired rounds with no post-hoc removal.
Substantial mean/median gaps, high variation and drift remain in the saved
results; intervals describe this run and do not remove systematic drift or
multiple-comparison concerns. Do not rank close means. Algebraic horizontal
reduction can change the accumulation loop itself, as the V3 control shows.

[Accuracy observations](ACCURACY.md) retain exact fixtures and independently
rounded MPFR references. Algebraic differences are observed, not bounded.
[Validation and archive provenance](ARTIFACTS.md) identify full logs, source
snapshots, binary hashes and the verified mirror. The local justfile/scripts
reproduce checks and extraction. No push, PR or release.
