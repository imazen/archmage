# External algebraic methods prototype

Unpublished, draft-only experiment: an actual no_std external library plus a
separate consumer. Production Magetypes/Archmage APIs and MSRV stay unchanged.
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

Measurements/reports are in progress. The justfile and scripts reproduce
checks and builds on eligible remote hosts through run-heavy. Full raw data
will be retained under `/mnt/v/output/archmage/algebraic-ext-2026-11-06/` on dev
and mirrored to tower. No push, PR or release.
