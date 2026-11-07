# Power routing, gamma and feature study

Generic power policies now use the selected arithmetic context; the independent
reference remains the original CORE implementation. A compact positive-normal
classifier and certified gamma specializations improve selected workloads.
Std still wins general power; whole-transfer wins depend on corpus and call
boundary. [Results and paired intervals](RESULTS.md) include the losses.

This draft also separates additive math families and makes Magemath a superset
of all 35 deprecated Magetypes methods at widths four/eight and optional sixteen.
See [families](../../magemath/docs/FEATURES.md),
[migration contracts](../../magemath/docs/MIGRATION.md) and
[54 measured cold builds](../../magemath/docs/BUILD-COST.md). No Zenmath–Magetypes
dependency edge is introduced. Normal currently shares precise arithmetic.

| Evidence stage | Effective clean coverage |
|---|---:|
| Original native policies / staged transfer / feedback | 112 groups, 5,728 cases |
| Original strict WASM policies | 60 groups, 2,980 cases |
| Final gamma native and NEON, separately | 28 groups, 236 cases each |
| Final targeted strict WASM | 3 groups, 153 cases |
| Final admitted-domain exact-bit gates | 22 cells, 1,108,465,743 comparisons |
| Family cold consumer builds | 18 configurations × 3 replicates |

Original dirty groups and targeted replacements are retained. Only the exact
clean replacement union supports conclusions. Native/NEON gamma source is
production `4da80e0a`; earlier routing/policy stages remain separately identified.
The baseline has 79 Rust files from `4ccaf1cf`; the gamma baseline has 64 Zenmath
Rust files from `9f466de6`. Manifest changes rename only unpublished packages.
`just validate-baseline` checks both frozen copies.

The original matrix includes all nine correct/MIDP/LOWP, Full/precondition,
fused/separate policies, whole-loop scalar feature contexts, ordinary scalar,
packed, historical, exponent-spread, boundary, log-bit, uniform and special
controls. Positive exponent-spread bases range roughly from 2^-60 to 2^60.
Historical bases are [0.01,4.01). No historical C timing is used for a speed ratio.
Gamma fixed and runtime-y controls are distinct. Inactive/tail precondition
lanes receive valid (1,0) fillers. Adapters assert their block contracts.

C0 controls use linear-srgb's actual continuous constants, not textbook sRGB.
The original staged transfer includes preprocessing/postprocessing memory passes;
the final C0 control is a scalar single-pass composition. Fused/separate original
composition is labeled. `linear-srgb =0.6.12` rational functions are separate
whole-transfer controls; exact source comparison is in
[linear-srgb-provenance.json](linear-srgb-provenance.json).

Native allocations occur outside timed iteration. WASM `matrix_kernel`
allocates case vectors, inputs and outputs per host invocation; targeted calls
amortize that work across 256 repeats. Original large WASM sizes use
`max(1,262144/n)` repeats. Within-platform comparisons include matched wrapper
costs; absolute native/WASM numbers include different wrappers.

[Accuracy](ACCURACY.md), [linked codegen](CODEGEN.md),
[validation](VALIDATION.md) and [artifact identities](ARTIFACTS.md) explain the
scope and reproduction commands. Raw results, exact binaries, source snapshots,
locks and failed attempts live outside git in the archive. `freeze.py` refuses
existing destinations and atomically preserves each actual executable before
subsequent builds. Run `just final-audit <archive-root>` for strict inventory,
replacement, source, binary, chunk-digest and cold-build checks.

Draft only: no push, release or publication.
