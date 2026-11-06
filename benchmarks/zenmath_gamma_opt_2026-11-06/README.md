# Certified fixed-gamma power experiment

Eight admitted-domain finalist paths passed403,078,452 exact CORE comparisons
and cross-path digest audits. Fixed scalar piecewise Estrin5 and packed global
Estrin13 decode/Estrin15 encode beat std in all eight power timing groups.
Whole C0 transfer has one measured regression; production overhead and
cross-platform integration remain main-lane work. No production API edits/push.

- [RESULTS.md](RESULTS.md): coverage, timings, C0 distinction and frozen identities.
- [BOUND.md](BOUND.md): exact-v full-mantissa certificate.
- [PIECEWISE-BOUND.md](PIECEWISE-BOUND.md): independent16-center certificate.
- [VECTOR-BOUND.md](VECTOR-BOUND.md): per-lane array4/8 argument.
- [NUMERIC-BOUNDS.md](NUMERIC-BOUNDS.md): initial generated budgets.
- [ARCHIVE.md](ARCHIVE.md): retained artifacts and byte-integrity audit.

`extract_finalists.py --run RUN --out tables.rs --manifest selected.json`
extracts902 bit-identical constants (7216bytes) from frozen proofs and verifies
every selected bit against archived compiled Rust. Selected global tables need
560bytes and scalar piecewise tables6656bytes; production linked size was not
measured here. Preserve proof DAG/order, outward rounding and independent
`zenmath::scalar::reference::powf` rescue. Ordinary scalar dispatch is not an oracle.

Use only admitted remote compute through `job.sh`/run-heavy. `justfile` holds
local command recipes; do not run benchmarks/builds on dev. Numerical modules
remain independent of the x86-only timing module and dependency. All results
retain the old4ccaf1cf feature graph; root handles the new feature split.
