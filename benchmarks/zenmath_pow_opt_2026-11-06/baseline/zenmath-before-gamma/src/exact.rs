//! Bit-exact libm-convention ports, behind the `exact` feature. Empty today.
//!
//! For callers whose contract is "identical bits to a specific libm on
//! every platform" (e.g. `zenfleet-core`'s `libm::log` determinism
//! requirement, `zensim`'s `PowForm::LibmPowf`/`PureRust` axis, and the
//! fdlibm/glibc vector ports in `zenmetrics-math`).
//!
//! Two conventions already exist in the workspace and must remain
//! distinct — they produce different bits:
//!
//! - **fdlibm / msun** (`zenmetrics-math/src/fdlibm.rs`, `fdlibm64.rs`):
//!   FreeBSD/SunPro `e_powf.c`/`e_expf.c`/`e_log.c` lane transcriptions,
//!   bit-identical to the `libm` crate. Vectorized with scalar `libm`
//!   fallback for exceptional lanes; `EXACT`/`POSITIVE_FINITE`/
//!   `POSITIVE_NORMAL` coverage gates compile out bookkeeping.
//! - **glibc** (`zenmetrics-math/src/glibc.rs`): Szabolcs Nagy's
//!   `e_powf.c` (glibc ≥2.28 `__powf`), different algorithm, different
//!   bits. Its exceptional lanes still call the host libm, so the current
//!   consumer implementation is not yet a complete portable port.
//!
//! Scalar exact ports follow the same rule: the algorithm is pinned to a
//! named libm convention, never "whatever the platform libm does".
//!
//! Vector exact ports need the fdlibm lane-transcription pattern
//! generalized beyond pow/exp/log/log2, plus f64 vector coverage
//! (currently only `f64x4` pow/log/log2 exist).

//! The migration inventory and version-selection review are maintained in
//! `zenmath/docs/INVENTORY.md`. Compatibility means a pinned implementation,
//! not correct rounding and not equality to every release of that lineage.
