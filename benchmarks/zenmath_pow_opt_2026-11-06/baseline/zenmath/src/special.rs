//! Scalar special functions, behind the `special` feature. Empty today.
//!
//! erf/erfc/gamma/beta/Student-t and friends are only needed by zenmetrics-style
//! consumers, so they stay behind a feature and codec, resize and hot-path metric
//! builds never compile them.
//!
//! Planned sources:
//!
//! - `zenmetrics-stats/src/special.rs`: `erfc` (Chebyshev, <1.2e-7 rel),
//!   `norm_cdf`, `ln_norm_cdf` (asymptotic tail), `ln_gamma` (Lanczos
//!   g=7/9 terms, <1e-15), `inc_beta` (Lentz continued fraction),
//!   `student_t_cdf`/`upper_tail`/`quantile`.
//! - `zenmetrics-nn/src/afine.rs`: `erf`, `gelu`, `softplus`, logistic
//!   scalers.
//! - `zenjpegai/src/lh.rs` + `zenanalyze` sigmoid/erf call sites.
//!
//! All entries must be `no_std` scalar f64/f32 with documented error
//! bounds, and each needs a matching vector form (or an explicit
//! "scalar-only by design" note) before the feature is considered stable.
