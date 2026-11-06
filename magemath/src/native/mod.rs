//! Reserved for ISA-specific kernels. Empty until a caller demonstrates a need.
//!
//! [`crate::portable`] provides matching bits and [`crate::relaxed`] provides
//! backend-dependent polynomial rounding with the same special-value handling.
//! This module is reserved for other ISA-specific choices, such as reciprocal
//! estimates or per-ISA tables, with separately documented contracts.
//!
//! A narrow input range alone does not make a kernel native: a range-specific
//! polynomial can still be portable. Never wire native kernels into a portable
//! dispatcher. The design record is `docs/DESIGN.md`.
