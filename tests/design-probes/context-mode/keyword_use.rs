//! This probes Rust syntax only; it does not add a magetypes spelling.
#![forbid(unsafe_code)]

#[keyword_attribute::accept(use(f32x8))]
pub fn accepts_keyword_argument() {}
