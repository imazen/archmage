//! Archmage and Magetypes usage patterns taken from zen* crates, reduced to
//! small kernels with scalar reference checks.
//!
//! Every pattern here compiles against **published 0.9.31** (`../published`)
//! and against the workspace candidate (`../candidate`), so a candidate change
//! that breaks one of them breaks a real consumer. Each item names the crate
//! and file it mirrors. Patterns that need unreleased API live in
//! `tests/zen-patterns/unreleased` instead.
//!
//! Inventory source: a read-only scan of zen* checkouts on 2026-11-15
//! (465 `#[attune]` sites in 41 shapes, 179 `attuned!`, 1,475 `incant!`,
//! 2,221 `#[arcane]`, 1,852 `#[rite]`, 765 `#[magetypes]`, 382
//! `#[autoversion]`). No zen crate used an API missing from 0.9.31.
#![allow(clippy::too_many_arguments)]

pub mod attune_families;
pub mod attune_helpers;
pub mod legacy_macros;
pub mod vector_ops;

#[cfg(test)]
mod reference {
    /// Relative-or-absolute closeness for approximation kernels.
    pub fn close(got: f32, want: f32, rel: f32) -> bool {
        if got == want {
            return true;
        }
        let diff = (got - want).abs();
        diff <= rel * want.abs().max(1e-30) || diff <= 1e-30
    }

    pub fn ramp(n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|i| lo + (hi - lo) * i as f32 / (n.max(2) - 1) as f32)
            .collect()
    }
}
