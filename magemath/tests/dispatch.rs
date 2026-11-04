//! The runtime dispatchers give the same bits whichever tiers the CPU reports.
//!
//! Lives in its own test binary: disabling tokens is process-wide, and the other
//! tests assert that the tokens they need are present.

mod common;

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use common::*;

#[test]
fn every_token_permutation_gives_the_same_bits() {
    let mut input = sweep_bits(1 << 12);
    input.extend(named_edges());
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        for op in OPS {
            let mut got = input.clone();
            slice_dispatched(op.name, &mut got);
            for (&g, &x) in got.iter().zip(&input) {
                assert_same(&perm.to_string(), op.name, x, g, (op.mirror)(x));
            }
        }
    });
    // x86-64 can disable V3, so the scalar fallback must really have run. NEON is
    // baseline on AArch64 and WASM's tier is decided at build time, so
    // `parity.rs` reaches their scalar tier through its proof wrappers instead.
    #[cfg(target_arch = "x86_64")]
    assert!(report.permutations_run >= 2, "{report}");
    let _ = report;
}
