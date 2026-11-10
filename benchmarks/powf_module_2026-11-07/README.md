# Powf implementation studies

The maintained implementation contract and per-ISA coverage are in
[POWF_REFERENCE.md](../../docs/POWF_REFERENCE.md). The latest bounded
MIDP/LOWP replacement comparisons are in [MIDLOW.md](MIDLOW.md), with
[resource measurements](MIDLOW-DEBUG.md), [review](MIDLOW-REVIEW.md),
[frozen plans](MIDLOW-PLAN.md), and [artifact identities](MIDLOW-ARTIFACTS.md).

The current harness benchmarks the actual library through literal const options,
including token-preserving scalar tails. Build with `just powf-midlow-build`;
verify with `just powf-midlow-verify BINARY`; run the frozen matrix with
`just powf-midlow-matrix BINARY DIRECTORY`. Generated probes live in Cargo's
output directory. Source checkout generation is unnecessary.

Ordinary tests retain directed oracle and policy checks. Use
`just powf-test-exhaustive` for all frozen rational fixtures and all 64 bounded
policies across the full scalar/vector/tail length matrix. The same assertions
and numerical bounds apply to both tiers. `--all-features` also enables this tier.

Historical experiments document earlier implementations; their timings must not
be applied to the current library. Complete sources and outputs remain in the
linked artifact inventories and the preserved history branch.

- [Initial runtime-options matrix](MATRIX.md) and [replacement study](REPLACEMENTS.md).
- [Const policy study](POLICIES.md) and [policy plan](POLICY-PLAN.md).
- [Fused versus separate polynomial study](NONFUSED.md) and [frozen plan](NONFUSED-PLAN.md).
- [HIGHP SIMD and C/Rust/provider comparisons](../powf_providers_2026-11-09/README.md).
