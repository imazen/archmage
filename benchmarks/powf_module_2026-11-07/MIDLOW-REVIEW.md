# Bounded power integration review

This review covers the changes after `458e0978`; the earlier canonical HIGHP
integration has its [separate review](../powf_providers_2026-11-09/HIGHP-SIMD.md).
The complete PR remains draft, with performance promotion held.

The four bounded families are implemented for every supported vector width:
portable/backend-dependent MIDP/LOWP, four domains, checked/assumed validation,
automatic/caller execution. All 64 encodings have single/vector/tail execution
coverage. This is implementation coverage, not a global mathematical-error
proof or a complete clean timing matrix.

Reviewed decisions:

- `PositiveGamma` uses a separate bit only with the Gamma domain. Invalid public
  encodings, including private clamp/edge bits, remain rejected. Runtime adapters
  decode to the same literal const instantiations; no dynamic policy enters the
  Horner graph. The public API snapshots cover the new domain, presets and
  scalar-with-proof entry point.
- Removing clamps relies on the existing admitted ranges and independent
  rational interval certificate. General nonnegative LOWP retains the guard
  needed for its admitted zero lane at huge exponents. Zero lanes in Gamma
  still receive their explicit final value. No unsafe conversion, fast-math
  flag or floating-environment change is introduced.
- BitExact selects fused arithmetic once per operation. Scalar representation
  can use compile-proven FMA through the existing primitive; baseline x86
  retains exact software without a per-node runtime detector. BackendDependent
  retains native relaxed semantics, including separate strict-WASM arithmetic.
- Scalar tails carry the same proof and const flags as packed values. The
  no-token Caller scalar API explicitly selects ScalarToken; substituting that
  for a different backend-dependent vector token is not a tail-equivalence
  strategy. Std-backed HIGHP does not acquire a bit-exact promise.
- Existing approximate entry points select private const clamp/edge policies.
  Their scalar composition stays in the caller's frame; this is compiler
  scheduling control, not another public policy. Actual-before assembly is the
  compiler oracle. An independently written mathematical composition is a
  numerical oracle and can already have different assembly before remapping.

All 19 local `just ci` stages pass in 675 s on final library source `295001e3`,
including soundness, generation, API snapshots, Miri, docs, native/no_std tests,
cross-target checks and the existing consumer codegen comparison to main.
The combined CI/first-warning-sweep run reports peak RSS 2.49 GiB, minimum
available 41,123 MiB and peak load 5.69. That wrapper exits 1 because the
additional warning sweep finds two test imports unused only on i686. Scoping
those imports changes no library code or assertions; the subsequent complete
24-configuration sweep is clean. Final benchmark Clippy is warning-denied and
passes; the harness's full 64-policy verifier also passes (32 s, peak RSS
3.00 GiB, minimum available 42,023 MiB, peak load 1.15).

Only new safe numerical grids are excluded by the explicit Miri caller list;
their scalar memory paths and FMA primitives retain interpreter coverage.
Every pre-existing Miri/test expectation remains. A newly introduced fixture
incorrectly treated repeated std power calls as a bit oracle; its std-only
comparison now checks ordinary numerical results, while all canonical
comparisons retain exact bits. Initial interpreter/compiler failures and all
repaired orchestration attempts remain in the sealed evidence.

Final strict WASM, both relaxed fusion choices, no-FMA Nehalem, i686 and native
M6 std/no_std tests pass as detailed in [MIDLOW.md](MIDLOW.md). Codegen has
72/72 identical x86 bodies and 18/18 identical native NEON bodies; four scalar
ARM bodies change scheduling with the same or fewer instructions. No existing
assembly or numerical gate is loosened.

Missing promotion evidence comes first: most timing cells are flagged; latest
quiet Intel/N1/WASM/i686 and first-use measurements are missing. Two flagged
backend-dependent MIDP cells exceed the cost-only 2% interval; software-FMA
ScalarCaller has a substantial measured cost. The full-domain LOWP accuracy
problem remains with fused and separate arithmetic. The explicit bounded
presets, measured errors, all failed intervals and all source/run versions are
documented; none is converted into an unqualified speed or quality claim.
