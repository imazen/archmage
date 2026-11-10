# PR147 independent review — 2026-11-10

**Do not merge the reviewed head.** The polynomial refits have useful measured
accuracy improvements, but the policy implementation has executable correctness
failures and qualified performance regressions. This review is for
`0464774e7c5c63a398041ddaa131b9bd4f7035c7`; the separate main reference is
`91a13739b73f46213ecfe7083c77297c531fc8ad`. Production sources were left unchanged.

Missing evidence comes first: no_std is blocked by a compile failure; quiet Intel,
ARM Linux and WASM timing matrices are absent. Thirty-eight of 96 AMD timing
cells fail cleanliness. The M6 timings are exploratory because its OS scheduler
does not provide verified core-class confinement. Numerical coverage is sampled,
not an exhaustive proof over all f32 inputs. Bit-distributed and exceptional-input
timing matrices, log1p accuracy sweeps, first-use dispatch costs and a matched
consumer binary-size comparison remain absent. These gaps prevent an unqualified
promotion claim independently of the demonstrated failures below.

## Required corrections

1. **Restore no_std compilation.** `logf/const_api.rs:99,199,236` calls the
   std-only scalar `f32::mul_add`. `cargo check -p magetypes --no-default-features`
   fails with three E0599 errors. Use the existing token-proven fused/relaxed
   primitives and retain the selected arithmetic policy throughout log1p.
2. **Make valid options work in both profiles.** There are 64 builder encodings
   per family. The debug shared-kernel table masks in LEGACY but has no LEGACY
   arms: 32/64 const configurations panic per family. It also drops BACKEND,
   so debug calls no longer select the requested relaxed arithmetic provider.
   Both runtime adapters recognize only 10/64 configurations; Caller, legacy
   precision and many valid combinations return UnsupportedOptions. Generate
   the complete table from the option axes and test all encodings against const
   calls, including rejected lanes. Do not hide this with release-only testing.
3. **Separate validation removal from valid-domain result handling.** Full and
   Specials accept exceptional inputs, so AssumeValid still promises their
   documented results. Current `Full + AssumeValid` turns NaN into a finite
   clamp result; `Specials + AssumeValid` turns zero into a finite logarithm.
   Removing validation is not permission to change results for accepted inputs.
4. **Correct signed zero and scalar error masks.** Specials log2(-0) returns NaN
   instead of -infinity; log1p(-0) returns +0. The scalar token log1p error mask
   is 0x000f rather than 0x0001 because it leaks its splat's lane mask.
5. **Resolve the dead LOWP implementation before claiming its measurements.**
   The outer LOWP branch always takes the floor-split cubic; the degree-2
   round-split branch inside its else is unreachable. Actual canonical LOWP
   has three FMAs, not two. Merely making the quadratic reachable is incorrect
   under the existing range: the exact rational mirror and directed MPFR give
   1.2516% relative error at 127.75 and 2.8733% at f32(127.99), above the 0.56%
   expectation. Fix the upper-binade reduction or explicitly propose a narrower
   domain before changing that graph. No expectation was weakened in this audit.
6. **Remove unsupported compatibility and zero-cost claims.** Separately compiled
   frozen main and PR executables disagree on ordinary V3 exp2_lowp and
   exp2_midp_unchecked inputs. Scalar legacy methods retain their literal old
   graph, while native routes change coefficients or exceptional handling.
   For example exp2_lowp at input 0x3f7fffff changes 0x3ffe938c to 0x3fffec5b.
   Checked native log2(-0) changes -infinity to NaN, and its NaN classification
   now differs from ScalarToken. The stated exhaustive zero-difference result
   cannot describe this frozen head against its predecessor. No sweep harness
   or source-identified raw sweep accompanies the PR's 17-file patch.
7. **Restore the cost bound and preserve caller proofs.** Qualified regressions
   below exceed 2%. The timing consumer retains 24 out-of-line expf/logf native
   kernel symbols plus three expf arithmetic closures and three error-formatting
   symbols. Const flags alone do not guarantee inlining. Add the token-preserving
   vector delegates used by powf and inspect the resulting hot loops; current
   configurable array entry points reconstruct a ScalarToken and select again.
   Measure the revised Caller/Automatic paths rather than assuming their cost.
8. **Rebase the stack correctly.** The PR still targets `fix/bounded-powf` at
   4a134564. Main contains that tree through squash commit 91a13739. Rebase the
   expf/logf change onto the merged main before retargeting; do not accidentally
   bring the pre-squash powf history back into the final PR.

Seven independent contract witnesses fail in debug; six fail in release (the
legacy-configuration panic is debug-only). The PR's own expf/logf unit suites each
fail two debug tests but pass in release. Generation, registry, token validation
and static soundness pass; no_std, formatting and strict clippy fail. Latest
observed GitHub state for this unchanged head: 32 successful and 24 failed checks.

## Numerical results

The new core graphs pass the sampled exactness gate: **49,408 rows, six output
contexts per row, zero non-NaN bit mismatches** against independent rational
fused/separate arithmetic. Each family has 3,088 structured/random inputs and
eight precision/arithmetic configurations. This includes powers-of-two,
near-one values, exponent boundaries and the documented exp2 rails.

The same outputs replay on AMD V3, AMD V4x, real M6 NEON, i686, strict WASM,
relaxed WASM with hardware fusion, relaxed WASM with FMA generation disabled,
and a no-FMA Nehalem CPU emulator. Automatic widths 1/4/8/16 agree with the
explicit native scalar token; BitExact also agrees with ScalarToken Caller.
BackendDependent intentionally follows the separate graph on scalar/i686/strict
WASM and the fused graph on matching native hardware. Both relaxed-engine
deterministic settings fused on this host; disabling engine FMA separately
exercised the nonfusing choice. Emulation supplies correctness evidence only.

Directed 256-bit MPFR sampled upper error estimates on native fused results:

| Kernel/range | Frozen legacy | PR147 canonical, actually executed |
|---|---:|---:|
| exp2 MIDP, [-126,127.5) | 1.329274 ULP | 0.730997 ULP |
| exp2 MIDP, sampled upper rail | 123.633149 ULP | 99.633149 ULP |
| exp2 LOWP, [-126,127.99] | 0.555731% relative | 0.029887% relative |
| log2 MIDP, positive normal | 3.429564 ULP | 2.023040 ULP |
| log2 LOWP, positive normal | nonzero bias at x=1 | 6.866177 ULP |

These are sampled maxima, not global bounds or correctly-rounded guarantees.
The currently executed LOWP exp2 cubic is a useful accuracy result at the same
FMA count. The unreachable quadratic is a different implementation and cannot
inherit those measurements. Positive subnormal log inputs remain expressly
inaccurate in Specials; this audit does not relabel that documented limitation
as full mathematical support.

## AMD Zen 5 timings

Frozen matrix: four operations, widths 4/8/16, native V3 and ScalarToken callers,
64/4096 values, streams and dependency chains. Inputs are ordinary values in
[0.5,1); chains renormalize the mantissa to retain the domain. Each of 96 cells
retains 50 interleaved rounds. An enforced cpuset contains one cache class.
58/96 cells meet every gate (26 native, 32 scalar); 38 remain inadmissible.
Initial CPU-placement refusals collected zero timed rounds and are preserved.

The frozen benchmark graph is independently validated: its outputs equal a
separately compiled main executable for 16,400 full-bit inputs in both ScalarToken
and V3 contexts. Controls are not calls through the PR's remapped legacy methods.

For V3 x4, 64-value streams, **ns per whole buffer**, with all 50 pairs clean:

| Implementation | ns | 95% paired change from corresponding main legacy |
|---|---:|---:|
| main log2_midp | 22.43 | reference |
| PR147 legacy log2_midp | 24.80 | +10.14% to +11.02% — FAIL |
| PR147 canonical MIDP BitExact Checked Automatic | 23.98 | +6.48% to +7.35% — FAIL |
| PR147 canonical MIDP BitExact AssumeValid Automatic | 17.39 | -22.87% to -22.09% — PASS |
| PR147 canonical MIDP BackendDependent AssumeValid Automatic | 16.63 | -26.19% to -25.48% — PASS |
| main log2_lowp | 12.81 | reference |
| PR147 canonical LOWP BitExact Checked Automatic | 21.29 | +65.93% to +66.44% — FAIL |
| PR147 canonical LOWP BitExact AssumeValid Automatic | 16.72 | +30.34% to +30.76% — FAIL |
| PR147 canonical LOWP BackendDependent AssumeValid Automatic | 15.34 | +19.33% to +20.27% — FAIL |
| main exp2_lowp | 13.29 | reference |
| PR147 canonical LOWP BitExact AssumeValid Automatic | 13.94 | +3.52% to +6.56% — FAIL |
| PR147 canonical LOWP BackendDependent AssumeValid Automatic | 13.11 | -2.98% to +0.68% — PASS, no demonstrated win |

The arithmetic options are literal const bits: MIDP checked 0, LOWP checked 8,
BitExact AssumeValid 32/40, BackendDependent AssumeValid 48/56. Automatic is
retained in all configured timings. The control uses the corresponding original
legacy method with its own documented validation and exceptional semantics.
The unchecked speedup is a different contract; it does not excuse a slowdown
in a remapped existing checked method.

Across the 58 qualified cells, PR147 legacy methods have 52 PASS, four FAIL and
two INCONCLUSIVE verdicts. Canonical BitExact checked has 22 PASS/36 FAIL;
BitExact AssumeValid has 30 PASS/28 FAIL; BackendDependent AssumeValid has
27 PASS/31 FAIL. No aggregate statistic overrides a failing cell. A confidence
interval crossing the 2% bound remains inconclusive; no extension was used to
seek a favorable result after the observed failures.

Named V3 consumer codegen corroborates the legacy log2_midp cost: 36 instructions
before, 40 after, with additional integer classification/masking. The original
and PR exp2_lowp both have 20 instructions and three FMAs, with changed
coefficients. Legacy log2_lowp retains 23 instructions/four FMAs. The identical
exp2_midp consumer bodies are folded together. Instruction counts exclude
alignment padding; these are consumer probes, not an explanation of every timing
difference. Cargo asm could not locate the thin-LTO assembly artifact; executable
objdump disassembly supplied the preserved source-identified bodies instead.

## Build diagnostics and resources

Three alternating target-cold default dev builds per revision, same rustc 1.99.1 /
LLVM 23.1.1, source paths remapped equally, dependencies rebuilt, OS cache warm:
main 3.20/3.11/3.13 s; PR147 3.22/3.21/3.23 s. Debug rlib sizes are
10,121,758 and 10,667,134 bytes respectively. These diagnostics do not reproduce
the PR's unexplained +7%/+13% figures under this explicit recipe, and establish
no universal compile-cost bound. GNU time peak RSS stays below 403,000 KiB.

Resource-wrapper summaries: initial validation 39 s / peak 2.40 GiB; native
protocol builds 66 s / 2.71 GiB; cross-target build/run 14 s / 0.45 GiB;
AMD timings 163 s / 0.26 GiB, minimum available 43,163 MiB, peak load 1.56;
three-pair cold builds 19 s / 0.38 GiB. M6 numerical build/run 12 s / peak
0.77 GiB. All run logs, inadmissible rounds, failed orchestration and raw results
are retained. Performance facts apply to their named host, caller, width, workload
and contract; they are not extrapolated to other ISAs.

## Recommended sequence

Keep the refits separate from policy plumbing in reviewable commits. First fix
no_std, the complete option truth table, accepted exceptional semantics, zero
signs and scalar masks, using the retained failing witnesses. Then expose the
same token-preserving vector delegation used by powf, measure its generated
code, and recover the existing-method cost bound. Decide between the measured
three-FMA LOWP exp2 refit and a correctly ranged quadratic using separate accuracy
and timing data. Update the contract docs and source-identified exhaustive
claims only after those results exist; rebase onto merged main and rerun the
unchanged promotion matrix. The current evidence supports useful numerical
work, not merging this head.
