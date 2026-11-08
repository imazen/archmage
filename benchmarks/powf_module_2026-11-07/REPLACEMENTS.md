# Power replacement comparisons, 2026-11-08

Promotion remains held. The canonical fused approximation is portable on its
explicit accepted domain; this is not std-level accuracy or full std semantics.
Existing lowp/midp methods remain unchanged. This study uses their actual calls,
std f32 power, promoted f64 power, integer-only std powi, runtime options and
caller-visible const options as separate controls. The original
[prevalidated-graph study](README.md) remains historical evidence for its own
runtime implementation; its ratios do not describe the new const caller path.

## Const caller code

The primary API carries `PowfOptions::new().with_domain(...)
.with_execution(...).bits()` as a u32 const argument through caller, vector,
backend and graph. The const route retains the vector representation inside a
matching feature region. Runtime adapters retain their explicit runtime names.

Rust 1.99.1 / LLVM 23.1.1 distribution-target x86 codegen, x4 Gamma:

| V3 probe | Normalized instructions including local callees | Branches |
|---|---:|---:|
| Const flags, literal 2.4 | 92 | 1 |
| Literal runtime options, literal 2.4 | 92 | 1 |
| Opaque runtime options, literal 2.4 | 276 | 11 |
| Const flags, opaque exponent | 113 | 5 |
| Literal runtime options, opaque exponent | 113 | 5 |
| Opaque runtime options, opaque exponent | 257 | 17 |

LLVM already specializes literal runtime options in this probe. The const API
makes the flags explicit throughout the generic caller chain; these counts do
not show an additional win over the optimized literal runtime form. Input
rejection remains. Counts include error paths and callees present in the emitted module.
Unresolved calls, including std/panic bodies and out-of-module helpers, are
listed and their instruction work is unmeasured. These are static counts.
The report contains 240 distinct Scalar/V3 probe bodies; LLVM coalesces several
identical domain variants. No existing assembly-equality gate was relaxed.

Retained Apple M6 NEON caller probes show the same specialization pattern:
x4 Gamma/literal exponent has 125 instructions and one branch for both const
and literal runtime options, versus 367 instructions and 11 branches with
opaque options. With an opaque exponent, const/literal-runtime options have
145 instructions/five branches, versus 355/15 for opaque options. All 240
distinct Scalar/NEON bodies are retained; an initial probe build inlined native
entries into main, so the diagnostic generator now preserves a plain caller
on guaranteed-feature targets. No production or measured binary changed.

## Mathematical accuracy

The pinned external `math_accuracy` bivariate adaptive search was connected to
the actual Rust implementations. Its three heuristics prioritize maximum,
mean and estimated maximum error. Eight implementations × three domains × two
actual backend paths × three modes give 144 searches. Each uses seed 271828183,
effort threshold 4096 and one thread. Source hashes, original GPL notices,
patches, command lines, every search log, admission/rejection evaluation counts
and witnesses are preserved outside git. Search arithmetic uses MPFR at 72
bits; every resulting witness is rechecked at 256 bits with directed MPFR.
This is a bounded search, not a certified global maximum-error computation.
Gamma search covers its positive interval; exact zero rails remain in the
common fixed corpus and the separate semantic comparison.

M6 numerical verification repeats all 26,857 common pairs through both Scalar
and actual NEON paths. It has zero independent-graph or canonical tier
mismatches and the same canonical maxima. Its system std f32 power has measured
maxima below 0.511, 0.510 and 0.509 ULP in the three domains, respectively;
promoted f64 remains below 0.500000002. These are platform-specific observations.

The union of 34 distinct new witnesses and the original 26,823 domain-tagged
pairs gives 26,857 common pairs for every implementation and both backends.
17,769 are accepted and 9,088 rejected. The accepted domain counts are 5,587,
5,625 and 6,557. Canonical result bits and rejection stages match the independent
rational/RNE graph, with zero scalar/V3 canonical differences. No coefficients
were changed. These results supersede the earlier sampled maxima.

| Domain | Canonical maximum measured ULP upper bound | Linux x86 std f32 power |
|---|---:|---:|
| PositiveNormal | 438.504 | 0.507 |
| NonNegativeNormal | 416.579 | 0.507 |
| Gamma | 47.902 | 0.502 |

Values are rounded upward. Promoted f64 power stays below 0.500000002 ULP on
this corpus. Correct rounding is checked using MPFR's direct f32 conversion
at both directed endpoints; a double enclosure that straddles an f32 midpoint
is insufficient for that classification. ULP error uses the true result's
binade, including gradual underflow. Finite-output and zero-rail behavior are
reported separately in the raw comparison. Std is a candidate, never truth.

On identical accepted Gamma pairs excluding base zero and exponent zero:

| Implementation | Scalar maximum measured ULP upper bound | V3 maximum measured ULP upper bound |
|---|---:|---:|
| Canonical | 47.902 | 47.902 |
| pow_midp / unchecked / precise | 48.562 | 47.902 |
| pow_lowp / unchecked | 93411.466 | 93408.852 |
| std f32 power | 0.502 | 0.502 |

Zero rails are a separate semantic difference, not silently removed failures:
checked midp/lowp can differ at admitted zero inputs, and unchecked midp has
large zero-rail errors. The raw report retains the full accepted corpus and
separate zero-rail-free summaries for every method and domain. Across nonzero
accepted inputs, native V3 midp matches the canonical graph in this corpus;
scalar midp follows its existing separate-rounding graph. The const design
improves portability/validation/caller integration, not these coefficients.

The canonical graph can replace existing approximations only where its domain
and measured accuracy satisfy the caller. Replacing std under a tight error
contract requires a more accurate algorithm. A finite startup screen cannot
certify arbitrary std power calls as bit-identical to this canonical graph.

The upstream search and hard-to-round tools address different questions:
[math_accuracy source](https://gitlab.inria.fr/zimmerma/math_accuracy/-/blob/master/binary64/check_sample2.c),
[accuracy study](https://members.loria.fr/PZimmermann/papers/accuracy.pdf),
[CORE-MATH worst-case references](https://core-math.gitlabpages.inria.fr/worst.html).

## Interleaved performance

The [frozen matrix and amendments](REPLACEMENT-MATRIX.md) define 672 native
cells, real controls, literal/opaque exponents, all widths, both lengths,
streams/chains, three domains and input distributions. All 50 retained paired
rounds enter zenbench MeanCi with 100,000 resamples. Ratios below one favor the
new const Caller API. These are pilots, not a complete promotion result.

Gamma 2.4, nonzero unorm8, x4 matching native caller, 64-element stream:

| CPU | Actual control | Const Caller/control | Paired 95% CI | Qualification |
|---|---|---:|---|---|
| Neoverse-N1 | pow_midp | 0.82121 | 0.81948–0.82288 | Clean pilot |
| Neoverse-N1 | pow_lowp | 1.50265 | 1.49688–1.50836 | Fails 2% replacement cost bound |
| Neoverse-N1 | std f32 powf per lane | 0.40202 | 0.40141–0.40265 | Clean pilot |
| Zen 4 | pow_midp | 0.87019 | 0.86146–0.88108 | Clean pilot |
| Zen 4 | pow_lowp | 1.48375 | 1.46871–1.50203 | Fails 2% replacement cost bound |
| Zen 4 | std f32 powf per lane | 0.40433 | 0.39580–0.41218 | Clean pilot |
| Apple M6 | pow_midp | 0.84189 | 0.83681–0.84722 | Exploratory, OS scheduling |
| Apple M6 | pow_lowp | 1.40322 | 1.39437–1.41259 | Exploratory, OS scheduling |
| Apple M6 | std f32 powf per lane | 0.31605 | 0.31434–0.31758 | Exploratory, OS scheduling |

All pilot resource flags are clean, unreliable=false, 50 pairs. M6 strict
placement refused before timing: Mach affinity tags and QoS do not enforce one
of its asymmetric core classes. The owner authorized interleaved exploratory
comparisons under OS scheduling. Their unrestricted placement is recorded and
never counts as strict promotion evidence. The original refusal is retained.
No native-target compiler flag is used.

The AMD ordinary matrix's first cell was unreliable. Intel's pilot timed out
behind the common benchmark admission owner, producing no timings. N1 cells
155 and 302 are invalid and preserved; continuation measures only previously
unmeasured cells once. Missing or invalid cells cannot pass promotion. No
favorable rerun replaces an existing cell. Full M6/N1 descriptive results and
remaining platform coverage are recorded with the final artifacts.

## Completed native matrices

Both M6 and N1 record all 672 ordinary cells and 50 pairs per cell. M6 has
275 resource-clean cells and 397 flagged cells; every M6 comparison remains
exploratory because core-class placement is unrestricted. N1 has 621 clean
cells and 51 invalid cells. Every pair remains in the direct paired analysis.
No aggregate result satisfies the every-cell promotion rule.

| N1 Automatic versus actual control | PASS | FAIL | INCONCLUSIVE | INADMISSIBLE |
|---|---:|---:|---:|---:|
| pow_midp | 589 | 16 | 16 | 51 |
| pow_midp_unchecked | 352 | 264 | 5 | 51 |
| pow_midp_precise | 585 | 19 | 17 | 51 |
| pow_lowp | 309 | 310 | 2 | 51 |
| pow_lowp_unchecked | 308 | 310 | 3 | 51 |
| std_f32_powf | 368 | 253 | 0 | 51 |
| std_promoted_f64_powf | 368 | 253 | 0 | 51 |

All 16 clean Automatic failures versus checked midp are matching-NEON
NonNegativeNormal cells. Their full cell identities and paired ratios follow.
All use literal exponents; `inv2.4` denotes f32 bits 3ed55555.

| Width / elements / work | Exponent | Inputs | Ratio | 95% CI |
|---|---|---|---:|---|
| x16 / 64 / chain | 2.4 | unorm8 | 1.031462 | 1.027336–1.037964 |
| x8 / 4096 / stream | 2.4 | unorm8 | 1.023559 | 1.021370–1.025636 |
| x16 / 4096 / chain | 2.4 | unorm8 | 1.053175 | 1.051279–1.055085 |
| x16 / 64 / chain | 2.4 | wide | 1.030201 | 1.024843–1.035698 |
| x16 / 4096 / chain | 2.4 | wide | 1.053010 | 1.048327–1.056935 |
| x16 / 64 / chain | inv2.4 | unorm8 | 1.032915 | 1.027971–1.037326 |
| x8 / 4096 / stream | inv2.4 | unorm8 | 1.024317 | 1.021814–1.026739 |
| x16 / 4096 / chain | inv2.4 | unorm8 | 1.054007 | 1.050996–1.057141 |
| x16 / 64 / chain | inv2.4 | wide | 1.031732 | 1.029882–1.033670 |
| x8 / 4096 / stream | inv2.4 | wide | 1.023797 | 1.021539–1.026395 |
| x16 / 4096 / chain | inv2.4 | wide | 1.053365 | 1.050903–1.055772 |
| x8 / 64 / stream | 0.5 | unorm8 | 1.024947 | 1.023002–1.027044 |
| x16 / 4096 / stream | 0.5 | unorm8 | 1.058460 | 1.055130–1.062849 |
| x8 / 64 / stream | 0.5 | wide | 1.024435 | 1.021061–1.027681 |
| x16 / 4096 / stream | 0.5 | wide | 1.056906 | 1.054440–1.059477 |
| x16 / 4096 / chain | 0.5 | wide | 1.037850 | 1.035782–1.039880 |

The following full-matrix Gamma x4/4096 stream cells are resource-clean on
both CPUs. They compare const Caller directly with every ordinary control.
M6's intervals are descriptive under OS scheduling. Prevalidated graph is a
diagnostic for checked-API cost, not the legacy/std replacement baseline.

| Actual control | N1 ratio [95% CI] | M6 exploratory ratio [95% CI] |
|---|---|---|
| pow_midp | 0.825371 [0.822617–0.827926] | 0.858938 [0.856215–0.861370] |
| pow_midp_unchecked | 1.224479 [1.220621–1.228377] | 1.252263 [1.248647–1.256671] |
| pow_midp_precise | 0.822885 [0.820499–0.825335] | 0.859846 [0.857514–0.862926] |
| pow_lowp | 1.503125 [1.497736–1.508589] | 1.408977 [1.404983–1.413816] |
| pow_lowp_unchecked | 1.490778 [1.485029–1.496325] | 1.407463 [1.402181–1.413474] |
| std_f32_powf | 0.386969 [0.386158–0.387842] | 0.314333 [0.313577–0.315380] |
| std_promoted_f64_powf | 0.223919 [0.223474–0.224412] | 0.152816 [0.152317–0.153248] |
| runtime_literal_caller | 0.998394 [0.995148–1.001472] | 1.000382 [0.997361–1.004028] |
| runtime_dynamic_caller | 1.002291 [0.999864–1.004713] | 1.003493 [1.000889–1.006806] |
| runtime_literal_auto | 0.998823 [0.996100–1.001534] | 1.001961 [0.999371–1.005349] |
| runtime_dynamic_auto | 0.992276 [0.989591–0.995119] | 1.001565 [0.998484–1.005222] |
| prevalidated_graph | 1.291519 [1.287984–1.295021] | 1.250139 [1.245715–1.255043] |

## Artifacts and resources

Frozen performance source: e4fad9b3caf96ddfaa4856a56cc11e0b7c25f3b0, tree
680f9e888427a9c8fd5db7bc179c6b31d59c7aa1. Actual reference executable hashes,
compiler/lockfile identity, all paired samples, invalid runs, original source,
MPFR inputs/outputs and generated assembly reports live under
`/mnt/v/output/archmage/powf-replacements-2026-11-08/`, mirrored under
`/mnt/tower/output/archmage/powf-replacements-2026-11-08/`.

Native callback/codegen build: 29 s, peak RSS 0.46 GiB, minimum available
21,810 MiB, peak load 14.73. Health checks and all three API snapshots: 22 s,
peak RSS 0.34 GiB, minimum available 22,161 MiB, peak load 13.86. M6 pilot:
3 s, BSD time peak RSS 0.02 GiB for the wrapper command, minimum reclaimable
page proxy 8,518 MiB, peak load 1.24; macOS used a monitored 2 GiB process-tree
guard, not a cgroup hard cap. That BSD statistic is not a measured whole-build
process-tree peak. Full logs preserve every resource wrapper line.

Cold consumer diagnostics use three fresh-target repetitions, Rust 1.99.1,
one codegen unit, thin LTO, cached dependencies and remapped source paths:

| Consumer | Build seconds (all repetitions) | Executable bytes (each repetition) |
|---|---|---:|
| Main / legacy | 4.433156 / 4.391728 / 4.422096 | 406,152 |
| Const candidate / same legacy source | 4.506343 / 4.457358 / 4.516223 | 406,168 |
| Const candidate / default bounded powf | 4.551558 / 4.536466 / 4.590131 | 422,008 |

These shared-host raw build diagnostics are not precise build-speed claims.
The third row changes the called API and its contract. The wrapper completed
in 41 s with peak RSS 0.35 GiB, minimum available 23,264 MiB, peak load 14.03;
per-build GNU-time RSS and exact binary hashes are retained.

All-pair analyses completed in 224 s (M6 data, including analyzer build) and
194 s (N1 data). Their respective wrapper peak RSS values were 0.53/0.04 GiB,
minimum available 22,924/23,571 MiB, and peak load 15.27 for both. The sealed
const implementation passed all 19 local CI stages and both semver checks in
696 s, peak RSS 1.00 GiB, minimum available 22,009 MiB, peak load 17.34.
The final study tools receive their separate lint/codegen checks; no existing
behavioral, numerical or assembly expectation is weakened.

Review checked the main-to-PR diff and the const migration against the previous
draft: coefficient bits and graph order are unchanged; generators and delegates
match; invalid const encodings fail compilation; exponent/base/range rejection
precedence and exact whole-vector masks remain covered. The native proof and
WASM probe select the arithmetic mode before evaluating the graph. No hand-written
unsafe Rust, library dependency, existing math behavior, accuracy threshold or
assembly-equality exception was introduced. Std FMA screening remains distinct
from direct std power admission. The external error-search tool is test-only,
with pinned original source and notices outside the production dependency graph.

Missing performance coverage still includes complete admitted AMD/Intel and
WASM replacement matrices, native first-use and physical below-V3/std fallback
costs. Intel's recorded pilot has no samples after an admission timeout. The
owner-approved M6 run is exploratory; it does not fill strict placement coverage.

Final local verification of the completed implementation/study source passed
all 19 CI stages, both semver checks, all 24 warning configurations, unchanged
assembly guards (1,623 x86 / 794 ARM / 794 WASM probes, zero reviewed
differences), and standalone all-target/all-feature clippy. It completed in
838 s, peak RSS 0.78 GiB, minimum available 21,588 MiB, peak load 17.83.
Explicit no_std power tests and i686 std/no_std checks pass (8 s, peak RSS
0.40 GiB); physical N1 full-payload tests pass with the NEON plan asserted
(6 s, peak RSS below the wrapper's 0.01 GiB display resolution). M6 study
clippy passes (11 s, BSD peak RSS 0.37 GiB, reclaimable-page proxy minimum
8,163 MiB, peak load 1.63). No benchmark runs overlapped M6 builds.

CI transport follow-up: the initial const-comparison head passed all 53
substantive jobs except WASM, which never started because the pinned Wasmtime
archive download returned HTTP 504 twice; the complete-matrix gate correctly
failed too. The immutable release-asset API delivers the same v48.0.1 archive
with published SHA-256
`4c2e31b68ad99e0a519f225a261fda099eb15f056d4a24fdb3c2a46517bde1df`.
The CI installer now uses that route with bounded retries and checks the
archive digest before extraction. A wrong-byte HTTP-success control fails
before installation. The installed v48.0.1 passes strict, fusing and nonfusing
unit/full-payload tests (24 s, peak RSS 0.10 GiB, minimum available 24,220 MiB,
peak load 12.30). No compiled library source or test expectation changes.
