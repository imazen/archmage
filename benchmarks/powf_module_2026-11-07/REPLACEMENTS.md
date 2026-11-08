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
rejection remains. Counts include error paths and local callees, exclude external
stdlib/panic implementation bodies, and are not dynamic instruction counts.
The report contains 240 distinct Scalar/V3 probe bodies; LLVM coalesces several
identical domain variants. No existing assembly-equality gate was relaxed.

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

The union of 34 distinct new witnesses and the original 26,823 domain-tagged
pairs gives 26,857 common pairs for every implementation and both backends.
17,769 are accepted and 9,088 rejected. The accepted domain counts are 5,587,
5,625 and 6,557. Canonical result bits and rejection stages match the independent
rational/RNE graph, with zero scalar/V3 canonical differences. No coefficients
were changed. These results supersede the earlier sampled maxima.

| Domain | Canonical maximum measured ULP upper bound | std f32 power |
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
