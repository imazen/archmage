# Local consumer inventory (2026-11-07)

Read-only ripgrep inspection of local checkouts, not downloaded source. The
14 public checkouts below were audited after a broader local manifest search.
Archived/alternate checkouts, Archmage's own examples and experiments, nested
copies, untracked files and Cargo output do not count toward this table.

The scanner finds literal method-call spellings in tracked Rust files and
records the HEAD, working-tree dirty state and matched file SHA256. It groups test/benchmark/example
paths separately and conservatively treats source after the first `cfg(test)`
as tests. Categories need source review: this is not a Rust resolver, cfg
expander, call graph or runtime profile. Counts are distinct method/source-line
pairs, not calls, instantiations, pixels, traffic or performance importance.
A line containing the same method several times counts once. Macro-instantiated
widths also count once. Comments on their own lines are excluded.

## Priorities for this patch

1. **Checked midp pow and transfer curves.** `pow_midp` has 41 candidate source
   lines; `pow_midp_precise` has four. Linear-srgb gamma uses widths4/8/16;
   zensim transfer curves use PQ/HLG; zenpixels PQ uses the precise alias;
   zentone uses pow in PQ/gamma paths. These inherit the checked exp2 NaN fix.
2. **Lowp pow through its unchecked alias.** Zenfilters has14
   `pow_lowp_unchecked` source lines across x86 and NEON/WASM implementations
   of gamma/contrast/protection. The name is an alias of checked lowp pow, so
   the clamped exp2 change reaches these paths too. Generic width8 matters.
3. **Actual checked exp/exp2 consumers.** There are five `exp_midp` lines,
   four `exp2_midp` and two `exp2_midp_precise` lines. Examples include zensim
   HLG decoding, metric weighting, zentone filmic curves and zensimx color.
   Include their surrounding operations and existing scalar tails in any
   application-level benchmark; isolated primitive ratios are not end-to-end costs.
4. **Portable FMA consumers.** The17 explicit `mul_add_portable` lines are15
   in zensr-micro's width8/16 convolution/portable-exp macro and two in the
   metric CLI's separable blur macro. They are f32 consumers. The scan also
   finds many bare `mul_add` spellings; those mix scalar/std and vector APIs
   and are not evidence of a particular fused dispatch path without resolution.

`cbrt_midp` (83 candidate lines) and `cbrt_lowp` (22) are prominent in
zensim/XYB/Oklab/quantization. All three cbrt methods are untouched here and
have no relaxed-FMA polynomial to switch. Their source frequency does not
justify adding them to an exp-clamp performance change.

No direct downstream `sigmoid_midp`, `silu_midp`, `exp2_lowp`, `exp_lowp` or
`pow_lowp` call is present in the audited production sections. The lowp aliases
still reach the changed clamp transitively. Retain activation correctness
coverage, but prioritize transfer/tonemapping workloads for further performance
work. The existing frozen mandatory matrix and recorded failures remain intact;
this inventory does not retroactively remove expensive cells or change promotion.

Unchecked midp exp/pow and unchecked logs also occur, especially through
zenmetrics-math and zenpixels BT.2446. Those omit the clamp changed by this
patch; their presence must not be counted as exposure to this exceptional fix.
Checked/unchecked and aliases therefore need separate consumer attribution.

Across these14 checkouts,19 of35 transcendental spellings occur in candidate
production sections. Absence is limited to this search scope, literal spellings
and inspected working trees; trait indirection or other checkouts can add uses.
Experimental zentone ACES2 and source-level exactness audits are excluded from
the production table. No consumer repository was synchronized, formatted or edited.

## Reviewed caller contexts

- **Zensim PQ/HLG:** width16 on V3/V4/V4x. The inspected macro explicitly
  excludes scalar; separate scalar/NEON/WASM entry points call its own canonical
  scalar bodies. A ScalarToken x4 microbenchmark is not that application's
  PQ/HLG path. Its scalar graph also owns its NaN/clamp contract, so a dependency
  upgrade needs application parity checks before any behavior claim.
- **Zenpixels PQ:** width16 generated for scalar, NEON, WASM, V3 and optional V4.
  This is a direct candidate for real scalar/NEON/WASM portability cost checks.
- **Zentone curves:** width8 across scalar, NEON, WASM and V3 for the reviewed
  BT.2408/BT.2446 curve functions. Its experimental ACES2 module is feature-gated
  and remains outside the production candidate counts above.
- **Zenfilters:** width8 x86 and generic width8 NEON/WASM implementations. Lowp
  pow alias use reaches the changed clamp on those selected paths.
- **Zenmetrics-math:** unchecked width8 and optional width16 policy operations;
  scalar tails use std when enabled, checked generic methods under no_std.
  Do not attribute the checked-clamp change to its unchecked vector methods.

These distinctions narrow the next application measurements to real width8/16
caller graphs and their actual tails. They do not erase the generic width4
regressions or authorize any new default.

## Inspected revisions

Dirty means the inspected working tree may differ from the listed committed
revision. Other agents' work was preserved. Nested old checkouts were not
counted again.

| Checkout | HEAD | Dirty | Candidate math source lines |
|---|---|---|---:|
| fast-ssim2 | `d4ad4fa10d04` | yes | 0 |
| jxl-encoder | `7b354ee6b014` | no | 8 |
| linear-srgb | `8900ffc87cd5` | no | 10 |
| ultrahdr | `b2dbf952ef6d` | yes | 0 |
| zenanalyze | `801d0963763d` | yes | 5 |
| zenjpeg | `762fa7128c21` | yes | 6 |
| zenmetrics | `097b5e4de9f8` | no | 21 |
| zenpipe | `4992aa08e0f0` | yes | 32 |
| zenpixels | `0d74f5eb91d9` | yes | 9 |
| zenquant | `39d6c4c69d42` | no | 3 |
| zensim | `6060039d8c0a` | no | 39 |
| zensimx | `1c67ce31b2b0` | yes | 47 |
| zensr | `1cbc449216d9` | no | 0 |
| zentone | `5ea015281452` | yes | 23 |

## Representative source

Paths below refer to the inspected local working trees; line numbers can move.

- linear-srgb: `src/simd.rs:117`, `src/tokens/x4.rs:117`, `src/tokens/x8.rs:178`.
- zenpixels: `zenpixels-convert/src/convert_kernels.rs:2164` and `src/hdr/bt2446a.rs:253` within that crate.
- zensim: `zensim/src/transfer.rs:339` (PQ), `:599` (HLG), `:627` (OOTF).
- zentone: `src/simd/curves.rs:104` (PQ), `:635` (filmic exp2), `src/simd/blocks.rs:281` (power).
- zenpipe: `zenfilters/src/simd/x86.rs:121`, `:163`, `wide_simd.rs:159`, `:195`.
- zenmetrics: `crates/zenmetrics-math/src/policy.rs:78`, `crates/mdctpsnr/src/lib.rs:473`, `crates/hdrvdp/src/simd_kernels.rs:109`.
- zensr: `crates/zensr-micro/src/simd.rs:62`, `:196`, macro widths at `:1043`/`:1045`.
- zenmetrics portable blur: `crates/zenmetrics-cli/src/metrics/ssimulacra1.rs:139`.

Reproduce against explicitly chosen local repositories:

```sh
just portability-usage /path/to/private-results.json /path/to/consumer-a /path/to/consumer-b
```

The output contains source excerpts and local paths. Keep it private when
inspecting private repositories. The public report includes only verified-public
checkout names and reviewed aggregate findings.
