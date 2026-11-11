# Fused portable reciprocal refinement — 2026-11-11

Portable Magemath LOWP/MIDP now use one canonical **fused** Newton graph.
Hardware availability selects the provider for the complete operation, not for
each Newton step. Magetypes' existing unfused portable methods are unchanged.

Reciprocal uses the same fixed integer seed followed by one/two iterations of
`r * fma(-x, r, 2)`. Rsqrt uses its fixed integer seed and
`y * fma(-((0.5*x)*y), y, 1.5)`, with the indicated multiplication rounding and
association. The accuracy contracts stay ≤6e-3 LOWP and ≤1e-4 MIDP on the
existing restricted domain. These are different rounded graphs from the prior
unfused iterations; scalar/vector and cross-ISA bit identity is retained by
using exact software FMA wherever hardware fusion cannot be proved.

## Provider selection

- V3/V4/V4x and NEON proofs instantiate native fused arithmetic directly.
- Scalar/array Automatic entries select a whole native feature kernel first.
  On x86 std fallback with FMA3 but no V3 proof, the graph can use `f32::mul_add`
  after one cached standard-library feature check. Its library body is outside
  wrapper instruction counts; no speed claim is made for that fallback.
- Caller execution retains its proof. Scalar Caller uses the existing fused
  primitive, which permits compile-proven hardware and otherwise exact software.
  No optional feature detection is available to a baseline no_std build.
- A compile-time relaxed-WASM build reuses **the same cached f32 fusion probe
  as powf**, after checking the relaxed token. It selects raw relaxed madd only
  when that probe proves fusion. Strict or non-fusing engines get packed exact
  software FMA. The hardware graph does not repeat token/probe selection per
  FMA; a testing disable before entry forces the software graph. An already
  selected operation retains its arithmetic choice.

Typed entries retain vector storage/backend and width; native proof types fold
the arithmetic selection at compile time. Scalar Automatic typed entries may
select the std FMA provider without changing vector representation. HIGHP and
BackendDependent estimate graphs bypass this portable-fusion selector.

## Validation

Native x86 and M6 std/no_std directed tests pass. Full x4 mantissa scans over
all 16,777,216 floats in [1,4) agree on portable maxima and witnesses across
Scalar, V3 and NEON:

| Graph | Maximum relative error | Input bits |
|---|---:|---|
| LOWP reciprocal | 3.362326737260e-3 | 3ff127fc |
| MIDP reciprocal | 1.141549399374e-5 | 3ff12999 |
| LOWP rsqrt | 1.752338672098e-3 | 406eb3c0 |
| MIDP rsqrt | 4.723778778297e-6 | 406eb88c |

Directed tests retain edge exponents, signed reciprocal inputs, masks and
matching-width tails. This scan is not a proof for every normal-domain bit
pattern. Wasmtime passes strict SIMD, relaxed fusion, forced non-fusion
(`cranelift-has-fma=false`), and disabled-relaxed-token tests. The provider tests
check the actual cached probe against the requested engine mode, then compare
hardware-selected and disabled-token software results bitwise. MSRV 1.89,
strict linting, full Magemath std/no_std tests and public API snapshots pass.

The V3 x8 MIDP reciprocal probe contains two native fused subtract instructions
and two multiplies, with no provider calls/branches. The relaxed-WASM x8 probe
calls `powf_f32_fuses` once before choosing a graph; its native arm contains four
relaxed madd operations for two chunks/two steps, with no per-FMA probe call.

Validation resource records: initial cross/scan run 9 s, peak-RSS 0.43 GiB;
final native/cross/lint/codegen/MSRV run 18 s, peak-RSS 0.51 GiB; subsequent
WASM adapter/full-suite/API review 29 s, peak-RSS 2.00 GiB. The M6 run reports
9 s and a wrapper RSS rounded to 0.00 GiB, which is not a zero-memory claim.

## One frozen timing run

**Overall UNRELIABLE; no replacement promotion.** Three groups have 50/50 clean
pairs; rsqrt stream has 22/50. Every round is retained; no retries/extensions.
Zenbench 0.1.11, Zen 5 V3 x8, enforced uniform cpuset, no target-cpu=native.
Source `f7559b2417d2df999ff427495deed79d470917c2`; source archive, executable
hash, compiler and frozen 4-group × 7-arm plan are in raw `SOURCE.json`.
A later WASM adapter removes test-mode per-FMA token checks; the measured x86
numerical graph and kernels are unchanged. No retiming is inferred from that.

All approximations below are Normal, AssumeValid, Caller proof retained. Times
are ns per 64 input values (stream) or 64 dependent eight-lane steps (chain).

| Graph | recip stream | recip chain | rsqrt stream (unqualified) | rsqrt chain |
|---|---:|---:|---:|---:|
| Portable LOWP unfused | 1.863 | 125.530 | 3.063 | 188.176 |
| Portable LOWP fused | 1.644 | 100.378 | 3.172 | 164.917 |
| Portable MIDP unfused | 3.211 | 226.856 | 5.360 | 321.171 |
| Portable MIDP fused | 3.061 | 177.889 | 5.079 | 274.920 |
| Existing Magetypes recip/rsqrt | 2.418 | 159.488 | 3.552 | 218.321 |
| Backend MIDP unchecked | 2.810 | 126.393 | 2.975 | 188.617 |
| HIGHP division/composition | 3.608 | 82.497 | 10.258 | 238.359 |

Zenbench's reciprocal chain bars:

```text
HIGHP division          █████████████████ 82.5ns
portable LOWP fused     ████████████████████ 100.4ns
portable LOWP unfused   █████████████████████████ 125.5ns
backend MIDP unchecked  ██████████████████████████ 126.4ns
magetypes recip         ████████████████████████████████ 159.5ns
portable MIDP fused     ████████████████████████████████████ 177.9ns
portable MIDP unfused   ██████████████████████████████████████████████ 226.9ns
```

[All-pair results](fusion-zen5-pairs.json) use 10,000 paired bootstrap resamples
of mean ratios, every round, no IQR removal. MIDP fused/unfused chain ratios:
reciprocal 0.784151 [0.783621, 0.784765]; rsqrt 0.855993
[0.855492, 0.856469]. Those improvements are supported by clean cells.

Portable MIDP remains slower than the legacy native graph: reciprocal ratio
1.115373 [1.114635, 1.116191]; rsqrt 1.259244 [1.258488, 1.260038]. Reciprocal
stream is also slower than legacy. Even fused/unfused reciprocal-stream
intervals have upper bounds 1.021613 (LOWP) and 1.044976 (MIDP), exceeding the
2% promotion bound despite lower means. This is not a complete performance pass.
Software-FMA cost versus the old unfused graph, M6/Neoverse/Intel timing, other
widths and long buffers remain unmeasured.

Timing run: rc=1 for retained admission failure, 9 s, peak-RSS 0.25 GiB,
minimum available 41975 MiB, peak load 0.73. Raw evidence lives under
`/mnt/v/output/archmage/recip-fusion-2026-11-11/`, mirrored under
`/mnt/tower/output/archmage/recip-fusion-2026-11-11/`.

`--measure --fusion-study` selects this frozen comparison in the study binary;
`run_local.py BINARY --fusion-study` retains the same strict placement/admission
rules. `just primitive-fusion-analyze RESULTS` computes the all-pair intervals;
`primitive-recip-fusion-wasm` exercises fusing/non-fusing engines and suppression.
