# Public powf policy permutation protocol

Freeze before timing. Source adds all 64 supported const encodings: 48 bounded
MIDP/LOWP configurations (three domains, two portability policies, two execution
policies, two validation policies) and 16 std HIGHP configurations (four domains,
two execution policies, two validation policies). Five named comparisons add the
actual legacy LOWP/MIDP, std f32, std f64 cast to f32, and dynamic runtime MIDP.
Each group interleaves all 69 implementations for 50 retained rounds.

56 cells: scalar x1 and scalar/native x4/x8/x16, buffers of 64/4096 values,
throughput streams and normalized dependency chains, literal exponents 2.4/0.5.
All implementations get identical in-memory input and output work. Inputs use
positive normalized 8-bit fractions; chain normalization maps output back into
[0.5,1). All range/validation configurations are valid on these inputs.
No native CPU compiler flag or fast math. Native means an actual matching proof.
x1 uses the public scalar/array boundary and has no separate native-token API.
The legacy x1 comparisons splat into the actual scalar f32x4 legacy methods
and extract lane zero; f32x1 has no legacy power methods.

Correctness: the unchanged independent rational four-graph fixtures are replayed
through all 48 bounded const encodings and their runtime adapters, widths 4/8/16,
every broadcast lane and mixed vectors with exact whole-vector error masks.
AssumeValid is called only for fixtures accepted by its corresponding checked
graph. BitExact selects the fused reference; BackendDependent selects native
relaxed FMA or separate arithmetic according to the actual execution context.
HIGHP is std delegation, tested for bit/classification equivalence to std and
independently measured against MPFR; finite probing is not a BitExact certificate.

Predeclared codegen/time comparisons form a partial order, not a universal
speed ladder:

- AssumeValid versus Checked: remove exponent/base/result-domain rejection work.
- Gamma versus NonNegativeNormal: remove computed-result rejection, replacing
  general input classification with explicit Gamma bounds. Timing can trade off.
- LOWP versus MIDP: shorter polynomial graphs, with measured larger error.
- BackendDependent versus BitExact: remove exact FMA correction/software work
  in scalar/strict-WASM/nonfusing-WASM caller contexts. Matching native FMA graphs
  should tie. Automatic scalar upgrades can also collapse this distinction.
- Caller versus Automatic: avoid autonomous selection when it would otherwise
  occur. A software BitExact caller can be slower than upgrading the operation;
  the saved dispatch work is not a promise of a faster complete power.
- Full HIGHP versus restricted HIGHP: omit domain checks. Caller/Automatic and
  Checked/AssumeValid Full HIGHP intentionally tie with direct std f32 power.
- Const BitExact/MIDP/Gamma/Checked/Automatic versus black-box runtime options:
  verify the policy decoder disappears from the const caller.

Inspect every emitted policy body, including compiler aliases. Count normalized
instructions, conditional branches, calls, multiplies, additions, FMAs and
software/probe references. Keep ties and reversals; never rewrite labels or
exclude an inconvenient pair to force the expected order. Static instruction
counts exclude external callees and do not substitute for timing.

Timing uses zenbench's unchanged strict admission and all-retained paired 95%
confidence analysis with 100,000 resamples. Cost upper <=1.02 passes the existing
2% bound; lower >1.02 fails; overlap is inconclusive. An optimization requires
upper <1.00. A correctly selected strict Intel pilot must be clean before its
full matrix. The owner-authorized M6 run uses explicit unrestricted scheduler
placement and is exploratory even when its resource gates are clean. Preserve
all flagged rounds, failures, binary hashes, source and resource logs.
No merge, promotion or numerical guarantee follows from an exploratory ranking.
