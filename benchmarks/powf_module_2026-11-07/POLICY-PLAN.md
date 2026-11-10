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

## Launcher correction

The first M6 run omitted zenbench's documented `ZENBENCH_LAUNCHER_PIDS`
registration for its waiting Python launcher. Preserve the complete original
run and its foreign-heavy-process observations; those observations alone do
not identify the process. The published 0.1.11 sampler already excludes the
measuring process itself. A separately labeled corrected launch uses the same
frozen binary, inputs, rounds and gates, registers only the waiting launcher's
actual PID and enables gate identity tracing. Every unrelated process and all
RAM/temperature thresholds remain visible and unchanged. This corrects explicit
owner registration; whether it explains the original flags requires evidence.
No further favorable reruns are authorized by this correction.

The first Intel launch refused before timing because its process boundary
spanned performance and efficiency cores. Preserve that zero-sample refusal.
Resolve the topology, confine the process to its performance CPU class, and run
the correctly registered pilot once before considering the full Intel matrix.

## HIGHP layout correction comparison

The original Full HIGHP native vector route converts `Result<Array>` to
`Result<Vector>`. M6 assembly emits byte reconstruction for that intermediate:
58 normalized x4 instructions versus 32 in the scalar-array route, each with
four std calls. Replacing `Result::map` alone reduces the native body to48;
separating validation from direct vector construction reduces it to38. These
are static counts excluding std callees, not timing ratios.

For the changed implementation, retain all64 configurations and the original
five comparisons, then append two explicitly named arms (71 total): the locally compiled
Result-map provider copied from c3abfbcd and the direct public array HIGHP API.
Freeze the same56 workloads, inputs,50 interleaved rounds and unchanged gates.
Compare corrected vector HIGHP to that local provider within each group,
and array HIGHP to direct std. Preserve both earlier69-arm matrices and their
foreign-process flags. This is a changed implementation, not a favorable retry
of the same binary. Extra retained codegen tables compare all three shapes.

The copied provider is labeled `local_HIGHP_ResultMap_reference`. It is not the
original library's full call path: in the current M6 module, its x4 wrapper
already optimizes to38 instructions, whereas the archived original public
method emitted58. Do not report their timing ratio as the original public
method's speedup. The actual before/after public-method assembly and the two
original69-arm timing runs are retained separately. The direct public array
HIGHP and direct std probes have matching instruction counts at every width.

## Intel final 71 admission continuation

The earlier correctly confined pilot collected zero samples while queued behind
a live owner. Before the final 71 implementation is timed, re-resolve that PID
and the common admission state. If the owner has exited, allow one fresh
correctly confined native x4/64 stream pilot, using the same frozen 56 cells,
50 rounds, inputs and strict gates. Run no timings alongside this lane's CI or
builds. A clean pilot permits the full matrix; any dirty result or admission
failure stops this continuation and remains recorded. Retain the pilot and
every full-matrix cell; do not select whichever repetition is favorable. The
prior zero-sample refusals and all M6 results remain unchanged. This introduces
no additional M6 timing attempt.
