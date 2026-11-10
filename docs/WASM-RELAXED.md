# WASM relaxed operations and portable contracts

Reviewed 2026-11-10 against the [current WASM numerics specification](https://webassembly.github.io/spec/core/exec/numerics.html#relaxed-operations)
and [Rust's stable intrinsics](https://doc.rust-lang.org/core/arch/wasm32/index.html).
This is a semantic inventory and API recommendation, not a speed comparison.

## Availability and testing

`Wasm128RelaxedToken` already proves `simd128,relaxed-simd`; callers can downcast
it with `.wasm128()` for existing Magetypes vectors. It does not yet implement
Magetypes' generic backend traits directly. Relaxed opcode emission requires a
compatible module build. Runtime validation proves availability; a probe cannot
make a strict-only engine load a module containing relaxed instructions.

With `archmage/testable_dispatch`, both WASM tokens can be disabled using
`dangerously_disable_token_process_wide`. The existing permutation helper now
covers both tiers. Disabling SIMD128 cascades to relaxed SIMD; disabling relaxed
SIMD leaves ordinary SIMD selectable. Normal builds preserve constant selection.
The testing feature adds atomic reads, so enable it only for development.
Previously acquired tokens and `from_context()` remain valid after disabling;
the switch suppresses later selection and cannot remove opcodes from the module.

The existing internal FMA selector now honors relaxed-token disabling. Relaxed
multiply-add falls back to separate multiply/add; the fused forms fall back to
exact software. A warmed engine-fusion cache cannot override a testing disable.
F32 and f64 keep separate fusion probes. No accuracy threshold is weakened.
ScalarToken portable FMA still uses its existing scalar/software routes; this
change does not add scalar-to-WASM-SIMD FMA dispatch.

`no_std` ScalarToken sqrt internally summons only compile-time-proven baseline
tokens (SSE2, NEON or WASM SIMD128), otherwise retaining integer software. Token
testing can select that software path. WASM sqrt needs ordinary SIMD128, not
relaxed SIMD. There is no standardized relaxed sqrt or relaxed reciprocal/rsqrt
opcode in the stable instruction set inspected here. Std scalar sqrt stays a
direct primitive call.

## Which operations merit first-class relaxed support?

Use `_relaxed` for a documented set of acceptable results, with exact input
assumptions named separately. It should not mean an engine-independent answer.
Keep existing precise methods and explicit portable fused graphs intact.

| Family | Contract needed | Recommendation |
|---|---|---|
| Multiply-add / negative multiply-add | Engine may fuse or round twice | Existing relaxed methods fit. Portable fused methods need the existing fusion probe and exact fallback. |
| Float min/max | NaN and opposite-signed zero handling may vary | Add explicit relaxed methods; preserve numeric min/max contracts. Restricted-domain fast paths must exclude those ambiguous inputs. |
| Float-to-int truncation | Finite inputs whose truncated integers are representable have exact results; exceptional results vary | A promising next API with explicit range assumptions or whole-vector validation. Do not substitute for saturating conversion. |
| Lane selection | All-zero/all-one mask lanes make bit selection and sign selection agree | A promising exact optimization once mask invariants and engine codegen are verified. Arbitrary bit masks require a distinct relaxed method. |
| Byte swizzle | Indices 0–15 select exactly; indices 128–255 produce zero; 16–127 vary | Keep general swizzle exact. Expose relaxed behavior explicitly or prove indices safe at the call site. |
| Q15 rounded multiplication | The pair (-32768, -32768) may wrap or saturate | A separate relaxed method, or an exact fast path excluding this pair. |
| Byte dot product / dot-add | Operand interpretation and intermediate saturation can vary | Start with a conservative nonnegative second operand 0–127, then specify accumulation/wrapping explicitly. General signed-byte dot needs a separate exact algorithm. |

These are recommendations, not newly shipped vector methods. There are twenty
standardized relaxed instructions across these families (Rust exposes additional
signed/unsigned aliases). The old token documentation's count of twenty-eight
was incorrect and has been corrected in the owning registry.

Engine choices are fixed during execution, but NaN encodings can still vary.
For this reason a fusion probe can choose the once-rounded operation; finite
startup samples cannot certify arbitrary native math as a portable reference.
WASM's [deterministic profile](https://webassembly.github.io/spec/core/appendix/profiles.html)
selects the prescribed relaxed behaviors; it is distinct from merely disabling
host FMA. Neither choice implies a global bound for composed transcendental math.

## Evidence and next promotion gates

`just wasm-selection-check` runs the fixed strict/relaxed matrix, std/no_std,
token permutations, exact sqrt fixtures and fused arithmetic. It separately
checks the fusion selector on fusing and nonfusing Wasmtime configurations.
`just sqrt-token-testing` verifies enabled/disabled/re-enabled no_std sqrt;
`just sqrt-check-exhaustive-no-std` opts into all f32 input patterns with constant
memory. Existing default sqrt spot checks remain enabled.

Before promoting any further family: test its complete ambiguity classes against
an independent oracle/set, prove any restricted input assumptions, inspect strict
and relaxed WASM plus native lowering, and use interleaved zenbench timings for
equal work. No speed ranking for those unimplemented families has been measured.
