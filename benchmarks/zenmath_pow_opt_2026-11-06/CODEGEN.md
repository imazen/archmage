# Linked codegen evidence

These are linked disassemblies, including thin LTO on native platforms.
The x86 file is the actual timed executable. ARM timing uses its stripped
copy; `.text` and `.rodata` are checked byte-identical to the retained named
build used for symbol extraction. Whole functions contain guards, setup and rescue;
whole-function instruction counts are not throughput estimates.

| Artifact under archive root | Disassembled executable SHA-256 |
|---|---|
| `sources/native/codegen-final-gamma-native-v6b` | `64ebda13f3ee500d9920b26bc7f1cfb7028dd4479e47b222b5fc95eb005e2b42` |
| `sources/arm-build/codegen-final-gamma-arm-v6c` | `142d25a32d37484662bd9aacbb8df39d273ec1f073fdf32ecad96719422daff1` |
| `sources/wasm/codegen-final-wasm-v4b` | `acd5c6418a76677d8e9792f8d26b42fda30c1c15f05d7e0238367a2855c9ce10` |

On x86, `gamma_timing::__attune_contexts_v3::<11>` starts at `0x73220`,
full disassembly line 21867, extracted as `function-004.s`. This is the
current once-buffer slice adapter. It copies the buffer once at entry (the
matched before/std controls copy too), then enters the shared intrinsic power.
The admitted decode polynomial at `0x73546..0x73627` (extract lines 186–239)
is packed `vmulpd`/`vaddpd`, using registers and constant loads. No calls or
addressable coefficient/scratch arrays occur in that polynomial span. The
whole function still has a 0x208-byte frame, guard state and fallback work;
this is not a claim of a stack-free complete power loop. The scalar fixed
current entry is `gamma_timing::fixed::<0, 2>` at `0x75f30`, extract
`function-008.s`; the independent baseline and std have separate extracts.

The actual timed ARM file has SHA-256
`ac0960ee7a24b48d8dfa85c66e9d4bbc2ec638e0e1212b0640ffc501cbf9e753`.
Its `.text` and `.rodata` are byte-identical to the named build above; both
section dumps and hashes are archived in `codegen-final-gamma-arm-v6c`.
This is code/data identity, not whole-file identity after stripping.

On ARM, `gamma_timing::power::<11>` starts at `0x2a460`, full line 32639,
extract `function-013.s`. The admitted first two-lane decode graph at
`0x2a820..0x2a8a0` (extract lines 242–274) consists of `fmul v*.2d` and
`fadd v*.2d`; the second two-lane block follows at `0x2a980`. The polynomial
has no helper call per operation. The four-input abstraction comprises two
128-bit f64 pairs. The complete function retains a 0x320-byte frame, range
classification, endpoint certificates and full-range fallback. This explains
why a packed polynomial improvement does not remove all generic power cost.

The initial integrated packed graph had a coefficient array-map helper and
addressable Estrin scratch in linked code. Production `4da80e0a` generates
named stages with the exact approved pair/carry/square DAG. The final extracts
above demonstrate the intended separate arithmetic without that helper in
the polynomial span. All earlier linked files and the rejected outlined NEON
experiment remain archived rather than being relabeled as final code.

The final WASM WAT has SHA-256
`d8c02bf2d3fed22f6f13ce5bfbedcc41e26e658db85ea0e0b6754ffd7a41ecda`.
It was printed from that exact timed module by wasmprinter 0.258.3.
`decode-polynomial.wat` starts at full WAT line 133650, function index 1052;
`encode-polynomial.wat` at 133962, index 1054. Their mangled symbols and
source lines are retained in the extraction identity. They contain respectively
32/26 and 36/30 `f64x2.mul`/`f64x2.add` instructions, with zero calls in either
polynomial function. The scalar certificate at line 111054, index 724,
also has zero calls. The whole packed entry and certificate retain calls for
complete surrounding work/rescue; no whole-kernel call-free claim is made.

Earlier native policy extracts (pre-gamma stage, binary `53872d…`) found no
calls in eight approximation bodies: fused MIDP/LOWP had 10/7 scalar FMA
instructions; separate controls had none. Those observations substantiate
context specialization at that stage, not byte identity with the final binary.
The same-input C CORE controls and perf profiles are retained under `c-control`
and `profiles-controls-v2`. They are instruction/profile comparisons, not
paired speed measurements. Historical C timings on another corpus are not
used as a speed ratio, and contraction was not changed to match them.

Reproduce native extraction with `just gamma-codegen <binary> <new-directory>`.
For WASM, build `analysis` with `--features wat-dump`, run `wat-dump` on the
module, then `extract-wat.py <wat> <module> <new-directory>`. Outputs refuse
existing destinations. Exact compiler, flags, binaries and source snapshots
are mapped in [ARTIFACTS.md](ARTIFACTS.md).
