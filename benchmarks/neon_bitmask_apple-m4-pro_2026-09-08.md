# NEON `bitmask` formulations — Apple M4 Pro, 2026-09-08

Host: Apple M4 Pro (aarch64), macOS 26.5.2, rustc 1.98.0, `--release`, no
`-Ctarget-cpu` override.
Harnesses: `~/tmp/bmbench` (isolated, raw `int32x4_t`/`uint8x16_t` in a `Vec`)
and `~/tmp/boolbench` (through the public `magetypes::simd::generic` types,
built against `main` and against the change).

`bitmask` packs one sign bit per lane into an integer. NEON has no `movemask`,
so the lowering is a choice. **Every candidate below was checked against a
scalar reference before timing** — 16 `i32x4` patterns, 256 `i16x8`, 4
`i64x2`, and all 65536 `u8x16` patterns. They are exactly equivalent; this is
purely a performance question, not a correctness one.

## Isolated (raw NEON types)

### `i32x4` -> 4-bit mask

| formulation | ns/vec | vs shipped |
|---|---|---|
| A: one `umov` per lane, shift/or (shipped) | 0.6185 | 1.000 |
| B: `cmlt` + AND weights + `vaddvq_u32` | 0.3824 | 0.618 |
| **C: `vshrq` to 0/1 + `vshlq` into position + `vaddvq_u32`** | **0.3183** | **0.515** |
| D: `vsraq` pairs + 2 extracts | 0.4401 | 0.712 |

### `i16x8` -> 8-bit mask

| formulation | ns/vec | vs shipped |
|---|---|---|
| A: one `umov` per lane (shipped) | 0.3107 | 1.000 |
| **B: `cmlt` + AND weights + `vaddvq_u16`** | **0.2216** | **0.713** |
| C: `cmlt` + AND weights + `vpadd_u16` tree | 0.3325 | 1.070 |

### `u8x16` -> 16-bit mask

| formulation | ns/vec | vs shipped |
|---|---|---|
| A: `vmulq` weights + `vpaddlq` widening tree (shipped) | 0.3895 | 1.000 |
| B: `cmlt` + AND weights + `vpaddlq` widening tree | 0.3908 | 1.003 |
| **C: `cmlt` + AND weights + `vpadd_u8` narrow tree** | **0.3325** | **0.854** |
| D: `cmlt` + AND weights + 2x `vaddv_u8` | 0.5326 | 1.367 |

### `i64x2` -> 2-bit mask

| formulation | ns/vec | vs shipped |
|---|---|---|
| **A: two `umov`s (shipped)** | **0.5403** | **1.000** |
| B: `cmlt` + AND weights + or | 0.5559 | 1.029 |

Two things worth keeping: a single `vaddv` over 16 bytes is the *worst* option
at 16 lanes (D, 1.37x) while being the best at 4 and 8 lanes, and at 2 lanes
the vector setup never pays for itself.

## Through the public API — the number that matters

Same host, loop over a `Vec` of the generic types, built against `main` and
against this change. Mask values vary per lane (an earlier run used
all-same-sign lanes, which makes the mask always `0` or `0xF` and is not a
valid `bitmask` benchmark — those numbers were discarded).

| op | before | after | delta |
|---|---|---|---|
| `i32x4::bitmask` | 0.40382 | 0.40429 | +0.1% |
| `i16x8::bitmask` | 0.93818 | 0.89886 | -4.2% |
| `u8x16::bitmask` | 0.42757 | 0.43554 | +1.9% |

**Neutral.** The isolated 1.94x / 1.40x / 1.17x does not reproduce through the
generic types. A dependency-chained variant was also tried and turned out to
measure memory latency (~5.2-5.7 ns/op both sides), not the operation.

The discrepancy is not fully explained. The most likely cause is that inlined
into a summing loop through the generic wrapper, LLVM already reassociates the
per-lane extraction into something equivalent, so the isolated harness was
measuring a form the library never actually emits.

## Conclusion

The shipped per-lane extraction is **not** leaving measurable performance on
the table through the public API. This change is offered as a code-shape
improvement (the 16-bit case replaces eight lane extracts with three
instructions) that measures neutral, not as a performance fix. Do not cite the
isolated table as a shipped speedup.
