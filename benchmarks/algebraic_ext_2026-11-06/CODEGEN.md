# External crate codegen

The external trait calls disappear in the inspected consumers, including without
LTO. This establishes feasibility through the existing public API. It does not
establish identical bytes, universal vectorization, or equal throughput.

[Extract index](codegen/index.json) records exact symbols, addresses and IR flags.
Each assembly extract starts with the full disassembly's line number and exact
**timed binary** SHA-256. [extract.py](extract.py) regenerates these extracts from
the archive. Functions include assertion cold paths; statements below about calls
and stack traffic refer to the hot loop only.

Native measurements ran on Ryzen 9 7900X with Rust 1.99.1 / LLVM 23.1.1.
ARM measurements ran on Neoverse N1 with Rust 1.99.0 / LLVM 23.1.1. The original
ARM snapshot accidentally queried the mise default compiler, 1.97.1. It remains
in the archive unchanged, alongside `arm/compiler-correction.json`, both timing
binaries' ELF `.comment` output and explicit `rustup run 1.99.0 rustc -vV`.
Both binaries and saved IR identify 1.99.0.

## Polynomial

The no-LTO V3 index/token/map consumer IR each contains 18 packed `<8 x float>`
add/multiply operations, with `reassoc nsz arcp contract`, and no references to
`algebraic_magetypes_prototype`. No `afn`, `nnan` or `ninf` flags appear in these
arithmetic extracts. The operation count includes an epilogue as well as the main
loop. See [index IR](codegen/off-v3-poly-index.ll),
[token IR](codegen/off-v3-poly-token.ll) and [local map IR](codegen/off-v3-poly-map.ll).

Actual linked V3 polynomial loops:

| LTO | Form | Function | Main loop | Packed FMAs / values |
|---|---|---|---|---|
| off | tokenless | `__attune_poly_index_f32_v3` at `0x176e00` | `0x176e90..0x176edb` | 6 / 16 |
| off | token | `__attune_poly_token_f32_v3` at `0x1770e0` | `0x177170..0x1771bb` | 6 / 16 |
| off | local map | `__attune_poly_map_f32_v3` at `0x171f80` | `0x172010..0x17205b` | 6 / 16 |
| thin | tokenless | `__attune_poly_index_f32_v3` at `0x7ac40` | `0x7acd0..0x7ad1b` | 6 / 16 |
| thin | token | `__attune_poly_token_f32_v3` at `0x7af20` | `0x7afb0..0x7affb` | 6 / 16 |
| thin | local map | `__attune_poly_map_f32_v3` at `0x75dc0` | `0x75e50..0x75e9b` | 6 / 16 |

These loops directly load/store vectors, without external calls or array stack
traffic. Instruction layout and pointer increments differ; no byte identity claim.
The baseline tokenless loop also loses the external call, but uses SSE packed
multiply/add and substantial unpack/shuffle work rather than V3 FMAs. Its exact
extract is [off baseline](codegen/off-baseline-poly-index.txt).

ARM tokenless/token/map polynomials likewise lower to packed NEON FMAs. The
main loop has 24 `fmla v*.4s` operations per 32 values, followed by an 8-value
cleanup loop with six FMAs. Neither loop calls the external crate or spills its
arrays. The saved `d8` register in the function frame is outside those loops.
See [off tokenless](codegen/off-neon-poly-index.txt),
[off token](codegen/off-neon-poly-token.txt), [off map](codegen/off-neon-poly-map.txt)
and corresponding `thin-` extracts. The repeated instructions and three-FMA
polynomial structure survive linking in both modes.

## Dot and reduction interaction

On V3, the algebraic horizontal reduction changes the optimizer's treatment of
the whole accumulation, not only the terminal sum:

- [off index](codegen/off-v3-dot-index.txt), function `0x173a10`, loop
  `0x173a80..0x173aa5`: two 256-bit FMAs per 16 values.
- [off ordinary reduction](codegen/off-v3-dot-ordinary-reduce.txt), function
  `0x17a7c0`, loop `0x17a830..0x17a87d`: four 128-bit FMAs per 16 values,
  with extraction/insertion work. The final reduction differs too.
- Both patterns survive thin linking, at loops `0x778c0..0x778e5` and
  `0x7e670..0x7e6bd` respectively. No hot-loop calls or stack traffic.

The ordinary-reduction control's means are 0.13786 / 0.14989 ns/value (off/thin)
versus 0.07496 / 0.08445 for algebraic reduction. These observations cannot be
attributed solely to the cost of a final horizontal instruction. Full controls,
including explicit fused and automatic scalar-loop cases, are in the
[CSV results](results/controls-native-off.csv).

ARM does not show that V3 width change. [Off index](codegen/off-neon-dot-index.txt)
is `cases_f32::{closure#15}` at `0xe3120`, main loop `0xe3168..0xe317c`:
two independent 128-bit accumulator FMAs per eight values. Its terminal sum is
`fadd v0.4s` then two `faddp` instructions. The
[ordinary-reduction control](codegen/off-neon-dot-ordinary-reduce.txt), closure 25
at `0xe3d30`, loop `0xe3d80..0xe3d9c`, has the same arithmetic widths and terminal
sum, with different address updates. Neither loop has calls or stack spills.
Thin loops at `0x1f794..0x1f7a8` and `0x203ac..0x203c8` retain these patterns.
This result belongs to the external-extension experiment; it does not replace
codegen findings from the separately frozen earlier matrix.

## Strict WASM

[Actual timed-module extracts](codegen/wasm-index.json) identify each function,
full-WAT line and module hash. For index/token/map in both modes, polynomial
bodies contain six `f32x4.mul` and six `f32x4.add` instructions, direct vector
loads/stores and no hot-loop calls or array spills. The three calls elsewhere
in each body are assertion failure paths. There is no packed relaxed FMA.
This is linked guest instruction evidence, not Wasmtime JIT native disassembly.

Optimized no-LTO IR provides packed flags evidence. Rust's thin-LTO `--emit`
IR is **pre-link** and some extracts are still scalar at that point; final
vectorization claims above use the actual linked assembly/WAT instead. There
was no extra unmeasured build substituted for the timing binaries.
