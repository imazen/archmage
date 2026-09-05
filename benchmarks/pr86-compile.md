# PR #86: direct arcane generation and checked SIMD storage

This is the final measurement after replacing the PR #85 post-pass with direct
native method templates and checked POD storage helpers. The earlier
`pr85-arcane-compile.md` records the initial attribute-only experiment.

## Implementation

Native generator templates interpolate `{arcane}` before methods. The shared
syntax helper emits `#[archmage::arcane(_self = archmage::ConcreteToken)]`;
`self` is the token proof and method bodies refer to it as `_self`. There is no
post-processing pass. Templates generate safe intrinsics, checked bit casts,
and fixed-array load/store helpers directly.

All 2,890 native backend methods are preserved. 2,206 use arcane (741 V3,
728 V4/V4x, 737 NEON); the remaining methods compose safe backend operations.
Explicit unsafe blocks in these three generated files fell from 242 in the
initial arcane experiment to zero. Memory operations now use two unsafe blocks
in a private storage module, plus audited unsafe POD trait implementations.
The macro still emits one unsafe feature boundary per arcane method.
Tokens and token-bearing public SIMD wrappers do not implement the POD trait.

The storage approach is inspired by
[fearless_simd's transmute module](https://github.com/linebender/fearless_simd/blob/main/fearless_simd/src/transmute.rs).
No new dependency is added. Scalar and WASM backend output is unchanged.

## Measurements

AMD Ryzen 9 9950X3D, Linux x86_64, rustc 1.98.1 (`48a229cea`, LLVM 22.1.8),
2026-09-04. No CPU/target-feature build flags. Six runs per variant and command,
using all six permutations of build order. Dependencies warmed first; only
magetypes cleaned before each measurement. Incremental compilation and compiler
wrapper disabled. No concurrent builds. Wall time includes Cargo.

| Command | main median (range) | #85 median (range) | #86 median (range) | #86 vs #85 |
| --- | --- | --- | --- | --- |
| check | 0.831 s (0.808–0.855) | 0.935 s (0.879–0.936) | 1.142 s (1.137–1.150) | +0.207 s (+22.1%) |
| release build | 1.054 s (1.033–1.069) | 1.156 s (1.102–1.163) | 1.352 s (1.305–1.413) | +0.195 s (+16.9%) |

Relative to main, #86 adds 0.312 s (+37.5%) for check and 0.298 s (+28.3%)
for release. These are library rebuilds with cached dependencies, not whole
workspace cold builds or downstream monomorphization. NEON is cfg-disabled in
these x86 timings. The difference includes macro expansion and emitted code,
not just procedural macro execution.

Commands: `cargo check --locked -p magetypes --features avx512` and
`cargo build --locked -p magetypes --features avx512 --release`.
Main baseline: `02611f5e8bfc771f5819b4f0c2d5d98d768391f1`.
PR #85 baseline: `c8f18fbef180509e5a87eae1d65c847d1d7aa791`.
Raw data: `pr86-compile.json` (`baseline` means #85, `arcane` means #86).

```sh
python3 benchmarks/pr85-arcane-compile.py /path/to/pr85 /path/to/pr86 /tmp/results.json /path/to/main
```

## Validation

- Full magetypes test suite (`--features avx512 --tests`): 1,604 passed, 5 ignored.
- Arcane macro/receiver tests: 41 passed.
- Generator regression verifies safe native output before formatting: passed.
- Generation, soundness scanner, summon checks, and regeneration idempotence: passed.
- Miri storage tests: 2 passed, covering bit preservation and weak alignment.
- Explicit rustc probes reject different-sized casts and non-POD bool destinations.
- x86/ARM checks, x86/ARM no-std checks, and WASM check: passed. WASM has two
  pre-existing unnecessary-parentheses warnings in unchanged generated code.
- Clippy for magetypes with AVX-512 and warnings denied: passed.
- Release `tf_inline_check` disassembly: zero residual calls in the kernel,
  for both #85 and #86. This is a sample codegen check, not a runtime benchmark.
- Formatting and diff whitespace checks: passed.

ARM was compile-checked, not runtime-tested. Other existing unsafe code outside
the three native backend files is outside this change's scope.
