# Scalar widen fix (#77): full-width widen + half select — Zen 5 measurements

Fix record for [#77]: the `ScalarToken` backend's `widen_low_*`/`widen_high_*`
ran 1.8x (M4) to 4.6x (Zen 5) slower per element than the byte-identical casts
written inline at the call site. This record documents the root cause (with
disassembly), the candidate bodies tried, and the measured result of the fix
that shipped in the same commit as this file.

- Machine: AMD Ryzen 9 9950X3D (Zen 5), WSL2, rustc 1.98.1, no
  `-Ctarget-cpu=native`.
- Baseline commit: 93f0c69 (PR #74 head, `feat/int-widen-narrow`); the fix is
  the commit containing this file.
- Command: `cargo bench -p magetypes --bench int_widen_narrow --features std`
  (binary invoked directly, 2 process runs per configuration).
- Trimmed raw output (scalar + v3_arcane labels of all runs):
  `int_widen_narrow_scalar_fix_zen5-9950x3d_2026-09-03.raw.txt`.

## Root cause

In `widen_low_u8_to_u16(self, a: [u8; 16]) -> [u16; 8]` written as
`core::array::from_fn(|i| a[i] as u16)`, the body reads only the low 8 bytes
of the by-value source array. LLVM's SROA pass promotes that 8-byte partition
of the (inlined) argument to an **`i64` scalar**; the 8 lane reads become
`lshr`/`trunc` extracts of that integer, and the SLP vectorizer reassembles
them as scalar-to-vector inserts instead of one vector load + extend. Final
IR, low half (bad):

```llvm
%v.i.sroa.0.0.copyload = load i64, ptr %data.i.i6, align 1
%.sroa.0.7.extract.shift = lshr i64 %v.i.sroa.0.0.copyload, 56
...                                  ; 7 more shift/trunc extracts
%2 = insertelement <8 x i16> poison, i16 %.sroa.0.0.extract.trunc, i64 0
...                                  ; 7 more inserts
```

vs the high half (`|i| a[i + 8] as u16`) of the *same function*, whose
partition stayed memory-resident and loop-vectorized cleanly:

```llvm
%11 = load <8 x i8>, ptr %v.i.sroa.4.0._8.i.sroa.0.0..sroa_idx, align 1
%12 = zext <8 x i8> %11 to <8 x i16>
```

x86-64 asm per 16-byte chunk, before the fix: the high half is 3 instructions
(`movq` + `punpcklbw` + `movdqu`), the low half is ~25 (a GPR load, five
`shr`-by-8k byte extracts, seven `movd` GPR->XMM transfers, a
`punpcklwd/dq/qdq` tree, and a `pand` mask). The inline-cast baseline gets the
3-instruction form for both halves plus 2x unrolling — hence the rate (not
fixed-cost) gap that grows with N.

## Candidate bodies (asm-classified, then measured)

Probed via `#[inline(never)]` wrappers around the exact bench loop shape;
"class" = byte-identical machine code groups (linker-verified by symbol
merging).

| candidate | body shape | class |
|---|---|---|
| status quo | `from_fn(\|i\| a[i] as u16)` per half | slow-low (1 half degrades) |
| explicit `while` loop per half | same reads, indexed writes | slow-low (identical code) |
| borrow `&a` before indexing | dissolved by inlining | slow-low (identical code) |
| whole-array copy first (`let c = a;`) | SROA eats the copy | slow-low (identical code) |
| iterator `zip` write per half | same reads | slow-low (identical code) |
| caller reordered (compute both, then store) | not an impl fix; control | slow-low — **ordering is not the trigger** |
| `a[..8].try_into()` + `.map()` (the PR's attempted fix) | half copied as `[u8; 8]` | **slow-both** — the 8-byte half copy itself is integer-promoted; BOTH halves degrade (matches the PR note: 18.3 ns vs 10.3 ns at N=256) |
| `as_chunks::<8>()` + per-half `from_fn` | same half-copy | slow-both (identical code) |
| explicit `u64::from_le_bytes` + shift extract | the promoted form, written by hand | slow-both (identical code) |
| **full-width widen, select half** (shipped) | `let f: [u16; 16] = from_fn(\|i\| a[i] as u16); from_fn(\|i\| f[i])` (`f[i + 8]` for high) | **fast — byte-identical to the inline-cast baseline** |

Why the winner works: reading *all* lanes in one `from_fn` keeps the source
array a memory-resident aggregate (no partition is a promotable 8-byte
integer), so each half loop-vectorizes to `load <8 x i8>` + `zext`; the unused
half is dead-code-eliminated (verified: exactly one `punpcklbw` per call site,
none wasted). The 256-bit scalar polyfill (`u8x32`) gets the same
byte-identical-to-baseline result.

## Measured result (scalar backend, widen: u8x16 -> 2x u16x8)

| N | before (2 runs) | after (2 runs) | via_array baseline | speedup |
|---|---|---|---|---|
| 16 | 1.79 / 1.83 ns | 1.00 / 1.02 ns | 0.93-0.94 ns | 1.8x |
| 256 | 22.4 / 22.3 ns | 7.24 / 7.51 ns | 7.15-7.44 ns | **3.1x — parity** |
| 4096 | 346 / 346 ns | 75.3 / 74.8 ns | 74.9-78.3 ns | **4.6x — parity** |
| 65536 | 5.5 / 5.5 us | 1.2 / 1.2 us | 1.2 us | **4.6x — parity** |
| 1M | 88.5 / 88.3 us | 22.1 / 22.1 us | 22.2-22.7 us | **4.0x — memory ceiling** |

## Widen chain (u8 -> u16 -> u32 accumulate), scalar

| N | before | after | via_array | speedup |
|---|---|---|---|---|
| 16 | 1.74 ns | 0.93 ns | 0.85-0.88 ns | 1.9x |
| 256 | 22.1 ns | 9.76 / 9.75 ns | 9.73-9.83 ns | **2.3x — parity** |
| 4096 | 356 ns | 184 / 183 ns | 183-184 ns | **1.9x — parity** |
| 65536 | 5.6 us | 3.0 us | 3.0-3.1 us | 1.9x |
| 1M | 89.3 us | 48.9 / 49.0 us | 49.0-49.3 us | 1.8x |

The N<=256 `widen_chain` anomaly from the issue (+46-94%) is gone at every
size.

## Not changed

- **narrow**: scalar narrow was already at parity here (71.4 vs 70.5 ns at
  4096; noise-level at 1M in both directions across runs) and its two-input
  merge shape does not exhibit the half-array promotion; left as is.
- **SIMD backends**: intrinsic impls untouched; `v3_arcane` labels moved
  within noise only (e.g. widen 4096: 54.9 -> 54.7 / 55.0 ns).

Correctness: `magetypes/tests/int_widen_narrow.rs` differential suite passes;
full `cargo test -p magetypes` green (42 suites); `just check-generated`
no-op; clippy zero warnings.

[#77]: https://github.com/imazen/archmage/issues/77
