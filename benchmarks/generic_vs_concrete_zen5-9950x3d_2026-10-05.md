# Generic `f32x8<T>` vs concrete inside and outside `#[arcane]` (Zen 5, 2026-10-05)

Bench: `magetypes/benches/generic_vs_concrete.rs` (zenbench `criterion_compat`).
Host: AMD Ryzen 9 9950X3D, Linux 7.0.0-34-generic, rustc 1.99.0 (b940084d7
2026-09-28), no `-Ctarget-cpu=native`. Source: d1e851e9 (main at 89e2e497 plus
documentation commits; no code change). Raw means: the `.raw.txt` file beside
this one.

Commands: run 1 `~/work/zen/scripts/run-heavy --mem 16G -- cargo bench -p
magetypes --bench generic_vs_concrete`; runs 2 and 3 the same `cargo bench`
under `nice -n 19`. zenbench printed its "0 rounds" warning on every group; the
means are still measured.

## Results (ns per call)

| Pattern | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Generic `f32x8<T>`, no `#[target_feature]` caller | 7.94 | 10.2 | 7.95 |
| Generic, no inline annotation, inside `#[arcane]` | 1.09 | 1.38 | 1.09 |
| Concrete `X64V3Token` inside `#[arcane]` | 1.09 | 1.37 | 1.08 |
| Concrete via `#[rite]` inside `#[arcane]` | 1.02 | 1.38 | 1.06 |
| Generic `#[inline(never)]` inside `#[arcane]` | 8.13 | 10.0 | 7.74 |
| Generic `#[inline(always)]` inside `#[arcane]` | 1.08 | 1.37 | 1.05 |
| Dot product, generic, no `#[target_feature]` caller | 8.27 | 10.5 | 8.08 |
| Dot product, generic inside `#[arcane]` | 0.88 | 1.15 | 0.88 |
| Dot product, concrete inside `#[arcane]` | 0.88 | 1.14 | 0.88 |

Run 2 is about 27% slower in every row, the clock rather than the code; the
ratios agree across runs. A generic function that does not inline into a
feature-enabled caller takes 7.3–7.6 times as long as one that does (9.1–9.4
times for the dot product). The 2026-02-26 version of this bench measured 18x
on a different build; that figure does not reproduce here.

## Assembly (`cargo asm -p magetypes --bench generic_vs_concrete`)

`#[inline(always)]` generic inside `#[arcane]` and the concrete V3 entry are
instruction-identical:

```asm
vmovups ymm0, ymmword ptr [rdi]
vaddps ymm0, ymm0, ymm0
vextractf128 xmm1, ymm0, 1
vaddps xmm0, xmm0, xmm1
vmovshdup xmm1, xmm0
vaddps xmm0, xmm0, xmm1
vshufpd xmm1, xmm0, xmm0, 1
vaddss xmm0, xmm0, xmm1
vzeroupper
ret
```

The `#[inline(never)]` generic loads with SSE, spills both halves to the
stack, and calls each backend method's `#[target_feature]` inner function:

```asm
movups xmm0, xmmword ptr [rdi]
movups xmm1, xmmword ptr [rdi + 16]
; ... four stores to the stack ...
call <X64V3Token as F32x8Backend>::add::__simd_inner_add
mov rdi, rbx
call <X64V3Token as F32x8Backend>::reduce_add::__simd_inner_reduce_add
```

Limits: one machine, one toolchain, an 8-lane kernel small enough that the call
overhead dominates. A larger kernel loses a different fraction to the boundary.
