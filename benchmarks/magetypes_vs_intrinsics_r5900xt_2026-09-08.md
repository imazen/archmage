# magetypes generic types vs raw intrinsics — r5900xt, 2026-09-08

Host: `r5900xt`, 32 cores, 60 GB, rustc 1.98.1, idle (load 0.00). `--release`,
no `-Ctarget-cpu` override. Library under test: `f46cf941` (archmage#111 head,
the revision zensr-micro pins).

Question this answers: **should a downstream crate write raw intrinsics instead
of depending on magetypes, and can raw intrinsics be faster?**

## The kernel

zensr-micro's `conv3x3` tap loop (`crates/zensr-micro/src/simd.rs:120`),
reproduced faithfully: two unaligned loads per tap plus two funnel shifts
(rather than three loads), four output blocks accumulated with FMA, 96 taps.
Written three ways at the same tier (x86-64-v3, AVX2+FMA):

1. **magetypes generic** — `f32x8::<X64V3Token>`, `concat_shift`, `mul_add`
2. **raw intrinsics** — `__m256`, `_mm256_loadu_ps`, `_mm256_permute2f128_ps` +
   `_mm256_alignr_epi8`, `_mm256_fmadd_ps`
3. **mixed** — magetypes types, dropping to raw intrinsics for the funnel shift
   via `into_repr()` / `from_repr(token, _)` and back

All three produce **bit-identical output** (max abs difference 0.0), asserted
before timing.

## Emitted instructions

Counted with `objdump -d` on the release binary:

| variant | vaddps | vbroadcastss | vfmadd213ps | vfmadd231ps | vmovups | vperm2f128 |
|---|---|---|---|---|---|---|
| magetypes | 4 | 12 | 8 | 4 | 5 | 1 |
| raw intrinsics | 4 | 12 | 8 | 4 | 5 | 1 |
| mixed | 4 | 12 | 8 | 4 | 5 | 1 |

Identical. There is no instruction for raw intrinsics to save.

## Throughput

Four runs of the same binary, best-of-9 inner timing each:

| run | raw / magetypes | mixed / magetypes |
|---|---|---|
| 1 | 0.9921 | 0.9654 |
| 2 | 1.0101 | 1.0056 |
| 3 | 0.9988 | 1.0043 |
| 4 | 1.0152 | 1.0011 |

Noise centred on 1.0. Run 1 in isolation would have supported "raw is 0.8%
faster, mixed 3.5% faster"; the repeats show that was code layout. Around
88-93 GFLOP/s for all three.

**Conclusion: raw intrinsics cannot be faster here, because they are the same
code.** Switching away from magetypes would also give up the four-tier
codegen — zensr writes one `#[magetypes(v3, neon, wasm128, scalar)]` body where
raw intrinsics need four hand-written kernels per algorithm.

## Mixing magetypes and intrinsics

`into_repr()` -> raw intrinsic -> `from_repr(token, _)` is the generic seam
(arch-specific `from_m256` and friends also exist). Variant 3 uses it for the
funnel shift and compiles to the identical instruction mix, so the seam costs
nothing. Dropping to intrinsics for a single operation inside a `#[magetypes]`
or `#[arcane]` body is free.

## Where a consumer's compile time actually goes

magetypes is a serialization point — crates above it cannot start until its
unit finishes — so this is the number that matters, not a share of a parallel
build. Unit only, dependencies pre-built, two runs each (identical to 0.02 s):

| what the consumer builds | serial time |
|---|---|
| archmage alone (tokens + `#[arcane]`/`#[rite]`/`#[magetypes]` macros) | 0.70 s |
| + magetypes, `w512` **off** | 1.70 s |
| + magetypes, `w512` on (crate default) | 2.27 s |
| + magetypes, `w512` + native `avx512` | 2.69 s |

Levers, largest first:

| lever | cost |
|---|---|
| native `avx512` | +0.42 s |
| `w512` on by default | **+0.57 s** |
| `concat_shift` on every backend (archmage#111) | +0.15 s |

The `w512` figure confirms the "saves ~25% build time" estimate already queued
in CLAUDE.md under QUEUED BREAKING CHANGES — it is 25% of the 2.27 s default.
It is the largest available win and roughly 4x `concat_shift`'s cost.

For reference, zensr-micro already declares
`magetypes = { default-features = false, features = ["std"] }`, which is the
right hygiene; its own `default = ["avx512"]` re-enables the lot because it
genuinely uses the `f32x16` AVX-512 tier. That is an informed trade, not an
oversight.

## Method notes

Two measurements were taken and discarded rather than reported:

- A first "archmage alone" run measured 0.06 s because the harness had already
  pre-built archmage — a no-op, not a compile.
- A `--features "std avx512"` row failed (`rc=1`) on shell quoting through ssh.

Both were re-run correctly; the numbers above are the corrected ones.
