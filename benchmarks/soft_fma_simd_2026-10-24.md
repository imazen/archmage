# Vectorizable software FMA: main vs candidate vs std, 2026-10-24

`mul_add_portable` falls back to software on the scalar backend when the build
does not guarantee an FMA instruction, and on WASM f64. Main (code measured
from `49ae6ae8`; unchanged through `d15aaf91`) ran one helper per lane: a branchy f32 round-to-odd, and libm's integer `fma_round`
(1,100 bytes, called per lane) for f64. The candidate
(`magetypes/src/soft_fma.rs`) processes whole lane arrays with no per-lane
branch:

- f32: exact product in f64, one add, narrow. A lane goes to the exact
  round-to-odd path only when the f64 sum is an f32 midpoint or in f32's
  subnormal range.
- f64: Dekker product and TwoSum, rounded to nearest. Lanes whose final add is
  exactly halfway after an inexact inner add, or that leave the safe exponent
  range, go to an exact integer FMA ported from libm 0.2.16.

Both vectorize on SSE2 and WASM SIMD (emitted assembly: packed `cvtps2pd`/
`subpd` on baseline x86-64, `f64x2` ops on simd128, one or two branches per
array).
Results are bit-identical to the previous implementation's contract.

## Method

Two harnesses, neither run on dev (shared with other agents):

1. **Interleaved, one binary.** main's code copied verbatim (its f32 helper;
   libm 0.2.16's `fma_round` for f64), the candidate's `soft_fma.rs` included
   by path (archived as `candidate_b.rs`: the non-test code of `06291ec2`,
   with one constant spelled differently but equal), and std's per-lane
   `mul_add`, in one zenbench group per shape so
   all three run in the same rounds. Each implementation must match std bit for
   bit before timing. Stream: 1,024 arrays. Chain: 1,024 dependent steps. Both
   in a baseline function and inside an `#[arcane]` V3 function (scalar types
   there still use software, since selection is by build configuration).
   Inputs: `normal` (the harness's mod-97 pattern) and `random` (random signs,
   significands and exponents, so main's data-dependent branches mispredict).
2. **Cross-process, per revision**, for the comparison with published 0.9.30
   (`e2dbab66`), whose magetypes cannot share a binary with main's: the
   `fma-isa` harness built per revision, ten alternating passes, pinned,
   with each process's background load on other cores recorded (runs over 5%
   are excluded).

Rust 1.99.0, release, `codegen-units=1`, baseline `x86_64-unknown-linux-gnu`
(no `target-cpu`), CPU 2 pinned, under `run-heavy`. Candidate commit for the
cross-process tables: `06291ec2` (same code as pushed). wasmtime 47.0.2.

## Interleaved: candidate vs main vs std (median ns per 1,024 operations)

**r7900x** (AMD Ryzen 9 7900X, Zen 4) (zenbench `unreliable`: [False, False], gate waits: [142, 129])

| Context | Type | Kind | main (ns) | candidate (ns) | std `mul_add` (ns) | candidate / main, normal [95% CI] | candidate / main, random [95% CI] | candidate / std |
|---|---|---|---:|---:|---:|---|---|---:|
| baseline | f32x1 | stream | 1118 | 761 | 1692 | 0.68× [0.67, 0.68] | 0.67× [0.66, 0.68] | 0.45× |
| baseline | f32x1 | chain | 2543 | 2229 | 1575 | 0.90× [0.88, 0.92] | 0.88× [0.87, 0.88] | 1.42× |
| baseline | f32x8 | stream | 10722 | 4145 | 13702 | 0.45× [0.42, 0.48] | 0.32× [0.31, 0.32] | 0.30× |
| baseline | f32x8 | chain | 11891 | 4269 | 12583 | 0.36× [0.36, 0.37] | 0.36× [0.36, 0.37] | 0.34× |
| baseline | f64x1 | stream | 4441 | 4707 | 1829 | 1.06× [1.03, 1.09] | 0.99× [0.99, 1.00] (n.s.) | 2.57× |
| baseline | f64x1 | chain | 5782 | 5965 | 1543 | 1.04× [1.01, 1.08] (n.s.) | 1.03× [1.02, 1.04] | 3.87× |
| baseline | f64x4 | stream | 18015 | 11764 | 6773 | 0.65× [0.64, 0.66] | 0.60× [0.60, 0.61] | 1.74× |
| baseline | f64x4 | chain | 15950 | 11120 | 6395 | 0.69× [0.68, 0.71] | 0.70× [0.69, 0.70] | 1.74× |
| inside V3 | f32x1 | stream | 629 | 780 | 56 | 1.22× [1.18, 1.25] | 1.20× [1.19, 1.21] | 13.95× |
| inside V3 | f32x1 | chain | 6735 | 2232 | 705 | 0.33× [0.33, 0.34] | 0.35× [0.33, 0.37] | 3.17× |
| inside V3 | f32x8 | stream | 11750 | 2911 | 773 | 0.25× [0.24, 0.27] | 0.24× [0.22, 0.26] | 3.76× |
| inside V3 | f32x8 | chain | 17211 | 4084 | 704 | 0.25× [0.24, 0.26] | 0.25× [0.24, 0.26] | 5.80× |
| inside V3 | f64x1 | stream | 4503 | 4631 | 119 | 1.01× [0.96, 1.05] (n.s.) | 1.00× [0.95, 1.05] (n.s.) | 39.03× |
| inside V3 | f64x1 | chain | 5722 | 5991 | 702 | 1.07× [1.04, 1.09] | 1.05× [1.04, 1.05] | 8.53× |
| inside V3 | f64x4 | stream | 18026 | 6723 | 776 | 0.41× [0.38, 0.44] | 0.34× [0.33, 0.34] | 8.66× |
| inside V3 | f64x4 | chain | 15949 | 10182 | 704 | 0.66× [0.64, 0.68] | 0.64× [0.61, 0.66] | 14.46× |

**i265** (Intel Core Ultra 7 265K; another agent loaded the box intermittently, hence the gate waits) (zenbench `unreliable`: [False, False], gate waits: [1686, 1222])

| Context | Type | Kind | main (ns) | candidate (ns) | std `mul_add` (ns) | candidate / main, normal [95% CI] | candidate / main, random [95% CI] | candidate / std |
|---|---|---|---:|---:|---:|---|---|---:|
| baseline | f32x1 | stream | 993 | 710 | 1224 | 0.71× [0.71, 0.72] | 0.72× [0.72, 0.72] | 0.58× |
| baseline | f32x1 | chain | 3771 | 3103 | 1358 | 0.82× [0.81, 0.83] | 0.82× [0.82, 0.82] | 2.28× |
| baseline | f32x8 | stream | 9283 | 4600 | 11915 | 0.50× [0.44, 0.56] | 0.48× [0.47, 0.49] | 0.39× |
| baseline | f32x8 | chain | 12041 | 5361 | 10961 | 0.45× [0.44, 0.47] | 0.46× [0.46, 0.47] | 0.49× |
| baseline | f64x1 | stream | 4389 | 3546 | 1126 | 0.81× [0.79, 0.83] | 0.78× [0.77, 0.79] | 3.15× |
| baseline | f64x1 | chain | 8351 | 5053 | 1436 | 0.61× [0.60, 0.64] | 0.61× [0.60, 0.61] | 3.52× |
| baseline | f64x4 | stream | 17635 | 8196 | 5069 | 0.47× [0.46, 0.48] | 0.23× [0.22, 0.25] | 1.62× |
| baseline | f64x4 | chain | 15880 | 9045 | 5530 | 0.57× [0.51, 0.63] | 0.60× [0.60, 0.61] | 1.64× |
| inside V3 | f32x1 | stream | 586 | 727 | 53 | 1.24× [1.24, 1.25] | 1.23× [1.22, 1.24] | 13.77× |
| inside V3 | f32x1 | chain | 8751 | 3277 | 679 | 0.37× [0.37, 0.38] | 0.38× [0.37, 0.38] | 4.82× |
| inside V3 | f32x8 | stream | 14971 | 2321 | 931 | 0.15× [0.14, 0.16] | 0.14× [0.13, 0.14] | 2.49× |
| inside V3 | f32x8 | chain | 17366 | 4447 | 684 | 0.26× [0.25, 0.26] | 0.27× [0.27, 0.27] | 6.50× |
| inside V3 | f64x1 | stream | 4419 | 3610 | 112 | 0.82× [0.81, 0.82] | 0.79× [0.79, 0.79] | 32.22× |
| inside V3 | f64x1 | chain | 8484 | 5079 | 692 | 0.60× [0.60, 0.61] | 0.60× [0.59, 0.62] | 7.34× |
| inside V3 | f64x4 | stream | 17940 | 4296 | 989 | 0.24× [0.21, 0.26] | 0.16× [0.16, 0.17] | 4.34× |
| inside V3 | f64x4 | chain | 17138 | 9306 | 700 | 0.60× [0.56, 0.65] | 0.56× [0.56, 0.57] | 13.29× |

## Cross-process: against published 0.9.30


baseline x86-64 build, r7900x

| Case | input | 0.9.30 (ns) | main (ns) | candidate (ns) | std `mul_add` (ns) | candidate / 0.9.30 | candidate / main |
|---|---|---:|---:|---:|---:|---:|---:|
| scalar f32x1 chain | normal | 1126 | 2474 | 2215 | 1567 | 1.97× | 0.90× |
| scalar f32x1 stream | normal | 1308 | 1117 | 756 | 1683 | 0.58× | 0.68× |
| scalar f32x8 chain | normal | 7284 | 7188 | 4220 | 12561 | 0.58× | 0.59× |
| scalar f32x8 stream | normal | 8402 | 9388 | 4090 | 13658 | 0.49× | 0.44× |
| scalar f64x1 chain | normal | 938 | 5705 | 5917 | 1566 | 6.31× | 1.04× |
| scalar f64x1 stream | normal | 963 | 4457 | 4654 | 1682 | 4.83× | 1.04× |
| scalar f64x4 chain | normal | 3738 | 16259 | 10703 | 6345 | 2.86× | 0.66× |
| scalar f64x4 stream | normal | 4152 | 17992 | 11625 | 6947 | 2.80× | 0.65× |
| inside V3 f32x8 chain | normal | 2448 | 7257 | 4728 | 703 | 1.93× | 0.65× |
| inside V3 f32x8 stream | normal | 5616 | 8127 | 2869 | 764 | 0.51× | 0.35× |
| inside V3 f64x4 chain | normal | 2420 | 16083 | 7762 | 703 | 3.21× | 0.48× |
| inside V3 f64x4 stream | normal | 2827 | 17985 | 6395 | 763 | 2.26× | 0.36× |
| scalar f32x1 chain | random | 1124 | 2475 | 2206 | 1567 | 1.96× | 0.89× |
| scalar f32x1 stream | random | 1315 | 1118 | 754 | 1687 | 0.57× | 0.67× |
| scalar f32x8 chain | random | 7304 | 7228 | 4219 | 12546 | 0.58× | 0.58× |
| scalar f32x8 stream | random | 8418 | 9774 | 4075 | 13696 | 0.48× | 0.42× |
| scalar f64x1 chain | random | 938 | 5711 | 5917 | 1564 | 6.31× | 1.04× |
| scalar f64x1 stream | random | 940 | 4753 | 4638 | 1686 | 4.93× | 0.98× |
| scalar f64x4 chain | random | 3731 | 16288 | 10706 | 6348 | 2.87× | 0.66× |
| scalar f64x4 stream | random | 4145 | 19295 | 11633 | 6946 | 2.81× | 0.60× |
| inside V3 f32x8 chain | random | 2439 | 7264 | 4725 | 703 | 1.94× | 0.65× |
| inside V3 f32x8 stream | random | 5616 | 8111 | 2869 | 760 | 0.51× | 0.35× |
| inside V3 f64x4 chain | random | 2423 | 16118 | 7745 | 703 | 3.20× | 0.48× |
| inside V3 f64x4 stream | random | 2881 | 19364 | 6514 | 760 | 2.26× | 0.34× |

WASM (wasmtime 47, r7900x)

| Case | input | 0.9.30 (ns) | main (ns) | candidate (ns) | std `mul_add` (ns) | candidate / 0.9.30 | candidate / main |
|---|---|---:|---:|---:|---:|---:|---:|
| scalar f32x1 chain | normal | 2398 | 2406 | 2248 | 2232 | 0.94× | 0.93× |
| scalar f32x1 stream | normal | 1746 | 1736 | 1210 | 1737 | 0.69× | 0.70× |
| scalar f32x8 chain | normal | 7508 | 7502 | 6715 | 10584 | 0.89× | 0.90× |
| scalar f32x8 stream | normal | 12655 | 12703 | 6442 | 15339 | 0.51× | 0.51× |
| scalar f64x1 chain | normal | 8945 | 10239 | 6720 | 7916 | 0.75× | 0.66× |
| scalar f64x1 stream | normal | 9370 | 9994 | 5955 | 6904 | 0.64× | 0.60× |
| scalar f64x4 chain | normal | 34877 | 37110 | 14348 | 29124 | 0.41× | 0.39× |
| scalar f64x4 stream | normal | 38020 | 39763 | 13664 | 27552 | 0.36× | 0.34× |
| wasm f32x4 chain | normal | 8598 | 8604 | 8595 | 5372 | 1.00× (overlap) | 1.00× (overlap) |
| wasm f32x4 stream | normal | 6234 | 6253 | 6238 | 6775 | 1.00× (overlap) | 1.00× (overlap) |
| wasm f64x2 chain | normal | 18746 | 22145 | 10056 | 15783 | 0.54× | 0.45× |
| wasm f64x2 stream | normal | 18994 | 20446 | 7667 | 14574 | 0.40× | 0.37× |

## Where it is still slower

- **Single-lane f32 streams inside a V3 function** (a long loop of `f32x1` in
  an `#[arcane]` function): 1.20–1.22× main on r7900x, 1.23–1.24× on i265.
  main's branchy helper has no call, so LLVM vectorizes that loop across
  elements (packed `vcvtps2pd`/`vsubpd`/`vblendvpd` in the disassembly); the
  candidate's out-of-line exact path keeps it scalar. Scalar tails are a few elements long, so
  this pattern needs over 8 elements of `f32x1` work in one loop to matter.
- **f64x1** on r7900x: 1.03–1.07× main in chains; streams within noise. On
  i265 f64x1 takes 0.60–0.82×.
- **Against 0.9.30 in baseline builds**: f32 vectors are faster than 0.9.30
  (0.48–0.58×), but f64 takes 2.3–6.3× and f32x1 chains 1.97× as long. 0.9.30
  reached the FMA instruction through a per-call CPU check, which magetypes no
  longer does inside vector arithmetic.
- **Against std's `mul_add`**: in baseline x86 builds std calls glibc's `fma`
  per lane (hardware via ifunc on these CPUs). The candidate is faster for f32
  arrays and f32x1 streams (0.30–0.58×), slower for f32x1 chains (1.4–2.3×),
  f64x1 (2.6–3.9×) and f64x4 (1.6–1.7×). Inside a V3
  function std inlines `vfmadd`, 3–39× faster than any software path. On WASM
  the candidate beats std's software `mul_add` for f64 (0.49–0.86×).

## Design steps that were measured and rejected

- Branch-free round-to-odd on every lane (first candidate): vectorized well, but
  single-lane chains took 2.4–2.9× main's time, since the error terms sat on the
  critical path.
- A 64-bit integer parity test for round-to-odd: SSE2 has no 64-bit integer
  compare, so LLVM emulates it; replaced by truncate-then-OR.
- An exact-path helper writing through `&mut`: the hot loop's value went
  through memory every iteration (f32x1 chain 1.42–1.48× main). Returning the
  array by value instead moved it through a general register (seen in the
  disassembly, not timed). Per-lane calls returning a float keep it in an XMM
  register.

## Data

`/mnt/v/output/archmage/soft_fma_simd_2026-10-24/` on `dev`: interleaved
harness (`interleaved/`, binary SHA-256 in `b.sha256`), per-process results
and harness (`cross-process/`), emitted assembly for x86-64, WASM and RISC-V
(`asm/`).

## Follow-up: a runtime check for f64 (same day)

std's `mul_add` on baseline x86-64 calls compiler-builtins' `fma`, which runs
CPUID on first use, caches a function pointer, and then makes one indirect
call per lane to a `vfmadd`. Results from hardware FMA and the software path
are bit-identical, so a runtime check changes speed, not output.

Three designs were compared interleaved in one binary on i265 and r7900x
(data: `interleaved-rt/`): the software path; archmage's cached
`X64V3Token::summon()` once per call, then one V3 call for the whole array;
and the same check with one V3 call per lane, operands in float registers. A
second process disabled the V3 token (`dangerously_disable_token_process_wide`)
to simulate a CPU without FMA.

- f64, per-lane calls: 0.09–0.43× the software path's time for every shape,
  both hosts, both inputs; faster than std (0.40–0.71×). Without FMA:
  1.00–1.07×, one row 1.14× (i265 f64x4 stream, random).
- f32: no design was a consistent win. Array calls lost on f32x8 chains
  (1.37–1.69×: the array goes through memory every step); per-lane calls lost
  on f32x8 streams (1.20–1.89×). The vectorized software path stays.
- Inside a V3 function every design inlines and runs at hardware speed.

Shipped: f64 per-lane with the cached check on x86_64 std builds; no_std
builds use a token only when `summon` is a compile-time constant. The final
code, cross-process on r7900x against the pushed array software (normal
inputs; random agrees):


| Case | 0.9.30 | main | pushed (array software) | candidate (f64 runtime check) | std | candidate / pushed | candidate / std | candidate / 0.9.30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar f32x1 chain | 1128 | 2490 | 2222 | 2220 | 1546 | 1.00× | 1.44× | 1.97× |
| scalar f32x1 stream | 1316 | 1118 | 756 | 758 | 1687 | 1.00× | 0.45× | 0.58× |
| scalar f32x8 chain | 7313 | 7243 | 4243 | 4228 | 12711 | 1.00× | 0.33× | 0.58× |
| scalar f32x8 stream | 8435 | 9370 | 4087 | 4090 | 13861 | 1.00× | 0.30× | 0.48× |
| scalar f64x1 chain | 940 | 5708 | 5950 | 942 | 1545 | 0.16× | 0.61× | 1.00× |
| scalar f64x1 stream | 943 | 4478 | 4698 | 1135 | 1785 | 0.24× | 0.64× | 1.20× |
| scalar f64x4 chain | 3754 | 16297 | 10746 | 4494 | 6316 | 0.42× | 0.71× | 1.20× |
| scalar f64x4 stream | 4158 | 18014 | 11744 | 3753 | 6758 | 0.32× | 0.56× | 0.90× |
| inside V3 f32x8 chain | 2450 | 7294 | 4740 | 4745 | 706 | 1.00× | 6.72× | 1.94× |
| inside V3 f32x8 stream | 5635 | 8153 | 2889 | 2880 | 763 | 1.00× | 3.77× | 0.51× |
| inside V3 f64x4 chain | 2432 | 16117 | 7760 | 729 | 703 | 0.09× | 1.04× | 0.30× |
| inside V3 f64x4 stream | 2844 | 18036 | 6370 | 765 | 765 | 0.12× | 1.00× | 0.27× |

### WASM: a cached probe of relaxed madd (not shipped)

The relaxed-SIMD proposal lets each environment choose fused or unfused
`madd` per operator but requires a fixed choice ("each environment globally
chooses a fixed projection for each operator"). A probe on a pair where the
two differ, cached in an atomic, can therefore select the single instruction
for the `_portable` forms. Measured in relaxed builds on r7900x, wasmtime 47:

Engine fuses (host FMA):


| Case | main | pushed | candidate (probe) | std | a * b + c | candidate / pushed |
|---|---:|---:|---:|---:|---:|---:|
| scalar f64x4 chain | 37140 | 14513 | 14510 | 29371 | 1109 | 1.00× |
| scalar f64x4 stream | 39898 | 13792 | 13803 | 27535 | 874 | 1.00× |
| wasm f32x4 chain | 8639 | 8636 | 2667 | 5450 | 1103 | 0.31× |
| wasm f32x4 stream | 6303 | 6298 | 1775 | 6770 | 574 | 0.28× |
| wasm f64x2 chain | 21816 | 10068 | 4188 | 15980 | 1108 | 0.42× |
| wasm f64x2 stream | 20486 | 7740 | 3410 | 14538 | 576 | 0.44× |

Engine rounds twice (`-C cranelift-has-fma=false`):


| Case | main | pushed | candidate (probe) | std | a * b + c | candidate / pushed |
|---|---:|---:|---:|---:|---:|---:|
| scalar f64x4 chain | 37170 | 14518 | 14519 | 29271 | 1110 | 1.00× |
| scalar f64x4 stream | 39944 | 13798 | 13814 | 27357 | 886 | 1.00× |
| wasm f32x4 chain | 8615 | 8652 | 10398 | 5439 | 1092 | 1.20× |
| wasm f32x4 stream | 6312 | 6302 | 6906 | 6801 | 575 | 1.10× |
| wasm f64x2 chain | 22018 | 10068 | 11618 | 15897 | 1092 | 1.15× |
| wasm f64x2 stream | 20509 | 7751 | 8471 | 14692 | 576 | 1.09× |

The probe is a large win on fusing engines and costs 9–20% on engines that
round twice, so it is held for a decision rather than shipped.
