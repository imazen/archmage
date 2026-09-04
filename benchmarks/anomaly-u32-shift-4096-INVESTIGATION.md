# The u32x4 uniform-shift "+34-40% at 4096" anomaly — root-caused (issue #76)

Investigation of the claim that a streaming u32x4 `load -> shift -> store`
kernel over 4096 elements is reproducibly +34-40% slower on Zen 5 with a
runtime uniform count (`_mm_srl_epi32(v, xmm_count)`, count hoisted) than with
an immediate (`_mm_srli_epi32::<3>`), while 16/256 show parity, 65536 shows
+9%, and 1M is noise.

- Machine: AMD Ryzen 9 9950X3D (Zen 5), WSL2 (kernel 7.0.0-30-generic),
  rustc 1.98.1 / LLVM 22.1.8, release profile, **no** `-Ctarget-cpu=native`
  (features come from `#[arcane]`'s `#[target_feature]`).
- Harness: `magetypes/examples/u32_shift_anomaly.rs` (this branch). Raw
  interleaved-run and batch outputs: `anomaly-u32-shift-4096-INVESTIGATION.raw.txt`.
- PMU counters were available (`perf stat` works in this WSL2 after
  `sysctl kernel.perf_event_paranoid=1`), including AMD core events
  (`de_no_dispatch_per_slot.*`, `de_dispatch_stall_cycle_dynamic_tokens_*`,
  `ls_bad_status2.stli_other`, `de_src_op_disp.*`).
- Shared box, background load avg ~9-10 of 32 threads throughout; every
  conclusion below is from interleaved or alternating-launch A/B designs that
  give both arms the same environment.

## TL;DR

**The gap is not a property of the shift instruction.** `vpsrld xmm,xmm,xmm`
measures 2.0c latency, 0.50c reciprocal throughput (2/clk) — same throughput
as the immediate form. Both loop bodies compile to the identical shape
(x4-unrolled, 128-bit, count hoisted, one instruction different). What the
original record measured is a **code-placement-keyed bistable dispatch state**
of this L1-resident streaming loop on Zen 5: the same machine code runs at
either ~0.026 ns/elem (~1.8 vec/cycle) or ~0.0285-0.031 ns/elem (+10-20%),
the state is selected by the binary's code layout (deterministic per binary,
stable across ASLR and processes), and it can afflict the *immediate* form
just as well — in one layout the const loop is the slow one and the uniform
loop the fast one, simultaneously, in the same second. Across the five
layouts measured, the uniform loop drew the degraded state in 4/5 and the
const loop in 1/5, so a naive single-binary benchmark will *usually* — but
not always, and not by a fixed margin — show "uniform slower".

The honest per-op cost of the 32-bit uniform shift on x86: one loop-invariant
`vmovd` (count setup), +1 cycle of shift latency (2c vs 1c), identical
throughput, and opacity to LLVM's algebraic optimizations. None of that
produces a stable throughput difference in this kernel when measured fairly.

## 1. Reproduction, and the first crack

Sequential measurement (kernels timed one after another, one process)
reproduces the claim: first run showed uniform128 +33% at 4096 (0.0376 vs
0.0282 ns/elem). But `uniform128_licm` — a variant whose source puts
`_mm_cvtsi32_si128` inside the loop and which compiles to a **byte-identical
loop** (LICM hoists it; verified by objdump) — measured 0.0280 in the same
run: identical machine code, 33% apart. The gap as originally measured is not
(only) instruction selection.

## 2. Codegen: the leading hypothesis is false

objdump of both loops (same in every binary variant):

```
const128  loop:  4x { vmovdqu load ; vpsrld $0x3,%xmm0,%xmm0 ; vmovdqu store } ; add/add/cmp/jne
uniform128 loop: vmovd %r8d,%xmm0 hoisted in preamble, then
                 4x { vmovdqu load ; vpsrld %xmm0,%xmm1,%xmm1 ; vmovdqu store } ; add/add/cmp/jne
```

Same unroll (x4), same 128-bit widths, same instruction count per element, no
auto-widening of the const version to 256-bit, no in-loop `vmovd`. The only
difference is the shift operand form. (LLVM sees `_mm_srli_epi32::<N>` as IR
`lshr` — stdarch implements it via `simd_shr` — while `_mm_srl_epi32` is the
opaque `llvm.x86.sse2.psrl.d` intrinsic; that asymmetry matters for algebra,
§6, but did not change this loop's shape.)

## 3. Per-instruction character (register-only loops, stable across all runs)

| microbench | cycles/op | meaning |
|---|---|---|
| `vpsrld xmm,xmm,xmm` serial chain | 1.98 | latency 2c (imm form is 1c) |
| `vpsrld xmm,xmm,xmm` 8 indep chains | 0.501 | throughput 2/clk — same as imm |
| `vpsrlvd` (per-lane) serial chain | 1.96 | latency 2c |
| `vpsrlvd` 8 indep chains | 0.502 | throughput 2/clk |

(The imm form's latency cannot be chain-measured — IR `lshr` chains fold —
but its streaming loop sustains 1.8 shifts/cycle, and it has no count
operand to wait for.) These numbers were identical in every window, fast or
slow: the *instruction* is never the thing that degrades.

## 4. The bistable state

With the artifact-proofed harness (interleaved round-robin sampling,
page-controlled arena buffers, identical reps), each 128-bit kernel at 4096
occupies one of (at least) two levels:

- **fast**: 0.0258-0.0273 ns/elem (~1.8 vec/cycle at the observed 5.33 GHz)
- **degraded**: 0.0285-0.0316 (+10-20%); srlv128 additionally shows a deeper
  ~0.049 level in some windows.

Which kernel sits at which level is **sticky per binary + time window** and
was observed in *both* assignments:

| batch (fresh process per spin, alternating) | const128 | uniform128 | note |
|---|---|---|---|
| binary A, window 1 (n=15 each) | 0.0262 [.0258-.0270] | 0.0304 [.0287-.0315] | uniform degraded; licm 0.0296 co-moves |
| binary B, window 2 (n=10 each) | 0.0288 | 0.0272 | **inverted** |
| binary B, window 2' (n=8 each) | 0.0290 | 0.0293 | both degraded; licm 0.0286 |

The identical-code pair (uniform128 vs uniform128_licm, different addresses)
always landed within noise of each other.

## 5. Same-second inversion: layout is the key

Two binaries differing only by a 208-byte code shift (an added unused
function), launched alternately in the same batch (n=8 each, same minute):

| binary | const128 loop top | uniform128 loop top | const128 | uniform128 |
|---|---|---|---|---|
| B | 0x1baf0 (0x30 mod 64, spans 3 lines) | 0x1bde0 (0x20, 2 lines) | **0.0285** [.0284-.0287] | 0.0272 [.0266-.0290] |
| C | 0x1bbc0 (0x00, 2 lines) | 0x1beb0 (0x30, 3 lines) | 0.0260 [.0257-.0262] | **0.0286** [.0284-.0287] |

The assignment inverts with the layout, deterministically, tightly, in the
same wall-clock window. **The same const128 machine code is 10% slower in
binary B than in binary C; uniform128 mirrors it inversely.** This is the
central result: the measured "uniform vs const" gap is a code-placement
draw, not the instruction.

Follow-ups that constrain (but do not fully identify) the placement key:

- Both 3-line-spanning loop bodies (start ≡ 0x30 mod 64, ~0x51-0x57-byte
  bodies) were degraded — but forcing every loop 64B-aligned did **not**
  guarantee the fast state: `-C llvm-args=-align-all-blocks=6` (binary D) and
  `-C llvm-args=-align-loops=64` (binary E) both left uniform128 degraded
  (0.0314 and 0.0301) with 64B-aligned 2-line loops, while const128 was fast
  (D wobbled between states). Loop-top alignment alone is neither necessary
  nor sufficient; other moved addresses (plausibly the other hot branches
  in the process) participate.
- ASLR on/off is irrelevant (the deciding bits are below page granularity,
  identical in every launch of a given binary): 6 ASLR-off and 6 normal
  launches behaved the same.

## 6. What the PMU says about the degraded state

Per 1s spin at 4096 (5.33G cycles), fast vs degraded:

| event | fast (either form) | degraded (uniform windows) |
|---|---|---|
| IPC | 7.3 (1.82 vec/cyc) | 6.1 (1.52 vec/cyc) |
| `de_no_dispatch_per_slot.backend_stalls` | 1.0-2.3G slots | 4.6-9.5G slots |
| `de_no_dispatch_per_slot.no_ops_from_frontend` | 2.7-3.9G | 1.9-2.4G (lower!) |
| `de_dispatch_stall_...part2.ag_tokens` | 5-11M cycles | **254-740M cycles (up to 14%)** |
| `de_src_op_disp.op_cache` share | 99.98% | 99.98% (never falls to decoder) |

Ruled out by measurement: STLF conflicts / 4K aliasing
(`ls_bad_status2.stli_other` ≈ 20-140K/5G cycles both states; buffer offsets
controlled and swept — exact 4K-aliased dst changes nothing), SMT-contention
slots (low), FP-scheduler / load-queue / store-queue / INT-PRF / retire-queue
token stalls (all noise-level), pipeline flushes (0), frequency (cycles/s
equal in fast and slow uniform runs), decoder fallback (op$ share unchanged).

So the degraded state is: same ops, same op-cache delivery, dispatch
backpressured with the **address-generation-scheduler token pool exhausted**
a double-digit percentage of cycles. A second degraded sub-mode observed on
const128 in one window shows low ag_tokens but elevated backend-stall slots
(different bottleneck signature, same ~0.029 level). The loop runs 12 memory
ops per 4-vector block at ~2.2 cycles — ~3.7 AGU ops/cycle against a 4-wide
AGU limit — so it sits on a knife edge where a placement-dependent frontend
delivery pattern decides whether the LS scheduler keeps up or congests. The
exact frontend property that flips it is not identifiable from public
counters (candidates: op$ fetch-window packing of the loop, branch-predictor
set interactions with other hot branches); what is proven is that it is
keyed by code placement and agnostic to the shift's operand form.

Why 4096 specifically: at 256 elements the loop runs only 64 iterations per
call between call/`black_box` serialization points and the queues never
reach steady-state congestion (and call overhead is ~30% anyway); at 65536+
the working set leaves L1, throughput drops to ~1 vec/cycle or less, AGU
pressure falls to ~2/cycle, and the knife edge disappears — every form
measures parity there.

## 7. Size table (3 interleaved process runs, median ns/elem)

4096 (the contested size) — note run 1 is binary A, runs 2-3 binary B:

| kernel | run 1 | run 2 | run 3 |
|---|---|---|---|
| copy128 (memcpy'd by LLVM) | 0.0120 | 0.0124 | 0.0122 |
| const128 | 0.0263 | 0.0297 | 0.0292 |
| uniform128 | 0.0314 | 0.0271 | 0.0273 |
| uniform128_licm | 0.0309 | 0.0275 | 0.0273 |
| srlv128 | 0.0335 | 0.0505 | 0.0494 |
| const256 | 0.0154 | 0.0156 | 0.0154 |
| uniform256 | 0.0149 | 0.0156 | 0.0155 |
| srlv256 | 0.0268 | 0.0214 | 0.0213 |

Uniform "wins" runs 2-3 by the same mechanism it lost run 1. Other sizes
(run 1 / run 2 / run 3, const128 vs uniform128): 256: .0320/.0362/.0368 vs
.0362/.0339/.0335 (same lottery, smaller stakes); 65536: .0325/.0341/.0328
vs .0325/.0340/.0325 (parity — the issue's "+9%" did not reproduce under
interleaving); 1M: .0619/.0637/.0619 vs .0631/.0649/.0627 (parity,
memory-bound). 256-bit forms are at parity at every size in every run.

## 8. Remediation verdicts (measured)

- **256-bit lowering (`_mm256_srl_epi32`): robust.** Both 256-bit forms are
  at parity in every window and ~1.7x faster per element at 4096 than the
  best 128-bit state (0.0149-0.0156). Twice the data per AGU op moves the
  loop off the knife edge entirely. The planned u32x8/u32x16 lowerings are
  unaffected by any of this.
- **`vpsrlvd` with a splatted count as the u32x4 lowering: rejected.** Never
  better than `_mm_srl_epi32`, frequently much worse (0.0335-0.0505 at 4096
  across windows; also slower at 256: ~0.055), despite identical measured
  latency/throughput in register-only loops — it draws the degraded states
  even more readily.
- **Forcing loop alignment (`-align-loops=64` / `-align-all-blocks=6`):
  rejected.** Does not reliably select the fast state (§5) and is not
  something a library can impose on downstream builds anyway.
- **Generator-side**: nothing to fix in the planned `_mm*_srl_epi32`
  lowering itself — count hoisting already happens (LICM handles even the
  in-loop `_mm_cvtsi32_si128` form; verified byte-identical). The right
  action for #76 is to ship the 32-bit family as specified and carry this
  file as the benchmark-interpretation record.
- **For CDEF-shaped users**: where the shift feeds further arithmetic (the
  add-combine probes here, and any real filter), per-element cost rises and
  the knife edge recedes; the uniform form's +1c latency is absorbed by the
  surrounding dependency graph. One real compiler-level cost of the opaque
  form to know about: LLVM algebraically refolds the *immediate* form across
  neighboring ops — it rewrote `(a >> 3) ^ (b >> 3)` into `(a ^ b) >> 3`
  (one shift instead of two) during harness construction — and can never do
  that with `psrl.d`. Kernels with foldable shift algebra keep an inherent
  const-form advantage no benchmark hygiene can remove.

## 9. What the original "+34-40%" was

A legitimate measurement of an unlucky (and sticky) layout draw, amplified by
sequential per-kernel measurement. This investigation reproduced +33% in its
own first sequential run — and then watched the same binary pair produce
-9%, +19%, 0%, and +16% as layout and window changed while identical-code
controls tracked each other. Benchmark records for L1-resident
streaming loops of this intensity on Zen 5 need: interleaved sampling,
identical-code controls, several fresh-process launches, and at least two
code layouts before a form-vs-form delta under ~20% is believable.

## Repro

```bash
cargo build --release -p magetypes --example u32_shift_anomaly
B=target/release/examples/u32_shift_anomaly

$B                                   # interleaved bench, all kernels/sizes
$B --sizes 4096 --rounds 9           # the contested size
$B --spin uniform128 4096 1          # one kernel, 1s, for perf stat
$B --trace uniform128 4096 60        # 50ms-slice time series
perf stat -e cycles,instructions,de_dispatch_stall_cycle_dynamic_tokens_part2.ag_tokens \
    $B --spin uniform128 4096 1      # needs kernel.perf_event_paranoid<=2
$B --spin micro_lat_srl 4096 1       # latency/throughput microbenches
# layout lottery: rebuild after any code change and re-run the spins;
# objdump -d $B | grep -A40 arcane_shr_const_128 for loop-top addresses
```

Investigated 2026-09-03 on the branch `agent/issue-76-anomaly` (base:
`feat/int-widen-narrow`). GitHub issue: imazen/archmage#76.
