# Measured results

The generic policy bypass is fixed. Accurate policies reach the selected
whole-kernel context; fused approximations specialize their arithmetic without
per-step software FMA calls. Reference policies remain independent. The compact
positive-normal guard preserves the CORE operation order. Certified gamma
specialization then improves selected exact-exponent inputs; it does not make
correct power universally faster than std.

Final source is production `4da80e0a`. The gamma baseline is `9f466de6`, already
including routing and feature changes. The earlier nine-policy matrix compares
against `4ccaf1cf` and must not be conflated with that later baseline.

| Final once-buffer dynamic workload | Before | Current | Matched std | Current − std paired 95% CI |
|---|---:|---:|---:|---:|
| V3 decode, value-uniform, 4,096 | 9.1208 | 3.4929 | 4.1552 | [-2.3098, +0.7893] |
| V3 encode, value-uniform, 4,096 | 9.5018 | 3.5419 | 3.7216 | [-1.1727, +0.8441] |
| NEON decode, value-uniform, 4,096 | 17.7374 | 9.2024 | 10.7912 | [-2.4957, -0.6820] |
| NEON encode, value-uniform, 4,096 | 18.5720 | 11.1330 | 11.3719 | [-1.4708, +0.8398] |

Values are ns/value. These once-buffer controls enter a matching feature
context once and copy the input buffer inside the call for every candidate,
including std. Four-input array calls have their own per-block ABI/dispatch
cost; they are reported separately in [all paired tables](PAIRED-TABLES.md).
The V3 intervals against std overlap zero here; NEON decode establishes an
advantage in this measured workload. Current versus before intervals are
negative for all four rows.

General exponent-spread power still loses: V3 once-buffer current 9.8608
versus std 3.5825 ns/value, CI [+5.6293,+6.9811]. NEON is 19.4970 versus
11.4199, CI [+7.3550,+8.8384]. The historical V3 once-buffer control regresses
from 9.8093 to 10.4944, CI [+0.1365,+1.2328]. Gamma guard/general-path costs
therefore remain visible; no universal dispatch or SIMD win is claimed.

Whole-transfer composition can change the ranking. The production NEON
C0/decode/bits/64 row is 1.5617 versus std 1.5538 ns/value, CI
[+0.0017,+0.0158]. The native counterpart's interval overlaps zero. The
independent prototype also lost that corpus: 1.1443833 versus 0.9674650,
CI [+0.1694829,+0.1840226]. The final control keeps this exact corpus and the
actual C0 cutoff/constants. Pow-only improvements do not establish a complete
transfer-function win.

The initial routing matrix includes all nine policies. For uniform decode at
1,024 values, accurate generic scalar falls from 29.6808 to 7.9303 ns/value;
MIDP fused Full from 20.1337 to 6.3636; LOWP fused precondition from 20.3492
to 4.0872. All nine paired rows and intervals are retained in the detailed
tables. Approximation rows retain their declared accuracy/range and explicit
fused/separate contracts; they are not substitutes for correct rounding.
The later native LOWP separate precondition selector keeps the inline path;
NEON uses its measured context path. Policy choices are per operation/contract,
not extrapolated across architectures.

Native original coverage is 60 power + 48 staged-transfer + 4 feedback groups,
5,728 cases. Original strict WASM covers 60 power groups / 2,980 cases. Final
gamma native and NEON each cover 28 groups / 236 cases, including fixed/dynamic
exponents, bits/value distributions, C0, short/large and general controls.
Three final WASM groups retain ordinary, policy, packed and std controls.
Every reported group must be gate-clean in the effective replacement union;
original dirty groups are preserved, not filtered round by round.

Final targeted WASM timing uses the same frozen guest and embedded Wasmtime
49.0.5 executable on Ryzen 7900X. Ordinary decode is 5.1228 versus std 4.6302
ns/value (CI [-0.7254,+1.7764]); encode is 4.4709 versus 4.7733
(CI [-1.5500,+0.9924]). Both are ties at this uncertainty. General power is
32.6147 versus 6.9405 (CI [+22.6782,+28.8682]). Earlier i265 targeted runs
had repeated heavy-process gate rejections and remain diagnostic; they are
not combined with these host-specific timing values. The original complete
60-group WASM matrix on i265 remains its own earlier stage.

The table intervals use zenbench MeanCi on direct differences from identical
raw rounds, with no extra exclusions. These are direct-all-round intervals,
distinct from zenbench's stored filtered-pair analyses. Marginal variation and
mean/median gaps can be large; a clean gate does not erase drift. Near ties
are left as ties. The original JSON, ordering, compensated samples, filtered
analyses and rejected runs remain available for independent review.

Feedback measures an amortized dependent workload, not scalar instruction
latency: scalar does 1,024 dependent steps, eight lanes do 128 vector steps.
The mapping includes abs, division and normalization and converges toward
fixed points. That initial stage was not remeasured after gamma integration.
Native V4x/Scalar full gamma correctness is covered; their final throughput
was not measured. Zen5 performance was not measured.
