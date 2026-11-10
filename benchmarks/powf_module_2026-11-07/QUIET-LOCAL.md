# Fresh local powf timings, 2026-11-09

The final library revision is 3283954d: redundant integral rounding is removed
only for ScalarToken, while native feature contexts retain the original RNE
conversion. No result-bit or numerical-bound change is admitted. Compiler:
Rust 1.99.1, baseline distribution build, explicit V3/V4x contexts, no native CPU
flag. The 64-cell matrix keeps 50interleaved pairs per cell and every round.

14/48 native/scalar/tail cells and7/12 ScalarCaller fallback cells have fully clean
admission;0/4 all-policy cells qualify. All 40 packed-native PositiveGamma MIDP
intervals pass the 2% cost bound against unchecked legacy MIDP, for both portable
and backend-dependent graphs; all 11 clean packed cells pass. Both LOWP families
pass all 40 intervals against legacy LOWP. Backend-dependent ScalarCaller MIDP
and LOWP pass 12/12 intervals, including 7 clean cells. Software portable FMA remains
slower in all 12 fallback intervals. These counts do not establish complete quiet
ISA coverage or authorize general promotion.

## Representative clean stream

Each new row is PositiveGamma/AssumeValid/Caller, x8, 4096 values, opaque exponent
2.4. Means are ns per complete buffer. Ratios use all-retained-round zenbench
paired MeanCi 95%, 100000 bootstrap resamples.

| Context and actual implementation | Mean ns | Named reference, mean ns | Paired ratio 95% CI |
|---|---:|---|---|
| V3 portable/midp/positive-gamma/assume/caller | 1381.05 | `legacy/pow_midp_unchecked` 1445.15 | [0.954216,0.957089] |
| V3 backend/midp/positive-gamma/assume/caller | 1379.60 | `legacy/pow_midp_unchecked` 1445.15 | [0.953063,0.956301] |
| V3 portable/lowp/positive-gamma/assume/caller | 949.73 | `legacy/pow_lowp` 1230.83 | [0.767555,0.774893] |
| V3 backend/lowp/positive-gamma/assume/caller | 949.41 | `legacy/pow_lowp` 1230.83 | [0.767505,0.774532] |
| ScalarCaller portable/midp/positive-gamma/assume/caller | 34339.30 | `legacy/pow_midp_unchecked` 14271.31 | [2.345612,2.485940] |
| ScalarCaller backend/midp/positive-gamma/assume/caller | 11432.22 | `legacy/pow_midp_unchecked` 14271.31 | [0.799880,0.802147] |
| ScalarCaller portable/lowp/positive-gamma/assume/caller | 25515.34 | `legacy/pow_lowp` 14031.26 | [1.816480,1.820442] |
| ScalarCaller backend/lowp/positive-gamma/assume/caller | 9542.29 | `legacy/pow_lowp` 14031.26 | [0.679243,0.680873] |

## Retained zenbench bars

The labels below are zenbench's original abbreviations. The table above gives
the complete implementation names and ns per buffer. No bars are rescaled.

V3:

```text
  backend/lowp/positive-gam…  ██████████████████ 949.4ns
  portable/lowp/positive-ga…  ██████████████████ 949.7ns
  backend/lowp/gamma/assume…  ███████████████████ 1.03µs
  portable/lowp/gamma/assum…  ███████████████████ 1.03µs
  legacy/pow_lowp             ███████████████████████ 1.23µs
  backend/midp/positive-gam…  ██████████████████████████ 1.38µs
  portable/midp/positive-ga…  ██████████████████████████ 1.38µs
  legacy/pow_midp_unchecked   ███████████████████████████ 1.45µs
  backend/midp/gamma/assume…  ████████████████████████████ 1.49µs
  portable/midp/gamma/assum…  ████████████████████████████ 1.49µs
  legacy/pow_midp             ██████████████████████████████████████████ 2.27µs
```

ScalarCaller:

```text
  backend/lowp/positive-gam…  ███████████ 9.54µs
  backend/lowp/gamma/assume…  ████████████ 10.02µs
  backend/midp/positive-gam…  ██████████████ 11.43µs
  backend/midp/gamma/assume…  ██████████████ 12.14µs
  legacy/pow_lowp             █████████████████ 14.03µs
  legacy/pow_midp_unchecked   █████████████████ 14.27µs
  legacy/pow_midp             █████████████████████ 17.66µs
  portable/lowp/positive-ga…  ██████████████████████████████ 25.52µs
  portable/lowp/gamma/assum…  ███████████████████████████████ 25.81µs
  portable/midp/gamma/assum…  ████████████████████████████████████████ 33.52µs
  portable/midp/positive-ga…  █████████████████████████████████████████ 34.34µs
```

## Preserved earlier implementations

The first quiet run on 1552af5a found a clean ScalarCaller backend MIDP
x8/4096 stream regression versus unchecked legacy: 14651.69 ns versus 14283.77 ns,
ratio 95% CI[1.022973,1.028966]. All-native intervals passed.

The all-backend truncation experiment on 26e6db7c fixed scalar cost but introduced
a clean V3 x16/64 chain regression at exponent 2.4: portable MIDP 53.55 ns versus
legacy unchecked 51.47 ns, ratio[1.026168,1.055266]; backend MIDP 54.93 ns,
ratio[1.042778,1.101244]. That general optimization is rejected. The final
scalar-only revision preserves native RNE and passes both affected comparisons.
All three runs and all 50 rounds per cell remain; these are changed implementations,
not repeated selection of favorable timings.

## Resources and evidence

The three 64-cell timing runs report 147/146/147 s respectively, each peak RSS
0.26 GiB. Final run minimum available 44497 MiB, peak load 1.84. No own build or
analysis overlapped timing. Final analysis 71 s, peak RSS 0.02 GiB. Exactness, full
64-policy/tail replay and Clippy before the final run: 84 s, peak RSS 4.52 GiB.

Evidence directory on dev: `/mnt/v/output/archmage/powf-mid-low-final-2026-11-09/`.
The independent mirror and hashes are indexed in
[MIDLOW-ARTIFACTS.md](MIDLOW-ARTIFACTS.md). Frozen commands, binaries, source
archives, partial/failed analysis attempts and every raw round are retained.
