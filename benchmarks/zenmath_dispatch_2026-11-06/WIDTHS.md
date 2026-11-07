# Packed width comparison

Ryzen 7900X, 65,536 values, ns/value. Same raw rounds and all-round paired 95% bootstrap intervals (10,000 resamples); no new exclusions. Power uses its clean repeat. Positive differences favor V3. These are slice workloads including input copy, not isolated instruction timings.

| Operation | V3 four inputs | V4x four inputs | V4x eight inputs | V4x4 − V3 [95% CI] | V4x8 − V3 [95% CI] |
|---|---:|---:|---:|---:|---:|
| exp | 1.413 | 2.215 | 1.440 | 0.803 [0.635, 0.973] | 0.027 [-0.095, 0.147] |
| ln | 1.591 | 3.049 | 4.307 | 1.458 [1.231, 1.706] | 2.717 [2.389, 3.066] |
| log2 | 1.682 | 2.901 | 4.161 | 1.219 [1.033, 1.413] | 2.479 [2.142, 2.830] |
| pow | 10.090 | 10.684 | 12.495 | 0.594 [-0.610, 1.840] | 2.405 [1.304, 3.469] |
| pow_rescue | 13.329 | 13.549 | 15.088 | 0.219 [-0.927, 1.327] | 1.759 [0.579, 2.913] |

Automatic x86 array/slice routing now prefers V3. The four-input V4x implementation uses four f64 lanes; the explicit eight-input candidate uses eight actual f64 AVX-512 lanes. Neither width is promoted automatically. The large-slice results support this conservative routing choice; they do not establish the fastest width for every short array, CPU or input family. In particular, the exp V3-versus-eight-lane interval can overlap zero. Zen 5 was not measured.

Packed pow is helpful on the common finite corpus and substantially slower on the special/rescue mixture. No universal packed speedup is claimed. ARM/WASM compare four-input packed code with scalar/context controls, not a nonexistent wider tier.
