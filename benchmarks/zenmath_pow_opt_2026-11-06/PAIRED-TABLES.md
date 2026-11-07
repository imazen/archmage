# Detailed paired tables

All values are ns/value. Differences are candidate minus the named control.
Intervals are direct all-round 95% bootstrap intervals using zenbench MeanCi,
with no extra exclusions. Dirty groups are replaced in full; their original
rows remain archived. An interval crossing zero does not establish an order.

Final scalar ordinary (all 28 groups per native platform)

| Platform / workload | Control | Candidate | Control mean | Candidate mean | Difference 95% CI |
|---|---|---|---:|---:|---:|
| V3: C0/decode/bits/4096 | std | current/ordinary | 1.1261 | 1.2794 | [-0.0825, +0.4043] |
| V3: C0/decode/bits/64 | std | current/ordinary | 1.1850 | 1.0918 | [-0.3278, +0.1522] |
| V3: C0/decode/values/4096 | std | current/ordinary | 5.3036 | 6.1074 | [-1.1332, +2.7801] |
| V3: C0/decode/values/64 | std | current/ordinary | 5.1048 | 4.7486 | [-1.9050, +1.1742] |
| V3: C0/encode/bits/4096 | std | current/ordinary | 1.0995 | 0.9608 | [-0.3946, +0.1242] |
| V3: C0/encode/bits/64 | std | current/ordinary | 1.1225 | 0.7731 | [-0.6234, -0.1083] |
| V3: C0/encode/values/4096 | std | current/ordinary | 5.9173 | 5.0615 | [-2.8665, +1.1527] |
| V3: C0/encode/values/64 | std | current/ordinary | 5.5747 | 4.4048 | [-2.8670, +0.4883] |
| V3: dynamic/decode/bits/4096 | std | current/ordinary | 3.8514 | 3.7816 | [-1.1670, +1.0166] |
| V3: dynamic/decode/bits/64 | std | current/ordinary | 4.6956 | 4.0369 | [-2.1230, +0.7628] |
| V3: dynamic/decode/values/4096 | std | current/ordinary | 3.7145 | 3.0795 | [-1.3042, -0.0165] |
| V3: dynamic/decode/values/64 | std | current/ordinary | 3.2297 | 3.5457 | [-0.5914, +1.2706] |
| V3: dynamic/encode/bits/4096 | std | current/ordinary | 4.0769 | 3.4121 | [-1.6889, +0.3909] |
| V3: dynamic/encode/bits/64 | std | current/ordinary | 4.1597 | 3.2082 | [-2.0745, +0.0480] |
| V3: dynamic/encode/values/4096 | std | current/ordinary | 4.3748 | 3.6691 | [-2.0457, +0.5962] |
| V3: dynamic/encode/values/64 | std | current/ordinary | 3.5663 | 4.4677 | [-0.3437, +2.1929] |
| V3: dynamic/general/16 | std | current/ordinary | 4.2426 | 10.1378 | [+3.9766, +7.9294] |
| V3: dynamic/general/65536 | std | current/ordinary | 3.8924 | 9.0838 | [+4.5861, +5.8254] |
| V3: dynamic/historical/16 | std | current/ordinary | 4.8110 | 9.2424 | [+2.6637, +6.2274] |
| V3: dynamic/historical/65536 | std | current/ordinary | 3.9877 | 9.6546 | [+4.9114, +6.4687] |
| V3: fixed/decode/bits/4096 | std | current/ordinary | 3.5538 | 3.6947 | [-0.8562, +1.1625] |
| V3: fixed/decode/bits/64 | std | current/ordinary | 3.9079 | 2.8751 | [-2.1454, -0.0844] |
| V3: fixed/decode/values/4096 | std | current/ordinary | 3.9227 | 3.3953 | [-1.6095, +0.5077] |
| V3: fixed/decode/values/64 | std | current/ordinary | 3.7721 | 3.8148 | [-1.1928, +1.3131] |
| V3: fixed/encode/bits/4096 | std | current/ordinary | 4.1869 | 3.5014 | [-1.8003, +0.4337] |
| V3: fixed/encode/bits/64 | std | current/ordinary | 3.7238 | 3.2833 | [-1.2755, +0.3551] |
| V3: fixed/encode/values/4096 | std | current/ordinary | 4.0161 | 3.1547 | [-1.9576, +0.1576] |
| V3: fixed/encode/values/64 | std | current/ordinary | 3.6014 | 4.6419 | [-0.2898, +2.4295] |
| NEON: C0/decode/bits/4096 | std | current/ordinary | 1.6693 | 1.6734 | [-0.1445, +0.1579] |
| NEON: C0/decode/bits/64 | std | current/ordinary | 1.5538 | 1.5617 | [+0.0017, +0.0158] |
| NEON: C0/decode/values/4096 | std | current/ordinary | 12.4790 | 12.9591 | [-0.0607, +0.9872] |
| NEON: C0/decode/values/64 | std | current/ordinary | 12.9200 | 12.5646 | [-1.5555, +0.5904] |
| NEON: C0/encode/bits/4096 | std | current/ordinary | 2.0992 | 2.0233 | [-0.2879, +0.1614] |
| NEON: C0/encode/bits/64 | std | current/ordinary | 2.1343 | 2.0629 | [-0.2456, +0.1284] |
| NEON: C0/encode/values/4096 | std | current/ordinary | 12.6566 | 12.5590 | [-1.5840, +1.3650] |
| NEON: C0/encode/values/64 | std | current/ordinary | 12.6971 | 12.5686 | [-1.2271, +0.9627] |
| NEON: dynamic/decode/bits/4096 | std | current/ordinary | 11.6615 | 11.2936 | [-1.8502, +0.9852] |
| NEON: dynamic/decode/bits/64 | std | current/ordinary | 10.5957 | 11.1293 | [+0.2282, +1.1279] |
| NEON: dynamic/decode/values/4096 | std | current/ordinary | 10.5681 | 11.0625 | [+0.0722, +1.1962] |
| NEON: dynamic/decode/values/64 | std | current/ordinary | 11.1776 | 11.1302 | [-1.0526, +0.8758] |
| NEON: dynamic/encode/bits/4096 | std | current/ordinary | 11.1996 | 11.1082 | [-1.3854, +1.1434] |
| NEON: dynamic/encode/bits/64 | std | current/ordinary | 11.2395 | 10.5513 | [-1.6333, -0.0621] |
| NEON: dynamic/encode/values/4096 | std | current/ordinary | 10.9398 | 10.8591 | [-1.1788, +0.8890] |
| NEON: dynamic/encode/values/64 | std | current/ordinary | 11.2027 | 11.6437 | [-0.6236, +1.4828] |
| NEON: dynamic/general/16 | std | current/ordinary | 11.1554 | 24.3761 | [+11.3016, +16.3243] |
| NEON: dynamic/general/65536 | std | current/ordinary | 11.4204 | 23.0825 | [+10.7188, +12.6546] |
| NEON: dynamic/historical/16 | std | current/ordinary | 11.1413 | 21.7354 | [+10.5226, +10.6680] |
| NEON: dynamic/historical/65536 | std | current/ordinary | 10.9072 | 21.7769 | [+10.1895, +11.6668] |
| NEON: fixed/decode/bits/4096 | std | current/ordinary | 10.0404 | 11.3551 | [+0.6779, +2.2804] |
| NEON: fixed/decode/bits/64 | std | current/ordinary | 10.7169 | 11.4583 | [-0.3992, +1.9509] |
| NEON: fixed/decode/values/4096 | std | current/ordinary | 10.9697 | 10.7223 | [-1.4364, +0.6874] |
| NEON: fixed/decode/values/64 | std | current/ordinary | 10.9291 | 11.1759 | [-1.0987, +1.3544] |
| NEON: fixed/encode/bits/4096 | std | current/ordinary | 10.1740 | 11.0684 | [+0.3941, +1.6989] |
| NEON: fixed/encode/bits/64 | std | current/ordinary | 10.1810 | 10.6191 | [+0.3718, +0.5493] |
| NEON: fixed/encode/values/4096 | std | current/ordinary | 10.1312 | 10.5501 | [+0.3137, +0.4910] |
| NEON: fixed/encode/values/64 | std | current/ordinary | 10.5046 | 11.0991 | [-0.0647, +0.9347] |

Final matched once-buffer controls

| Platform / workload | Control | Candidate | Control mean | Candidate mean | Difference 95% CI |
|---|---|---|---:|---:|---:|
| V3: dynamic/decode/values/4096 | before/slice-context | current/slice-context | 9.1208 | 3.4929 | [-6.7560, -4.4456] |
| V3: dynamic/decode/values/4096 | std/slice-context | current/slice-context | 4.1552 | 3.4929 | [-2.3098, +0.7893] |
| V3: dynamic/encode/values/4096 | before/slice-context | current/slice-context | 9.5018 | 3.5419 | [-7.3348, -4.6289] |
| V3: dynamic/encode/values/4096 | std/slice-context | current/slice-context | 3.7216 | 3.5419 | [-1.1727, +0.8441] |
| V3: dynamic/general/65536 | before/slice-context | current/slice-context | 10.1768 | 9.8608 | [-0.9138, +0.2574] |
| V3: dynamic/general/65536 | std/slice-context | current/slice-context | 3.5825 | 9.8608 | [+5.6293, +6.9811] |
| V3: dynamic/historical/65536 | before/slice-context | current/slice-context | 9.8093 | 10.4944 | [+0.1365, +1.2328] |
| V3: dynamic/historical/65536 | std/slice-context | current/slice-context | 4.2319 | 10.4944 | [+5.6524, +6.9086] |
| NEON: dynamic/decode/values/4096 | before/slice-context | current/slice-context | 17.7374 | 9.2024 | [-8.8625, -7.9085] |
| NEON: dynamic/decode/values/4096 | std/slice-context | current/slice-context | 10.7912 | 9.2024 | [-2.4957, -0.6820] |
| NEON: dynamic/encode/values/4096 | before/slice-context | current/slice-context | 18.5720 | 11.1330 | [-8.4496, -6.7932] |
| NEON: dynamic/encode/values/4096 | std/slice-context | current/slice-context | 11.3719 | 11.1330 | [-1.4708, +0.8398] |
| NEON: dynamic/general/65536 | before/slice-context | current/slice-context | 19.5514 | 19.4970 | [-1.2080, +0.9742] |
| NEON: dynamic/general/65536 | std/slice-context | current/slice-context | 11.4199 | 19.4970 | [+7.3550, +8.8384] |
| NEON: dynamic/historical/65536 | before/slice-context | current/slice-context | 19.5314 | 20.3964 | [-0.3913, +2.2296] |
| NEON: dynamic/historical/65536 | std/slice-context | current/slice-context | 11.2516 | 20.3964 | [+8.1465, +10.3778] |

Final WASM ordinary and packed controls

| Platform / workload | Control | Candidate | Control mean | Candidate mean | Difference 95% CI |
|---|---|---|---:|---:|---:|
| SIMD128: gamma-decode/uniform/stream1024 | std/control | ordinary/correct | 4.6302 | 5.1228 | [-0.7254, +1.7764] |
| SIMD128: gamma-decode/uniform/stream1024 | old-ordinary/correct | ordinary/correct | 30.8329 | 5.1228 | [-28.8239, -22.7899] |
| SIMD128: gamma-decode/uniform/stream1024 | old-packed-wasm128/correct | packed-wasm128/correct | 26.7057 | 8.9604 | [-20.6877, -14.8687] |
| SIMD128: gamma-encode/uniform/stream1024 | std/control | ordinary/correct | 4.7733 | 4.4709 | [-1.5500, +0.9924] |
| SIMD128: gamma-encode/uniform/stream1024 | old-ordinary/correct | ordinary/correct | 31.5295 | 4.4709 | [-30.5325, -23.7549] |
| SIMD128: gamma-encode/uniform/stream1024 | old-packed-wasm128/correct | packed-wasm128/correct | 27.8132 | 11.8511 | [-18.9424, -13.0657] |
| SIMD128: positive/per-value/stream16 | std/control | ordinary/correct | 6.9405 | 32.6147 | [+22.6782, +28.8682] |
| SIMD128: positive/per-value/stream16 | old-ordinary/correct | ordinary/correct | 32.1243 | 32.6147 | [-2.7939, +3.7850] |
| SIMD128: positive/per-value/stream16 | old-packed-wasm128/correct | packed-wasm128/correct | 30.8682 | 27.3372 | [-5.8613, -1.2981] |

Initial routing stage, before gamma specialization (decode uniform, 1,024 values)

| Platform / workload | Control | Candidate | Control mean | Candidate mean | Difference 95% CI |
|---|---|---|---:|---:|---:|
| V3: gamma-decode/uniform/stream1024 | old-scalar/correct | current/correct | 29.6808 | 7.9303 | [-24.5355, -19.2043] |
| V3: gamma-decode/uniform/stream1024 | old-scalar/midp-full-fused | current/midp-full-fused | 20.1337 | 6.3636 | [-16.0958, -11.4582] |
| V3: gamma-decode/uniform/stream1024 | old-scalar/lowp-full-fused | current/lowp-full-fused | 19.4755 | 6.2188 | [-15.7159, -10.9172] |
| V3: gamma-decode/uniform/stream1024 | old-scalar/midp-full-separate | current/midp-full-separate | 10.9942 | 8.2298 | [-4.3941, -1.2475] |
| V3: gamma-decode/uniform/stream1024 | old-scalar/lowp-full-separate | current/lowp-full-separate | 9.3731 | 6.6940 | [-4.8530, -0.5944] |
| V3: gamma-decode/uniform/stream1024 | old-scalar/midp-pre-fused | current/midp-pre-fused | 19.7868 | 6.4053 | [-15.8555, -10.9932] |
| V3: gamma-decode/uniform/stream1024 | old-scalar/lowp-pre-fused | current/lowp-pre-fused | 20.3492 | 4.0872 | [-18.9708, -13.7093] |
| V3: gamma-decode/uniform/stream1024 | old-scalar/midp-pre-separate | current/midp-pre-separate | 12.3923 | 9.4626 | [-5.5858, -0.2770] |
| V3: gamma-decode/uniform/stream1024 | old-scalar/lowp-pre-separate | current/lowp-pre-separate | 5.7421 | 3.8577 | [-3.6813, -0.4946] |

