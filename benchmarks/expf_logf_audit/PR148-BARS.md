# PR148 retained zenbench bars
Means are ns per whole buffer. Tables preserve full implementation names; bars
are copied verbatim from zenbench. All Intel sections below are clean with all
50 pairs and enforced placement. M6 is unconfined, resource-flagged and
exploratory. Ordered sections include the review patch, not submitted PR148.

## Intel submitted PR148: log2 MIDP, x4/64 stream

| Actual implementation | Mean ns |
|---|---:|
| main_91a13739_legacy | 24.022 |
| PR148_legacy_method | 25.769 |
| PR148_canonical_BitExact_Checked_Automatic | 21.057 |
| PR148_canonical_BitExact_AssumeValid_Automatic | 15.403 |
| PR148_canonical_BackendDependent_AssumeValid_Automatic | 14.243 |
| std_f32_per_lane | 74.122 |

```text
  PR148_canonical_BackendDe…  ████████ 14.2ns
  PR148_canonical_BitExact_…  █████████ 15.4ns
  PR148_canonical_BitExact_…  ████████████ 21.1ns
  main_91a13739_legacy        ██████████████ 24.0ns
  PR148_legacy_method         ███████████████ 25.8ns
  std_f32_per_lane            ███████████████████████████████████████████ 74.1ns
```

## Intel ordered-comparison fix: log2 MIDP, x4/64 stream

| Actual implementation | Mean ns |
|---|---:|
| main_91a13739_legacy | 23.795 |
| PR148_legacy_method | 23.803 |
| PR148_canonical_BitExact_Checked_Automatic | 19.836 |
| PR148_canonical_BitExact_AssumeValid_Automatic | 15.097 |
| PR148_canonical_BackendDependent_AssumeValid_Automatic | 13.936 |
| std_f32_per_lane | 76.168 |

```text
  PR148_canonical_BackendDe…  ████████ 13.9ns
  PR148_canonical_BitExact_…  █████████ 15.1ns
  PR148_canonical_BitExact_…  ███████████ 19.8ns
  main_91a13739_legacy        █████████████ 23.8ns
  PR148_legacy_method         █████████████ 23.8ns
  std_f32_per_lane            ███████████████████████████████████████████ 76.2ns
```

## Intel explicit range: exp2, x8/64 stream, native

| Actual implementation | Mean ns |
|---|---:|
| main_exp2_midp | 13.831 |
| main_exp2_lowp | 7.926 |
| MIDP_finite_fused_assume_token | 13.943 |
| MIDP_normal_fused_assume_token | 8.587 |
| MIDP_normal_fused_checked_token | 10.843 |
| MIDP_normal_relaxed_assume_token | 8.602 |
| LOWP_finite_cubic_fused_assume_token | 7.876 |
| LOWP_normal_cubic_fused_assume_token | 6.368 |
| LOWP_normal_quadratic_fused_assume_token | 5.386 |
| LOWP_normal_quadratic_fused_checked_token | 7.712 |
| LOWP_normal_quadratic_relaxed_assume_token | 5.442 |
| MIDP_public_BitExact_AssumeValid_Automatic | 15.140 |
| LOWP_public_BitExact_AssumeValid_Automatic | 9.464 |

```text
  LOWP_normal_quadratic_fus…  ███████████████ 5.4ns
  LOWP_normal_quadratic_rel…  ███████████████ 5.4ns
  LOWP_normal_cubic_fused_a…  ██████████████████ 6.4ns
  LOWP_normal_quadratic_fus…  ██████████████████████ 7.7ns
  LOWP_finite_cubic_fused_a…  ██████████████████████ 7.9ns
  main_exp2_lowp              ███████████████████████ 7.9ns
  MIDP_normal_fused_assume_…  ████████████████████████ 8.6ns
  MIDP_normal_relaxed_assum…  ████████████████████████ 8.6ns
  LOWP_public_BitExact_Assu…  ███████████████████████████ 9.5ns
  MIDP_normal_fused_checked…  ███████████████████████████████ 10.8ns
  main_exp2_midp              ███████████████████████████████████████ 13.8ns
  MIDP_finite_fused_assume_…  ████████████████████████████████████████ 13.9ns
  MIDP_public_BitExact_Assu…  ███████████████████████████████████████████ 15.1ns
```

## Intel explicit range: exp2, x8/64 stream, ScalarToken

| Actual implementation | Mean ns |
|---|---:|
| main_exp2_midp | 152.116 |
| main_exp2_lowp | 119.022 |
| MIDP_finite_fused_assume_token | 348.814 |
| MIDP_normal_fused_assume_token | 339.953 |
| MIDP_normal_fused_checked_token | 324.380 |
| MIDP_normal_relaxed_assume_token | 91.148 |
| LOWP_finite_cubic_fused_assume_token | 224.233 |
| LOWP_normal_cubic_fused_assume_token | 217.850 |
| LOWP_normal_quadratic_fused_assume_token | 173.803 |
| LOWP_normal_quadratic_fused_checked_token | 162.131 |
| LOWP_normal_quadratic_relaxed_assume_token | 82.376 |
| MIDP_public_BitExact_AssumeValid_Automatic | 163.726 |
| LOWP_public_BitExact_AssumeValid_Automatic | 131.392 |

```text
  LOWP_normal_quadratic_rel…  ██████████ 82.4ns
  MIDP_normal_relaxed_assum…  ███████████ 91.1ns
  main_exp2_lowp              ██████████████ 119.0ns
  LOWP_public_BitExact_Assu…  ████████████████ 131.4ns
  main_exp2_midp              ██████████████████ 152.1ns
  LOWP_normal_quadratic_fus…  ████████████████████ 162.1ns
  MIDP_public_BitExact_Assu…  ████████████████████ 163.7ns
  LOWP_normal_quadratic_fus…  █████████████████████ 173.8ns
  LOWP_normal_cubic_fused_a…  ██████████████████████████ 217.9ns
  LOWP_finite_cubic_fused_a…  ███████████████████████████ 224.2ns
  MIDP_normal_fused_checked…  ███████████████████████████████████████ 324.4ns
  MIDP_normal_fused_assume_…  █████████████████████████████████████████ 340.0ns
  MIDP_finite_fused_assume_…  ██████████████████████████████████████████ 348.8ns
```

## M6 ordered-comparison fix: exploratory log2 MIDP x4/64 stream

| Actual implementation | Mean ns |
|---|---:|
| main_91a13739_legacy | 18.785 |
| PR148_legacy_method | 18.792 |
| PR148_canonical_BitExact_Checked_Automatic | 18.706 |
| PR148_canonical_BitExact_AssumeValid_Automatic | 13.045 |
| PR148_canonical_BackendDependent_AssumeValid_Automatic | 13.062 |
| std_f32_per_lane | 86.436 |

```text
  PR148_canonical_BitExact_…  ██████ 13.0ns
  PR148_canonical_BackendDe…  ██████ 13.1ns
  PR148_canonical_BitExact_…  █████████ 18.7ns
  main_91a13739_legacy        █████████ 18.8ns
  PR148_legacy_method         █████████ 18.8ns
  std_f32_per_lane            ███████████████████████████████████████████ 86.4ns
```
