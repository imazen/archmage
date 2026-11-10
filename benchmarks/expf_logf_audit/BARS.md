# Retained zenbench bars

All numbers in the adjacent tables are ns per whole buffer. Bars are copied
verbatim from zenbench; full implementation names remain in the tables.
AMD groups below have 50/50 clean pairs. Mac groups are exploratory and
resource-flagged; their intervals cannot pass the promotion gate.

## AMD V3: exp2/lowp/4/native/64/stream

| Actual implementation | Mean ns |
|---|---:|
| main_91a13739_legacy | 13.294 |
| PR147_legacy_method | 13.228 |
| PR147_canonical_BitExact_Checked_Automatic | 16.367 |
| PR147_canonical_BitExact_AssumeValid_Automatic | 13.942 |
| PR147_canonical_BackendDependent_AssumeValid_Automatic | 13.107 |
| std_f32_per_lane | 71.862 |

```text
  PR147_canonical_BackendDe…  ████████ 13.1ns
  PR147_legacy_method         ████████ 13.2ns
  main_91a13739_legacy        ████████ 13.3ns
  PR147_canonical_BitExact_…  ████████ 13.9ns
  PR147_canonical_BitExact_…  ██████████ 16.4ns
  std_f32_per_lane            ███████████████████████████████████████████ 71.9ns
```

## AMD V3: log2/midp/4/native/64/stream

| Actual implementation | Mean ns |
|---|---:|
| main_91a13739_legacy | 22.430 |
| PR147_legacy_method | 24.802 |
| PR147_canonical_BitExact_Checked_Automatic | 23.979 |
| PR147_canonical_BitExact_AssumeValid_Automatic | 17.385 |
| PR147_canonical_BackendDependent_AssumeValid_Automatic | 16.627 |
| std_f32_per_lane | 79.561 |

```text
  PR147_canonical_BackendDe…  █████████ 16.6ns
  PR147_canonical_BitExact_…  █████████ 17.4ns
  main_91a13739_legacy        ████████████ 22.4ns
  PR147_canonical_BitExact_…  █████████████ 24.0ns
  PR147_legacy_method         █████████████ 24.8ns
  std_f32_per_lane            ███████████████████████████████████████████ 79.6ns
```

## AMD V3: log2/lowp/4/native/64/stream

| Actual implementation | Mean ns |
|---|---:|
| main_91a13739_legacy | 12.809 |
| PR147_legacy_method | 12.791 |
| PR147_canonical_BitExact_Checked_Automatic | 21.287 |
| PR147_canonical_BitExact_AssumeValid_Automatic | 16.723 |
| PR147_canonical_BackendDependent_AssumeValid_Automatic | 15.338 |
| std_f32_per_lane | 79.345 |

```text
  PR147_legacy_method         ███████ 12.8ns
  main_91a13739_legacy        ███████ 12.8ns
  PR147_canonical_BackendDe…  ████████ 15.3ns
  PR147_canonical_BitExact_…  █████████ 16.7ns
  PR147_canonical_BitExact_…  ████████████ 21.3ns
  std_f32_per_lane            ███████████████████████████████████████████ 79.3ns
```

## M6 NEON exploratory: log2/midp/4/native/64/stream

| Actual implementation | Mean ns |
|---|---:|
| main_91a13739_legacy | 18.685 |
| PR147_legacy_method | 24.058 |
| PR147_canonical_BitExact_Checked_Automatic | 18.688 |
| PR147_canonical_BitExact_AssumeValid_Automatic | 12.911 |
| PR147_canonical_BackendDependent_AssumeValid_Automatic | 12.936 |
| std_f32_per_lane | 86.041 |

```text
  PR147_canonical_BitExact_…  ██████ 12.9ns
  PR147_canonical_BackendDe…  ██████ 12.9ns
  main_91a13739_legacy        █████████ 18.7ns
  PR147_canonical_BitExact_…  █████████ 18.7ns
  PR147_legacy_method         ████████████ 24.1ns
  std_f32_per_lane            ███████████████████████████████████████████ 86.0ns
```

## M6 NEON exploratory: log2/lowp/4/native/64/stream

| Actual implementation | Mean ns |
|---|---:|
| main_91a13739_legacy | 11.152 |
| PR147_legacy_method | 11.290 |
| PR147_canonical_BitExact_Checked_Automatic | 17.697 |
| PR147_canonical_BitExact_AssumeValid_Automatic | 12.021 |
| PR147_canonical_BackendDependent_AssumeValid_Automatic | 12.089 |
| std_f32_per_lane | 85.917 |

```text
  main_91a13739_legacy        ██████ 11.2ns
  PR147_legacy_method         ██████ 11.3ns
  PR147_canonical_BitExact_…  ██████ 12.0ns
  PR147_canonical_BackendDe…  ██████ 12.1ns
  PR147_canonical_BitExact_…  █████████ 17.7ns
  std_f32_per_lane            ███████████████████████████████████████████ 85.9ns
```
