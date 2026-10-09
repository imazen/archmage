# Runtime results

Median of five process medians; bracketed values are their minimum and maximum.
Positive changes mean more elapsed time. Ranges are not confidence intervals.

## release, 256×256

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 5.701 [5.674, 5.736] | +0.00% | 1.585 [1.582, 1.587] | +0.00% |
| body_default | 6.974 [6.958, 6.976] | +22.32% | 1.598 [1.588, 1.604] | +0.82% |
| body_operation | 6.975 [6.968, 7.004] | +22.34% | 1.594 [1.583, 1.599] | +0.60% |

## release, 512×512

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 125.439 [124.979, 125.515] | +0.00% | 6.188 [6.152, 6.212] | +0.00% |
| body_default | 149.580 [148.987, 149.699] | +19.25% | 6.171 [6.165, 6.187] | -0.28% |
| body_operation | 149.143 [149.017, 149.635] | +18.90% | 6.180 [6.151, 6.299] | -0.13% |

## ship, 256×256

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 3.443 [3.431, 3.458] | +0.00% | 1.373 [1.361, 1.380] | +0.00% |
| body_default | 3.438 [3.417, 3.445] | -0.14% | 1.379 [1.366, 1.381] | +0.40% |
| body_operation | 3.426 [3.410, 3.467] | -0.49% | 1.370 [1.365, 1.374] | -0.22% |

## ship, 512×512

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 78.065 [77.875, 78.528] | +0.00% | 5.821 [5.804, 5.941] | +0.00% |
| body_default | 78.109 [78.046, 78.244] | +0.06% | 5.863 [5.825, 5.894] | +0.73% |
| body_operation | 78.216 [78.131, 78.320] | +0.19% | 5.856 [5.839, 5.866] | +0.61% |
