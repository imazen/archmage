# Runtime results

Median of five process medians; bracketed values are their minimum and maximum.
Positive changes mean more elapsed time. Ranges are not confidence intervals.

## release, 256×256

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 6.352 [6.183, 6.488] | +0.00% | 2.247 [2.152, 2.334] | +0.00% |
| body_default | 7.497 [7.387, 7.708] | +18.03% | 2.214 [2.138, 2.287] | -1.45% |

## release, 512×512

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 122.679 [121.846, 123.082] | +0.00% | 6.851 [6.699, 6.953] | +0.00% |
| body_default | 145.634 [145.277, 146.080] | +18.71% | 6.856 [6.684, 6.922] | +0.07% |

## ship, 256×256

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 4.045 [3.320, 4.196] | +0.00% | 1.984 [1.323, 2.058] | +0.00% |
| body_default | 4.074 [3.348, 4.141] | +0.72% | 1.986 [1.338, 2.163] | +0.07% |

## ship, 512×512

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 76.556 [75.761, 76.737] | +0.00% | 6.464 [5.644, 6.525] | +0.00% |
| body_default | 76.884 [75.977, 77.025] | +0.43% | 6.474 [5.660, 6.592] | +0.15% |
