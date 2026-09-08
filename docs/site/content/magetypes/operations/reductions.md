+++
title = "Reductions"
weight = 3
+++

`reduce_add`, `reduce_min`, and `reduce_max` collapse vector lanes to a scalar.
The [generic luma example](@/magetypes/dispatch/types-and-dispatch.md) shows a
complete reduction loop, f64 outer accumulator, and scalar tail.

Float addition order can differ by backend and vector width. Cancellation can
make the discrepancy large; it is not always a tiny fixed error. Define the
acceptable domain and tolerance for the algorithm, and test long inputs as well
as one vector. Min/max reductions also need an explicit NaN policy.

For comparison-generated integer masks:

| Method | Result |
|---|---|
| `any_true()` | At least one true lane |
| `all_true()` | Every lane true |
| `bitmask()` | Packed lane sign bits in lane order |

The return integer type of `bitmask()` is shape-specific; an eight-lane vector
need not return `u8`. Use the API's actual result type. The fused predicate
scans in `zenpixels-convert/src/scan.rs` and `zenpng/src/simd/scan.rs` combine
violations across blocks before reducing, avoiding a serial scalar decision for
every small vector. Const generics remove checks no longer needed by a scan.

Terminal byte reductions such as sum and absolute-difference sums avoid
materializing intermediate widened vectors when the caller only needs a scalar.
Use widening when later vector arithmetic actually needs those lanes.
