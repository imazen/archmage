# CORE-MATH Rust ports

Draft implementation, 2026-11-05. The owner requested Rust ports of the subset
used by zen crates, with correctly rounded defaults and separately pinned
compatibility implementations. Consumer migration is not part of this change.

Source: [CORE-MATH](https://gitlab.inria.fr/core-math/core-math), commit
`660067dfd3e751c8f07acc31c899ff410aa7206e`. Each port retains its upstream MIT
notice. Shipping code is safe Rust, has no C dependency, and supports no_std.
The contract assumes nearest-even rounding and gradual underflow. It does not
emulate errno, exception flags, dynamic rounding modes, or NaN payloads.

## Coverage and remaining work

Missing: consumer-used f64 exp/log/log2/log10/pow, trig and inverse trig, and
cancellation-safe functions. Production call sites still need a precision-aware
inventory that excludes tests and examples. Compatibility extraction remains
separate; the existing zenmetrics-math ports have not moved.

Present in `scalar`: f32 exp2f, expf, log2f, logf, log10f, powf, cbrtf, sinf,
cosf, tanf, atanf, atan2f, tanhf, log1pf and expm1f. `sincosf` composes the
individual functions. Binary64 cbrt and exp2 are ported. Explicit f64 FMA uses
safe software arithmetic unless the std configuration guarantees hardware.
Ordinary multiplication followed by addition keeps two roundings.

## Validation to date

On Core Ultra 7 265K, Rust 1.99.0, baseline x86-64, no_std default, debug:
200,000 MPFR records passed for each of the 15 f32 functions and the two f64
functions. These sampled checks are not an exhaustive correctness proof.
`tools/core_math_oracle.c` and `tools/core_math_oracle64.c` generate independent
directed MPFR bounds, increasing precision until both bounds round to the same
result. The Rust checker compares exact bits, including signed zero; NaNs compare
by classification. Missing, empty or malformed record files are errors.

Source and logs are in the current experiment directory `~/tmp/zenmath-port`
on the measurement host; durable artifact mirroring is pending. Logs
`validation2.log` through `validation7.log` record the commands and resource
wrapper results. The tangent, atan2, cbrt and exp2 build/check runs respectively
reported peak RSS 0.16, 0.15, 0.15 and 0.15 GiB, all exit status zero.

Still required: upstream hard cases, checked-in regression vectors, release and
std/FMA configurations, independent FMA regression coverage, ARM/WASM/i686/no-FMA
runs, MSRV, clippy/rustdoc, codegen and performance comparisons. No cross-platform
or performance conclusion for these new ports follows from the earlier C study.
