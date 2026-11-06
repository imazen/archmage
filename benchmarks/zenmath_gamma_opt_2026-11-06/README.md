# Certified fixed-gamma power experiment

Experimental scalar specialization for exact f32 exponents `0x4019999a`
(decode) and `0x3ed55555` (encode). No production source or public API changes.
The certificate returns a value only when rigorous outward endpoints round to
one f32; otherwise it calls independent `zenmath::scalar::reference::powf`.

[BOUND.md](BOUND.md) derives the certificate and separate Horner/Estrin budgets.
[NUMERIC-BOUNDS.md](NUMERIC-BOUNDS.md) records the generated domain and budgets.
Constants use actual C0-continuous linear-srgb source `8900ffc87cd50b249bae19c99d874431e055d888`.
The power oracle uses exact f32 arguments/exponents; a whole-transfer f64 curve
oracle is a different contract.

Initial scalar smoke covers all52 kind/degree/layout combinations (degrees8..20).
Independent directed MPFR checks passed on13687 decode and13690 encode inputs,
including inverse images of output midpoints, boundaries and fallback rails.
Containment is required even when CORE rescue returns the correct output bits.
This corpus is boundary-biased; its rescue counts do not represent throughput inputs.
No exhaustive domain sweep is claimed yet. Scalar and packed measurements are in [RESULTS.md](RESULTS.md).
The16-center degree5 scalar candidate wins all8 scalar groups against std;
packed global-coefficient degree13 decode/15 encode leads array4/8 results.
[PIECEWISE-BOUND.md](PIECEWISE-BOUND.md) and [VECTOR-BOUND.md](VECTOR-BOUND.md)
state their certificate arguments. All use the same exact CORE contract.

The first linked binary SHA256 is
`935622b2dcc84fda918cc796801602a190bfa4cf818478efcf674866995218f3`.
Rust1.99.1, empty actual Cargo rustflags, release thin LTO/codegen-units1.
Linked assembly confirms all16 V3 certificate timing contexts inline the loop
and polynomial arithmetic, with conditional CORE rescue calls only. Both std
contexts inline their loops and call system libm powf. f64 operations stay separate.

Artifacts on r5900xt: `/home/lilith/tmp/zenmath-gamma-opt-sol/runs/initial-v3/`.
The successful build log is adjacent `build-initial-v3.log`:
run-heavy rc0/29s/peak RSS0.48GiB/min available47417MiB/peak load2.19.
The earlier v2 clippy failure is retained separately. Source/binary/compiler
identities and actual Cargo fingerprints are in `identity/`; `oracle-results.csv`
retains all52 sample rescue counts. Durable mirror content audit pending; all frozen source/binaries retained.

All builds/toolchains/temp/output are local to the isolated task directory;
NAS is used for durable artifacts only. Timing uses zenbench with its shared
lock and resource gate; output arrays and executable paths are warmed first.
Numerical code is independent of the x86-only timing module/dependency.
