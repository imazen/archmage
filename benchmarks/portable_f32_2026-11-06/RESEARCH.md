# Candidate source review (2026-11-06)

All four requested functions are evaluated directly: exp, ln, log2, pow.
No missing method is replaced with std. Published crate sources were obtained
with `cargo read`; exact versions and dependency graph are in Cargo.lock.
lib.rs access returned HTTP403; cargo-read and the registry resolved versions.

| Candidate | Version/source | Accuracy evidence/claim | Coverage and caveats |
|---|---|---|---|
| Rust std | Compiler/platform recorded per run | [Official f32 docs](https://doc.rust-lang.org/std/primitive.f32.html) specify unspecified precision | All four; libc and runtime dispatch can affect bits. Same-binary native/Nehalem comparison isolates runtime CPU effects on one OS/libc. |
| zenmath CORE-MATH Rust port | Workspace parent 8eacda2f | Existing independent MPFR validation and upstream hard cases; correctly rounded intended contract | All four, scalar; do not confuse the Rust port with upstream C. |
| magemath CORE-MATH Rust vector port | Workspace parent 8eacda2f | Same contract, four packed f64 intermediates | All four, including per-lane pow exponent. |
| Rust libm | 0.2.16 | [Primary source](https://github.com/rust-lang/libm); no blanket correctly rounded guarantee identified | All four. `force-soft-floats` is a separate build; it disables optional arch implementations but does not constitute proof of portability or accuracy. |
| fpmath | 0.2.0 | [Author docs](https://github.com/eduardosm/rust-fpmath) claim error less than 0.5+2^-10 ULP for most functions, explicitly excluding certain Bessel cases, not these four | All four; pure Rust/no_std is not itself a bit-identity proof. Test the target matrix. |
| micromath | 2.1.0 | [Author docs](https://github.com/tarcieri/micromath) explicitly trade precision for performance/size | All four via `micromath::F32`. Its source test tolerances (exp/ln/log2 .001 relative, pow .002) are sample expectations, not global bounds. ln uses approximate inverse below1; log2=ln*LOG2_E, pow=exp(y*ln(x)). |
| metallic | 0.3.1 | [Author correctness status](https://github.com/jdh8/metallic-rs#correctness-status) claims correctly rounded f32/f64, backed by exhaustive unary CORE-MATH parity plus hard-case testing | All four; this study independently tests claimed accuracy and target equality. f64 mul_add lowering deserves explicit no-FMA/WASM scrutiny. |
| magetypes midp/lowp | Workspace parent 8eacda2f | [Existing domains and accuracy limits](../../docs/transcendentals.md) | Unary methods and scalar-exponent pow. Fixed2.4 is packed; general per-lane pow uses a splat/pow/extract API adapter, not optimized general SIMD pow. |

# Maximum-error search

[math_accuracy](https://gitlab.inria.fr/zimmerma/math_accuracy) pinned commit
`c2b897fb` contains `binary64/check_sample.c` and `check_sample2.c`, which also
support USE_FLOAT. Upstream's max mode samples bit-index intervals, retains
20 promising intervals, splits them, and exhausts sufficiently small cells.
The [accuracy paper](https://members.loria.fr/PZimmermann/papers/accuracy.pdf)
explains why observed maxima are lower bounds on worst error.

`search.py` is an independently written adaptation of that algorithm, with
explicit beam/sample/threshold controls, seeded sampling, Rust candidate batch
evaluation and directed MPFR scoring. Each level retains input/output files
and a live JSON record. It searches implementation error, distinct from the
[CORE-MATH hard-to-round corpus](https://core-math.gitlabpages.inria.fr/worst.html),
which locates mathematical answers near rounding boundaries. Search and
sampling do not prove a global maximum or bound. No upstream GPL source is
copied into the Rust harness.

# Numerical definitions

Correctness percentage compares exact output bits with the independently
correctly rounded reference, with NaN payload/sign excluded. Signed-zero
mismatches and finite/nonfinite/classification mismatches are counted separately.
Adjacent representable-step distance screens candidates; it is not true ULP error.
True-real error divides distance from the mathematical result by
`2^max(-149, floor(log2(abs(real_result)))-23)`; the denominator is the f32
spacing in that binade, floored at minimum-subnormal spacing. Directed MPFR
at128 bits (increased until both endpoints round to the same f32) brackets the
reference. Directed double endpoint conversion plus outward-nextafter
subtraction/division conservatively brackets measured error. Overflow and
nonfinite answers are classification checks, not finite ULP measurements.
Eligibility also requires signed zeros, specials and tested output equality.
