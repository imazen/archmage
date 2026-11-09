# Std versus Rust libm: frozen comparison

This is a new provider comparison after the 64-policy implementation study.
Rust `libm` 0.2.16 is pinned, with its default `arch` feature. It is a benchmark
and numerical-reference dependency only; no production provider changes.
The local crate source and current docs confirm `libm::powf(f32,f32)`.
Std f32 power is the platform provider. They are not assumed bit-identical,
equally accurate, or interchangeable on exceptional inputs.

Reuse the 56 workload axes and input generation of POLICY-PLAN.md: scalar x1;
scalar/native x4/x8/x16; 64/4096 values; streams/normalized chains; f32 exponents
2.4/0.5. Each group has four named arms: std and Rust libm, each with literal
and opaque uniform exponent. Both providers use equal array/slice work with
baseline distribution compiler settings, one codegen unit and thin LTO.
Literal exponent specialization is a separate measured dimension; never
attribute its benefit to the general opaque-exponent provider.

Each cell retains 50 interleaved zenbench rounds. Predeclared paired edges:
libm/std literal, libm/std opaque, std opaque/literal, libm opaque/literal.
Use all retained rounds and MeanCi 100,000 resamples. Strict cost upper 95%
<=1.02 passes, lower >1.02 fails, overlap is inconclusive; optimization needs
upper <1.00. Resource flags or unverified placement prohibit promotion.

One Intel performance-core-confined pilot precedes its full matrix; stop after
a dirty pilot or failed admission. M6 uses the owner's exploratory scheduler
policy, records all 56 cells once and retains every resource flag. This new
provider experiment does not retry any old policy cell or replace old data.
No favorable rerun or gate change. Exclude this lane's builds/tests while timing.

Numerical outputs use the unchanged 27,810 domain-tagged pairs plus 99 preserved
Full-domain special witnesses. Compare each provider independently with
MPFR 256; retain actual outputs, class differences, correctly-rounded counts,
sampled ULP maxima and exact witnesses. Assert mixed-lane width consistency
within each provider at 1/4/8/16; compare non-NaN results across Intel, M6,
no-FMA emulation and strict/relaxed WASM separately. NaN payloads are unspecified.
Finite samples do not prove a whole-domain error or portability bound.

Source provenance: Rust libm https://docs.rs/libm/0.2.16/libm/fn.powf.html ;
https://github.com/rust-lang/compiler-builtins . Its source identifies the
FreeBSD/SunPro power algorithm. The clean zenmetrics checkout 097b5e4d also has
a lane transcription in `crates/zenmetrics-math/src/fdlibm.rs`; inspect it as a
separate portability lead, not as an already-admitted backend or zenbench result.
Its historical Instant/median example supplies no gate for this experiment.
