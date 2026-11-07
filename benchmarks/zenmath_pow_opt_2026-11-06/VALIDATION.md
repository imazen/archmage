# Validation and scope

No established expectation, numerical bound or skip was relaxed. The independent
CORE reference is retained; tests use `scalar::reference` and reference policy,
not the newly selected facade. Gamma acceptance additionally requires its
analytic certificate and exact equality on every admitted input.

| Stage / selected command | Verified outcome |
|---|---|
| Native post-gamma full `cargo test -p zenmath -p magemath --all-features --release --lib --tests --examples`, `--doc`, strict all-target clippy | Passed; combined wrapper later failed only on absent aarch64-none target, preserved |
| Installed missing target, no_std/MSRV/no-FMA/unification and 48 external feature consumers | Passed, 92 s / 0.31 GiB wrapper |
| ARM post-gamma full library/integration/example/docs/lint and 48 external feature consumers | Passed, 575 s / 0.96 GiB |
| Final `4da80e0a` native focused core/policy/acceleration, strict library lint, MSRV 1.89, no_std, qemu Nehalem no-FMA | Passed, 19 s / 0.43 GiB |
| Final unrolled native/NEON gamma tests and strict library clippy | Passed before final numerical gates |
| Legacy 35 methods × 3 widths, kernels/array/slice and mode-width parity | Passed Scalar/V3/V4x and Scalar/NEON; WASM Scalar/SIMD128 passed, 83 s / 1.27 GiB wrapper |
| Final exact admitted gamma domains | 22 cells / 1,108,465,743 comparisons passed; raw/chunk/ordered digests match |
| Final family cold build/run | 54 fresh targets passed; 171 s / 0.29 GiB |

`final-validation.sh` retains the full selected commands. `final-focused.sh`
rechecks the final unrolled production source; the earlier whole-suite and
superset stages are identified separately because their arithmetic/surfaces
were unchanged by unrolling. The focused no-FMA command uses
`CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER='qemu-x86_64 -cpu Nehalem'`;
its runner executed tests, rather than merely compiling a target.

The legacy parity generator covers every length from zero through one width,
then two complete blocks plus a three-element tail. Fixtures are lane-distinct,
with edge/special cases and nontrivial exact power exponents. Width sixteen
requires `w512`; all-feature V4x validates real supported wide contexts. Public
`splat_t`, `from_array_t` and `to_array` replace private representation access;
no Magetypes visibility was widened. Canonical regeneration reads Magemath's
owned shared bodies. Deprecated-source extraction is a documented one-time
migration tool, not a permanent generator dependency.

Feature tests cover all eight family unions and plain/proof/runtime backends
for each crate, with missing-family compile-fail cases. An external consumer
unifies Archmage std without Zenmath runtime-dispatch; LLVM checks and outputs
verify this does not silently turn on probing. Default-false primitives-only
is an explicitly authorized draft migration, not an implicit quality switch.
Normal shares precise arithmetic. Configurable accurate rescue can still
compile the private core even without public correct-rounding names.

Final gamma domain paths: native V3 ordinary/array4, V4x actual array8,
portable Scalar array4; NEON ordinary/array4/array8(two four-input blocks);
strict WASM ordinary/array4/array8(two blocks); and a separate no-SIMD module
loaded with both SIMD and relaxed SIMD disabled. Each path has both exact
exponent families. This is a bounded family validation, not all 2^64 pairs.
Wasmtime validation and timing use the retained matching 49.0.5 engine.

The actual `just final-audit <archive-root>` recipe passed (3 s / 0.30 GiB) and requires exact unique group/case inventories,
clean complete replacements, all final source bytes, every immutable snapshot
and executable hash, the full chunk coverage and output digests, two feature
consumer matrices, all 54 cold binaries, and all 902 literal bits. Large raw
results remain in the mirrored artifact archive. No macOS connection, push,
release or publication was performed. Post-gamma dependency feedback timing,
wide gamma throughput, and a causal dependency-edge build experiment were not
measured; existing feedback and wide exact parity evidence are labeled as such.
