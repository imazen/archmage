# PR #85: arcane token receivers versus handwritten target-feature functions

Baseline: PR #85 at `c8f18fbef180509e5a87eae1d65c847d1d7aa791`.
Historical attribute-only experiment: `d38493e` (preserved locally on
`experiment/pr85-arcane-measured`). See [pr86-compile.md](pr86-compile.md) for
the final direct-generator/storage implementation and measurements.

The generator emits `#[archmage::arcane(_self = archmage::X64V3Token)]`
(and the corresponding V4/V4x/NEON token) directly on backend trait methods.
`self` supplies the token proof; no additional parameter is introduced.
Macro discovery now consults the declared receiver type when there is no
separate token parameter. Normal token validation and feature lookup are reused.
707 V3, 728 V4/V4x, and 700 NEON methods are converted. As in the baseline,
methods without unsafe blocks, scalar, and WASM are unchanged.

## Compile measurements

AMD Ryzen 9 9950X3D, Linux x86_64, rustc 1.98.1
(`48a229cea`, LLVM 22.1.8), 2026-09-04. Default CPU target; no native CPU flags.
Six runs per variant per command, alternating order. Dependencies were warmed
first; only magetypes was cleaned before each timed invocation. Incremental
compilation disabled, compiler wrapper disabled. Wall time includes Cargo.
No other builds were run concurrently with the measurements.

| Command | Baseline median (range) | arcane median (range) | Difference |
| --- | --- | --- | --- |
| `cargo check --locked -p magetypes --features avx512` | 0.929 s (0.909–0.931) | 1.140 s (1.138–1.147) | +0.212 s (+22.8%) |
| `cargo build --locked -p magetypes --features avx512 --release` | 1.146 s (1.144–1.153) | 1.342 s (1.272–1.377) | +0.196 s (+17.1%) |

The attribute version costs approximately 0.2 seconds per fresh magetypes
compilation on this machine. These measure library builds with cached
dependencies, not whole-workspace cold builds or downstream monomorphization.
The x86 measurements compile the 1,435 V3/V4 methods; NEON is cfg-disabled.
The difference includes macro expansion and its emitted token assertions and
wrappers, so it should not be attributed solely to procedural macro execution.
No runtime performance comparison was performed in this experiment.

Raw measurements: `pr85-arcane-compile.json`.
Reproduce with two checkouts (baseline and experiment):

```sh
python3 benchmarks/pr85-arcane-compile.py /path/to/baseline /path/to/experiment /tmp/results.json
```

## Validation

- `cargo run -p xtask -- generate`: passed, including soundness and summon checks.
- `cargo check -p magetypes --features avx512`: passed.
- Same check with `--target aarch64-unknown-linux-gnu`: passed.
- Arcane macro tests and new token-receiver regression: 41 passed.
- Magetypes library tests: 14 passed, 4 ignored.
- Cross-architecture parity and cross-width adversarial tests: passed on host.
- `cargo fmt --all` and `git diff --check`: passed.

The new regression exercises safe AVX2 value intrinsics on a trait impl with
an owned token receiver, a borrowed receiver, `_self` in the body, and const
generic forwarding. ARM was compile-checked, not runtime-tested.
