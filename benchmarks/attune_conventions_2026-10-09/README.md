# Attune convention expansion allocations — 2026-10-09

Revisions:

- Before: `67e78a83c93157ef602e5782ce670bb2f2cf61ea`.
- Initial implementation: `bb79a827601e7bb1dfe09d20059fcc5519c4cb94`.
- Final implementation: `d437eadefa95333948ca7140d95325dc01e21f80`.

Host: i265, Core Ultra 7 265K, x86_64 Linux. Rust 1.98.1
(`48a229cea`, LLVM 22.1.8). The existing `profile_allocations` unit-test harness
runs 1,000 expansions per case and reports integer averages. It uses proc_macro2's
standalone backend in the debug test profile. Input lexing is excluded; allocation
calls bypassing its test allocator are not counted.

```sh
cargo test -p archmage-macros --lib profile_allocations -- --ignored --nocapture
```

Before and initial runs used source archives with `CARGO_INCREMENTAL=0`; the final
run used the working checkout. Runs were serial under run-heavy with a 16 GiB
memory cap and eight build jobs. Raw logs and parsed results are adjacent to this
file. Nanosecond observations are retained for provenance, not evidence of a
compile-time speed change: these are single averaged runs, not repeated consumer
cold builds.

## Observations

Nine of ten existing cases have unchanged allocation counts and bytes from before
to final. All measured legacy cases (`arcane`, `arcane-dispatch`, `rite`,
`magetypes`, `autoversion`) are unchanged. `attune-compose` changes from 742
allocations / 42,699 bytes to 743 / 42,723 bytes per expansion. The initial
implementation's extra allocations in direct/all cases were eliminated by lazy
proof bindings, returning the rewritten stream without an unused capture, and
using allocation-free punctuated iteration.

The existing harness does not isolate a call that successfully inherits a proof;
these observations describe the added overhead on its existing fixtures. No
end-user cold-compile or runtime-performance claim follows from these counts.
The broader draft cold-build acceptance gate remains open.

Resource-wrapper output:

```text
before + initial: rc=0 7s | peak-RSS 0.32GiB | min-avail 28581MiB | peak-load 0.72
final profile:   rc=0 2s | peak-RSS 0.24GiB | min-avail 28895MiB | peak-load 0.27
final checks:    rc=0 91s | peak-RSS 0.34GiB | min-avail 28495MiB | peak-load 1.47
```

## Correctness validation

Final checks passed: macro unit tests; the convention integration suite with
no default features and with AVX-512 enabled; macro-crate all-feature Clippy with
warnings denied; existing attune selection/attribute/inline tests; sibling
resolution; soundness regressions; and unchanged legacy expansion snapshots plus
input/output compile suites. Execution was native x86-64. Enabling the AVX-512
Cargo feature is not an AVX-512 execution coverage claim.

```sh
cargo test -p archmage-macros
cargo test --test attune_conventions --no-default-features
cargo test --test attune_conventions --features avx512
cargo clippy -p archmage-macros --all-targets --all-features -- -D warnings
cargo test -p archmage --test attune --test attune_selection \
  --test attune_attributes --test attune_inline --test arcane_sibling_resolution \
  --test soundness_exploits --test macro_expand -- --test-threads=1
```

The full validation log is retained at
`~/tmp/attune-conventions-final-checks.log` with SHA-256:
`75af38219c1f2785d49e8c14520d81bdd0bd4f7847cfad50b8c82aa161abc9d9`.
