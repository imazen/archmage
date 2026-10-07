# Validation — 2026-10-07

The experiment's correctness, privacy and assembly commands completed
successfully; see [run.log](run.log). The artifact generator and measured tables
were committed in `aae8294d`; compiler and dependency provenance in `2fa2f8d7`.
Neither commit changes the library or macro implementation.

`cargo xtask ci` then returned 0 on the draft. Generation left the tree clean;
format, soundness, registry/token validation, API parity, verifier self-tests,
native clippy/tests, no-std checks, no-features integration, public API snapshots,
and documentation checks passed.

The existing runner skipped Miri, ARM cross-tests/clippy, and WASM cross-tests
because the required tools were unavailable. This is not cross-platform or
Miri validation. Existing xtask warnings were emitted; the generated experiment's
successful build emitted no warnings.

Resource-wrapper record for the local CI command:
`done rc=0 163s | peak-RSS 1.16GiB | min-avail 20757MiB | peak-load 2.02`.
This is a validation run, not a performance comparison.

The draft contains only experiment and design files. No attune API has been
implemented, and the main bookmark remains at the pinned source baseline.
