# Independent expf/logf audit

The subsequent [PR148 review](PR148.md) and [focused Magemath ownership study](MAGEMATH-LAYER.md)
retain separate source identities and evidence. The historical PR147 report below remains unchanged.

This unpublished workspace reviews [PR147](https://github.com/imazen/archmage/pull/147)
at `0464774e7c5c63a398041ddaa131b9bd4f7035c7`, against merged main
`91a13739b73f46213ecfe7083c77297c531fc8ad`. It adds review tools and failing
contract witnesses; it does not repair or approve that implementation.

Read [REVIEW.md](REVIEW.md) for findings, bounds, costs and missing coverage.

The tools are deliberately outside `magetypes/examples/` and ordinary workspace
tests. `tests/regressions.rs` expresses intended contracts: **seven tests fail
in debug, six in release on the reviewed head**. Those failures are the review's
reproducible witnesses, not waived shipping tests.

```sh
just expf-logf-audit-build
just expf-logf-audit-contracts release
cargo test --manifest-path benchmarks/expf_logf_audit/Cargo.toml --test regressions
cargo test --release --manifest-path benchmarks/expf_logf_audit/Cargo.toml --test regressions
```

`contracts` enumerates every valid builder configuration for both families and
prints const/runtime outcomes. `evaluate` reads `E HEXBITS` / `L HEXBITS`, in
homogeneous groups of 16. It prints the input, literal option bits, ScalarToken
Caller, Automatic array widths 1/4/8/16, and explicit native scalar-token results.
The quality analyzer checks exact rational fused/separate graphs independently
of directed 256-bit MPFR mathematical accuracy. Sampled maxima are not global
error guarantees. Use `--native-separate` for strict/nonfusing WASM, i686, or
the explicit no-FMA emulator execution. Engine deterministic selection alone
does not force a nonfusing relaxed multiply-add; the study also disables engine
FMA generation and verifies the resulting graph.

`legacy` is compiled unchanged against the frozen main archive and PR. Set
`AUDIT_FROZEN=1` in the PR build to validate the benchmark's frozen-source copy
against the separate main binary. `freeze.py` extracts the six original kernels
at the pinned main revision; its only API adaptations are public splat
construction and passing the existing proof token explicitly. Regenerate and
format together:

```sh
python3 -B benchmarks/expf_logf_audit/freeze.py
cargo fmt --manifest-path benchmarks/expf_logf_audit/Cargo.toml -p archmage-expf-logf-audit
```

`run_matrix.py` freezes 96 ordinary-input cells before running: four operations,
widths 4/8/16, ScalarToken and native callers, 64/4096 values, dependency chains
and independent streams. Each cell compares six explicitly named implementations
in 50 interleaved rounds. It retains every round and dirty cell. Run under a
verified process CPU boundary and resource cap; `--exploratory` explicitly opts
out of class confinement and prevents promotion evidence. `analyze_timing`
uses zenbench's paired ratio confidence analysis with every retained round.
No-regression requires the upper 95% bound to be at most 1.02 in every mandatory
clean cell; optimization additionally requires an upper bound below 1.00 in an
intended cell. Missing or dirty cells remain unproven. These requirements are
unchanged from the powf study.

`codegen` supplies named V3 consumer functions for frozen/new methods. Inspect
the executable with `just expf-logf-audit-codegen OUTPUT`. `build_cost.py` performs
three alternating target-cold default dev builds per revision, recording GNU
time peak RSS and debug rlib bytes. It keeps the OS cache warm, rebuilds dependencies,
and remaps each source path to the same debug-information prefix.

Large raw results and executable witnesses live outside Git. The review records
their checksummed artifact identity and the native source/compiler identities.
