# Profiling arcane

Profile the **consumer**, `magetypes`, to measure execution of the proc macro.
Profiling `archmage-macros` itself measures the compiler building the macro library.
These are separate from rustc's work on the macro's generated output.

## Working commands

The installed tools tested here are nightly-2026-09-02 (rustc 1.100.0-nightly,
5db7f4be8) and summarize 12.0.3 (measureme revision 569c4c3b).
The [upstream summarize instructions](https://github.com/rust-lang/measureme/blob/master/summarize/README.md)
install the binary this way; `measureme-tools` is not needed for these commands:

```sh
cargo install --git https://github.com/rust-lang/measureme --branch stable summarize
```

A single consumer profile, with dependencies warmed first:

```sh
export CARGO_INCREMENTAL=0
unset RUSTC_WRAPPER
cargo +nightly-2026-09-02 build --locked -p magetypes --features avx512
cargo +nightly-2026-09-02 clean -p magetypes
mkdir -p /tmp/arcane-profile
cargo +nightly-2026-09-02 rustc --locked -p magetypes --features avx512 --lib -- \
  -Zmacro-stats -Zself-profile=/tmp/arcane-profile
summarize summarize /tmp/arcane-profile/magetypes-*.mm_profdata
```

Use a new output directory for each run, so the glob names exactly one trace.
The current trace extension is `.mm_profdata`, not `.pft`. The repeated word in
`summarize summarize` is correct: executable followed by subcommand.
Use `--release` consistently for a release profile. The example above profiles a
dev library build, whereas the independent wall-clock benchmark also measures
`cargo check` and release builds without profiling overhead.

- `-Zmacro-stats` reports expansion counts, lines, and bytes; it does not time macros.
- `expand_proc_macro` times the proc-macro boundary, including bridge overhead.
  It does not provide a breakdown of Rust functions inside arcane. In this consumer,
  all 1,473 such events correspond to arcane attributes: 1,469 short-path native
  attributes plus four existing qualified attributes. ARM is cfg-disabled here.
- `expand_crate` inclusive time includes macro execution. Do not add inclusive
  time to the child event time. Compare self times when attributing costs.
- Type checking and MIR/borrow checking measure downstream compiler work, including
  generated code. A faster macro need not reduce that work.
- `-Zself-profile-events=default,args` can add invocation labels for trace analysis,
  but is expensive: the exploratory run spent about 198 ms allocating query strings.
  Repeated measurements below use default events, without `args`.
- For function-level hotspots, sample rustc with native stack profiling and debug
  symbols for build dependencies, or extract the proc_macro2 transformation into a
  benchmark executable. The latter uses proc_macro2's fallback backend, so confirm
  wins in the actual compiler bridge too. This is the useful suggestion in the
  [profiling forum thread](https://users.rust-lang.org/t/profiling-a-proc-macro/64274/5).

The [Rust compiler profiling guide](https://rustc-dev-guide.rust-lang.org/profiling.html)
describes self-profile and native profiling. The other linked
[syn benchmark discussion](https://users.rust-lang.org/t/interesting-compile-time-benchmark-for-proc-macro-with-syn-vs-w-o-syn/139051)
reports a cold-build comparison with different clippy behavior; it does not establish
that removing syn helps arcane. Arcane already parses only signatures and leaves
function bodies as opaque tokens (`LightFn`).

For native stack sampling on a host that permits perf events, warm the macro and
its dependencies with symbols, then profile a consumer rebuild:

```sh
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_BUILD_OVERRIDE_DEBUG=true
unset RUSTC_WRAPPER
cargo +nightly-2026-09-02 build --locked -p magetypes --features avx512
cargo +nightly-2026-09-02 clean -p magetypes
perf record -e cpu-clock -F 999 --call-graph dwarf -o /tmp/arcane-perf.data -- \
  cargo +nightly-2026-09-02 build --locked -p magetypes --features avx512
perf report -i /tmp/arcane-perf.data
```

The sampling command was attempted here with `/usr/bin/perf`, but the host rejected
software events (`perf_event_paranoid=4`). No function-level sampling results are
claimed. The default `perf` on PATH also had a missing shared library; the system
binary worked up to the permissions check. No host security settings were changed.

## First optimization

Return the original token stream when it contains no `Self` identifier to replace
or no `incant` identifier to rewrite. The recursive scan looks at identifiers and
groups, not serialized text: string literals cannot trigger it. When a target is
present, the existing rewrite is retained. This avoids allocating replacement
groups for the common case without changing emitted code or feature validation.
The scan adds work when the target is present; the timings below concern the
magetypes workload, not every possible incant-heavy consumer.

Baseline: PR #86 at `44622b9`. Same Ryzen 9 9950X3D machine as the earlier PR86
measurements. Six alternating runs per variant, cached dependencies, magetypes
cleaned before each run, incremental and compiler wrapper disabled, no concurrent
builds. `profile-arcane.py` saves traces, summaries, compiler version, patch, and
compact results. Reproduce with two worktrees and a fresh output directory:

```sh
python3 benchmarks/profile-arcane.py /path/to/baseline /path/to/candidate /tmp/profiles
```

| Nightly self-profile event | Before median (range) | After median (range) |
| --- | --- | --- |
| Proc-macro execution | 95.50 ms (94.81–98.50) | 92.36 ms (88.95–93.42) |
| Expansion excluding children | 85.73 ms (79.31–87.06) | 83.65 ms (79.72–86.33) |
| Type checking | 271.77 ms (268.63–273.61) | 271.96 ms (268.58–273.17) |
| Borrow checking | 126.12 ms (124.09–126.61) | 125.96 ms (123.76–126.84) |

Proc-macro execution improves by **3.14 ms (3.3%)**. Macro stats are identical:
1,469 short-path expansions / 922,154 bytes, plus four qualified expansions /
6,830 bytes. Type checking and borrow checking are effectively unchanged.

Independent, uninstrumented stable 1.98.1 measurements using
`pr85-arcane-compile.py`, six alternating runs per command:

| Command | Before median (range) | After median (range) | Median change |
| --- | --- | --- | --- |
| check | 1.124 s (1.081–1.153) | 1.109 s (1.090–1.125) | −15 ms (−1.4%) |
| release build | 1.341 s (1.338–1.350) | 1.332 s (1.328–1.338) | −9 ms (−0.7%) |

This is a small improvement; check ranges overlap. Do not interpret the entire
wall-time delta as proc-macro execution savings or mix nightly instrumented times
with stable uninstrumented times. Raw results: `arcane-self-profile.json` and
`arcane-fast-path-compile.json`. Full local traces are in
`/tmp/arcane-profile-comparison`; large binary traces are not committed.

## Next investigations

The remaining proc-macro execution budget is about 92 ms for this workload.
Removing more cloning/parsing may help, but larger overall gains likely require
reducing the compiler work created by each expansion. Any change to inner
functions, proof assertions, or target-feature boundaries needs separate semantic
and codegen validation. This optimization leaves those structures unchanged.

## Validation

- 121 macro unit tests passed, including nested identifier scans and rewrite equivalence.
- 41 arcane/receiver integration tests passed.
- All five macro expansion test groups passed: unchanged snapshots, unexpanded
  inputs, standalone expanded outputs, and existing known-failure fixtures.
- Full magetypes tests: 1,604 passed, five ignored.
- ARM compile check with the avx512 feature passed.
- Formatting and whitespace checks passed.

These validate unchanged expansion behavior, not a new proof of token authenticity.
The previously discussed token-name/tag validation is untouched.
