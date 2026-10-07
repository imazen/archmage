# Stable migration diagnostic probe

This tests warning delivery, not archmage conversion. No production macro is
modified and no attune replacement is compiled. Run on Rust 1.98.1, x86_64,
2026-10-07; see [results.json](results.json) and [warnings.txt](warnings.txt).

The isolated proc-macro crate emits a per-invocation deprecated local constant
for an identity function attribute and expression macro. A separate macro is
deprecated at its definition. run.py compiles the crate and consumer with rustc,
checks JSON diagnostics, executes assertions, checks deny behavior, and verifies
that direct proc_macro Diagnostic use still requires an unstable feature.

Observed: dynamic multiline notes point at the invocation; allow/expect work;
deny fails compilation; expression arguments execute once. Three deprecations
were emitted: dynamic attribute, dynamic expression and static macro definition.
No JSON suggested_replacement was emitted, so this mechanism alone does not
provide a cargo fix edit. All asserted probe outcomes passed.

Reproduce with `just run /absolute/fresh/output/path`, or run `python3 run.py
--out /absolute/fresh/output/path` under the resource-limited build wrapper.
Every command and complete output is retained in that new directory. The runner
currently targets a Linux host (.so proc-macro output); it is not portability or
MSRV validation. No compilation-cost comparison was performed.

The generated helper pattern still needs testing in actual expansion contexts:
trait/impl methods, const and async bodies, multiple generated variants, caller
lint scopes and macro nesting. The probe does not justify injecting this helper
unchanged into all production macros. See the [migration contract](../../ATTUNE_MIGRATION.md).
