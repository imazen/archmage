# Constructor API and prelude audit, 2026-09-27

The large diff has two causes: an intended change to shared constructor modes,
and unintended prelude exports. The latter is fixed before publication.

Baseline: `1a0ea59` (before constructor modes). Expanded surface: `7d557b90`
(after `_with_token`). Correction: `cd6cdc40`.

## What was intended

- Thirty public generic vector names now alias a shared core with `Explicit`
  fixed as its mode. The core has a second, sealed mode parameter. The contextual
  aliases fix `Context`. This avoids ambiguous inherent constructor lookup on a
  defaulted mode parameter while retaining the old one-parameter spelling.
- Shared methods and vector conversions are mode-generic. Their snapshot paths
  and signatures therefore change from the old struct to `core_types`, with an
  additional `M` parameter and mode-preserving input/output types.
- Contextual constructors, owned conversions between modes, and public
  `_with_token` alternatives are new surface. They are real additions, not
  whitespace. Native `from_raw` is an additional feature-checked constructor.
- The constructor modes, shared core, and contextual aliases are public under
  `magetypes::simd::generic`; they support explicit type annotations and bounds.

The old aliases still accept token-taking calls. Tests cover legacy inference,
contextual construction, generic token helpers, and owned boundary conversions.
Generated layout assertions retain size/alignment equality with each backend
representation. The underlying type's Rust path has changed; diagnostic
`type_name` strings must not be assumed identical. Borrowed vectors and containers
cannot be converted between modes by the owned `From` implementations.

## What was not intended

`magetypes::prelude` used `pub use crate::simd::generic::*`. Adding modes and
modules to `generic` therefore exported five extra names through the prelude:
`ConstructorMode`, `Context`, `Explicit`, `core_types`, and `local`.
Neither prelude source file had been deliberately edited during the constructor
work. The unchanged glob was the cause of the expanded surface.

The corrected prelude uses a generator-produced list of vector names plus its
existing three helper traits and `SimdToken`. The generator derives the vector
list and `w512` guards from the same type inventory as the implementations.
The [audit data](../benchmarks/constructor_api_audit_2026-09-27.json) confirms the
original **34 top-level names**, with no missing or additional names, for x86,
ARM, and WASM. A compile regression imports conflicting downstream names through
a second glob, so accidental reintroduction becomes an ambiguity error.

## Why snapshots remain large

The snapshotter assigns canonical paths to re-exported definitions and annotates
other paths with `[also: ...]`. Removing `prelude::core_types` changes the chosen
canonical path of many methods back to `simd::generic::core_types`. Methods on a
type alias remain callable even when their snapshot line is listed under the
underlying type rather than `prelude`.

For the x86 default snapshot, the reported prelude lines changed from 2,994 at
the baseline to 4,947 before this correction, then 44 afterward. That is **not**
a count of methods added or removed: the original methods now appear under the
core's canonical path. Snapshot summary counts include re-export paths and
must not be treated as unique API counts or as a semver checker.

Reproduce the name audit with `just constructor-api-audit` (or its Python command).
This audit is not a complete semver proof against a published release. No
`cargo semver-checks` release check or publication was performed in this change.

## Validation

`use(...)` and legacy `define(...)` integration tests passed on native x86 and
AArch64 under QEMU. The removed option has a parser regression asserting its
migration diagnostic. WASM test compilation, i686 library compilation, and
x86 all-feature library Clippy passed. Broad WASM/i686 checks still emit warnings
in unchanged source outside this change; logs are retained at
`/home/lilith/data/archmage/use-review/2026-09-27/`.
