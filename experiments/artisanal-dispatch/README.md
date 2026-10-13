# Artisanal dispatch prototype

Self-contained SIMD proofs and explicit dispatch, independent of Archmage. This is
an unpublished experiment, not a replacement or a migration requirement.
It lives on Archmage main in a separate Cargo workspace with `publish = false`;
the released crates do not depend on it. Run `just artisanal-check` from the
repository root for its native validation.

The implementation is **422 formatted Rust lines**, including the tier table,
proof construction, macros, comments and blank lines: eight pages at 55 lines per
page. See [the audit](AUDIT.md) for the safety argument, dependencies and limits.
Tests and generated expansions are additional material.

Three primitives, all for **free functions**:

```rust
use artisanal_dispatch::{bridge, dispatch, tier};

#[tier(v3)]
fn helper_v3(x: u32) -> u32 { x + 1 }

#[bridge(v3)]
fn work_v3(x: u32) -> u32 { helper_v3(x) }

#[bridge(scalar)]
fn work_scalar(x: u32) -> u32 { x + 1 }

let answer = dispatch!(work(41); v3, scalar);
assert_eq!(answer, 42);
```

* `#[tier(v3)]` gives one function its feature context and architecture gate.
* `#[bridge(v3)]` also generates `work_v3_t(V3, x)`, with the same visibility.
* `dispatch!` tries the explicit list in order, calling the `_t` siblings. A scalar
  fallback is required. Enable this crate's `std` feature for native CPU detection;
  the default `no_std` mode uses build-time guarantees only.

Inside a feature context, call the direct function. Outside, use dispatch or an
existing proof: `work_v3_t(token, x)`. `V3::from_context()` constructs a proof in a
feature context checked by rustc. Stronger proofs convert to weaker ones through
`From`; they cannot upgrade. There is no implicit probing inside a bridge.

Methods remain ordinary Rust and can call these free kernels. Receivers, associated
functions and trait methods are rejected. Generic types, const parameters,
lifetimes, where clauses, nested free functions and `impl Trait` are supported.

Bodies default to `#[inline]`; `#[inline(never)]` is accepted. Proof bridges use
`#[inline(always)]`. Both carry the same user `cfg` and visibility. There is no
implicit tier selection, body cloning, caller rewriting or surrounding-impl parsing.

## Inspect and reproduce

* [Macro implementation](macros/src/lib.rs)
* [Complete tier registry](macros/src/tiers.rs)
* [Runtime entry point](src/lib.rs)
* [Expanded example](artifacts/codegen-expanded.rs)
* [Expanded x86 proofs](artifacts/proofs-expanded.rs)
* [Assembly excerpts](artifacts/codegen.txt)

`just check` runs the checks listed in [AUDIT.md](AUDIT.md). It requires an x86-64
V3 host, Rust 1.99 and 1.89, the listed cross targets, QEMU, the AArch64 GNU linker,
`cargo expand`, and a `WASMTIME` environment variable pointing to Wasmtime.
`just ui`, `just codegen`, and `just count` expose the individual checks. Run builds
under the resource controls appropriate for the machine.
