# Version-gating newer-stable intrinsics above the MSRV

archmage's MSRV is **1.89**. When a hardware path needs a `core::arch` intrinsic
that stabilized *above* it — e.g. the aarch64 NEON-f16 converters `vcvt_f32_f16`
/ `vcvt_f16_f32`, stable since **1.94** (`stdarch_neon_f16`) — a static `cfg` on
the intrinsic would drag the whole crate's MSRV up to 1.94. Instead the path is
picked by **toolchain version** with `rustversion`, inside `#[cfg(target_arch =
…)]`: the HW kernel compiles on rustc ≥ X, a branchless software fallback below.
**No build script, no MSRV bump.**

| Intrinsic state | Mechanism |
|---|---|
| **Stable since a known version** (NEON-f16 @ 1.94) | `rustversion` + `target_arch` gate, both arms covered by normal CI |

```rust
#[cfg(target_arch = "aarch64")]
#[rustversion::since(1.94)]   // HW exists → runtime token decides HW vs software
fn neon_f16_decode_select(token: NeonToken, ..) { if fp16 { hw } else { software } }

#[cfg(target_arch = "aarch64")]
#[rustversion::before(1.94)]  // HW not compiled → software only, no MSRV bump
fn neon_f16_decode_select(token: NeonToken, ..) { software }
```

The `since`/`before` gate selects whether the HW impl *exists*; the runtime
token (`Arm64V2Token::summon()`, proving `fp16`) selects whether to *use* it.

Both arms are covered by the **normal** CI matrix: the MSRV 1.89 `cargo check`
compiles the `before` software arm below the bound, and the stable
`test-aarch64`/`test-cross` jobs compile **and run** the `since` HW arm above it.
The exact `1.94` literal is not independently re-validated — it's documented in
std, and a wrong bound surfaces at the next toolchain reaching the gap.

A **nightly-only** intrinsic (no stable version to name) would instead need a
try-compile `build.rs` probe gated on nightly; archmage ships none today.

## Adding the next stable intrinsic

1. Write the HW kernel inside `#[cfg(target_arch = "<arch>")]`, gated
   `#[rustversion::since(<ver>)]`.
2. Add the paired `*_select` helper: `#[rustversion::since(<ver>)]` (HW + runtime
   token) and `#[rustversion::before(<ver>)]` (software only).
3. The normal CI covers both arms; if `<ver>` is uncertain, add a transient
   2-cell `cross` matrix at `<ver> - 1` / `<ver>` to pin it, then remove it.
