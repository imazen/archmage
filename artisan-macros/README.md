# artisan-macros

Auditable, convention-forward proc-macros for sound SIMD tier dispatch.

Two macros, one file, user-owned feature strings. No tier registry, no codegen step, no body rewriting. Read the crate top to bottom in a sitting, fork it if you want to.

Status: **design-phase skeleton.** See [DESIGN.md](./DESIGN.md) for the full sketch.

## The shape

```rust
use artisan_macros::{cpu_tier, chain};

#[cpu_tier(enable = "avx2,fma,bmi1,bmi2,f16c,lzcnt,popcnt,movbe")]
fn compute_v3(data: &[f32]) -> f32 { /* hand-written AVX2+FMA */ }

#[cpu_tier(enable = "sse4.2,popcnt")]
fn compute_v2(data: &[f32]) -> f32 { /* hand-written SSE4.2 */ }

#[cpu_tier(enable = "neon")]
fn compute_neon(data: &[f32]) -> f32 { /* hand-written NEON */ }

fn compute_scalar(data: &[f32]) -> f32 { data.iter().sum() }

#[chain(
    x86_64 = [
        compute_v3 = "avx2,fma,bmi1,bmi2,f16c,lzcnt,popcnt,movbe",
        compute_v2 = "sse4.2,popcnt",
    ],
    aarch64 = [
        compute_neon = "neon",
    ],
    default = compute_scalar,
)]
pub fn compute(data: &[f32]) -> f32 {}
```

Each arch chain is a trampoline sequence: `compute_v3`'s cache miss falls through to `compute_v2`'s cache miss falls through to `compute_scalar`. One `AtomicU8` per tier, tri-state (empty/false/true). The entry function is a compile-time arch switch — no per-call-site dispatch tree.

## What it is

- ~500 LoC proc-macro crate, single `lib.rs`
- Emits `#[target_feature]` + `#[cfg(target_arch)]` + `#[inline]` via `#[cpu_tier]`
- Emits a trampoline chain + atomic caches + entry dispatcher via `chain!`
- `#![forbid(unsafe_code)]`-compatible downstream (macro-emitted `unsafe` is exempt)
- Constant-CPUID axiom is the only soundness assumption beyond what Rust already requires

## What it is not

- Not a replacement for archmage's token system, `#[magetypes]`, `#[autoversion]`, or `incant!`
- Not a tier registry — users own their feature strings and suffixes
- Not a benchmarking or testing framework — test hooks are provided behind a feature flag, parity harness lives elsewhere

## License

MIT OR Apache-2.0
