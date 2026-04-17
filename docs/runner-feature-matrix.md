# Runner Feature Matrix (CI-observed)

Captured from CI run [`24479154394`](https://github.com/imazen/archmage/actions/runs/24479154394) on commit `ce5169b` (2026-04-15). Source artifacts: `cpu-survey-test-aarch64-*` from the `test-aarch64` job.

This is **observed** behavior — what `is_aarch64_feature_detected!` and `IsProcessorFeaturePresent` actually return on each runner — not what the silicon physically supports. The Windows column under-reports its hardware (see "Detection gaps" below).

## AArch64 features

| Feature | `ubuntu-24.04-arm`<br/>Neoverse N2 | `macos-14`<br/>Apple M1 (Virtual) | `macos-latest`<br/>Apple M1 (Virtual) | `windows-11-arm`<br/>Neoverse N2 |
|---|:---:|:---:|:---:|:---:|
| `neon`         | ✅ | ✅ | ✅ | ✅ |
| `crc`          | ✅ | ✅ | ✅ | ✅ |
| `rdm`          | ✅ | ✅ | ✅ | **❌** |
| `dotprod`      | ✅ | ✅ | ✅ | ✅ |
| `fp16`         | ✅ | ✅ | ✅ | **❌** |
| `fhm`          | ✅ | ✅ | ✅ | **❌** |
| `fcma`         | ✅ | ✅ | ✅ | **❌** |
| `bf16`         | ✅ | ❌ | ❌ | **❌** |
| `i8mm`         | ✅ | ❌ | ❌ | **❌** |
| `jsconv`       | ✅ | ✅ | ✅ | ✅ |
| `frintts`      | ✅ | ✅ | ✅ | **❌** |
| `aes`          | ✅ | ✅ | ✅ | ✅ |
| `sha2`         | ✅ | ✅ | ✅ | ✅ |
| `sha3`         | ✅ | ✅ | ✅ | **❌** |
| `sm4`          | ✅ | ❌ | ❌ | ❌ |
| `lse`          | ✅ | ✅ | ✅ | ✅ |
| `rcpc`         | ✅ | ✅ | ✅ | ✅ |
| `rcpc2`        | ✅ | ✅ | ✅ | **❌** |
| `sve`          | ✅ | ❌ | ❌ | ✅ |
| `sve2`         | ✅ | ❌ | ❌ | ✅ |
| `sve2-aes`     | ❌ | ❌ | ❌ | ❌ |
| `sve2-bitperm` | ✅ | ❌ | ❌ | ✅ |
| `sve2-sha3`    | ✅ | ❌ | ❌ | ✅ |
| `sve2-sm4`     | ✅ | ❌ | ❌ | ✅ |
| `f32mm`        | ❌ | ❌ | ❌ | ❌ |
| `f64mm`        | ❌ | ❌ | ❌ | ❌ |
| `dit`          | ❌ | ✅ | ✅ | ❌ |
| `sb`           | ✅ | ✅ | ✅ | ❌ |
| `ssbs`         | ❌ | ✅ | ✅ | ❌ |
| `paca`         | ✅ | ✅ | ✅ | **❌** |
| `pacg`         | ✅ | ✅ | ✅ | **❌** |
| `dpb`          | ✅ | ✅ | ✅ | **❌** |
| `dpb2`         | ✅ | ✅ | ✅ | **❌** |
| `flagm`        | ✅ | ✅ | ✅ | **❌** |
| `rand`         | ❌ | ❌ | ❌ | ❌ |
| `bti`, `mte`, `tme` | ❌ | ❌ | ❌ | ❌ |

**Bold ❌** = present in hardware, **not detected by the OS** (false negative).

## Token availability (current archmage tokens)

| Token | Linux ARM | macOS 14 | macOS latest | Windows ARM |
|---|:---:|:---:|:---:|:---:|
| `NeonToken`        | ✅ | ✅ | ✅ | ✅ |
| `NeonAesToken`     | ✅ | ✅ | ✅ | ✅ |
| `NeonSha3Token`    | ✅ | ✅ | ✅ | **❌** |
| `NeonCrcToken`     | ✅ | ✅ | ✅ | ✅ |
| `Arm64V2Token`     | ✅ | ✅ | ✅ | **❌** |
| `Arm64V3Token`     | ✅ | ❌ | ❌ | **❌** |

## Detection gaps to fix

The Windows ARM runner uses **the same silicon** as the Linux ARM runner (Neoverse N2 / Cobalt 100), but `is_aarch64_feature_detected!` returns `false` for 14 features that physically exist. Root cause: stdarch's Windows backend at `library/std_detect/src/detect/os/windows/aarch64.rs` only probes features that have a corresponding `PF_ARM_*` constant in `IsProcessorFeaturePresent` — and Microsoft only exposes ~17 ARM features, while Linux's `HWCAP`/`HWCAP2` covers ~50.

### Where this hurts archmage tokens today

| Token | False-negative on Windows? | Why | Fix |
|---|---|---|---|
| `NeonSha3Token` | yes | requires `sha3`, Windows doesn't probe it | platform baseline override OR MIDR check |
| `Arm64V2Token`  | yes | requires (likely) `rdm`+`fp16`+`dotprod`, Windows misses `rdm` and `fp16` | platform baseline override (Win11 on ARM ⇒ ARMv8.2-A floor ⇒ all three guaranteed) |
| `Arm64V3Token`  | yes (separately also false on Apple M1) | requires `bf16`+`i8mm`, Windows reports both as `false`. Cobalt N2 actually has them. | requires per-CPU MIDR detection on Windows; not OS-baseline guaranteed |

### Recommended Windows ARM overrides

Windows 11 on ARM officially requires Snapdragon 850 / 8cx-class or later (all Cortex-A76-derived, ARMv8.2-A). That baseline guarantees:

- `rdm` (mandatory in ARMv8.1-A)
- `lse` (mandatory in ARMv8.1-A) — already detected via `PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE`
- `fp16`, `dotprod` (commonly mandatory in ARMv8.2-A profiles, but technically optional in ARM ARM — verify per-CPU before assuming)
- `paca`, `pacg` (FEAT_PAuth, mandatory in ARMv8.3-A) — verify ARMv8.3 floor before assuming

Conservative override that's safe:

```rust
#[cfg(all(target_arch = "aarch64", target_os = "windows"))]
fn rdm_present() -> bool { true }   // Win11-on-ARM mandates ARMv8.1+
#[cfg(all(target_arch = "aarch64", not(target_os = "windows")))]
fn rdm_present() -> bool { std::arch::is_aarch64_feature_detected!("rdm") }
```

Apply the same pattern for any feature mandated by the Win11-on-ARM hardware floor, but **do not** override features that are merely "common but optional" in the baseline (e.g. `bf16`, `i8mm`) — those need real per-CPU detection (MIDR or capability probing).

## Token gaps — what we don't have yet

### 1. SVE tokens (Linux ARM + Windows ARM both have SVE2)

CI confirms `sve` and `sve2` light up on both Cobalt 100 runners. archmage has no token for them. Blockers (Rust nightly, `repr(scalable)` not yet accepted, SVE types can't cross trait boundaries) keep this from being a stable token tier today, but a research-preview token gated behind a `nightly-sve` cargo feature is feasible.

Suggested tiers when the type system catches up:

- `Aarch64SveToken` — `sve` only (Neoverse V1, Graviton 3, Fujitsu A64FX)
- `Aarch64Sve2Token` — `sve` + `sve2` (Neoverse N2/V2, Cobalt 100, Graviton 4)
- `Aarch64Sve2BitpermToken` — adds bitperm/sha3/sm4 (separately observable in CI)

### 2. Apple Silicon platform token

M1 has a deterministic feature set: every shipping M1 chip is identical. Runtime detection on macOS aarch64 is wasted work. An `AppleSiliconM1Token` (or just promoting Apple-detected runs to `Arm64V2Token` via compile-time `cfg(target_os = "macos", target_arch = "aarch64")`) would let M1 hit the v2 path without the `is_aarch64_feature_detected!` overhead.

M2+ adds `bf16` and `i8mm` — would need a separate `AppleSiliconM2Token` or runtime detection of those two features specifically.

### 3. `Arm64V3Token` on Windows ARM

Cobalt 100 has `bf16` and `i8mm` in hardware. Windows reports both as `false`. This is the highest-impact false negative — Windows ARM CI runs use top-of-line hardware but get the v1 codepath. Options:

- Read `MIDR_EL1` to identify Neoverse N2 specifically, override to `true` (requires kernel cooperation; Windows may not allow EL0 access)
- Capability-probe at startup: try executing a `bfdot` instruction inside a SEH/`__try` block, see if it faults — works but adds startup cost and is brittle
- Document the gap and accept v1 codepath on Windows ARM until Microsoft adds `PF_ARM_V82_BF16_INSTRUCTIONS_AVAILABLE`

### 4. `f32mm` / `f64mm` (SVE matrix multiply)

Not on any current GitHub runner. Would light up on Graviton 4 (Neoverse V2) or Apple M4 SME. Defer until runner fleet has the hardware.

### 5. macOS-only features that we don't tokenize

`dit` and `ssbs` light up on macOS but not Linux/Windows. These are security/timing-side-channel features, not compute — no archmage compute kernel needs them.

## Notes

- `macos-latest` and `macos-14` are **the same Apple M1 Virtual** silicon. There is currently no GitHub-hosted macOS runner with M2+. To exercise `bf16`/`i8mm` on Apple Silicon, self-hosted runners are needed.
- The Windows runner reports `cpu: ARM Neoverse N2 (Cobalt 100)` from the example's CPU identification, but that comes from MIDR/CPUID-style probing — not from `IsProcessorFeaturePresent`. The detection gap is purely in the OS's per-feature query API.
- Linux `ubuntu-24.04-arm` is the most feature-complete CI target. Use it for the broadest aarch64 coverage; treat Windows ARM as an x-platform smoke test that can't validate v2/v3-tier dispatch correctly today.
