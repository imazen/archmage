# magemath slice entries vs per-chunk calls

Host r7900x (Ryzen 9 7900X, Zen 4), rustc 1.99.1, no `-C target-cpu`, thin LTO.
Source: parent commit 1cb042ba3741 plus this package. One process run, exploratory
(zenbench `inference=exploratory`); not measured on Zen 5, Intel or ARM.

    cargo bench --bench slices

BitExact MIDP presets (pow Gamma n = 2.4, exp2, log2). Mean time per whole slice:

| function | len | array8 per chunk | caller `#[attune]` kernel | slice entry |
|---|---|---|---|---|
| pow  | 64   | 170 ns  | 39 ns   | 37 ns   |
| exp2 | 64   | 179 ns  | 21 ns   | 20 ns   |
| log2 | 64   | 175 ns  | 15 ns   | 14 ns   |
| pow  | 4096 | 10.8 µs | 2.29 µs | 2.32 µs |
| exp2 | 4096 | 11.2 µs | 1.33 µs | 1.36 µs |
| log2 | 4096 | 11.1 µs | 0.92 µs | 0.94 µs |

- `array8`: `try_*_array_with::<_, 8>` per chunk, Automatic execution each call.
- `attune`: the README pattern, vector entries with Caller execution.
- `slice`: `try_*_slice_with`.

The per-chunk array calls take 4.6x to 12x as long as the slice entry. The slice
entry and a hand-written kernel are within 3% of each other.

---

# 16-lane AVX-512 slice widening and option granularity (`2026-11-17`, Zen 5)

Host `dev` (Ryzen 9 9950X3D, Zen 5, single core pinned to CCD1 core 8), source `d538e9bd06f2`
(`feat/0.9.32-magemath-profiles`), no `-C target-cpu`, `magemath` `avx512` enabled
(`Execution::Upgrade` selects 16-lane AVX-512 `zmm` loops), `n = 30` (`CV = 0.2%–1.4%`),
run via `run-heavy --mem 24G`.

## `slices` (`len = 4096`, inputs uniform in `[0.01, 3.91)`, `run-heavy` peak-RSS `0.14 GiB`)

| function | `array8` per chunk | caller 8-lane `#[attune]` | 16-lane `slice` entry | speedup vs `array8` | speedup vs 8-lane `attune` |
|---|---|---|---|---|---|
| `pow` (`gamma`, `n = 2.4`) | `10.35 µs` | `2.18 µs` | `927.8 ns` | `11.2x` | `2.35x` |
| `exp2` | `9.96 µs` | `1.00 µs` | `519.2 ns` | `19.2x` | `1.93x` |
| `log2` | `1.33 µs` | `735.7 ns` | `497.0 ns` | `2.68x` | `1.48x` |

## `options_granularity` (`len = 4096`, `run-heavy` peak-RSS `0.59 GiB`)

```
taskset -c 8 cargo bench --bench options_granularity -- --cpu-policy=unrestricted
```

Sampled benchmark inputs (all strictly valid across every configuration in their group):
- `log2_4096`: `near_one` (`4096` normals in `[0.75, 1.25)`)
- `exp2_4096`: `small_sym` (`4096` finite values in `[-0.45, 0.45)`, including `0.0` at index `2048`)
- `cbrt_4096` and `powf_4096`: `unit_pos` (`4096` normals in `[0.01, 0.99)`)

### `log2_4096`
| configuration | mean ± std | vs `default_normals` | profile domain / notes |
|---|---|---|---|
| `default_normals` (`LogfOptions::new()`) | `492.8 ns ± 1.8 ns` | `1.00x` | checked positive normals (`[MIN_POSITIVE, MAX]`) |
| `admit_zero` (`.admit_zero()`) | `534.3 ns ± 2.9 ns` | `+8.4%` | adds off-critical-path `±0 -> -inf` blend |
| `admit_subnormals` (`.admit_subnormals()`) | `647.8 ns ± 3.0 ns` | `+31.5%` | `* 2^23` pre-scale + blend on critical latency path before division + range check |
| `admit_all_floats` (`.admit_all_floats()`) | `572.2 ns ± 2.9 ns` | `+16.1%` | subnormal scale + IEEE blends (`+inf`, `<0`, `NaN`), infallible (no `any_true()` branch) |
| `near_one_range` (`.near_one_range()`) | `381.1 ns ± 0.9 ns` | `-22.7%` (`1.29x` faster) | `[0.70710677, 1.4142134]` single-octave zero-reduction (`e = 0`), checked, bit-exact |
| `near_one_assume` (`.near_one_range().assume_in_bounds()`) | `231.6 ns ± 0.9 ns` | `-53.0%` (`2.13x` faster) | zero-reduction + unchecked |

### `exp2_4096`
| configuration | mean ± std | vs `default_finite` | profile domain / notes |
|---|---|---|---|
| `default_finite` (`ExpfOptions::new()`) | `532.5 ns ± 0.8 ns` | `1.00x` | checked finite inputs, internal `[-126, 128]` output clamp |
| `range_at_127_5_clamped` (`.input_range(-126.0, 127.5)`) | `667.6 ns ± 1.5 ns` | `+25.4%` (`+64.3%` vs unclamped) | `127.5` upper endpoint retains output clamp so `round_ties_even(127.5) = 128` stays finite |
| `unclamped_exp2_range` (`.unclamped_exp2_range()`) | `406.4 ns ± 1.9 ns` | `-23.7%` (`1.31x` faster) | `[-126.0, 127.0]`, elides internal `[-126, 128]` clamp |
| `small_exp2_range` (`.small_exp2_range()`) | `309.3 ns ± 4.6 ns` | `-41.9%` (`1.72x` faster) | `[-0.5, 0.5]` single-octave zero-reduction (`n = 0`), checked, bit-exact (`<= 0.78 ULP`) |
| `small_exp2_clamp_oob` (`.small_exp2_range().clamp_out_of_bounds()`) | `329.9 ns ± 4.7 ns` | `-38.0%` (`+6.7%` vs checked) | infallible input clamp + `NaN` propagation |
| `small_exp2_assume` (`.small_exp2_range().assume_in_bounds()`) | `191.8 ns ± 5.1 ns` | `-64.0%` (`2.78x` faster) | zero-reduction + unchecked |

### `cbrt_4096`
| configuration | mean ± std | vs `default_bounded` | profile domain / notes |
|---|---|---|---|
| `default_bounded` (`CbrtfOptions::new()`) | `791.0 ns ± 3.2 ns` | `1.00x` | checked `±0` and normals strictly below `f32::MAX / 3` (rejects nonzero subnormals; elides denorm scale) |
| `strict_normal_range` (`.input_range(0.01, 4.0)`) | `806.8 ns ± 3.4 ns` | `+2.0%` | checked `[0.01, 4.0]` normal sub-interval, elides denorm scale |
| `subnormals_assume_in_bounds` (`.admit_subnormals().assume_in_bounds()`) | `1.04 µs ± 2.1 ns` | `+31.5%` (`15%` faster than full) | `* 2^24` / `* 2^-8` subnormal normalization without `>= 1e36` / `±inf` / `NaN` path |
| `unit_range_subnormals` (`.unit_range()`) | `1.13 µs ± 1.9 ns` | `+42.9%` | checked `[-1.0, 1.0]` including subnormals |
| `admit_all_floats` (`.admit_all_floats()`) | `1.22 µs ± 3.1 ns` | `+54.2%` | full IEEE `f32` domain |

### `powf_4096` (`n = 1.0 / 2.4`)
| configuration | mean ± std | vs `gamma_error_on_oob` | profile domain / notes |
|---|---|---|---|
| `gamma_assume_in_bounds` (`.gamma_domain().assume_in_bounds()`) | `753.5 ns ± 2.4 ns` | `-21.1%` | unchecked `{+0.0} ∪ [2^-16, 2^16]` (exp `[0.0, 4.0]`), elides internal `y*log2(x)` clamp |
| `srgb_encode_domain` (`.srgb_encode_domain()`) | `947.7 ns ± 4.0 ns` | `-0.8%` (`1.53x` faster than unranged) | checked `{+0.0} ∪ [MIN_POSITIVE, 1.0]` (exp `[0.0, 0.99]`); sign-aware 4-corner range elision |
| `gamma_error_on_oob` (`.gamma_domain()`) | `955.5 ns ± 2.2 ns` | `1.00x` | checked `{+0.0} ∪ [2^-16, 2^16]` (exp `[0.0, 4.0]`), elides internal `y*log2(x)` clamp |
| `gamma_clamp_oob` (`.gamma_domain().clamp_out_of_bounds()`) | `1.13 µs ± 2.8 ns` | `+18.3%` | infallible input clamp into `{+0.0} ∪ [2^-16, 2^16]` + `NaN` propagation, still elides internal `y*log2(x)` clamp |
| `unranged_admit_zero` (`.admit_zero_base()`) | `1.45 µs ± 4.3 ns` | `+51.8%` | checked positive normals + `+0.0` (for `n >= 0`), retains internal `y*log2(x)` clamp |
| `unranged_zero_and_subnormals` (`.admit_zero_base().admit_subnormals()`) | `1.64 µs ± 6.7 ns` | `+71.6%` (`+13.1%` vs zero-only) | adds subnormal base normalization (`{+0.0} ∪ [f32::from_bits(1), MAX]`) |

---

# Discrete `*Domain` enums, opt-in `Subnormals`, `const_exponent`, and `Log1pf` (`2026-11-18`, Zen 5)

Host `dev` (Ryzen 9 9950X3D, Zen 5, single core pinned to CCD1 core 8 via `taskset -c 8`),
no `-C target-cpu`, `magemath` `avx512` enabled (`Execution::Upgrade` 16-lane AVX-512 `zmm`
slice loops), `len = 4096`, run via `run-heavy --mem 24G` (`peak-RSS 0.43 GiB`, `min-avail 39391 MiB`).

### `powf_4096` (`const_exponent` algebraic & general fast-paths + `Subnormals::FlushOrClamp`)
| configuration | mean ± mad | vs `gamma_error_on_oob` | domain / notes |
|---|---|---|---|
| `unit_const_exp_0_5_sqrt` (`PowfDomain::Unit + const_exponent(0.5)`) | `251.8 ns ± 1 ns` | `-69.7%` (`3.30x` faster) | checked `{+0.0} ∪ [MIN_POSITIVE, 1.0]`, IEEE hardware `vsqrtps` fast-path |
| `unit_const_exp_2_0_mul` (`PowfDomain::Unit + const_exponent(2.0)`) | `365.6 ns ± 2 ns` | `-56.0%` (`2.27x` faster) | checked `{+0.0} ∪ [MIN_POSITIVE, 1.0]`, IEEE hardware `vmulps` fast-path |
| `unit_const_exp_inv_24_assume` (`PowfDomain::Unit + const_exponent(1.0 / 2.4) + assume_in_bounds()`) | `666.5 ns ± 2 ns` | `-19.7%` | unchecked `{+0.0} ∪ [MIN_POSITIVE, 1.0]`, constant `1/2.4` (`0` exponent prep + range check elided) |
| `gamma_assume_in_bounds` (`PowfDomain::HdrGamma + assume_in_bounds()`) | `686.7 ns ± 2 ns` | `-17.3%` | unchecked `{+0.0} ∪ [2^-16, 2^16]` (runtime `n = 1/2.4`, exp `[0.0, 4.0]`) |
| `unit_const_exp_inv_24` (`PowfDomain::Unit + const_exponent(1.0 / 2.4)`) | `827.1 ns ± 3 ns` | `-0.4%` (`1.60x` faster than unranged) | checked `{+0.0} ∪ [MIN_POSITIVE, 1.0]`, compile-time `1/2.4 * [-126, 1]` range-check elision |
| `gamma_error_on_oob` (`PowfDomain::HdrGamma`) | `830.4 ns ± 4 ns` | `1.00x` | checked `{+0.0} ∪ [2^-16, 2^16]` (runtime `n = 1/2.4`, exp `[0.0, 4.0]`) |
| `srgb_encode_runtime_exp` (`PowfDomain::UnitEncode`) | `831.5 ns ± 3 ns` | `+0.1%` | checked `{+0.0} ∪ [MIN_POSITIVE, 1.0]` (runtime `n = 1/2.4`, exp `[0.0, 0.99]`) |
| `gamma_flush_subnormals` (`PowfDomain::HdrGamma + flush_or_clamp_subnormals()`) | `1.00 µs ± 9 ns` | `+20.7%` | shoves nonzero subnormals to `+0.0` while still rejecting `< 0`, `> 2^16`, `±inf`, `NaN` |
| `gamma_clamp_oob` (`PowfDomain::HdrGamma + clamp_out_of_bounds()`) | `1.02 µs ± 4 ns` | `+23.1%` | infallible input clamp into `{+0.0} ∪ [2^-16, 2^16]` + `NaN` propagation |
| `unranged_admit_zero` (`PowfDomain::NonNegative`) | `1.33 µs ± 4 ns` | `+59.6%` | checked positive normals + `+0.0` (`n >= 0`), retains internal `y*log2(x)` clamp |
| `unranged_zero_and_subnormals` (`PowfDomain::NonNegative + admit_subnormals()`) | `1.52 µs ± 2 ns` | `+82.6%` | adds `* 2^23` subnormal base normalization |

### `cbrt_4096` (opt-in `Subnormals` on `CbrtfDomain::SignedUnit` vs `FlushOrClamp` vs `Normalize`)
| configuration | mean ± mad | vs `default_bounded` | domain / notes |
|---|---|---|---|
| `signed_unit_assume_in_bounds` (`CbrtfDomain::SignedUnit + assume_in_bounds()`) | `640.3 ns ± 5 ns` | `-9.9%` (`1.57x` faster than subnorm) | unchecked `[-1.0, 1.0]`, elides both range check and subnormal rescaling |
| `strict_normal_range` (`.input_range(0.01, 4.0)`) | `688.6 ns ± 5 ns` | `-3.1%` | checked `[0.01, 4.0]` `bf16`-bounded normal interval, elides subnormal check |
| `default_bounded` (`CbrtfDomain::Bounded`) | `710.4 ns ± 4 ns` | `1.00x` | checked `±0.0` and normal `|x| < f32::MAX / 3` (`Subnormals::Reject` default) |
| `signed_unit_reject_subnormals` (`CbrtfDomain::SignedUnit` / `.unit_range()`) | `796.8 ns ± 2 ns` | `+12.2%` (`-20.7%` vs `admit_subnormals`) | checked `[-1.0, 1.0]` with opt-in `Subnormals::Reject` default (no `* 2^24` rescale) |
| `signed_unit_flush_subnormals` (`CbrtfDomain::SignedUnit + flush_or_clamp_subnormals()`) | `969.9 ns ± 2 ns` | `+36.5%` | shoves `(-MIN_POSITIVE, +MIN_POSITIVE)` subnormals to `±0.0`, rejects `|x| > 1` / `NaN` |
| `signed_unit_admit_subnormals` (`CbrtfDomain::SignedUnit + admit_subnormals()`) | `1.01 µs ± 2 ns` | `+41.5%` | checked `[-1.0, 1.0]` with exact `* 2^24` / `* 2^-8` subnormal normalization |
| `admit_all_floats` (`CbrtfDomain::AllFloats`) | `1.10 µs ± 2 ns` | `+54.8%` | full IEEE `f32` domain |

### `log1p_4096` (dedicated `magemath::log1pf::Log1pf`)
| configuration | mean ± mad | vs `default_gt_neg_one` | domain / notes |
|---|---|---|---|
| `around_zero_assume` (`Log1pfDomain::AroundZero + assume_in_bounds()`) | `347.5 ns ± 1 ns` | `-59.1%` (`2.45x` faster) | `[-0.25, 0.25]`, elides exponent extraction + `1+x >= 2` Kahan branch + checks |
| `around_zero_checked` (`Log1pfDomain::AroundZero`) | `533.5 ns ± 2 ns` | `-37.2%` (`1.59x` faster) | checked `[-0.25, 0.25]`, elides exponent extraction + `1+x >= 2` Kahan branch |
| `default_gt_neg_one` (`Log1pfDomain::GreaterThanNegOne`) | `849.7 ns ± 4 ns` | `1.00x` | checked `(-1.0, MAX]` (`Subnormals::Reject`) |

### `log2_4096` & `exp2_4096` (with `bf16` bounds & `Subnormals::FlushOrClamp`)
| group / configuration | mean ± mad | vs group default | notes |
|---|---|---|---|
| `log2`: `near_one_assume` (`LogfDomain::SingleOctave + assume_in_bounds()`) | `217.1 ns ± 1 ns` | `-49.4%` (`1.98x` faster) | single-octave zero-reduction + unchecked |
| `log2`: `near_one_range` (`LogfDomain::SingleOctave`) | `318.2 ns ± 2 ns` | `-25.9%` (`1.35x` faster) | checked `[0.70710677, 1.4142134]` single-octave zero-reduction |
| `log2`: `default_normals` (`LogfDomain::PositiveNormals`) | `429.2 ns ± 5 ns` | `1.00x` | checked `[MIN_POSITIVE, MAX]` |
| `log2`: `admit_zero` (`LogfDomain::NonNegative`) | `481.6 ns ± 2 ns` | `+12.2%` | `±0.0 -> -inf` |
| `log2`: `admit_subnormals` (`.admit_subnormals()`) | `575.2 ns ± 3 ns` | `+34.0%` | `* 2^23` subnormal normalization |
| `log2`: `zero_flush_subnormals` (`NonNegative + flush_or_clamp_subnormals()`) | `579.9 ns ± 2 ns` | `+35.1%` | shoves `(0, MIN_POSITIVE)` subnormals to `0.0 -> -inf` under `Checked` |
| `exp2`: `small_exp2_assume` (`ExpfDomain::SingleOctave + assume_in_bounds()`) | `181.0 ns ± 2 ns` | `-61.0%` (`2.56x` faster) | `[-0.5, 0.5]` zero-reduction + unchecked |
| `exp2`: `small_exp2_range` (`ExpfDomain::SingleOctave`) | `257.4 ns ± 1 ns` | `-44.6%` (`1.80x` faster) | checked `[-0.5, 0.5]` zero-reduction |
| `exp2`: `unclamped_exp2_range` (`ExpfDomain::NormalOutput`) | `343.2 ns ± 1 ns` | `-26.1%` (`1.35x` faster) | checked `[-126.0, 127.0]`, elides internal `[-126, 128]` clamp |
| `exp2`: `default_finite` (`ExpfDomain::Finite`) | `464.3 ns ± 3 ns` | `1.00x` | checked finite inputs |
| `exp2`: `range_at_127_5_clamped` (`.input_range(-126.0, 127.5)`) | `570.3 ns ± 3 ns` | `+22.8%` | `bf16`-exact `127.5` upper bound keeps pre-`round()` clamp active |
