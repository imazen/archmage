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

Host `dev` (Ryzen 9 9950X3D, Zen 5, single core pinned to CCD1 core 8), no `-C target-cpu`,
`magemath` `avx512` enabled (`Execution::Upgrade` selects 16-lane AVX-512 `zmm` loops),
`n = 30` (`CV = 0.2%–1.4%`), run via `run-heavy --mem 24G`.

## `slices` (`len = 4096`, `run-heavy` peak-RSS `0.14 GiB`)

| function | `array8` per chunk | caller 8-lane `#[attune]` | 16-lane `slice` entry | speedup vs `array8` | speedup vs 8-lane `attune` |
|---|---|---|---|---|---|
| `pow` (`gamma`, `n = 2.4`) | `10.35 µs` | `2.18 µs` | `927.8 ns` | `11.2x` | `2.35x` |
| `exp2` | `9.96 µs` | `1.00 µs` | `519.2 ns` | `19.2x` | `1.93x` |
| `log2` | `1.33 µs` | `735.7 ns` | `497.0 ns` | `2.68x` | `1.48x` |

## `options_granularity` (`len = 4096`, `run-heavy` peak-RSS `0.59 GiB`)

```
taskset -c 8 cargo bench --bench options_granularity -- --cpu-policy=unrestricted
```

### `log2_4096`
| configuration | mean ± std | vs `default_normals` | notes |
|---|---|---|---|
| `default_normals` (`LogfOptions::new()`) | `492.8 ns ± 1.8 ns` | `1.00x` | positive normal inputs |
| `admit_zero` (`.admit_zero()`) | `534.3 ns ± 2.9 ns` | `+8.4%` | off-critical-path `±0 -> -inf` blend |
| `admit_subnormals` (`.admit_subnormals()`) | `647.8 ns ± 3.0 ns` | `+31.5%` | `* 2^23` pre-scale + blend on critical latency path before division + range check |
| `admit_all_floats` (`.admit_all_floats()`) | `572.2 ns ± 2.9 ns` | `+16.1%` | subnormal scale + IEEE blends (`+inf`, `<0`, `NaN`), infallible (no `any_true()` branch) |
| `near_one_range` (`.near_one_range()`) | `381.1 ns ± 0.9 ns` | `-22.7%` (`1.29x` faster) | `[0.70710677, 1.4142134]` single-octave zero-reduction (`e = 0`), checked, bit-exact |
| `near_one_assume` (`.near_one_range().assume_in_bounds()`) | `231.6 ns ± 0.9 ns` | `-53.0%` (`2.13x` faster) | zero-reduction + unchecked |

### `exp2_4096`
| configuration | mean ± std | vs `default_finite` | notes |
|---|---|---|---|
| `default_finite` (`ExpfOptions::new()`) | `532.5 ns ± 0.8 ns` | `1.00x` | finite inputs, output clamped at `[-126, 128]` |
| `range_at_127_5_clamped` (`.input_range(-126.0, 127.5)`) | `667.6 ns ± 1.5 ns` | `+25.4%` (`+64.3%` vs unclamped) | `127.5` upper endpoint retains output clamp so `round_ties_even(127.5) = 128` stays finite |
| `unclamped_exp2_range` (`.unclamped_exp2_range()`) | `406.4 ns ± 1.9 ns` | `-23.7%` (`1.31x` faster) | `[-126.0, 127.0]`, elides internal `[-126, 128]` clamp |
| `small_exp2_range` (`.small_exp2_range()`) | `309.3 ns ± 4.6 ns` | `-41.9%` (`1.72x` faster) | `[-0.5, 0.5]` single-octave zero-reduction (`n = 0`), checked, bit-exact (`<= 0.78 ULP`) |
| `small_exp2_clamp_oob` (`.small_exp2_range().clamp_out_of_bounds()`) | `329.9 ns ± 4.7 ns` | `-38.0%` (`+6.7%` vs checked) | infallible input clamp + `NaN` propagation |
| `small_exp2_assume` (`.small_exp2_range().assume_in_bounds()`) | `191.8 ns ± 5.1 ns` | `-64.0%` (`2.78x` faster) | zero-reduction + unchecked |

### `cbrt_4096`
| configuration | mean ± std | vs `default_bounded` | notes |
|---|---|---|---|
| `default_bounded` (`CbrtfOptions::new()`) | `791.0 ns ± 3.2 ns` | `1.00x` | bounded normal inputs (`\|x\| < 1e38`, subnormals flush to `±0.0`) |
| `strict_normal_range` (`.input_range(0.01, 4.0)`) | `806.8 ns ± 3.4 ns` | `+2.0%` | checked normal sub-interval, elides denorm scale |
| `subnormals_assume_in_bounds` (`.admit_subnormals().assume_in_bounds()`) | `1.04 µs ± 2.1 ns` | `+31.5%` (`15%` faster than full) | `* 2^24` / `* 2^-8` subnormal normalization without `>= 1e36` / `±inf` / `NaN` path |
| `unit_range_subnormals` (`.unit_range()`) | `1.13 µs ± 1.9 ns` | `+42.9%` | checked `[-1.0, 1.0]` including subnormals |
| `admit_all_floats` (`.admit_all_floats()`) | `1.22 µs ± 3.1 ns` | `+54.2%` | full IEEE `f32` domain |

### `powf_4096` (`n = 1.0 / 2.4`)
| configuration | mean ± std | vs `gamma_error_on_oob` | notes |
|---|---|---|---|
| `gamma_assume_in_bounds` (`.gamma_domain().assume_in_bounds()`) | `753.5 ns ± 2.4 ns` | `-21.1%` | unchecked `[0.01, 4.0]` |
| `srgb_encode_domain` (`.srgb_encode_domain()`) | `947.7 ns ± 4.0 ns` | `-0.8%` (`1.53x` faster than unranged) | `[MIN_POSITIVE, 1.0]` with `exp [0.0, 0.99]`; sign-aware 4-corner range elision |
| `gamma_error_on_oob` (`.gamma_domain()`) | `955.5 ns ± 2.2 ns` | `1.00x` | checked `[0.01, 4.0]`, elides internal `y*log2(x)` clamp |
| `gamma_clamp_oob` (`.gamma_domain().clamp_out_of_bounds()`) | `1.13 µs ± 2.8 ns` | `+18.3%` | infallible input clamp + `NaN` propagation, still elides internal `y*log2(x)` clamp |
| `unranged_admit_zero` (`.admit_zero_base()`) | `1.45 µs ± 4.3 ns` | `+51.8%` | unranged positive normals + `±0.0` |
| `unranged_zero_and_subnormals` (`.admit_zero_base().admit_subnormals()`) | `1.64 µs ± 6.7 ns` | `+71.6%` (`+13.1%` vs zero-only) | adds subnormal base normalization |
