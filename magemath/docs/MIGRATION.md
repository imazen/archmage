# Transcendental migration

Magemath provides all 35 deprecated Magetypes method names at widths four,
eight and sixteen. All require `configurable`; width sixteen additionally
requires `w512`, which `avx512` enables. Widths four and eight leave the
512-bit backends disabled unless explicitly requested.

For exact legacy behavior use `legacy::kernels` (width eight),
`legacy::kernels::f32x4` or `legacy::kernels::f32x16`. The matching arrays and
slices are `legacy::{array,slice}::<name>_f32x4/8/16`. These preserve each
backend's original rounding and exceptional-value behavior, including legacy
NaN behavior. NaN payloads remain unspecified. Signed zeros are compared by bits.

The existing `portable`, `relaxed`, and `unfused` families also provide all
names at these widths. Their established special-value handling stays intact:
portable uses fused rounding on every backend; relaxed uses backend rounding;
unfused uses separate multiply/add rounding. They deliberately propagate NaNs
where legacy min/max-based formulas can produce a finite value. Moving a caller
from an inherent legacy method to portable therefore is a contract choice,
not an exact replacement on all exceptional inputs. `*_unchecked` retains its
original input-domain requirement and makes no additional out-of-domain promise.

| Legacy method | Width 4 | Width 8 | Width 16 (`w512`) | Compatibility note |
|---|---|---|---|---|
| `log2_lowp` | Yes | Yes | Yes | Original rounding and special handling |
| `log2_lowp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `exp2_lowp` | Yes | Yes | Yes | Original rounding and special handling |
| `exp2_lowp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `ln_lowp` | Yes | Yes | Yes | Original rounding and special handling |
| `ln_lowp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `exp_lowp` | Yes | Yes | Yes | Original rounding and special handling |
| `exp_lowp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `log10_lowp` | Yes | Yes | Yes | Original rounding and special handling |
| `log10_lowp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `pow_lowp` | Yes | Yes | Yes | Original rounding and special handling |
| `pow_lowp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `log2_midp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `log2_midp` | Yes | Yes | Yes | Original rounding and special handling |
| `log2_midp_precise` | Yes | Yes | Yes | Retained compatibility spelling |
| `exp2_midp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `exp2_midp` | Yes | Yes | Yes | Original rounding and special handling |
| `exp2_midp_precise` | Yes | Yes | Yes | Retained compatibility spelling |
| `ln_midp` | Yes | Yes | Yes | Original rounding and special handling |
| `ln_midp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `ln_midp_precise` | Yes | Yes | Yes | Retained compatibility spelling |
| `exp_midp` | Yes | Yes | Yes | Original rounding and special handling |
| `exp_midp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `exp_midp_precise` | Yes | Yes | Yes | Retained compatibility spelling |
| `sigmoid_midp` | Yes | Yes | Yes | Original rounding and special handling |
| `silu_midp` | Yes | Yes | Yes | Original rounding and special handling |
| `log10_midp` | Yes | Yes | Yes | Original rounding and special handling |
| `log10_midp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `log10_midp_precise` | Yes | Yes | Yes | Retained compatibility spelling |
| `pow_midp` | Yes | Yes | Yes | Original rounding and special handling |
| `pow_midp_unchecked` | Yes | Yes | Yes | Existing unchecked domain |
| `pow_midp_precise` | Yes | Yes | Yes | Retained compatibility spelling |
| `cbrt_lowp` | Yes | Yes | Yes | Original rounding and special handling |
| `cbrt_midp` | Yes | Yes | Yes | Original rounding and special handling |
| `cbrt_midp_precise` | Yes | Yes | Yes | Retained compatibility spelling |

The six log/exp/power `*_midp_precise` spellings remain aliases of their
existing midp bodies; `cbrt_midp_precise` retains its distinct refinement.
Cbrt bodies now live in Magemath and do not call deprecated inherent methods.
Public `splat_t`, `from_array_t`, and `to_array` carry proofs and representations
across the crate boundary; no Magetypes visibility change is needed.

`src/legacy_impl.rs` is the canonical Magemath arithmetic template. Normal
regeneration (`just magemath-generate`) reads only that source and Magemath's
inventory. `tools/extract-legacy-migration.py` records the one-time migration
from the deprecated generated Magetypes bodies; it is not a regeneration
dependency and is to be removed before those implementations are retired.
Future arithmetic changes belong in Magemath's canonical template.

`legacy_superset` compares every name and enabled width with the old methods
on explicitly required available ISAs, with exact bits except unspecified NaN
payloads. It exercises kernels, arrays, every tail length through one width,
and a two-block-plus-tail slice. `width_parity` compares the three existing
rounding families against their width-eight controls, including multi-block
slices; unchecked cases stay within their domain.
