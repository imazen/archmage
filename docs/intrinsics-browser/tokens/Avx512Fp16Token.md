# Avx512Fp16Token — AVX-512FP16

Proof that AVX-512 FP16 (half-precision) is available.

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b, avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe, pclmulqdq, aes, avx512f, avx512bw, avx512cd, avx512dq, avx512vl, avx512fp16
**Total intrinsics:** 935 (918 safe, 17 unsafe, 0 stable, 935 unstable/unknown)

## Usage

```rust
use archmage::{Avx512Fp16Token, SimdToken};

if let Some(token) = Avx512Fp16Token::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: Avx512Fp16Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(16) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: Avx512Fp16Token, chunk: &mut [f32; 16]) {
    let v = _mm512_loadu_ps(chunk.as_ptr());  // safe inside #[rite]
    let doubled = _mm512_add_ps(v, v);
    _mm512_storeu_ps(chunk.as_mut_ptr(), doubled);
}
// Use #![forbid(unsafe_code)] — import_intrinsics provides safe memory ops.
```



## All Intrinsics

### Unstable/Nightly (935 intrinsics)

| Name | Description | Instruction |
|------|-------------|-------------|
| `_mm256_abs_ph` | Finds the absolute value of each packed half-precision (16-b... |  |
| `_mm256_add_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm256_castpd_ph` | Cast vector of type `__m256d` to type `__m256h`. This intrin... |  |
| `_mm256_castph128_ph256` | Cast vector of type `__m128h` to type `__m256h`. The upper 8... |  |
| `_mm256_castph256_ph128` | Cast vector of type `__m256h` to type `__m128h`. This intrin... |  |
| `_mm256_castph_pd` | Cast vector of type `__m256h` to type `__m256d`. This intrin... |  |
| `_mm256_castph_ps` | Cast vector of type `__m256h` to type `__m256`. This intrins... |  |
| `_mm256_castph_si256` | Cast vector of type `__m256h` to type `__m256i`. This intrin... |  |
| `_mm256_castps_ph` | Cast vector of type `__m256` to type `__m256h`. This intrins... |  |
| `_mm256_castsi256_ph` | Cast vector of type `__m256i` to type `__m256h`. This intrin... |  |
| `_mm256_cmp_ph_mask` | Compare packed half-precision (16-bit) floating-point elemen... |  |
| `_mm256_cmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm256_conj_pch` | Compute the complex conjugates of complex numbers in a, and ... |  |
| `_mm256_cvtepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm256_cvtepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm256_cvtepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm256_cvtepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm256_cvtepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm256_cvtepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm256_cvtpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm256_cvtph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm256_cvtph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm256_cvtph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm256_cvtph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm256_cvtph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm256_cvtph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm256_cvtph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm256_cvtsh_h` | Copy the lower half-precision (16-bit) floating-point elemen... |  |
| `_mm256_cvttph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm256_cvttph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm256_cvttph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm256_cvttph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm256_cvttph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm256_cvttph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm256_cvtxph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm256_cvtxps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm256_div_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm256_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm256_fcmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm256_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm256_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm256_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm256_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm256_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm256_fmul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm256_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm256_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm256_fpclass_ph_mask` | Test packed half-precision (16-bit) floating-point elements ... | vfpclassph |
| `_mm256_getexp_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm256_getmant_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm256_load_ph` | Load 256-bits (composed of 16 packed half-precision (16-bit)... |  |
| `_mm256_loadu_ph` | Load 256-bits (composed of 16 packed half-precision (16-bit)... |  |
| `_mm256_mask3_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm256_mask3_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm256_mask3_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm256_mask3_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm256_mask3_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm256_mask3_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm256_mask3_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm256_mask3_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm256_mask_add_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm256_mask_blend_ph` | Blend packed half-precision (16-bit) floating-point elements... |  |
| `_mm256_mask_cmp_ph_mask` | Compare packed half-precision (16-bit) floating-point elemen... |  |
| `_mm256_mask_cmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm256_mask_conj_pch` | Compute the complex conjugates of complex numbers in a, and ... |  |
| `_mm256_mask_cvtepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm256_mask_cvtepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm256_mask_cvtepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm256_mask_cvtepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm256_mask_cvtepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm256_mask_cvtepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm256_mask_cvtpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm256_mask_cvtph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm256_mask_cvtph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm256_mask_cvtph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm256_mask_cvtph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm256_mask_cvtph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm256_mask_cvtph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm256_mask_cvtph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm256_mask_cvttph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm256_mask_cvttph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm256_mask_cvttph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm256_mask_cvttph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm256_mask_cvttph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm256_mask_cvttph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm256_mask_cvtxph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm256_mask_cvtxps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm256_mask_div_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm256_mask_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm256_mask_fcmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm256_mask_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm256_mask_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm256_mask_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm256_mask_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm256_mask_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm256_mask_fmul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm256_mask_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm256_mask_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm256_mask_fpclass_ph_mask` | Test packed half-precision (16-bit) floating-point elements ... | vfpclassph |
| `_mm256_mask_getexp_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm256_mask_getmant_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm256_mask_max_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm256_mask_min_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm256_mask_mul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm256_mask_mul_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm256_mask_rcp_ph` | Compute the approximate reciprocal of packed 16-bit floating... | vrcpph |
| `_mm256_mask_reduce_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm256_mask_roundscale_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm256_mask_rsqrt_ph` | Compute the approximate reciprocal square root of packed hal... | vrsqrtph |
| `_mm256_mask_scalef_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm256_mask_sqrt_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm256_mask_sub_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm256_maskz_add_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm256_maskz_cmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm256_maskz_conj_pch` | Compute the complex conjugates of complex numbers in a, and ... |  |
| `_mm256_maskz_cvtepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm256_maskz_cvtepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm256_maskz_cvtepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm256_maskz_cvtepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm256_maskz_cvtepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm256_maskz_cvtepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm256_maskz_cvtpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm256_maskz_cvtph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm256_maskz_cvtph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm256_maskz_cvtph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm256_maskz_cvtph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm256_maskz_cvtph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm256_maskz_cvtph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm256_maskz_cvtph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm256_maskz_cvttph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm256_maskz_cvttph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm256_maskz_cvttph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm256_maskz_cvttph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm256_maskz_cvttph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm256_maskz_cvttph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm256_maskz_cvtxph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm256_maskz_cvtxps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm256_maskz_div_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm256_maskz_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm256_maskz_fcmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm256_maskz_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm256_maskz_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm256_maskz_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm256_maskz_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm256_maskz_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm256_maskz_fmul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm256_maskz_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm256_maskz_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm256_maskz_getexp_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm256_maskz_getmant_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm256_maskz_max_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm256_maskz_min_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm256_maskz_mul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm256_maskz_mul_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm256_maskz_rcp_ph` | Compute the approximate reciprocal of packed 16-bit floating... | vrcpph |
| `_mm256_maskz_reduce_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm256_maskz_roundscale_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm256_maskz_rsqrt_ph` | Compute the approximate reciprocal square root of packed hal... | vrsqrtph |
| `_mm256_maskz_scalef_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm256_maskz_sqrt_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm256_maskz_sub_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm256_max_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm256_min_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm256_mul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm256_mul_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm256_permutex2var_ph` | Shuffle half-precision (16-bit) floating-point elements in a... |  |
| `_mm256_permutexvar_ph` | Shuffle half-precision (16-bit) floating-point elements in a... |  |
| `_mm256_rcp_ph` | Compute the approximate reciprocal of packed 16-bit floating... | vrcpph |
| `_mm256_reduce_add_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm256_reduce_max_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm256_reduce_min_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm256_reduce_mul_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm256_reduce_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm256_roundscale_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm256_rsqrt_ph` | Compute the approximate reciprocal square root of packed hal... | vrsqrtph |
| `_mm256_scalef_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm256_set1_ph` | Broadcast the half-precision (16-bit) floating-point value a... |  |
| `_mm256_set_ph` | Set packed half-precision (16-bit) floating-point elements i... |  |
| `_mm256_setr_ph` | Set packed half-precision (16-bit) floating-point elements i... |  |
| `_mm256_setzero_ph` | Return vector of type __m256h with all elements set to zero |  |
| `_mm256_sqrt_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm256_store_ph` | Store 256-bits (composed of 16 packed half-precision (16-bit... |  |
| `_mm256_storeu_ph` | Store 256-bits (composed of 16 packed half-precision (16-bit... |  |
| `_mm256_sub_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm256_undefined_ph` | Return vector of type `__m256h` with indetermination element... |  |
| `_mm256_zextph128_ph256` | Cast vector of type `__m256h` to type `__m128h`. The upper 8... |  |
| `_mm512_abs_ph` | Finds the absolute value of each packed half-precision (16-b... |  |
| `_mm512_add_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm512_add_round_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm512_castpd_ph` | Cast vector of type `__m512d` to type `__m512h`. This intrin... |  |
| `_mm512_castph128_ph512` | Cast vector of type `__m128h` to type `__m512h`. The upper 2... |  |
| `_mm512_castph256_ph512` | Cast vector of type `__m256h` to type `__m512h`. The upper 1... |  |
| `_mm512_castph512_ph128` | Cast vector of type `__m512h` to type `__m128h`. This intrin... |  |
| `_mm512_castph512_ph256` | Cast vector of type `__m512h` to type `__m256h`. This intrin... |  |
| `_mm512_castph_pd` | Cast vector of type `__m512h` to type `__m512d`. This intrin... |  |
| `_mm512_castph_ps` | Cast vector of type `__m512h` to type `__m512`. This intrins... |  |
| `_mm512_castph_si512` | Cast vector of type `__m512h` to type `__m512i`. This intrin... |  |
| `_mm512_castps_ph` | Cast vector of type `__m512` to type `__m512h`. This intrins... |  |
| `_mm512_castsi512_ph` | Cast vector of type `__m512i` to type `__m512h`. This intrin... |  |
| `_mm512_cmp_ph_mask` | Compare packed half-precision (16-bit) floating-point elemen... |  |
| `_mm512_cmp_round_ph_mask` | Compare packed half-precision (16-bit) floating-point elemen... |  |
| `_mm512_cmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_cmul_round_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_conj_pch` | Compute the complex conjugates of complex numbers in a, and ... |  |
| `_mm512_cvt_roundepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm512_cvt_roundepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm512_cvt_roundepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm512_cvt_roundepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm512_cvt_roundepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm512_cvt_roundepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm512_cvt_roundpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm512_cvt_roundph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm512_cvt_roundph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm512_cvt_roundph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm512_cvt_roundph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm512_cvt_roundph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm512_cvt_roundph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm512_cvt_roundph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm512_cvtepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm512_cvtepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm512_cvtepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm512_cvtepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm512_cvtepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm512_cvtepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm512_cvtpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm512_cvtph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm512_cvtph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm512_cvtph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm512_cvtph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm512_cvtph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm512_cvtph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm512_cvtph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm512_cvtsh_h` | Copy the lower half-precision (16-bit) floating-point elemen... |  |
| `_mm512_cvtt_roundph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm512_cvtt_roundph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm512_cvtt_roundph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm512_cvtt_roundph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm512_cvtt_roundph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm512_cvtt_roundph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm512_cvttph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm512_cvttph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm512_cvttph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm512_cvttph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm512_cvttph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm512_cvttph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm512_cvtx_roundph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm512_cvtx_roundps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm512_cvtxph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm512_cvtxps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm512_div_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm512_div_round_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm512_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm512_fcmadd_round_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm512_fcmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_fcmul_round_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm512_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm512_fmadd_round_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm512_fmadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm512_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm512_fmaddsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm512_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm512_fmsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm512_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm512_fmsubadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm512_fmul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm512_fmul_round_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm512_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm512_fnmadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm512_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm512_fnmsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm512_fpclass_ph_mask` | Test packed half-precision (16-bit) floating-point elements ... | vfpclassph |
| `_mm512_getexp_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm512_getexp_round_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm512_getmant_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm512_getmant_round_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm512_load_ph` | Load 512-bits (composed of 32 packed half-precision (16-bit)... |  |
| `_mm512_loadu_ph` | Load 512-bits (composed of 32 packed half-precision (16-bit)... |  |
| `_mm512_mask3_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm512_mask3_fcmadd_round_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm512_mask3_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm512_mask3_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm512_mask3_fmadd_round_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm512_mask3_fmadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm512_mask3_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm512_mask3_fmaddsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm512_mask3_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm512_mask3_fmsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm512_mask3_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm512_mask3_fmsubadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm512_mask3_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm512_mask3_fnmadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm512_mask3_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm512_mask3_fnmsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm512_mask_add_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm512_mask_add_round_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm512_mask_blend_ph` | Blend packed half-precision (16-bit) floating-point elements... |  |
| `_mm512_mask_cmp_ph_mask` | Compare packed half-precision (16-bit) floating-point elemen... |  |
| `_mm512_mask_cmp_round_ph_mask` | Compare packed half-precision (16-bit) floating-point elemen... |  |
| `_mm512_mask_cmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_mask_cmul_round_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_mask_conj_pch` | Compute the complex conjugates of complex numbers in a, and ... |  |
| `_mm512_mask_cvt_roundepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm512_mask_cvt_roundepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm512_mask_cvt_roundepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm512_mask_cvt_roundepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm512_mask_cvt_roundepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm512_mask_cvt_roundepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm512_mask_cvt_roundpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm512_mask_cvt_roundph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm512_mask_cvt_roundph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm512_mask_cvt_roundph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm512_mask_cvt_roundph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm512_mask_cvt_roundph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm512_mask_cvt_roundph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm512_mask_cvt_roundph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm512_mask_cvtepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm512_mask_cvtepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm512_mask_cvtepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm512_mask_cvtepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm512_mask_cvtepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm512_mask_cvtepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm512_mask_cvtpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm512_mask_cvtph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm512_mask_cvtph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm512_mask_cvtph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm512_mask_cvtph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm512_mask_cvtph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm512_mask_cvtph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm512_mask_cvtph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm512_mask_cvtt_roundph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm512_mask_cvtt_roundph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm512_mask_cvtt_roundph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm512_mask_cvtt_roundph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm512_mask_cvtt_roundph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm512_mask_cvtt_roundph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm512_mask_cvttph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm512_mask_cvttph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm512_mask_cvttph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm512_mask_cvttph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm512_mask_cvttph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm512_mask_cvttph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm512_mask_cvtx_roundph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm512_mask_cvtx_roundps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm512_mask_cvtxph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm512_mask_cvtxps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm512_mask_div_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm512_mask_div_round_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm512_mask_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm512_mask_fcmadd_round_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm512_mask_fcmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_mask_fcmul_round_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_mask_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm512_mask_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm512_mask_fmadd_round_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm512_mask_fmadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm512_mask_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm512_mask_fmaddsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm512_mask_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm512_mask_fmsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm512_mask_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm512_mask_fmsubadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm512_mask_fmul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm512_mask_fmul_round_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm512_mask_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm512_mask_fnmadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm512_mask_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm512_mask_fnmsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm512_mask_fpclass_ph_mask` | Test packed half-precision (16-bit) floating-point elements ... | vfpclassph |
| `_mm512_mask_getexp_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm512_mask_getexp_round_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm512_mask_getmant_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm512_mask_getmant_round_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm512_mask_max_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm512_mask_max_round_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm512_mask_min_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm512_mask_min_round_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm512_mask_mul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm512_mask_mul_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm512_mask_mul_round_pch` | Multiply the packed complex numbers in a and b, and store th... | vfmulcph |
| `_mm512_mask_mul_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm512_mask_rcp_ph` | Compute the approximate reciprocal of packed 16-bit floating... | vrcpph |
| `_mm512_mask_reduce_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm512_mask_reduce_round_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm512_mask_roundscale_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm512_mask_roundscale_round_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm512_mask_rsqrt_ph` | Compute the approximate reciprocal square root of packed hal... | vrsqrtph |
| `_mm512_mask_scalef_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm512_mask_scalef_round_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm512_mask_sqrt_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm512_mask_sqrt_round_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm512_mask_sub_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm512_mask_sub_round_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm512_maskz_add_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm512_maskz_add_round_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm512_maskz_cmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_maskz_cmul_round_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_maskz_conj_pch` | Compute the complex conjugates of complex numbers in a, and ... |  |
| `_mm512_maskz_cvt_roundepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm512_maskz_cvt_roundepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm512_maskz_cvt_roundepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm512_maskz_cvt_roundepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm512_maskz_cvt_roundepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm512_maskz_cvt_roundepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm512_maskz_cvt_roundpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm512_maskz_cvt_roundph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm512_maskz_cvt_roundph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm512_maskz_cvt_roundph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm512_maskz_cvt_roundph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm512_maskz_cvt_roundph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm512_maskz_cvt_roundph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm512_maskz_cvt_roundph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm512_maskz_cvtepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm512_maskz_cvtepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm512_maskz_cvtepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm512_maskz_cvtepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm512_maskz_cvtepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm512_maskz_cvtepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm512_maskz_cvtpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm512_maskz_cvtph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm512_maskz_cvtph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm512_maskz_cvtph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm512_maskz_cvtph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm512_maskz_cvtph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm512_maskz_cvtph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm512_maskz_cvtph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm512_maskz_cvtt_roundph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm512_maskz_cvtt_roundph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm512_maskz_cvtt_roundph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm512_maskz_cvtt_roundph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm512_maskz_cvtt_roundph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm512_maskz_cvtt_roundph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm512_maskz_cvttph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm512_maskz_cvttph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm512_maskz_cvttph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm512_maskz_cvttph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm512_maskz_cvttph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm512_maskz_cvttph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm512_maskz_cvtx_roundph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm512_maskz_cvtx_roundps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm512_maskz_cvtxph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm512_maskz_cvtxps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm512_maskz_div_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm512_maskz_div_round_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm512_maskz_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm512_maskz_fcmadd_round_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm512_maskz_fcmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_maskz_fcmul_round_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm512_maskz_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm512_maskz_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm512_maskz_fmadd_round_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm512_maskz_fmadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm512_maskz_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm512_maskz_fmaddsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm512_maskz_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm512_maskz_fmsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm512_maskz_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm512_maskz_fmsubadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm512_maskz_fmul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm512_maskz_fmul_round_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm512_maskz_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm512_maskz_fnmadd_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm512_maskz_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm512_maskz_fnmsub_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm512_maskz_getexp_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm512_maskz_getexp_round_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm512_maskz_getmant_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm512_maskz_getmant_round_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm512_maskz_max_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm512_maskz_max_round_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm512_maskz_min_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm512_maskz_min_round_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm512_maskz_mul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm512_maskz_mul_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm512_maskz_mul_round_pch` | Multiply the packed complex numbers in a and b, and store th... | vfmulcph |
| `_mm512_maskz_mul_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm512_maskz_rcp_ph` | Compute the approximate reciprocal of packed 16-bit floating... | vrcpph |
| `_mm512_maskz_reduce_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm512_maskz_reduce_round_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm512_maskz_roundscale_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm512_maskz_roundscale_round_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm512_maskz_rsqrt_ph` | Compute the approximate reciprocal square root of packed hal... | vrsqrtph |
| `_mm512_maskz_scalef_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm512_maskz_scalef_round_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm512_maskz_sqrt_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm512_maskz_sqrt_round_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm512_maskz_sub_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm512_maskz_sub_round_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm512_max_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm512_max_round_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm512_min_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm512_min_round_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm512_mul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm512_mul_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm512_mul_round_pch` | Multiply the packed complex numbers in a and b, and store th... | vfmulcph |
| `_mm512_mul_round_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm512_permutex2var_ph` | Shuffle half-precision (16-bit) floating-point elements in a... |  |
| `_mm512_permutexvar_ph` | Shuffle half-precision (16-bit) floating-point elements in a... |  |
| `_mm512_rcp_ph` | Compute the approximate reciprocal of packed 16-bit floating... | vrcpph |
| `_mm512_reduce_add_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm512_reduce_max_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm512_reduce_min_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm512_reduce_mul_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm512_reduce_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm512_reduce_round_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm512_roundscale_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm512_roundscale_round_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm512_rsqrt_ph` | Compute the approximate reciprocal square root of packed hal... | vrsqrtph |
| `_mm512_scalef_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm512_scalef_round_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm512_set1_ph` | Broadcast the half-precision (16-bit) floating-point value a... |  |
| `_mm512_set_ph` | Set packed half-precision (16-bit) floating-point elements i... |  |
| `_mm512_setr_ph` | Set packed half-precision (16-bit) floating-point elements i... |  |
| `_mm512_setzero_ph` | Return vector of type __m512h with all elements set to zero |  |
| `_mm512_sqrt_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm512_sqrt_round_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm512_store_ph` | Store 512-bits (composed of 32 packed half-precision (16-bit... |  |
| `_mm512_storeu_ph` | Store 512-bits (composed of 32 packed half-precision (16-bit... |  |
| `_mm512_sub_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm512_sub_round_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm512_undefined_ph` | Return vector of type `__m512h` with indetermination element... |  |
| `_mm512_zextph128_ph512` | Cast vector of type `__m128h` to type `__m512h`. The upper 2... |  |
| `_mm512_zextph256_ph512` | Cast vector of type `__m256h` to type `__m512h`. The upper 1... |  |
| `_mm_abs_ph` | Finds the absolute value of each packed half-precision (16-b... |  |
| `_mm_add_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm_add_round_sh` | Add the lower half-precision (16-bit) floating-point element... | vaddsh |
| `_mm_add_sh` | Add the lower half-precision (16-bit) floating-point element... | vaddsh |
| `_mm_castpd_ph` | Cast vector of type `__m128d` to type `__m128h`. This intrin... |  |
| `_mm_castph_pd` | Cast vector of type `__m128h` to type `__m128d`. This intrin... |  |
| `_mm_castph_ps` | Cast vector of type `__m128h` to type `__m128`. This intrins... |  |
| `_mm_castph_si128` | Cast vector of type `__m128h` to type `__m128i`. This intrin... |  |
| `_mm_castps_ph` | Cast vector of type `__m128` to type `__m128h`. This intrins... |  |
| `_mm_castsi128_ph` | Cast vector of type `__m128i` to type `__m128h`. This intrin... |  |
| `_mm_cmp_ph_mask` | Compare packed half-precision (16-bit) floating-point elemen... |  |
| `_mm_cmp_round_sh_mask` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_cmp_sh_mask` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_cmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm_cmul_round_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_cmul_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_comi_round_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_comi_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_comieq_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_comige_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_comigt_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_comile_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_comilt_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_comineq_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_conj_pch` | Compute the complex conjugates of complex numbers in a, and ... |  |
| `_mm_cvt_roundi32_sh` | Convert the signed 32-bit integer b to a half-precision (16-... | vcvtsi2sh |
| `_mm_cvt_roundi64_sh` | Convert the signed 64-bit integer b to a half-precision (16-... | vcvtsi2sh |
| `_mm_cvt_roundsd_sh` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2sh |
| `_mm_cvt_roundsh_i32` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2si |
| `_mm_cvt_roundsh_i64` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2si |
| `_mm_cvt_roundsh_sd` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2sd |
| `_mm_cvt_roundsh_ss` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2ss |
| `_mm_cvt_roundsh_u32` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2usi |
| `_mm_cvt_roundsh_u64` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2usi |
| `_mm_cvt_roundss_sh` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sh |
| `_mm_cvt_roundu32_sh` | Convert the unsigned 32-bit integer b to a half-precision (1... | vcvtusi2sh |
| `_mm_cvt_roundu64_sh` | Convert the unsigned 64-bit integer b to a half-precision (1... | vcvtusi2sh |
| `_mm_cvtepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm_cvtepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm_cvtepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm_cvtepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm_cvtepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm_cvtepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm_cvti32_sh` | Convert the signed 32-bit integer b to a half-precision (16-... | vcvtsi2sh |
| `_mm_cvti64_sh` | Convert the signed 64-bit integer b to a half-precision (16-... | vcvtsi2sh |
| `_mm_cvtpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm_cvtph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm_cvtph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm_cvtph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm_cvtph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm_cvtph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm_cvtph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm_cvtph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm_cvtsd_sh` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2sh |
| `_mm_cvtsh_h` | Copy the lower half-precision (16-bit) floating-point elemen... |  |
| `_mm_cvtsh_i32` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2si |
| `_mm_cvtsh_i64` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2si |
| `_mm_cvtsh_sd` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2sd |
| `_mm_cvtsh_ss` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2ss |
| `_mm_cvtsh_u32` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2usi |
| `_mm_cvtsh_u64` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2usi |
| `_mm_cvtsi128_si16` | Copy the lower 16-bit integer in a to dst |  |
| `_mm_cvtsi16_si128` | Copy 16-bit integer a to the lower elements of dst, and zero... |  |
| `_mm_cvtss_sh` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sh |
| `_mm_cvtt_roundsh_i32` | Convert the lower half-precision (16-bit) floating-point ele... | vcvttsh2si |
| `_mm_cvtt_roundsh_i64` | Convert the lower half-precision (16-bit) floating-point ele... | vcvttsh2si |
| `_mm_cvtt_roundsh_u32` | Convert the lower half-precision (16-bit) floating-point ele... | vcvttsh2usi |
| `_mm_cvtt_roundsh_u64` | Convert the lower half-precision (16-bit) floating-point ele... | vcvttsh2usi |
| `_mm_cvttph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm_cvttph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm_cvttph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm_cvttph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm_cvttph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm_cvttph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm_cvttsh_i32` | Convert the lower half-precision (16-bit) floating-point ele... | vcvttsh2si |
| `_mm_cvttsh_i64` | Convert the lower half-precision (16-bit) floating-point ele... | vcvttsh2si |
| `_mm_cvttsh_u32` | Convert the lower half-precision (16-bit) floating-point ele... | vcvttsh2usi |
| `_mm_cvttsh_u64` | Convert the lower half-precision (16-bit) floating-point ele... | vcvttsh2usi |
| `_mm_cvtu32_sh` | Convert the unsigned 32-bit integer b to a half-precision (1... | vcvtusi2sh |
| `_mm_cvtu64_sh` | Convert the unsigned 64-bit integer b to a half-precision (1... | vcvtusi2sh |
| `_mm_cvtxph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm_cvtxps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm_div_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm_div_round_sh` | Divide the lower half-precision (16-bit) floating-point elem... | vdivsh |
| `_mm_div_sh` | Divide the lower half-precision (16-bit) floating-point elem... | vdivsh |
| `_mm_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm_fcmadd_round_sch` | Multiply the lower complex number in a by the complex conjug... | vfcmaddcsh |
| `_mm_fcmadd_sch` | Multiply the lower complex number in a by the complex conjug... | vfcmaddcsh |
| `_mm_fcmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm_fcmul_round_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_fcmul_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm_fmadd_round_sch` | Multiply the lower complex numbers in a and b, accumulate to... | vfmaddcsh |
| `_mm_fmadd_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmadd |
| `_mm_fmadd_sch` | Multiply the lower complex numbers in a and b, accumulate to... | vfmaddcsh |
| `_mm_fmadd_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmadd |
| `_mm_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm_fmsub_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmsub |
| `_mm_fmsub_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmsub |
| `_mm_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm_fmul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm_fmul_round_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_fmul_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm_fnmadd_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmadd |
| `_mm_fnmadd_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmadd |
| `_mm_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm_fnmsub_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmsub |
| `_mm_fnmsub_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmsub |
| `_mm_fpclass_ph_mask` | Test packed half-precision (16-bit) floating-point elements ... | vfpclassph |
| `_mm_fpclass_sh_mask` | Test the lower half-precision (16-bit) floating-point elemen... | vfpclasssh |
| `_mm_getexp_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm_getexp_round_sh` | Convert the exponent of the lower half-precision (16-bit) fl... | vgetexpsh |
| `_mm_getexp_sh` | Convert the exponent of the lower half-precision (16-bit) fl... | vgetexpsh |
| `_mm_getmant_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm_getmant_round_sh` | Normalize the mantissas of the lower half-precision (16-bit)... | vgetmantsh |
| `_mm_getmant_sh` | Normalize the mantissas of the lower half-precision (16-bit)... | vgetmantsh |
| `_mm_load_ph` | Load 128-bits (composed of 8 packed half-precision (16-bit) ... |  |
| `_mm_load_sh` | Load a half-precision (16-bit) floating-point element from m... |  |
| `_mm_loadu_ph` | Load 128-bits (composed of 8 packed half-precision (16-bit) ... |  |
| `_mm_mask3_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm_mask3_fcmadd_round_sch` | Multiply the lower complex number in a by the complex conjug... | vfcmaddcsh |
| `_mm_mask3_fcmadd_sch` | Multiply the lower complex number in a by the complex conjug... | vfcmaddcsh |
| `_mm_mask3_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm_mask3_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm_mask3_fmadd_round_sch` | Multiply the lower complex numbers in a and b, accumulate to... | vfmaddcsh |
| `_mm_mask3_fmadd_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmadd |
| `_mm_mask3_fmadd_sch` | Multiply the lower complex numbers in a and b, accumulate to... | vfmaddcsh |
| `_mm_mask3_fmadd_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmadd |
| `_mm_mask3_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm_mask3_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm_mask3_fmsub_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmsub |
| `_mm_mask3_fmsub_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmsub |
| `_mm_mask3_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm_mask3_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm_mask3_fnmadd_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmadd |
| `_mm_mask3_fnmadd_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmadd |
| `_mm_mask3_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm_mask3_fnmsub_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmsub |
| `_mm_mask3_fnmsub_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmsub |
| `_mm_mask_add_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm_mask_add_round_sh` | Add the lower half-precision (16-bit) floating-point element... | vaddsh |
| `_mm_mask_add_sh` | Add the lower half-precision (16-bit) floating-point element... | vaddsh |
| `_mm_mask_blend_ph` | Blend packed half-precision (16-bit) floating-point elements... |  |
| `_mm_mask_cmp_ph_mask` | Compare packed half-precision (16-bit) floating-point elemen... |  |
| `_mm_mask_cmp_round_sh_mask` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_mask_cmp_sh_mask` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_mask_cmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm_mask_cmul_round_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_mask_cmul_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_mask_conj_pch` | Compute the complex conjugates of complex numbers in a, and ... |  |
| `_mm_mask_cvt_roundsd_sh` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2sh |
| `_mm_mask_cvt_roundsh_sd` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2sd |
| `_mm_mask_cvt_roundsh_ss` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2ss |
| `_mm_mask_cvt_roundss_sh` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sh |
| `_mm_mask_cvtepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm_mask_cvtepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm_mask_cvtepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm_mask_cvtepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm_mask_cvtepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm_mask_cvtepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm_mask_cvtpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm_mask_cvtph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm_mask_cvtph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm_mask_cvtph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm_mask_cvtph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm_mask_cvtph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm_mask_cvtph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm_mask_cvtph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm_mask_cvtsd_sh` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2sh |
| `_mm_mask_cvtsh_sd` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2sd |
| `_mm_mask_cvtsh_ss` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2ss |
| `_mm_mask_cvtss_sh` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sh |
| `_mm_mask_cvttph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm_mask_cvttph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm_mask_cvttph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm_mask_cvttph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm_mask_cvttph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm_mask_cvttph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm_mask_cvtxph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm_mask_cvtxps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm_mask_div_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm_mask_div_round_sh` | Divide the lower half-precision (16-bit) floating-point elem... | vdivsh |
| `_mm_mask_div_sh` | Divide the lower half-precision (16-bit) floating-point elem... | vdivsh |
| `_mm_mask_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm_mask_fcmadd_round_sch` | Multiply the lower complex number in a by the complex conjug... | vfcmaddcsh |
| `_mm_mask_fcmadd_sch` | Multiply the lower complex number in a by the complex conjug... | vfcmaddcsh |
| `_mm_mask_fcmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm_mask_fcmul_round_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_mask_fcmul_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_mask_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm_mask_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm_mask_fmadd_round_sch` | Multiply the lower complex numbers in a and b, accumulate to... | vfmaddcsh |
| `_mm_mask_fmadd_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmadd |
| `_mm_mask_fmadd_sch` | Multiply the lower complex numbers in a and b, accumulate to... | vfmaddcsh |
| `_mm_mask_fmadd_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmadd |
| `_mm_mask_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm_mask_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm_mask_fmsub_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmsub |
| `_mm_mask_fmsub_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmsub |
| `_mm_mask_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm_mask_fmul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm_mask_fmul_round_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_mask_fmul_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_mask_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm_mask_fnmadd_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmadd |
| `_mm_mask_fnmadd_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmadd |
| `_mm_mask_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm_mask_fnmsub_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmsub |
| `_mm_mask_fnmsub_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmsub |
| `_mm_mask_fpclass_ph_mask` | Test packed half-precision (16-bit) floating-point elements ... | vfpclassph |
| `_mm_mask_fpclass_sh_mask` | Test the lower half-precision (16-bit) floating-point elemen... | vfpclasssh |
| `_mm_mask_getexp_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm_mask_getexp_round_sh` | Convert the exponent of the lower half-precision (16-bit) fl... | vgetexpsh |
| `_mm_mask_getexp_sh` | Convert the exponent of the lower half-precision (16-bit) fl... | vgetexpsh |
| `_mm_mask_getmant_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm_mask_getmant_round_sh` | Normalize the mantissas of the lower half-precision (16-bit)... | vgetmantsh |
| `_mm_mask_getmant_sh` | Normalize the mantissas of the lower half-precision (16-bit)... | vgetmantsh |
| `_mm_mask_load_sh` | Load a half-precision (16-bit) floating-point element from m... |  |
| `_mm_mask_max_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm_mask_max_round_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vmaxsh |
| `_mm_mask_max_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vmaxsh |
| `_mm_mask_min_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm_mask_min_round_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vminsh |
| `_mm_mask_min_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vminsh |
| `_mm_mask_move_sh` | Move the lower half-precision (16-bit) floating-point elemen... |  |
| `_mm_mask_mul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm_mask_mul_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm_mask_mul_round_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_mask_mul_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vmulsh |
| `_mm_mask_mul_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_mask_mul_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vmulsh |
| `_mm_mask_rcp_ph` | Compute the approximate reciprocal of packed 16-bit floating... | vrcpph |
| `_mm_mask_rcp_sh` | Compute the approximate reciprocal of the lower half-precisi... | vrcpsh |
| `_mm_mask_reduce_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm_mask_reduce_round_sh` | Extract the reduced argument of the lower half-precision (16... | vreducesh |
| `_mm_mask_reduce_sh` | Extract the reduced argument of the lower half-precision (16... | vreducesh |
| `_mm_mask_roundscale_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm_mask_roundscale_round_sh` | Round the lower half-precision (16-bit) floating-point eleme... | vrndscalesh |
| `_mm_mask_roundscale_sh` | Round the lower half-precision (16-bit) floating-point eleme... | vrndscalesh |
| `_mm_mask_rsqrt_ph` | Compute the approximate reciprocal square root of packed hal... | vrsqrtph |
| `_mm_mask_rsqrt_sh` | Compute the approximate reciprocal square root of the lower ... | vrsqrtsh |
| `_mm_mask_scalef_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm_mask_scalef_round_sh` | Scale the packed single-precision (32-bit) floating-point el... | vscalefsh |
| `_mm_mask_scalef_sh` | Scale the packed single-precision (32-bit) floating-point el... | vscalefsh |
| `_mm_mask_sqrt_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm_mask_sqrt_round_sh` | Compute the square root of the lower half-precision (16-bit)... | vsqrtsh |
| `_mm_mask_sqrt_sh` | Compute the square root of the lower half-precision (16-bit)... | vsqrtsh |
| `_mm_mask_store_sh` | Store the lower half-precision (16-bit) floating-point eleme... |  |
| `_mm_mask_sub_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm_mask_sub_round_sh` | Subtract the lower half-precision (16-bit) floating-point el... | vsubsh |
| `_mm_mask_sub_sh` | Subtract the lower half-precision (16-bit) floating-point el... | vsubsh |
| `_mm_maskz_add_ph` | Add packed half-precision (16-bit) floating-point elements i... | vaddph |
| `_mm_maskz_add_round_sh` | Add the lower half-precision (16-bit) floating-point element... | vaddsh |
| `_mm_maskz_add_sh` | Add the lower half-precision (16-bit) floating-point element... | vaddsh |
| `_mm_maskz_cmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm_maskz_cmul_round_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_maskz_cmul_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_maskz_conj_pch` | Compute the complex conjugates of complex numbers in a, and ... |  |
| `_mm_maskz_cvt_roundsd_sh` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2sh |
| `_mm_maskz_cvt_roundsh_sd` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2sd |
| `_mm_maskz_cvt_roundsh_ss` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2ss |
| `_mm_maskz_cvt_roundss_sh` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sh |
| `_mm_maskz_cvtepi16_ph` | Convert packed signed 16-bit integers in a to packed half-pr... | vcvtw2ph |
| `_mm_maskz_cvtepi32_ph` | Convert packed signed 32-bit integers in a to packed half-pr... | vcvtdq2ph |
| `_mm_maskz_cvtepi64_ph` | Convert packed signed 64-bit integers in a to packed half-pr... | vcvtqq2ph |
| `_mm_maskz_cvtepu16_ph` | Convert packed unsigned 16-bit integers in a to packed half-... | vcvtuw2ph |
| `_mm_maskz_cvtepu32_ph` | Convert packed unsigned 32-bit integers in a to packed half-... | vcvtudq2ph |
| `_mm_maskz_cvtepu64_ph` | Convert packed unsigned 64-bit integers in a to packed half-... | vcvtuqq2ph |
| `_mm_maskz_cvtpd_ph` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ph |
| `_mm_maskz_cvtph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2w |
| `_mm_maskz_cvtph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2dq |
| `_mm_maskz_cvtph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2qq |
| `_mm_maskz_cvtph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uw |
| `_mm_maskz_cvtph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2udq |
| `_mm_maskz_cvtph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2uqq |
| `_mm_maskz_cvtph_pd` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2pd |
| `_mm_maskz_cvtsd_sh` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2sh |
| `_mm_maskz_cvtsh_sd` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2sd |
| `_mm_maskz_cvtsh_ss` | Convert the lower half-precision (16-bit) floating-point ele... | vcvtsh2ss |
| `_mm_maskz_cvtss_sh` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sh |
| `_mm_maskz_cvttph_epi16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2w |
| `_mm_maskz_cvttph_epi32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2dq |
| `_mm_maskz_cvttph_epi64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2qq |
| `_mm_maskz_cvttph_epu16` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uw |
| `_mm_maskz_cvttph_epu32` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2udq |
| `_mm_maskz_cvttph_epu64` | Convert packed half-precision (16-bit) floating-point elemen... | vcvttph2uqq |
| `_mm_maskz_cvtxph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2psx |
| `_mm_maskz_cvtxps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2phx |
| `_mm_maskz_div_ph` | Divide packed half-precision (16-bit) floating-point element... | vdivph |
| `_mm_maskz_div_round_sh` | Divide the lower half-precision (16-bit) floating-point elem... | vdivsh |
| `_mm_maskz_div_sh` | Divide the lower half-precision (16-bit) floating-point elem... | vdivsh |
| `_mm_maskz_fcmadd_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmaddcph |
| `_mm_maskz_fcmadd_round_sch` | Multiply the lower complex number in a by the complex conjug... | vfcmaddcsh |
| `_mm_maskz_fcmadd_sch` | Multiply the lower complex number in a by the complex conjug... | vfcmaddcsh |
| `_mm_maskz_fcmul_pch` | Multiply packed complex numbers in a by the complex conjugat... | vfcmulcph |
| `_mm_maskz_fcmul_round_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_maskz_fcmul_sch` | Multiply the lower complex numbers in a by the complex conju... | vfcmulcsh |
| `_mm_maskz_fmadd_pch` | Multiply packed complex numbers in a and b, accumulate to th... | vfmaddcph |
| `_mm_maskz_fmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmadd |
| `_mm_maskz_fmadd_round_sch` | Multiply the lower complex numbers in a and b, accumulate to... | vfmaddcsh |
| `_mm_maskz_fmadd_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmadd |
| `_mm_maskz_fmadd_sch` | Multiply the lower complex numbers in a and b, accumulate to... | vfmaddcsh |
| `_mm_maskz_fmadd_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmadd |
| `_mm_maskz_fmaddsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmaddsub |
| `_mm_maskz_fmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsub |
| `_mm_maskz_fmsub_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmsub |
| `_mm_maskz_fmsub_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfmsub |
| `_mm_maskz_fmsubadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfmsubadd |
| `_mm_maskz_fmul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm_maskz_fmul_round_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_maskz_fmul_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_maskz_fnmadd_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmadd |
| `_mm_maskz_fnmadd_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmadd |
| `_mm_maskz_fnmadd_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmadd |
| `_mm_maskz_fnmsub_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vfnmsub |
| `_mm_maskz_fnmsub_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmsub |
| `_mm_maskz_fnmsub_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vfnmsub |
| `_mm_maskz_getexp_ph` | Convert the exponent of each packed half-precision (16-bit) ... | vgetexpph |
| `_mm_maskz_getexp_round_sh` | Convert the exponent of the lower half-precision (16-bit) fl... | vgetexpsh |
| `_mm_maskz_getexp_sh` | Convert the exponent of the lower half-precision (16-bit) fl... | vgetexpsh |
| `_mm_maskz_getmant_ph` | Normalize the mantissas of packed half-precision (16-bit) fl... | vgetmantph |
| `_mm_maskz_getmant_round_sh` | Normalize the mantissas of the lower half-precision (16-bit)... | vgetmantsh |
| `_mm_maskz_getmant_sh` | Normalize the mantissas of the lower half-precision (16-bit)... | vgetmantsh |
| `_mm_maskz_load_sh` | Load a half-precision (16-bit) floating-point element from m... |  |
| `_mm_maskz_max_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm_maskz_max_round_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vmaxsh |
| `_mm_maskz_max_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vmaxsh |
| `_mm_maskz_min_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm_maskz_min_round_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vminsh |
| `_mm_maskz_min_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vminsh |
| `_mm_maskz_move_sh` | Move the lower half-precision (16-bit) floating-point elemen... |  |
| `_mm_maskz_mul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm_maskz_mul_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm_maskz_mul_round_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_maskz_mul_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vmulsh |
| `_mm_maskz_mul_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_maskz_mul_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vmulsh |
| `_mm_maskz_rcp_ph` | Compute the approximate reciprocal of packed 16-bit floating... | vrcpph |
| `_mm_maskz_rcp_sh` | Compute the approximate reciprocal of the lower half-precisi... | vrcpsh |
| `_mm_maskz_reduce_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm_maskz_reduce_round_sh` | Extract the reduced argument of the lower half-precision (16... | vreducesh |
| `_mm_maskz_reduce_sh` | Extract the reduced argument of the lower half-precision (16... | vreducesh |
| `_mm_maskz_roundscale_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm_maskz_roundscale_round_sh` | Round the lower half-precision (16-bit) floating-point eleme... | vrndscalesh |
| `_mm_maskz_roundscale_sh` | Round the lower half-precision (16-bit) floating-point eleme... | vrndscalesh |
| `_mm_maskz_rsqrt_ph` | Compute the approximate reciprocal square root of packed hal... | vrsqrtph |
| `_mm_maskz_rsqrt_sh` | Compute the approximate reciprocal square root of the lower ... | vrsqrtsh |
| `_mm_maskz_scalef_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm_maskz_scalef_round_sh` | Scale the packed single-precision (32-bit) floating-point el... | vscalefsh |
| `_mm_maskz_scalef_sh` | Scale the packed single-precision (32-bit) floating-point el... | vscalefsh |
| `_mm_maskz_sqrt_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm_maskz_sqrt_round_sh` | Compute the square root of the lower half-precision (16-bit)... | vsqrtsh |
| `_mm_maskz_sqrt_sh` | Compute the square root of the lower half-precision (16-bit)... | vsqrtsh |
| `_mm_maskz_sub_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm_maskz_sub_round_sh` | Subtract the lower half-precision (16-bit) floating-point el... | vsubsh |
| `_mm_maskz_sub_sh` | Subtract the lower half-precision (16-bit) floating-point el... | vsubsh |
| `_mm_max_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vmaxph |
| `_mm_max_round_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vmaxsh |
| `_mm_max_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vmaxsh |
| `_mm_min_ph` | Compare packed half-precision (16-bit) floating-point elemen... | vminph |
| `_mm_min_round_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vminsh |
| `_mm_min_sh` | Compare the lower half-precision (16-bit) floating-point ele... | vminsh |
| `_mm_move_sh` | Move the lower half-precision (16-bit) floating-point elemen... |  |
| `_mm_mul_pch` | Multiply packed complex numbers in a and b, and store the re... | vfmulcph |
| `_mm_mul_ph` | Multiply packed half-precision (16-bit) floating-point eleme... | vmulph |
| `_mm_mul_round_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_mul_round_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vmulsh |
| `_mm_mul_sch` | Multiply the lower complex numbers in a and b, and store the... | vfmulcsh |
| `_mm_mul_sh` | Multiply the lower half-precision (16-bit) floating-point el... | vmulsh |
| `_mm_permutex2var_ph` | Shuffle half-precision (16-bit) floating-point elements in a... |  |
| `_mm_permutexvar_ph` | Shuffle half-precision (16-bit) floating-point elements in a... |  |
| `_mm_rcp_ph` | Compute the approximate reciprocal of packed 16-bit floating... | vrcpph |
| `_mm_rcp_sh` | Compute the approximate reciprocal of the lower half-precisi... | vrcpsh |
| `_mm_reduce_add_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm_reduce_max_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm_reduce_min_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm_reduce_mul_ph` | Reduce the packed half-precision (16-bit) floating-point ele... |  |
| `_mm_reduce_ph` | Extract the reduced argument of packed half-precision (16-bi... | vreduceph |
| `_mm_reduce_round_sh` | Extract the reduced argument of the lower half-precision (16... | vreducesh |
| `_mm_reduce_sh` | Extract the reduced argument of the lower half-precision (16... | vreducesh |
| `_mm_roundscale_ph` | Round packed half-precision (16-bit) floating-point elements... | vrndscaleph |
| `_mm_roundscale_round_sh` | Round the lower half-precision (16-bit) floating-point eleme... | vrndscalesh |
| `_mm_roundscale_sh` | Round the lower half-precision (16-bit) floating-point eleme... | vrndscalesh |
| `_mm_rsqrt_ph` | Compute the approximate reciprocal square root of packed hal... | vrsqrtph |
| `_mm_rsqrt_sh` | Compute the approximate reciprocal square root of the lower ... | vrsqrtsh |
| `_mm_scalef_ph` | Scale the packed half-precision (16-bit) floating-point elem... | vscalefph |
| `_mm_scalef_round_sh` | Scale the packed single-precision (32-bit) floating-point el... | vscalefsh |
| `_mm_scalef_sh` | Scale the packed single-precision (32-bit) floating-point el... | vscalefsh |
| `_mm_set1_ph` | Broadcast the half-precision (16-bit) floating-point value a... |  |
| `_mm_set_ph` | Set packed half-precision (16-bit) floating-point elements i... |  |
| `_mm_set_sh` | Copy half-precision (16-bit) floating-point elements from a ... |  |
| `_mm_setr_ph` | Set packed half-precision (16-bit) floating-point elements i... |  |
| `_mm_setzero_ph` | Return vector of type __m128h with all elements set to zero |  |
| `_mm_sqrt_ph` | Compute the square root of packed half-precision (16-bit) fl... | vsqrtph |
| `_mm_sqrt_round_sh` | Compute the square root of the lower half-precision (16-bit)... | vsqrtsh |
| `_mm_sqrt_sh` | Compute the square root of the lower half-precision (16-bit)... | vsqrtsh |
| `_mm_store_ph` | Store 128-bits (composed of 8 packed half-precision (16-bit)... |  |
| `_mm_store_sh` | Store the lower half-precision (16-bit) floating-point eleme... |  |
| `_mm_storeu_ph` | Store 128-bits (composed of 8 packed half-precision (16-bit)... |  |
| `_mm_sub_ph` | Subtract packed half-precision (16-bit) floating-point eleme... | vsubph |
| `_mm_sub_round_sh` | Subtract the lower half-precision (16-bit) floating-point el... | vsubsh |
| `_mm_sub_sh` | Subtract the lower half-precision (16-bit) floating-point el... | vsubsh |
| `_mm_ucomieq_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_ucomige_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_ucomigt_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_ucomile_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_ucomilt_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_ucomineq_sh` | Compare the lower half-precision (16-bit) floating-point ele... |  |
| `_mm_undefined_ph` | Return vector of type `__m128h` with indetermination element... |  |


