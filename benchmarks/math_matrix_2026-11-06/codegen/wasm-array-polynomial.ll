; Selected arithmetic from wasm/math_matrix_study-ba178c59a0ecbf47.ll
; Symbol _RNCNvNtCsdrsRtign2uv_17math_matrix_study9algebraic9cases_f32s9_0B5_
%5 = fmul reassoc nsz arcp contract <4 x float> %4, splat (float f0x3E2AAAAB)
%6 = fadd reassoc nsz arcp contract <4 x float> %5, splat (float 5.000000e-01)
%7 = fmul reassoc nsz arcp contract <4 x float> %6, %4
%8 = fadd reassoc nsz arcp contract <4 x float> %7, splat (float 1.000000e+00)
%9 = fmul reassoc nsz arcp contract <4 x float> %8, %4
%10 = fadd reassoc nsz arcp contract <4 x float> %9, splat (float 1.000000e+00)
%11 = fmul reassoc nsz arcp contract <4 x float> %_21.sroa.0.0.copyload.i16.i2, splat (float f0x3E2AAAAB)
%12 = fadd reassoc nsz arcp contract <4 x float> %11, splat (float 5.000000e-01)
%13 = fmul reassoc nsz arcp contract <4 x float> %12, %_21.sroa.0.0.copyload.i16.i2
%14 = fadd reassoc nsz arcp contract <4 x float> %13, splat (float 1.000000e+00)
%15 = fmul reassoc nsz arcp contract <4 x float> %14, %_21.sroa.0.0.copyload.i16.i2
%16 = fadd reassoc nsz arcp contract <4 x float> %15, splat (float 1.000000e+00)
