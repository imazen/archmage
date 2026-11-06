; Selected arithmetic from wasm/math_matrix_study-ba178c59a0ecbf47.ll
; Symbol _RNCNvNtCsdrsRtign2uv_17math_matrix_study9algebraic9cases_f32sd_0B5_
%8 = fmul reassoc nsz arcp contract <4 x float> %7, %6
%9 = fadd reassoc nsz arcp contract <4 x float> %8, %3
%10 = fmul reassoc nsz arcp contract <4 x float> %_21.sroa.0.0.copyload.i18.i4, %_21.sroa.0.0.copyload.i.i2
%11 = fadd reassoc nsz arcp contract <4 x float> %10, %4
%rdx.op = fadd reassoc nsz arcp contract <4 x float> %9, %11
%12 = tail call reassoc nsz arcp contract float @llvm.vector.reduce.fadd.v4f32(float 0.000000e+00, <4 x float> %rdx.op)
