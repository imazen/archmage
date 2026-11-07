; Selected arithmetic from arm/math_matrix_study-107ee31fd7b8ef38.ll
; Symbol _RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32s9_0B5_
%5 = fmul reassoc nsz arcp contract <32 x float> %wide.vec, splat (float f0x3E2AAAAB)
%6 = fadd reassoc nsz arcp contract <32 x float> %5, splat (float 5.000000e-01)
%7 = fmul reassoc nsz arcp contract <32 x float> %6, %wide.vec
%8 = fadd reassoc nsz arcp contract <32 x float> %7, splat (float 1.000000e+00)
%9 = fmul reassoc nsz arcp contract <32 x float> %8, %wide.vec
%interleaved.vec = fadd reassoc nsz arcp contract <32 x float> %9, splat (float 1.000000e+00)
%13 = fmul reassoc nsz arcp contract <4 x float> %12, splat (float f0x3E2AAAAB)
%14 = fadd reassoc nsz arcp contract <4 x float> %13, splat (float 5.000000e-01)
%15 = fmul reassoc nsz arcp contract <4 x float> %14, %12
%16 = fadd reassoc nsz arcp contract <4 x float> %15, splat (float 1.000000e+00)
%17 = fmul reassoc nsz arcp contract <4 x float> %16, %12
%18 = fadd reassoc nsz arcp contract <4 x float> %17, splat (float 1.000000e+00)
%20 = fmul reassoc nsz arcp contract <4 x float> %19, splat (float f0x3E2AAAAB)
%21 = fadd reassoc nsz arcp contract <4 x float> %20, splat (float 5.000000e-01)
%22 = fmul reassoc nsz arcp contract <4 x float> %21, %19
%23 = fadd reassoc nsz arcp contract <4 x float> %22, splat (float 1.000000e+00)
%24 = fmul reassoc nsz arcp contract <4 x float> %23, %19
%25 = fadd reassoc nsz arcp contract <4 x float> %24, splat (float 1.000000e+00)
