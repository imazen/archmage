; Selected arithmetic from arm/math_matrix_study-107ee31fd7b8ef38.ll
; Symbol _RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32s0_0B5_
%10 = fmul reassoc nsz arcp contract <4 x float> %wide.load7, %wide.load
%11 = fmul reassoc nsz arcp contract <4 x float> %wide.load8, %wide.load4
%12 = fmul reassoc nsz arcp contract <4 x float> %wide.load9, %wide.load5
%13 = fmul reassoc nsz arcp contract <4 x float> %wide.load10, %wide.load6
%14 = fadd reassoc nsz arcp contract <4 x float> %10, %vec.phi
%15 = fadd reassoc nsz arcp contract <4 x float> %11, %vec.phi1
%16 = fadd reassoc nsz arcp contract <4 x float> %12, %vec.phi2
%17 = fadd reassoc nsz arcp contract <4 x float> %13, %vec.phi3
%bin.rdx = fadd reassoc nsz arcp contract <4 x float> %15, %14
%bin.rdx11 = fadd reassoc nsz arcp contract <4 x float> %16, %bin.rdx
%bin.rdx12 = fadd reassoc nsz arcp contract <4 x float> %17, %bin.rdx11
%19 = tail call reassoc nsz arcp contract float @llvm.vector.reduce.fadd.v4f32(float 0.000000e+00, <4 x float> %bin.rdx12)
%23 = fmul reassoc nsz arcp contract <4 x float> %wide.load18, %wide.load17
%24 = fadd reassoc nsz arcp contract <4 x float> %23, %vec.phi16
%26 = tail call reassoc nsz arcp contract float @llvm.vector.reduce.fadd.v4f32(float 0.000000e+00, <4 x float> %24)
%_30.i = fmul reassoc nsz arcp contract float %b.i, %a.i
%_29.i = fadd reassoc nsz arcp contract float %_30.i, %sum.sroa.0.08.i
