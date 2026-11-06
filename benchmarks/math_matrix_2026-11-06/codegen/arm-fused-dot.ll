; Selected arithmetic from arm/math_matrix_study-107ee31fd7b8ef38.ll
; Symbol _RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32sh_0B5_
%_5.i.i = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %_3.i2.sroa.0.0.copyload.i, <4 x float> %_3.i1.sroa.0.0.copyload.i, <4 x float> %sum.sroa.0.026.i)
%_9.i.i = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %_3.i2.sroa.4.0.copyload.i, <4 x float> %_3.i1.sroa.4.0.copyload.i, <4 x float> %sum.sroa.4.027.i)
%4 = fadd <4 x float> %_5.i.i, %_9.i.i
%pair.i.i = fadd <4 x float> %_8.i14.i, %_9.i15.i
%pair1.i.i = fadd <4 x float> %pair.i.i, %_11.i.i
