; Selected arithmetic from r7900x/math_matrix_study-d62ccc0b1a7ac227.ll
; Symbol _RNvNtCsl8sLjJBmgi8_17math_matrix_study9algebraic29___attune_dot_algebraic_f32_v3
%10 = fmul reassoc nsz arcp contract <8 x float> %wide.load18, %wide.load
%11 = fmul reassoc nsz arcp contract <8 x float> %wide.load19, %wide.load15
%12 = fmul reassoc nsz arcp contract <8 x float> %wide.load20, %wide.load16
%13 = fmul reassoc nsz arcp contract <8 x float> %wide.load21, %wide.load17
%14 = fadd reassoc nsz arcp contract <8 x float> %10, %vec.phi
%15 = fadd reassoc nsz arcp contract <8 x float> %11, %vec.phi12
%16 = fadd reassoc nsz arcp contract <8 x float> %12, %vec.phi13
%17 = fadd reassoc nsz arcp contract <8 x float> %13, %vec.phi14
%bin.rdx = fadd reassoc nsz arcp contract <8 x float> %15, %14
%bin.rdx22 = fadd reassoc nsz arcp contract <8 x float> %16, %bin.rdx
%bin.rdx23 = fadd reassoc nsz arcp contract <8 x float> %17, %bin.rdx22
%19 = tail call reassoc nsz arcp contract float @llvm.vector.reduce.fadd.v8f32(float 0.000000e+00, <8 x float> %bin.rdx23)
%23 = fmul reassoc nsz arcp contract <4 x float> %wide.load29, %wide.load28
%24 = fadd reassoc nsz arcp contract <4 x float> %23, %vec.phi27
%26 = tail call reassoc nsz arcp contract float @llvm.vector.reduce.fadd.v4f32(float 0.000000e+00, <4 x float> %24)
%_0.i = fmul reassoc nsz arcp contract float %b, %a
%_0.i3 = fadd reassoc nsz arcp contract float %_0.i, %sum.sroa.0.010
