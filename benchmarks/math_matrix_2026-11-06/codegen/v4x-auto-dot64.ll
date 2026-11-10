; Selected arithmetic from r7900x/math_matrix_study-d62ccc0b1a7ac227.ll
; Symbol _RNvNtCsl8sLjJBmgi8_17math_matrix_study9algebraic30___attune_dot_algebraic_f64_v4x
%10 = fmul reassoc nsz arcp contract <8 x double> %wide.load18, %wide.load
%11 = fmul reassoc nsz arcp contract <8 x double> %wide.load19, %wide.load15
%12 = fmul reassoc nsz arcp contract <8 x double> %wide.load20, %wide.load16
%13 = fmul reassoc nsz arcp contract <8 x double> %wide.load21, %wide.load17
%14 = fadd reassoc nsz arcp contract <8 x double> %10, %vec.phi
%15 = fadd reassoc nsz arcp contract <8 x double> %11, %vec.phi12
%16 = fadd reassoc nsz arcp contract <8 x double> %12, %vec.phi13
%17 = fadd reassoc nsz arcp contract <8 x double> %13, %vec.phi14
%bin.rdx = fadd reassoc nsz arcp contract <8 x double> %15, %14
%bin.rdx22 = fadd reassoc nsz arcp contract <8 x double> %16, %bin.rdx
%bin.rdx23 = fadd reassoc nsz arcp contract <8 x double> %17, %bin.rdx22
%19 = tail call reassoc nsz arcp contract double @llvm.vector.reduce.fadd.v8f64(double 0.000000e+00, <8 x double> %bin.rdx23)
%23 = fmul reassoc nsz arcp contract <4 x double> %wide.load29, %wide.load28
%24 = fadd reassoc nsz arcp contract <4 x double> %23, %vec.phi27
%26 = tail call reassoc nsz arcp contract double @llvm.vector.reduce.fadd.v4f64(double 0.000000e+00, <4 x double> %24)
%_0.i = fmul reassoc nsz arcp contract double %b, %a
%_0.i3 = fadd reassoc nsz arcp contract double %_0.i, %sum.sroa.0.010
