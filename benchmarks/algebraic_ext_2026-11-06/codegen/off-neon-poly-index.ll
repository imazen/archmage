; arm/codegen/off/algebraic_ext_consumer-e0d9c317500875ca.ll
; define internal void @_RNCNvNtCs6wCdmQibB4y_22algebraic_ext_consumer9generated9cases_f32s1_0B5_(ptr noalias nofree nonnull readonly captures(none) %_1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, ptr noalias nofree nonnull readonly align 4 captures(none) %y.0, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %o.0, i64 noundef range(i64 0, 2305843009213693952) %o.1) unnamed_addr #1 personality ptr @rust_eh_personality !guid !380 {
%4 = fmul reassoc nsz arcp contract <32 x float> %wide.vec, splat (float f0x3E2AAAAB)
%5 = fadd reassoc nsz arcp contract <32 x float> %4, splat (float 5.000000e-01)
%6 = fmul reassoc nsz arcp contract <32 x float> %5, %wide.vec
%7 = fadd reassoc nsz arcp contract <32 x float> %6, splat (float 1.000000e+00)
%8 = fmul reassoc nsz arcp contract <32 x float> %7, %wide.vec
%interleaved.vec = fadd reassoc nsz arcp contract <32 x float> %8, splat (float 1.000000e+00)
%12 = fmul reassoc nsz arcp contract <4 x float> %11, splat (float f0x3E2AAAAB)
%13 = fadd reassoc nsz arcp contract <4 x float> %12, splat (float 5.000000e-01)
%14 = fmul reassoc nsz arcp contract <4 x float> %13, %11
%15 = fadd reassoc nsz arcp contract <4 x float> %14, splat (float 1.000000e+00)
%16 = fmul reassoc nsz arcp contract <4 x float> %15, %11
%17 = fadd reassoc nsz arcp contract <4 x float> %16, splat (float 1.000000e+00)
%19 = fmul reassoc nsz arcp contract <4 x float> %18, splat (float f0x3E2AAAAB)
%20 = fadd reassoc nsz arcp contract <4 x float> %19, splat (float 5.000000e-01)
%21 = fmul reassoc nsz arcp contract <4 x float> %20, %18
%22 = fadd reassoc nsz arcp contract <4 x float> %21, splat (float 1.000000e+00)
%23 = fmul reassoc nsz arcp contract <4 x float> %22, %18
%24 = fadd reassoc nsz arcp contract <4 x float> %23, splat (float 1.000000e+00)
