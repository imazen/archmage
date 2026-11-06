; arm/codegen/off/algebraic_ext_consumer-e0d9c317500875ca.ll
; define internal void @_RNCNvNtCs6wCdmQibB4y_22algebraic_ext_consumer9generated9cases_f32sd_0B5_(ptr noalias nofree nonnull readonly captures(none) %_1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %y.0, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %o.0, i64 noundef range(i64 0, 2305843009213693952) %o.1) unnamed_addr #1 personality ptr @rust_eh_personality !guid !715 {
%4 = fmul reassoc nsz arcp contract <8 x float> %3, %2
%6 = fadd reassoc nsz arcp contract <8 x float> %4, %5
%10 = tail call reassoc nsz arcp contract float @llvm.vector.reduce.fadd.v8f32(float 0.000000e+00, <8 x float> %6)
