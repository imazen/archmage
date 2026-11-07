; arm/codegen/off/algebraic_ext_consumer-e0d9c317500875ca.ll
; define internal void @_RNCNvNtCs6wCdmQibB4y_22algebraic_ext_consumer9generated9cases_f32sn_0B5_(ptr noalias nofree nonnull readonly captures(none) %_1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %y.0, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %o.0, i64 noundef range(i64 0, 2305843009213693952) %o.1) unnamed_addr #1 personality ptr @rust_eh_personality !guid !882 {
%4 = fmul reassoc nsz arcp contract <4 x float> %3, %2
%5 = fadd reassoc nsz arcp contract <4 x float> %4, %sum.sroa.0.0.i
%8 = fmul reassoc nsz arcp contract <4 x float> %7, %6
%9 = fadd reassoc nsz arcp contract <4 x float> %8, %sum.sroa.12.0.i
%11 = fadd <4 x float> %5, %9
%pair.i.i = fadd <4 x float> %_8.i22.i, %_9.i23.i
%pair1.i.i = fadd <4 x float> %pair.i.i, %_11.i24.i
