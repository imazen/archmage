; arm/codegen/thin/algebraic_ext_consumer-742d703ec4c30b3b.ll
; define internal void @_RNCNvNtCseI5deSllD5x_22algebraic_ext_consumer9generated9cases_f32sn_0B5_(ptr noalias nofree nonnull readonly captures(none) %_1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %y.0, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %o.0, i64 noundef range(i64 0, 2305843009213693952) %o.1) unnamed_addr #1 personality ptr @rust_eh_personality !guid !861 {
%_5.i14.i = fmul reassoc nsz arcp contract float %_3.i3.sroa.0.0.copyload.i, %_3.i4.sroa.0.0.copyload.i
%_5.i14.1.i = fmul reassoc nsz arcp contract float %_3.i3.sroa.4.0.copyload.i, %_3.i4.sroa.4.0.copyload.i
%_5.i14.2.i = fmul reassoc nsz arcp contract float %_3.i3.sroa.5.0.copyload.i, %_3.i4.sroa.5.0.copyload.i
%_5.i14.3.i = fmul reassoc nsz arcp contract float %_3.i3.sroa.6.0.copyload.i, %_3.i4.sroa.6.0.copyload.i
%_5.i14.4.i = fmul reassoc nsz arcp contract float %_3.i3.sroa.7.0.copyload.i, %_3.i4.sroa.7.0.copyload.i
%_5.i14.5.i = fmul reassoc nsz arcp contract float %_3.i3.sroa.8.0.copyload.i, %_3.i4.sroa.8.0.copyload.i
%_5.i14.6.i = fmul reassoc nsz arcp contract float %_3.i3.sroa.9.0.copyload.i, %_3.i4.sroa.9.0.copyload.i
%_5.i14.7.i = fmul reassoc nsz arcp contract float %_3.i3.sroa.10.0.copyload.i, %_3.i4.sroa.10.0.copyload.i
%_5.i.i = fadd reassoc nsz arcp contract float %_5.i14.i, %sum.sroa.0.0.vec.extract.i
%_5.i.1.i = fadd reassoc nsz arcp contract float %_5.i14.1.i, %sum.sroa.0.4.vec.extract.i
%_5.i.2.i = fadd reassoc nsz arcp contract float %_5.i14.2.i, %sum.sroa.0.8.vec.extract.i
%_5.i.3.i = fadd reassoc nsz arcp contract float %_5.i14.3.i, %sum.sroa.0.12.vec.extract.i
%_5.i.4.i = fadd reassoc nsz arcp contract float %_5.i14.4.i, %sum.sroa.12.16.vec.extract.i
%_5.i.5.i = fadd reassoc nsz arcp contract float %_5.i14.5.i, %sum.sroa.12.20.vec.extract.i
%_5.i.6.i = fadd reassoc nsz arcp contract float %_5.i14.6.i, %sum.sroa.12.24.vec.extract.i
%_5.i.7.i = fadd reassoc nsz arcp contract float %_5.i14.7.i, %sum.sroa.12.28.vec.extract.i
%3 = fadd <4 x float> %sum.sroa.0.12.vec.insert.i, %sum.sroa.12.28.vec.insert.i
%pair.i.i = fadd <4 x float> %_8.i22.i, %_9.i23.i
%pair1.i.i = fadd <4 x float> %pair.i.i, %_11.i24.i
