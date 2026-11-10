; arm/codegen/thin/algebraic_ext_consumer-742d703ec4c30b3b.ll
; define internal void @_RNCNvNtCseI5deSllD5x_22algebraic_ext_consumer9generated9cases_f32sd_0B5_(ptr noalias nofree nonnull readonly captures(none) %_1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %y.0, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %o.0, i64 noundef range(i64 0, 2305843009213693952) %o.1) unnamed_addr #1 personality ptr @rust_eh_personality !guid !693 {
%_5.i13.i = fmul reassoc nsz arcp contract float %_3.i.sroa.0.0.copyload.i, %_3.i3.sroa.0.0.copyload.i
%_5.i13.1.i = fmul reassoc nsz arcp contract float %_3.i.sroa.4.0.copyload.i, %_3.i3.sroa.4.0.copyload.i
%_5.i13.2.i = fmul reassoc nsz arcp contract float %_3.i.sroa.5.0.copyload.i, %_3.i3.sroa.5.0.copyload.i
%_5.i13.3.i = fmul reassoc nsz arcp contract float %_3.i.sroa.6.0.copyload.i, %_3.i3.sroa.6.0.copyload.i
%_5.i13.4.i = fmul reassoc nsz arcp contract float %_3.i.sroa.7.0.copyload.i, %_3.i3.sroa.7.0.copyload.i
%_5.i13.5.i = fmul reassoc nsz arcp contract float %_3.i.sroa.8.0.copyload.i, %_3.i3.sroa.8.0.copyload.i
%_5.i13.6.i = fmul reassoc nsz arcp contract float %_3.i.sroa.9.0.copyload.i, %_3.i3.sroa.9.0.copyload.i
%_5.i13.7.i = fmul reassoc nsz arcp contract float %_3.i.sroa.10.0.copyload.i, %_3.i3.sroa.10.0.copyload.i
%_5.i.i = fadd reassoc nsz arcp contract float %_5.i13.i, %sum.sroa.0.0.vec.extract.i
%_5.i.1.i = fadd reassoc nsz arcp contract float %_5.i13.1.i, %sum.sroa.0.4.vec.extract.i
%_5.i.2.i = fadd reassoc nsz arcp contract float %_5.i13.2.i, %sum.sroa.0.8.vec.extract.i
%_5.i.3.i = fadd reassoc nsz arcp contract float %_5.i13.3.i, %sum.sroa.0.12.vec.extract.i
%_5.i.4.i = fadd reassoc nsz arcp contract float %_5.i13.4.i, %sum.sroa.12.16.vec.extract.i
%_5.i.5.i = fadd reassoc nsz arcp contract float %_5.i13.5.i, %sum.sroa.12.20.vec.extract.i
%_5.i.6.i = fadd reassoc nsz arcp contract float %_5.i13.6.i, %sum.sroa.12.24.vec.extract.i
%_5.i.7.i = fadd reassoc nsz arcp contract float %_5.i13.7.i, %sum.sroa.12.28.vec.extract.i
%_0.i.i.i.i.i.1.i = fadd reassoc nsz arcp contract float %_5.i.1.i, %_5.i.i
%_0.i.i.i.i.i.2.i = fadd reassoc nsz arcp contract float %_0.i.i.i.i.i.1.i, %_5.i.2.i
%_0.i.i.i.i.i.3.i = fadd reassoc nsz arcp contract float %_0.i.i.i.i.i.2.i, %_5.i.3.i
%_0.i.i.i.i.i.4.i = fadd reassoc nsz arcp contract float %_0.i.i.i.i.i.3.i, %_5.i.4.i
%_0.i.i.i.i.i.5.i = fadd reassoc nsz arcp contract float %_0.i.i.i.i.i.4.i, %_5.i.5.i
%_0.i.i.i.i.i.6.i = fadd reassoc nsz arcp contract float %_0.i.i.i.i.i.5.i, %_5.i.6.i
%_0.i.i.i.i.i.7.i = fadd reassoc nsz arcp contract float %_0.i.i.i.i.i.6.i, %_5.i.7.i
