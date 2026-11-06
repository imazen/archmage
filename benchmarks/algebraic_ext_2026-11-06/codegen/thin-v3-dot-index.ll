; native/codegen/thin/algebraic_ext_consumer-c1d3e2ef8dc61653.ll
; internal fastcc void @_RNvNtCs3b3W98B2qce_22algebraic_ext_consumer9generated25___attune_dot_index_f32_v3(ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %y.0, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %out.0, i64 noundef range(i64 0, 2305843009213693952) %out.1) unnamed_addr #5 personality ptr @rust_eh_personality !guid !4252 {
%_5.i13 = fmul reassoc nsz arcp contract float %b.sroa.0.0.vec.extract, %a.sroa.0.0.vec.extract
%_5.i13.1 = fmul reassoc nsz arcp contract float %b.sroa.0.4.vec.extract, %a.sroa.0.4.vec.extract
%_5.i13.2 = fmul reassoc nsz arcp contract float %b.sroa.0.8.vec.extract, %a.sroa.0.8.vec.extract
%_5.i13.3 = fmul reassoc nsz arcp contract float %b.sroa.0.12.vec.extract, %a.sroa.0.12.vec.extract
%_5.i13.4 = fmul reassoc nsz arcp contract float %b.sroa.0.16.vec.extract, %a.sroa.0.16.vec.extract
%_5.i13.5 = fmul reassoc nsz arcp contract float %b.sroa.0.20.vec.extract, %a.sroa.0.20.vec.extract
%_5.i13.6 = fmul reassoc nsz arcp contract float %b.sroa.0.24.vec.extract, %a.sroa.0.24.vec.extract
%_5.i13.7 = fmul reassoc nsz arcp contract float %b.sroa.0.28.vec.extract, %a.sroa.0.28.vec.extract
%_5.i = fadd reassoc nsz arcp contract float %_5.i13, %sum.sroa.0.0.vec.extract
%_5.i.1 = fadd reassoc nsz arcp contract float %_5.i13.1, %sum.sroa.0.4.vec.extract
%_5.i.2 = fadd reassoc nsz arcp contract float %_5.i13.2, %sum.sroa.0.8.vec.extract
%_5.i.3 = fadd reassoc nsz arcp contract float %_5.i13.3, %sum.sroa.0.12.vec.extract
%_5.i.4 = fadd reassoc nsz arcp contract float %_5.i13.4, %sum.sroa.0.16.vec.extract
%_5.i.5 = fadd reassoc nsz arcp contract float %_5.i13.5, %sum.sroa.0.20.vec.extract
%_5.i.6 = fadd reassoc nsz arcp contract float %_5.i13.6, %sum.sroa.0.24.vec.extract
%_5.i.7 = fadd reassoc nsz arcp contract float %_5.i13.7, %sum.sroa.0.28.vec.extract
%_0.i.i.i.i.i.1 = fadd reassoc nsz arcp contract float %_5.i.1, %_5.i
%_0.i.i.i.i.i.2 = fadd reassoc nsz arcp contract float %_5.i.2, %_0.i.i.i.i.i.1
%_0.i.i.i.i.i.3 = fadd reassoc nsz arcp contract float %_5.i.3, %_0.i.i.i.i.i.2
%_0.i.i.i.i.i.4 = fadd reassoc nsz arcp contract float %_5.i.4, %_0.i.i.i.i.i.3
%_0.i.i.i.i.i.5 = fadd reassoc nsz arcp contract float %_5.i.5, %_0.i.i.i.i.i.4
%_0.i.i.i.i.i.6 = fadd reassoc nsz arcp contract float %_5.i.6, %_0.i.i.i.i.i.5
%_0.i.i.i.i.i.7 = fadd reassoc nsz arcp contract float %_5.i.7, %_0.i.i.i.i.i.6
