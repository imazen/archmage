; native/codegen/thin/algebraic_ext_consumer-c1d3e2ef8dc61653.ll
; internal fastcc void @_RNvNtCs3b3W98B2qce_22algebraic_ext_consumer9generated41___attune_dot_index_ordinary_reduce_f32_v3(ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %y.0, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %out.0, i64 noundef range(i64 0, 2305843009213693952) %out.1) unnamed_addr #5 personality ptr @rust_eh_personality !guid !5070 {
%_5.i14 = fmul reassoc nsz arcp contract float %b.sroa.0.0.vec.extract, %a.sroa.0.0.vec.extract
%_5.i14.1 = fmul reassoc nsz arcp contract float %b.sroa.0.4.vec.extract, %a.sroa.0.4.vec.extract
%_5.i14.2 = fmul reassoc nsz arcp contract float %b.sroa.0.8.vec.extract, %a.sroa.0.8.vec.extract
%_5.i14.3 = fmul reassoc nsz arcp contract float %b.sroa.0.12.vec.extract, %a.sroa.0.12.vec.extract
%_5.i14.4 = fmul reassoc nsz arcp contract float %b.sroa.0.16.vec.extract, %a.sroa.0.16.vec.extract
%_5.i14.5 = fmul reassoc nsz arcp contract float %b.sroa.0.20.vec.extract, %a.sroa.0.20.vec.extract
%_5.i14.6 = fmul reassoc nsz arcp contract float %b.sroa.0.24.vec.extract, %a.sroa.0.24.vec.extract
%_5.i14.7 = fmul reassoc nsz arcp contract float %b.sroa.0.28.vec.extract, %a.sroa.0.28.vec.extract
%_5.i = fadd reassoc nsz arcp contract float %_5.i14, %sum.sroa.0.0.vec.extract
%_5.i.1 = fadd reassoc nsz arcp contract float %_5.i14.1, %sum.sroa.0.4.vec.extract
%_5.i.2 = fadd reassoc nsz arcp contract float %_5.i14.2, %sum.sroa.0.8.vec.extract
%_5.i.3 = fadd reassoc nsz arcp contract float %_5.i14.3, %sum.sroa.0.12.vec.extract
%_5.i.4 = fadd reassoc nsz arcp contract float %_5.i14.4, %sum.sroa.0.16.vec.extract
%_5.i.5 = fadd reassoc nsz arcp contract float %_5.i14.5, %sum.sroa.0.20.vec.extract
%_5.i.6 = fadd reassoc nsz arcp contract float %_5.i14.6, %sum.sroa.0.24.vec.extract
%_5.i.7 = fadd reassoc nsz arcp contract float %_5.i14.7, %sum.sroa.0.28.vec.extract
%5 = fadd <4 x float> %3, %4
%7 = fadd <4 x float> %5, %6
%9 = fadd <4 x float> %7, %8
