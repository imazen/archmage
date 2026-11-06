; native/codegen/off/algebraic_ext_consumer-7fab387ac6137b08.ll
; internal fastcc void @_RNvNtCs8KlgWgeHDc4_22algebraic_ext_consumer9generated41___attune_dot_index_ordinary_reduce_f32_v3(ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %y.0, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %out.0, i64 noundef range(i64 0, 2305843009213693952) %out.1) unnamed_addr #5 personality ptr @rust_eh_personality !guid !5109 {
%2 = fmul reassoc nsz arcp contract <8 x float> %.sroa.021.0.copyload, %.sroa.022.0.copyload
%3 = fmul reassoc nsz arcp contract <8 x float> %.sroa.021.0.copyload, %.sroa.022.0.copyload
%sum.sroa.0.28.vec.insert53 = fadd reassoc nsz arcp contract <8 x float> %6, %sum.sroa.0.0
%8 = fmul reassoc nsz arcp contract <8 x float> %.sroa.021.0.copyload.1, %.sroa.022.0.copyload.1
%9 = fmul reassoc nsz arcp contract <8 x float> %.sroa.021.0.copyload.1, %.sroa.022.0.copyload.1
%sum.sroa.0.28.vec.insert53.1 = fadd reassoc nsz arcp contract <8 x float> %12, %sum.sroa.0.28.vec.insert53
%15 = fmul reassoc nsz arcp contract <8 x float> %.sroa.021.0.copyload.epil, %.sroa.022.0.copyload.epil
%16 = fmul reassoc nsz arcp contract <8 x float> %.sroa.021.0.copyload.epil, %.sroa.022.0.copyload.epil
%sum.sroa.0.28.vec.insert53.epil = fadd reassoc nsz arcp contract <8 x float> %19, %sum.sroa.0.0.epil.init
%20 = fadd reassoc nsz arcp contract <8 x float> %.lcssa, %sum.sroa.0.0.lcssa
%23 = fadd <4 x float> %21, %22
%25 = fadd <4 x float> %23, %24
%27 = fadd <4 x float> %25, %26
