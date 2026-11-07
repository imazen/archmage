; native/codegen/off/algebraic_ext_consumer-7fab387ac6137b08.ll
; internal fastcc void @_RNvNtCs8KlgWgeHDc4_22algebraic_ext_consumer9generated25___attune_dot_index_f32_v3(ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %y.0, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %out.0, i64 noundef range(i64 0, 2305843009213693952) %out.1) unnamed_addr #5 personality ptr @rust_eh_personality !guid !4272 {
%2 = fmul reassoc nsz arcp contract <8 x float> %.sroa.0.0.copyload, %.sroa.023.0.copyload
%3 = fadd reassoc nsz arcp contract <8 x float> %2, %sum.sroa.0.0
%5 = fmul reassoc nsz arcp contract <8 x float> %.sroa.0.0.copyload.1, %.sroa.023.0.copyload.1
%6 = fadd reassoc nsz arcp contract <8 x float> %5, %3
%9 = fmul reassoc nsz arcp contract <8 x float> %.sroa.0.0.copyload.epil, %.sroa.023.0.copyload.epil
%10 = fadd reassoc nsz arcp contract <8 x float> %9, %sum.sroa.0.0.epil.init
%11 = tail call reassoc nsz arcp contract float @llvm.vector.reduce.fadd.v8f32(float 0.000000e+00, <8 x float> %.lcssa)
