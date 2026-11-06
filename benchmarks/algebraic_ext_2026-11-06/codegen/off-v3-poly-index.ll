; native/codegen/off/algebraic_ext_consumer-7fab387ac6137b08.ll
; internal fastcc void @_RNvNtCs8KlgWgeHDc4_22algebraic_ext_consumer9generated26___attune_poly_index_f32_v3(ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %out.0, i64 noundef range(i64 0, 2305843009213693952) %out.1) unnamed_addr #5 personality ptr @rust_eh_personality !guid !4629 {
%2 = fmul reassoc nsz arcp contract <8 x float> %.sroa.069.0.copyload, splat (float f0x3E2AAAAB)
%3 = fadd reassoc nsz arcp contract <8 x float> %2, splat (float 5.000000e-01)
%4 = fmul reassoc nsz arcp contract <8 x float> %.sroa.069.0.copyload, %3
%5 = fadd reassoc nsz arcp contract <8 x float> %4, splat (float 1.000000e+00)
%6 = fmul reassoc nsz arcp contract <8 x float> %.sroa.069.0.copyload, %5
%7 = fadd reassoc nsz arcp contract <8 x float> %6, splat (float 1.000000e+00)
%9 = fmul reassoc nsz arcp contract <8 x float> %.sroa.069.0.copyload.1, splat (float f0x3E2AAAAB)
%10 = fadd reassoc nsz arcp contract <8 x float> %9, splat (float 5.000000e-01)
%11 = fmul reassoc nsz arcp contract <8 x float> %.sroa.069.0.copyload.1, %10
%12 = fadd reassoc nsz arcp contract <8 x float> %11, splat (float 1.000000e+00)
%13 = fmul reassoc nsz arcp contract <8 x float> %.sroa.069.0.copyload.1, %12
%14 = fadd reassoc nsz arcp contract <8 x float> %13, splat (float 1.000000e+00)
%17 = fmul reassoc nsz arcp contract <8 x float> %.sroa.069.0.copyload.epil, splat (float f0x3E2AAAAB)
%18 = fadd reassoc nsz arcp contract <8 x float> %17, splat (float 5.000000e-01)
%19 = fmul reassoc nsz arcp contract <8 x float> %.sroa.069.0.copyload.epil, %18
%20 = fadd reassoc nsz arcp contract <8 x float> %19, splat (float 1.000000e+00)
%21 = fmul reassoc nsz arcp contract <8 x float> %.sroa.069.0.copyload.epil, %20
%22 = fadd reassoc nsz arcp contract <8 x float> %21, splat (float 1.000000e+00)
