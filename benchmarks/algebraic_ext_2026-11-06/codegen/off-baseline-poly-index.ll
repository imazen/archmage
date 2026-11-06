; native/codegen/off/algebraic_ext_consumer-7fab387ac6137b08.ll
; define internal void @_RNCNvNtCs8KlgWgeHDc4_22algebraic_ext_consumer9generated9cases_f32s1_0B5_(ptr noalias nofree nonnull readonly captures(none) %_1, ptr noalias nofree noundef nonnull readonly align 4 captures(none) %x.0, i64 noundef range(i64 0, 2305843009213693952) %x.1, ptr noalias nofree nonnull readonly align 4 captures(none) %y.0, i64 noundef range(i64 0, 2305843009213693952) %y.1, ptr noalias nofree noundef nonnull writeonly align 4 captures(none) %o.0, i64 noundef range(i64 0, 2305843009213693952) %o.1) unnamed_addr #2 personality ptr @rust_eh_personality !guid !451 {
%105 = fmul reassoc nsz arcp contract <16 x float> %104, splat (float f0x3E2AAAAB)
%106 = fadd reassoc nsz arcp contract <16 x float> %105, splat (float 5.000000e-01)
%110 = fmul reassoc nsz arcp contract <16 x float> %106, %109
%111 = fadd reassoc nsz arcp contract <16 x float> %110, splat (float 1.000000e+00)
%115 = fmul reassoc nsz arcp contract <16 x float> %111, %114
%119 = fmul reassoc nsz arcp contract <16 x float> %118, splat (float f0x3E2AAAAB)
%120 = fadd reassoc nsz arcp contract <16 x float> %119, splat (float 5.000000e-01)
%124 = fmul reassoc nsz arcp contract <16 x float> %120, %123
%125 = fadd reassoc nsz arcp contract <16 x float> %124, splat (float 1.000000e+00)
%129 = fmul reassoc nsz arcp contract <16 x float> %125, %128
%interleaved.vec = fadd reassoc nsz arcp contract <32 x float> %130, splat (float 1.000000e+00)
%134 = fmul reassoc nsz arcp contract <4 x float> %133, splat (float f0x3E2AAAAB)
%135 = fadd reassoc nsz arcp contract <4 x float> %134, splat (float 5.000000e-01)
%136 = fmul reassoc nsz arcp contract <4 x float> %135, %133
%137 = fadd reassoc nsz arcp contract <4 x float> %136, splat (float 1.000000e+00)
%138 = fmul reassoc nsz arcp contract <4 x float> %137, %133
%139 = fadd reassoc nsz arcp contract <4 x float> %138, splat (float 1.000000e+00)
%141 = fmul reassoc nsz arcp contract <4 x float> %140, splat (float f0x3E2AAAAB)
%142 = fadd reassoc nsz arcp contract <4 x float> %141, splat (float 5.000000e-01)
%143 = fmul reassoc nsz arcp contract <4 x float> %142, %140
%144 = fadd reassoc nsz arcp contract <4 x float> %143, splat (float 1.000000e+00)
%145 = fmul reassoc nsz arcp contract <4 x float> %144, %140
%146 = fadd reassoc nsz arcp contract <4 x float> %145, splat (float 1.000000e+00)
