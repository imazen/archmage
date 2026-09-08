.section .text.asm_patterns::x86_impl::__arcane_concat_shift_f32x8_v3,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_concat_shift_f32x8_v3,@function
asm_patterns::x86_impl::__arcane_concat_shift_f32x8_v3:
	.cfi_startproc
	vmovdqa ymm0, ymmword ptr [rsi]
	vperm2i128 ymm1, ymm0, ymmword ptr [rdx], 33
	vpalignr ymm0, ymm1, ymm0, 4
	vmovdqa ymmword ptr [rdi], ymm0
	vzeroupper
	ret
