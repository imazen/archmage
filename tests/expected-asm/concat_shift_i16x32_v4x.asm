.section .text.asm_patterns::x86_impl::__arcane_concat_shift_i16x32_v4x,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_concat_shift_i16x32_v4x,@function
asm_patterns::x86_impl::__arcane_concat_shift_i16x32_v4x:
	.cfi_startproc
	vmovdqa64 zmm0, zmmword ptr [rsi]
	vmovdqa64 zmm1, zmmword ptr [rdx]
	vshufi64x2 zmm2, zmm0, zmm1, 78
	valignq zmm0, zmm1, zmm0, 2
	vpalignr zmm0, zmm2, zmm0, 2
	vmovdqa64 zmmword ptr [rdi], zmm0
	vzeroupper
	ret
