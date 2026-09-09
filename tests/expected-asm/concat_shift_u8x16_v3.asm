.section .text.asm_patterns::x86_impl::__arcane_concat_shift_u8x16_v3,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_concat_shift_u8x16_v3,@function
asm_patterns::x86_impl::__arcane_concat_shift_u8x16_v3:
	.cfi_startproc
	vmovdqa xmm0, xmmword ptr [rdx]
	vpalignr xmm0, xmm0, xmmword ptr [rsi], 3
	vmovdqa xmmword ptr [rdi], xmm0
	ret
