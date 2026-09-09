.section .text.asm_patterns::x86_impl::__arcane_load_array_ref,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_load_array_ref,@function
asm_patterns::x86_impl::__arcane_load_array_ref:
	.cfi_startproc
	vmovups ymm0, ymmword ptr [rsi]
	vmovaps ymmword ptr [rdi], ymm0
	vzeroupper
	ret
