.section .text.asm_patterns::load_array_ref::__simd_inner_load_array_ref,"ax",@progbits
	.p2align	4
.type	asm_patterns::load_array_ref::__simd_inner_load_array_ref,@function
asm_patterns::load_array_ref::__simd_inner_load_array_ref:
	.cfi_startproc
	vmovups ymm0, ymmword ptr [rsi]
	vmovaps ymmword ptr [rdi], ymm0
	vzeroupper
	ret
