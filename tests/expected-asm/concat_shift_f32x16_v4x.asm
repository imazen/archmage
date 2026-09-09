.section .text.asm_patterns::x86_impl::__arcane_concat_shift_f32x16_v4x,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_concat_shift_f32x16_v4x,@function
asm_patterns::x86_impl::__arcane_concat_shift_f32x16_v4x:
	.cfi_startproc
	vmovaps zmm0, zmmword ptr [rsi]
	vmovaps zmm1, zmmword ptr [rip + .LCPI31_0]
	vpermi2ps zmm1, zmm0, zmmword ptr [rdx]
	vmovaps zmmword ptr [rdi], zmm1
	vzeroupper
	ret
