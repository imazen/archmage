.section .text.asm_patterns::load_first_chunk_128::__simd_inner_load_first_chunk_128,"ax",@progbits
	.p2align	4
.type	asm_patterns::load_first_chunk_128::__simd_inner_load_first_chunk_128,@function
asm_patterns::load_first_chunk_128::__simd_inner_load_first_chunk_128:
	.cfi_startproc
	cmp rdx, 3
	jbe .LBB43_2
	vmovups xmm0, xmmword ptr [rsi]
	vmovaps xmmword ptr [rdi], xmm0
	ret
.LBB43_2:
	push rax
	.cfi_def_cfa_offset 16
	lea rdi, [rip + .Lanon.5a5f59e739c2d80c2f4dd981e804d024.41]
	call qword ptr [rip + core::option::unwrap_failed@GOTPCREL]
