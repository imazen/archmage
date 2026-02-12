.section .text.asm_patterns::load_first_chunk::__simd_inner_load_first_chunk,"ax",@progbits
	.p2align	4
.type	asm_patterns::load_first_chunk::__simd_inner_load_first_chunk,@function
asm_patterns::load_first_chunk::__simd_inner_load_first_chunk:
	.cfi_startproc
	cmp rdx, 7
	jbe .LBB40_2
	vmovups ymm0, ymmword ptr [rsi]
	vmovaps ymmword ptr [rdi], ymm0
	vzeroupper
	ret
.LBB40_2:
	push rax
	.cfi_def_cfa_offset 16
	lea rdi, [rip + .Lanon.5a5f59e739c2d80c2f4dd981e804d024.31]
	call qword ptr [rip + core::option::unwrap_failed@GOTPCREL]
