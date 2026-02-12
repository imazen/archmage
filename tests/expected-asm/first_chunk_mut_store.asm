.section .text.asm_patterns::store_first_chunk_mut::__simd_inner_store_first_chunk_mut,"ax",@progbits
	.p2align	4
.type	asm_patterns::store_first_chunk_mut::__simd_inner_store_first_chunk_mut,@function
asm_patterns::store_first_chunk_mut::__simd_inner_store_first_chunk_mut:
	.cfi_startproc
	cmp rdx, 7
	jbe .LBB45_2
	vmovaps ymm0, ymmword ptr [rdi]
	vmovups ymmword ptr [rsi], ymm0
	vzeroupper
	ret
.LBB45_2:
	push rax
	.cfi_def_cfa_offset 16
	lea rdi, [rip + .Lanon.5a5f59e739c2d80c2f4dd981e804d024.42]
	call qword ptr [rip + core::option::unwrap_failed@GOTPCREL]
