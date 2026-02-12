.section .text.asm_patterns::load_try_into::__simd_inner_load_try_into,"ax",@progbits
	.p2align	4
.type	asm_patterns::load_try_into::__simd_inner_load_try_into,@function
asm_patterns::load_try_into::__simd_inner_load_try_into:
	.cfi_startproc
	cmp rdx, 7
	jbe .LBB38_2
	vmovups ymm0, ymmword ptr [rsi]
	vmovaps ymmword ptr [rdi], ymm0
	vzeroupper
	ret
.LBB38_2:
	push rax
	.cfi_def_cfa_offset 16
	lea rcx, [rip + .Lanon.5a5f59e739c2d80c2f4dd981e804d024.30]
	mov esi, 8
	xor edi, edi
	call qword ptr [rip + core::slice::index::slice_index_fail@GOTPCREL]
