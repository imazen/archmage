.section .text.asm_patterns::x86_impl::__arcane_load_try_into,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_load_try_into,@function
asm_patterns::x86_impl::__arcane_load_try_into:
	.cfi_startproc
	cmp rdx, 7
	jbe .LBB21_2
	vmovups ymm0, ymmword ptr [rsi]
	vmovaps ymmword ptr [rdi], ymm0
	vzeroupper
	ret
.LBB21_2:
	push rax
	.cfi_def_cfa_offset 16
	lea rcx, [rip + .Lanon.bbb2f47a6bbb2a68548d5b0cc4d77b31.13]
	mov esi, 8
	xor edi, edi
	call qword ptr [rip + core::slice::index::slice_index_fail@GOTPCREL]
