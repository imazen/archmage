.section .text.asm_patterns::x86_impl::__arcane_load_first_chunk_256,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_load_first_chunk_256,@function
asm_patterns::x86_impl::__arcane_load_first_chunk_256:
	.cfi_startproc
	cmp rdx, 7
	jbe .LBB25_2
	vmovups ymm0, ymmword ptr [rsi]
	vmovaps ymmword ptr [rdi], ymm0
	vzeroupper
	ret
.LBB25_2:
	push rax
	.cfi_def_cfa_offset 16
	lea rdi, [rip + .Lanon.bbb2f47a6bbb2a68548d5b0cc4d77b31.16]
	call qword ptr [rip + core::option::unwrap_failed@GOTPCREL]
