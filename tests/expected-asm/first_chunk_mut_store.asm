.section .text.asm_patterns::x86_impl::__arcane_store_first_chunk_mut,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_store_first_chunk_mut,@function
asm_patterns::x86_impl::__arcane_store_first_chunk_mut:
	.cfi_startproc
	cmp rdx, 7
	jbe .LBB29_2
	vmovaps ymm0, ymmword ptr [rdi]
	vmovups ymmword ptr [rsi], ymm0
	vzeroupper
	ret
.LBB29_2:
	push rax
	.cfi_def_cfa_offset 16
	lea rdi, [rip + .Lanon.bbb2f47a6bbb2a68548d5b0cc4d77b31.17]
	call qword ptr [rip + core::option::unwrap_failed@GOTPCREL]
