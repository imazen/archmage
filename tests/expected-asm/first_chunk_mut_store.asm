.section .text.asm_patterns::x86_impl::__arcane_store_first_chunk_mut,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_store_first_chunk_mut,@function
asm_patterns::x86_impl::__arcane_store_first_chunk_mut:
	.cfi_startproc
	cmp rdx, 7
	jbe .LBB28_2
	vmovaps ymm0, ymmword ptr [rdi]
	vmovups ymmword ptr [rsi], ymm0
	vzeroupper
	ret
.LBB28_2:
	push rax
	.cfi_def_cfa_offset 16
	lea rdi, [rip + .Lanon.ff48498752e7de0fb7d567d3ea463b08.17]
	call qword ptr [rip + core::option::unwrap_failed@GOTPCREL]
