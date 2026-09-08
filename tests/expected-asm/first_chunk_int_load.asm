.section .text.asm_patterns::x86_impl::__arcane_load_first_chunk_i,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_load_first_chunk_i,@function
asm_patterns::x86_impl::__arcane_load_first_chunk_i:
	.cfi_startproc
	cmp rdx, 31
	jbe .LBB23_2
	vmovups ymm0, ymmword ptr [rsi]
	vmovaps ymmword ptr [rdi], ymm0
	vzeroupper
	ret
.LBB23_2:
	push rax
	.cfi_def_cfa_offset 16
	lea rdi, [rip + .Lanon.ff48498752e7de0fb7d567d3ea463b08.14]
	call qword ptr [rip + core::option::unwrap_failed@GOTPCREL]
