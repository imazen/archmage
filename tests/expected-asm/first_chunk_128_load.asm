.section .text.asm_patterns::x86_impl::__arcane_load_first_chunk_128,"ax",@progbits
	.p2align	4
.type	asm_patterns::x86_impl::__arcane_load_first_chunk_128,@function
asm_patterns::x86_impl::__arcane_load_first_chunk_128:
	.cfi_startproc
	cmp rdx, 3
	jbe .LBB24_2
	vmovups xmm0, xmmword ptr [rsi]
	vmovaps xmmword ptr [rdi], xmm0
	ret
.LBB24_2:
	push rax
	.cfi_def_cfa_offset 16
	lea rdi, [rip + .Lanon.ff48498752e7de0fb7d567d3ea463b08.15]
	call qword ptr [rip + core::option::unwrap_failed@GOTPCREL]
