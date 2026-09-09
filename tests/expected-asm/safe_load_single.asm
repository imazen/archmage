.section .text.safe_memory_overhead::x86_impl::__arcane_safe_load_single,"ax",@progbits
	.p2align	4
.type	safe_memory_overhead::x86_impl::__arcane_safe_load_single,@function
safe_memory_overhead::x86_impl::__arcane_safe_load_single:
	.cfi_startproc
	vmovups ymm0, ymmword ptr [rsi]
	vmovaps ymmword ptr [rdi], ymm0
	vzeroupper
	ret
