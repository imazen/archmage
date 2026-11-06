# Source r7900x/math_matrix_study-d62ccc0b1a7ac227.s:160990
# Exact symbol _RNvNtCsl8sLjJBmgi8_17math_matrix_study9algebraic29___attune_dot_algebraic_f32_v3
_RNvNtCsl8sLjJBmgi8_17math_matrix_study9algebraic29___attune_dot_algebraic_f32_v3:
	.cfi_startproc
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movq	%rsi, 8(%rsp)
	movq	%rcx, (%rsp)
	cmpq	%rcx, %rsi
	jne	.LBB1008_1
	movq	%rsi, (%rsp)
	movq	%r9, 16(%rsp)
	cmpq	%r9, %rsi
	jne	.LBB1008_4
	movq	%rsi, %rax
	andq	$7, %rax
	movq	%rax, (%rsp)
	jne	.LBB1008_16
	testq	%rsi, %rsi
	je	.LBB1008_19
	cmpq	$32, %rsi
	jae	.LBB1008_9
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%eax, %eax
	jmp	.LBB1008_13
.LBB1008_9:
	movabsq	$2305843009213693920, %rax
	andq	%rsi, %rax
	leaq	(,%rsi,4), %rcx
	andq	$-128, %rcx
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%r9d, %r9d
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm2, %xmm2, %xmm2
	vxorps	%xmm3, %xmm3, %xmm3
	.p2align	4
.LBB1008_10:
	vmovups	(%rdx,%r9), %ymm4
	vmovups	32(%rdx,%r9), %ymm5
	vmovups	64(%rdx,%r9), %ymm6
	vmovups	96(%rdx,%r9), %ymm7
	vfmadd231ps	(%rdi,%r9), %ymm4, %ymm0
	vfmadd231ps	32(%rdi,%r9), %ymm5, %ymm1
	vfmadd231ps	64(%rdi,%r9), %ymm6, %ymm2
	vfmadd231ps	96(%rdi,%r9), %ymm7, %ymm3
	subq	$-128, %r9
	cmpq	%r9, %rcx
	jne	.LBB1008_10
	vaddps	%ymm0, %ymm1, %ymm0
	vaddps	%ymm2, %ymm3, %ymm1
	vaddps	%ymm0, %ymm1, %ymm0
	vextractf128	$1, %ymm0, %xmm1
	vaddps	%xmm1, %xmm0, %xmm0
	vshufpd	$1, %xmm0, %xmm0, %xmm1
	vaddps	%xmm1, %xmm0, %xmm0
	vmovshdup	%xmm0, %xmm1
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	%rax, %rsi
	je	.LBB1008_18
	testb	$24, %sil
	je	.LBB1008_17
.LBB1008_13:
	vxorps	%xmm1, %xmm1, %xmm1
	vblendps	$1, %xmm0, %xmm1, %xmm0
	.p2align	4
.LBB1008_14:
	vmovups	(%rdx,%rax,4), %xmm1
	vfmadd231ps	(%rdi,%rax,4), %xmm1, %xmm0
	addq	$4, %rax
	cmpq	%rax, %rsi
	jne	.LBB1008_14
	vshufpd	$1, %xmm0, %xmm0, %xmm1
	vaddps	%xmm1, %xmm0, %xmm0
	vmovshdup	%xmm0, %xmm1
	vaddss	%xmm1, %xmm0, %xmm0
.LBB1008_18:
	vmovss	%xmm0, (%r8)
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
	.p2align	4
.LBB1008_17:
	.cfi_def_cfa_offset 32
	vmovss	(%rdx,%rax,4), %xmm1
	vfmadd231ss	(%rdi,%rax,4), %xmm1, %xmm0
	incq	%rax
	cmpq	%rax, %rsi
	jne	.LBB1008_17
	jmp	.LBB1008_18
.LBB1008_1:
	leaq	.Lalloc_1588954c8aea71d945a6f9763cc6cdfe(%rip), %r9
	leaq	8(%rsp), %rsi
	movq	%rsp, %rdx
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	callq	*_RINvNtCskADBYZwJmSU_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
.LBB1008_4:
	leaq	.Lalloc_0f99138800b46912980550b9cb589245(%rip), %r9
	movq	%rsp, %rsi
	leaq	16(%rsp), %rdx
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	callq	*_RINvNtCskADBYZwJmSU_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
.LBB1008_16:
	leaq	.Lalloc_53973d2fe29b4adba8bb7390b5678745(%rip), %rdx
	leaq	.Lalloc_37c45c22558f47ff37a5feb5b1439d82(%rip), %r9
	movq	%rsp, %rsi
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	callq	*_RINvNtCskADBYZwJmSU_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
.LBB1008_19:
	leaq	.Lalloc_1958a884e0e4eb44cf896dc63a62cea9(%rip), %rdx
	xorl	%edi, %edi
	xorl	%esi, %esi
	callq	*_RNvNtCskADBYZwJmSU_4core9panicking18panic_bounds_check@GOTPCREL(%rip)
.Lfunc_end1008:
