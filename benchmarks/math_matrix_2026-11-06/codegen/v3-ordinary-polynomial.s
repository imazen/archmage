# Source r7900x/math_matrix_study-d62ccc0b1a7ac227.s:160416
# Exact symbol _RNvNtCsl8sLjJBmgi8_17math_matrix_study9algebraic27___attune_poly_vector_f32_v3
_RNvNtCsl8sLjJBmgi8_17math_matrix_study9algebraic27___attune_poly_vector_f32_v3:
	.cfi_startproc
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movq	%rsi, 8(%rsp)
	movq	%rdx, (%rsp)
	cmpq	%rdx, %rsi
	jne	.LBB1002_1
	movq	%rsi, 16(%rsp)
	movq	%r8, (%rsp)
	cmpq	%r8, %rsi
	jne	.LBB1002_5
	movq	%rsi, %rax
	andq	$7, %rax
	movq	%rax, (%rsp)
	jne	.LBB1002_10
	testq	%rsi, %rsi
	je	.LBB1002_15
	cmpq	$8, %rsi
	jne	.LBB1002_11
	xorl	%eax, %eax
	jmp	.LBB1002_14
.LBB1002_11:
	movq	%rsi, %rax
	shrq	$3, %rax
	movabsq	$288230376151711742, %rdx
	andq	%rax, %rdx
	movl	$32, %r8d
	xorl	%eax, %eax
	vbroadcastss	.LCPI1002_0(%rip), %ymm0
	vbroadcastss	.LCPI1002_1(%rip), %ymm1
	vbroadcastss	.LCPI1002_2(%rip), %ymm2
	.p2align	4
.LBB1002_12:
	vmovups	-32(%rdi,%r8), %ymm3
	vmulps	%ymm0, %ymm3, %ymm4
	vaddps	%ymm1, %ymm4, %ymm4
	vmulps	%ymm4, %ymm3, %ymm4
	vaddps	%ymm2, %ymm4, %ymm4
	vmulps	%ymm4, %ymm3, %ymm3
	vaddps	%ymm2, %ymm3, %ymm3
	vmovups	%ymm3, -32(%rcx,%r8)
	addq	$2, %rax
	vmovups	(%rdi,%r8), %ymm3
	vmulps	%ymm0, %ymm3, %ymm4
	vaddps	%ymm1, %ymm4, %ymm4
	vmulps	%ymm4, %ymm3, %ymm4
	vaddps	%ymm2, %ymm4, %ymm4
	vmulps	%ymm4, %ymm3, %ymm3
	vaddps	%ymm2, %ymm3, %ymm3
	vmovups	%ymm3, (%rcx,%r8)
	addq	$64, %r8
	cmpq	%rax, %rdx
	jne	.LBB1002_12
	testb	$8, %sil
	je	.LBB1002_15
.LBB1002_14:
	shlq	$5, %rax
	vbroadcastss	.LCPI1002_0(%rip), %ymm0
	vmovups	(%rdi,%rax), %ymm1
	vmulps	%ymm0, %ymm1, %ymm0
	vbroadcastss	.LCPI1002_1(%rip), %ymm2
	vaddps	%ymm2, %ymm0, %ymm0
	vmulps	%ymm0, %ymm1, %ymm0
	vbroadcastss	.LCPI1002_2(%rip), %ymm2
	vaddps	%ymm2, %ymm0, %ymm0
	vmulps	%ymm0, %ymm1, %ymm0
	vaddps	%ymm2, %ymm0, %ymm0
	vmovups	%ymm0, (%rcx,%rax)
.LBB1002_15:
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.LBB1002_1:
	.cfi_def_cfa_offset 32
	leaq	.Lalloc_ec86ae65837b1d217f22476a567a127a(%rip), %r9
	leaq	8(%rsp), %rsi
	jmp	.LBB1002_2
.LBB1002_5:
	leaq	.Lalloc_b954a05a288705bdae6e8d345172b0fc(%rip), %r9
	leaq	16(%rsp), %rsi
.LBB1002_2:
	movq	%rsp, %rdx
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	callq	*_RINvNtCskADBYZwJmSU_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
.LBB1002_10:
	leaq	.Lalloc_53973d2fe29b4adba8bb7390b5678745(%rip), %rdx
	leaq	.Lalloc_e377e4691427651d98bcb1a720c99fef(%rip), %r9
	movq	%rsp, %rsi
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	callq	*_RINvNtCskADBYZwJmSU_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
.Lfunc_end1002:
