# Source r7900x/math_matrix_study-d62ccc0b1a7ac227.s:163972
# Exact symbol _RNvNtCsl8sLjJBmgi8_17math_matrix_study9algebraic36___attune_poly_array_algebraic_f32_v3
_RNvNtCsl8sLjJBmgi8_17math_matrix_study9algebraic36___attune_poly_array_algebraic_f32_v3:
	.cfi_startproc
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movq	%rsi, 8(%rsp)
	movq	%rdx, (%rsp)
	cmpq	%rdx, %rsi
	jne	.LBB1034_1
	movq	%rsi, 16(%rsp)
	movq	%r8, (%rsp)
	cmpq	%r8, %rsi
	jne	.LBB1034_5
	movq	%rsi, %rax
	andq	$7, %rax
	movq	%rax, (%rsp)
	jne	.LBB1034_10
	testq	%rsi, %rsi
	je	.LBB1034_15
	cmpq	$8, %rsi
	jne	.LBB1034_11
	xorl	%eax, %eax
	jmp	.LBB1034_14
.LBB1034_11:
	movq	%rsi, %rax
	shrq	$3, %rax
	movabsq	$288230376151711742, %rdx
	andq	%rax, %rdx
	movl	$32, %r8d
	xorl	%eax, %eax
	vbroadcastss	.LCPI1034_0(%rip), %ymm0
	vbroadcastss	.LCPI1034_1(%rip), %ymm1
	vbroadcastss	.LCPI1034_2(%rip), %ymm2
	.p2align	4
.LBB1034_12:
	vmovups	-32(%rdi,%r8), %ymm3
	vmovaps	%ymm0, %ymm4
	vfmadd213ps	%ymm1, %ymm3, %ymm4
	vfmadd213ps	%ymm2, %ymm3, %ymm4
	vfmadd213ps	%ymm2, %ymm3, %ymm4
	vmovups	%ymm4, -32(%rcx,%r8)
	vmovups	(%rdi,%r8), %ymm3
	vmovaps	%ymm0, %ymm4
	vfmadd213ps	%ymm1, %ymm3, %ymm4
	vfmadd213ps	%ymm2, %ymm3, %ymm4
	vfmadd213ps	%ymm2, %ymm3, %ymm4
	addq	$2, %rax
	vmovups	%ymm4, (%rcx,%r8)
	addq	$64, %r8
	cmpq	%rax, %rdx
	jne	.LBB1034_12
	testb	$8, %sil
	je	.LBB1034_15
.LBB1034_14:
	shlq	$5, %rax
	vmovups	(%rdi,%rax), %ymm0
	vbroadcastss	.LCPI1034_0(%rip), %ymm1
	vbroadcastss	.LCPI1034_1(%rip), %ymm2
	vbroadcastss	.LCPI1034_2(%rip), %ymm3
	vfmadd231ps	%ymm1, %ymm0, %ymm2
	vfmadd213ps	%ymm3, %ymm0, %ymm2
	vfmadd213ps	%ymm3, %ymm0, %ymm2
	vmovups	%ymm2, (%rcx,%rax)
.LBB1034_15:
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.LBB1034_1:
	.cfi_def_cfa_offset 32
	leaq	.Lalloc_5a34f9299c065b0902f76e5a7a205f93(%rip), %r9
	leaq	8(%rsp), %rsi
	jmp	.LBB1034_2
.LBB1034_5:
	leaq	.Lalloc_ac4a105c14647a3841f29e521af59ec7(%rip), %r9
	leaq	16(%rsp), %rsi
.LBB1034_2:
	movq	%rsp, %rdx
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	callq	*_RINvNtCskADBYZwJmSU_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
.LBB1034_10:
	leaq	.Lalloc_53973d2fe29b4adba8bb7390b5678745(%rip), %rdx
	leaq	.Lalloc_809008a91022f932ff624f4d6cd759cf(%rip), %r9
	movq	%rsp, %rsi
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	callq	*_RINvNtCskADBYZwJmSU_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
.Lfunc_end1034:
