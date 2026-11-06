# Source r7900x/math_matrix_study-d62ccc0b1a7ac227.s:161799
# Exact symbol _RNvNtCsl8sLjJBmgi8_17math_matrix_study9algebraic30___attune_dot_algebraic_f64_v4x
_RNvNtCsl8sLjJBmgi8_17math_matrix_study9algebraic30___attune_dot_algebraic_f64_v4x:
	.cfi_startproc
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movq	%rsi, 8(%rsp)
	movq	%rcx, (%rsp)
	cmpq	%rcx, %rsi
	jne	.LBB1015_1
	movq	%rsi, (%rsp)
	movq	%r9, 16(%rsp)
	cmpq	%r9, %rsi
	jne	.LBB1015_4
	movq	%rsi, %rax
	andq	$3, %rax
	movq	%rax, (%rsp)
	jne	.LBB1015_16
	testq	%rsi, %rsi
	je	.LBB1015_19
	cmpq	$32, %rsi
	jae	.LBB1015_9
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%eax, %eax
	jmp	.LBB1015_13
.LBB1015_9:
	movabsq	$1152921504606846944, %rax
	andq	%rsi, %rax
	movq	%rsi, %rcx
	andq	$-32, %rcx
	shlq	$3, %rcx
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%r9d, %r9d
	vxorpd	%xmm1, %xmm1, %xmm1
	vxorpd	%xmm2, %xmm2, %xmm2
	vxorpd	%xmm3, %xmm3, %xmm3
	.p2align	4
.LBB1015_10:
	vmovupd	(%rdx,%r9), %zmm4
	vmovupd	64(%rdx,%r9), %zmm5
	vmovupd	128(%rdx,%r9), %zmm6
	vmovupd	192(%rdx,%r9), %zmm7
	vfmadd231pd	(%rdi,%r9), %zmm4, %zmm0
	vfmadd231pd	64(%rdi,%r9), %zmm5, %zmm1
	vfmadd231pd	128(%rdi,%r9), %zmm6, %zmm2
	vfmadd231pd	192(%rdi,%r9), %zmm7, %zmm3
	addq	$256, %r9
	cmpq	%r9, %rcx
	jne	.LBB1015_10
	vaddpd	%zmm0, %zmm1, %zmm0
	vaddpd	%zmm2, %zmm3, %zmm1
	vaddpd	%zmm0, %zmm1, %zmm0
	vextractf64x4	$1, %zmm0, %ymm1
	vaddpd	%zmm1, %zmm0, %zmm0
	vextractf128	$1, %ymm0, %xmm1
	vaddpd	%xmm1, %xmm0, %xmm0
	vshufpd	$1, %xmm0, %xmm0, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	cmpq	%rax, %rsi
	je	.LBB1015_18
	testb	$28, %sil
	je	.LBB1015_17
.LBB1015_13:
	vmovq	%xmm0, %xmm0
	.p2align	4
.LBB1015_14:
	vmovupd	(%rdx,%rax,8), %ymm1
	vfmadd231pd	(%rdi,%rax,8), %ymm1, %ymm0
	addq	$4, %rax
	cmpq	%rax, %rsi
	jne	.LBB1015_14
	vextractf128	$1, %ymm0, %xmm1
	vaddpd	%xmm1, %xmm0, %xmm0
	vshufpd	$1, %xmm0, %xmm0, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
.LBB1015_18:
	vmovsd	%xmm0, (%r8)
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
	.p2align	4
.LBB1015_17:
	.cfi_def_cfa_offset 32
	vmovsd	(%rdx,%rax,8), %xmm1
	vfmadd231sd	(%rdi,%rax,8), %xmm1, %xmm0
	incq	%rax
	cmpq	%rax, %rsi
	jne	.LBB1015_17
	jmp	.LBB1015_18
.LBB1015_1:
	leaq	.Lalloc_b3aec13970de27901a79e7459996f9a4(%rip), %r9
	leaq	8(%rsp), %rsi
	movq	%rsp, %rdx
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	callq	*_RINvNtCskADBYZwJmSU_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
.LBB1015_4:
	leaq	.Lalloc_25e4573d996386c71a2f8db6a9cde4e7(%rip), %r9
	movq	%rsp, %rsi
	leaq	16(%rsp), %rdx
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	callq	*_RINvNtCskADBYZwJmSU_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
.LBB1015_16:
	leaq	.Lalloc_53973d2fe29b4adba8bb7390b5678745(%rip), %rdx
	leaq	.Lalloc_80e3c6a1b4f9984bf15187564d9e1df5(%rip), %r9
	movq	%rsp, %rsi
	xorl	%edi, %edi
	xorl	%ecx, %ecx
	callq	*_RINvNtCskADBYZwJmSU_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
.LBB1015_19:
	leaq	.Lalloc_4fcf08b97b127104b2fbb1fdd0ed1d1a(%rip), %rdx
	xorl	%edi, %edi
	xorl	%esi, %esi
	callq	*_RNvNtCskADBYZwJmSU_4core9panicking18panic_bounds_check@GOTPCREL(%rip)
.Lfunc_end1015:
