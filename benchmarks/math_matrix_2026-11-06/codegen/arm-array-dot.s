# Source arm/math_matrix_study-107ee31fd7b8ef38.s:2768
# Exact symbol _RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32sd_0B5_
_RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32sd_0B5_:
	.cfi_startproc
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	stp	x29, x30, [sp, #32]
	add	x29, sp, #32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	cmp	x2, x4
	str	x2, [sp, #8]
	stur	x4, [x29, #-8]
	b.ne	.LBB20_12
	cmp	x2, x6
	stur	x2, [x29, #-8]
	str	x6, [sp, #16]
	b.ne	.LBB20_13
	ands	x8, x2, #0x7
	stur	x8, [x29, #-8]
	b.ne	.LBB20_14
	cbz	x2, .LBB20_15
	lsr	x8, x2, #3
	cmp	x2, #15
	b.hi	.LBB20_6
	movi	v4.2d, #0000000000000000
	movi	v5.2d, #0000000000000000
	mov	x9, xzr
	b	.LBB20_9
.LBB20_6:
	movi	d0, #0000000000000000
	movi	d2, #0000000000000000
	and	x9, x8, #0x3fffffffffffffe
	movi	d4, #0000000000000000
	movi	d5, #0000000000000000
	mov	x10, x1
	movi	d1, #0000000000000000
	movi	d3, #0000000000000000
	mov	x11, x3
	movi	d6, #0000000000000000
	movi	d7, #0000000000000000
	and	x12, x8, #0x3fffffffffffffe
.LBB20_7:
	ldp	q16, q17, [x10, #32]
	subs	x12, x12, #2
	ldp	q18, q22, [x11, #32]
	ldp	q19, q23, [x11], #64
	ldp	q20, q21, [x10], #64
	fmul	v16.4s, v18.4s, v16.4s
	fmul	v17.4s, v22.4s, v17.4s
	fmul	v19.4s, v19.4s, v20.4s
	fmul	v18.4s, v23.4s, v21.4s
	uzp2	v20.4s, v19.4s, v16.4s
	uzp2	v21.4s, v18.4s, v17.4s
	zip1	v22.2s, v19.2s, v16.2s
	zip2	v23.2s, v19.2s, v16.2s
	zip2	v16.4s, v19.4s, v16.4s
	zip1	v24.2s, v18.2s, v17.2s
	uzp2	v19.4s, v20.4s, v19.4s
	zip2	v20.2s, v18.2s, v17.2s
	zip2	v17.4s, v18.4s, v17.4s
	uzp2	v18.4s, v21.4s, v18.4s
	fadd	v7.2s, v22.2s, v7.2s
	fadd	v6.2s, v23.2s, v6.2s
	fadd	v3.2s, v16.2s, v3.2s
	fadd	v5.2s, v24.2s, v5.2s
	fadd	v1.2s, v19.2s, v1.2s
	fadd	v4.2s, v20.2s, v4.2s
	fadd	v2.2s, v17.2s, v2.2s
	fadd	v0.2s, v18.2s, v0.2s
	b.ne	.LBB20_7
	faddp	s16, v4.2s
	faddp	s5, v5.2s
	cmp	x8, x9
	faddp	s6, v6.2s
	faddp	s4, v7.2s
	faddp	s2, v2.2s
	faddp	s3, v3.2s
	faddp	s0, v0.2s
	faddp	s1, v1.2s
	mov	v5.s[1], v16.s[0]
	mov	v4.s[1], v6.s[0]
	mov	v5.s[2], v2.s[0]
	mov	v4.s[2], v3.s[0]
	mov	v5.s[3], v0.s[0]
	mov	v4.s[3], v1.s[0]
	b.eq	.LBB20_11
.LBB20_9:
	lsl	x11, x9, #5
	sub	x8, x8, x9
	add	x10, x3, x11
	add	x11, x1, x11
.LBB20_10:
	ldp	q1, q0, [x11], #32
	ldp	q3, q2, [x10], #32
	subs	x8, x8, #1
	fmla	v5.4s, v2.4s, v0.4s
	fmla	v4.4s, v3.4s, v1.4s
	b.ne	.LBB20_10
.LBB20_11:
	fadd	v0.4s, v4.4s, v5.4s
	faddp	v0.4s, v0.4s, v0.4s
	faddp	s0, v0.2s
	str	s0, [x5]
	.cfi_def_cfa wsp, 48
	ldp	x29, x30, [sp, #32]
	add	sp, sp, #48
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB20_12:
	.cfi_restore_state
	adrp	x5, .Lalloc_f4b38bff303ddffba9379adbb22e8fac
	add	x5, x5, :lo12:.Lalloc_f4b38bff303ddffba9379adbb22e8fac
	add	x1, sp, #8
	sub	x2, x29, #8
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.LBB20_13:
	adrp	x5, .Lalloc_d21f18c16577e9b25d27dc69ab63f0e6
	add	x5, x5, :lo12:.Lalloc_d21f18c16577e9b25d27dc69ab63f0e6
	sub	x1, x29, #8
	add	x2, sp, #16
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.LBB20_14:
	adrp	x2, .Lalloc_53973d2fe29b4adba8bb7390b5678745
	add	x2, x2, :lo12:.Lalloc_53973d2fe29b4adba8bb7390b5678745
	adrp	x5, .Lalloc_23ed1d52e387cf2d33baf4a8e3210577
	add	x5, x5, :lo12:.Lalloc_23ed1d52e387cf2d33baf4a8e3210577
	sub	x1, x29, #8
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.LBB20_15:
	adrp	x2, .Lalloc_a628bfb195b3c315e28e6381e9571022
	add	x2, x2, :lo12:.Lalloc_a628bfb195b3c315e28e6381e9571022
	mov	x0, xzr
	mov	x1, xzr
	bl	_RNvNtCslpfLnFkgLmC_4core9panicking18panic_bounds_check
.Lfunc_end20:
