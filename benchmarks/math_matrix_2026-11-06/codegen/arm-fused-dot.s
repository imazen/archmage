# Source arm/math_matrix_study-107ee31fd7b8ef38.s:3260
# Exact symbol _RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32sh_0B5_
_RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32sh_0B5_:
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
	b.ne	.LBB24_7
	cmp	x2, x6
	stur	x2, [x29, #-8]
	str	x6, [sp, #16]
	b.ne	.LBB24_8
	ands	x8, x2, #0x7
	stur	x8, [x29, #-8]
	b.ne	.LBB24_9
	cbz	x2, .LBB24_10
	movi	v0.2d, #0000000000000000
	movi	v1.2d, #0000000000000000
	lsr	x8, x2, #3
	add	x9, x3, #16
	add	x10, x1, #16
.LBB24_5:
	ldp	q2, q3, [x10, #-16]
	subs	x8, x8, #1
	ldp	q4, q5, [x9, #-16]
	add	x9, x9, #32
	add	x10, x10, #32
	fmla	v1.4s, v4.4s, v2.4s
	fmla	v0.4s, v5.4s, v3.4s
	b.ne	.LBB24_5
	fadd	v0.4s, v1.4s, v0.4s
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
.LBB24_7:
	.cfi_restore_state
	adrp	x5, .Lalloc_e576b96f0dc2e4784c706c64e47fe7a7
	add	x5, x5, :lo12:.Lalloc_e576b96f0dc2e4784c706c64e47fe7a7
	add	x1, sp, #8
	sub	x2, x29, #8
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.LBB24_8:
	adrp	x5, .Lalloc_8d828d968891132c8ea180c21fc45803
	add	x5, x5, :lo12:.Lalloc_8d828d968891132c8ea180c21fc45803
	sub	x1, x29, #8
	add	x2, sp, #16
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.LBB24_9:
	adrp	x2, .Lalloc_53973d2fe29b4adba8bb7390b5678745
	add	x2, x2, :lo12:.Lalloc_53973d2fe29b4adba8bb7390b5678745
	adrp	x5, .Lalloc_373cc5cc9bb30c07aa345d65d85e96f9
	add	x5, x5, :lo12:.Lalloc_373cc5cc9bb30c07aa345d65d85e96f9
	sub	x1, x29, #8
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.LBB24_10:
	adrp	x2, .Lalloc_6b57d84239c40a257a81522c86edee99
	add	x2, x2, :lo12:.Lalloc_6b57d84239c40a257a81522c86edee99
	mov	x0, xzr
	mov	x1, xzr
	bl	_RNvNtCslpfLnFkgLmC_4core9panicking18panic_bounds_check
.Lfunc_end24:
