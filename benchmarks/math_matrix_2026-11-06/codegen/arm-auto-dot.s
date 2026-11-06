# Source arm/math_matrix_study-107ee31fd7b8ef38.s:1371
# Exact symbol _RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32s0_0B5_
_RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32s0_0B5_:
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
	b.ne	.LBB8_16
	cmp	x2, x6
	stur	x2, [x29, #-8]
	str	x6, [sp, #16]
	b.ne	.LBB8_17
	ands	x8, x2, #0x7
	stur	x8, [x29, #-8]
	b.ne	.LBB8_18
	cbz	x2, .LBB8_19
	cmp	x2, #16
	b.hs	.LBB8_6
	movi	d0, #0000000000000000
	mov	x8, xzr
	b	.LBB8_10
.LBB8_6:
	movi	v0.2d, #0000000000000000
	movi	v1.2d, #0000000000000000
	and	x8, x2, #0x1ffffffffffffff0
	movi	v2.2d, #0000000000000000
	movi	v3.2d, #0000000000000000
	add	x9, x1, #32
	add	x10, x3, #32
	and	x11, x2, #0x1ffffffffffffff0
.LBB8_7:
	ldp	q4, q5, [x9, #-32]
	subs	x11, x11, #16
	ldp	q6, q7, [x10, #-32]
	ldp	q16, q17, [x9], #64
	ldp	q18, q19, [x10], #64
	fmla	v0.4s, v6.4s, v4.4s
	fmla	v1.4s, v7.4s, v5.4s
	fmla	v2.4s, v18.4s, v16.4s
	fmla	v3.4s, v19.4s, v17.4s
	b.ne	.LBB8_7
	fadd	v0.4s, v1.4s, v0.4s
	fadd	v1.4s, v3.4s, v2.4s
	subs	x9, x2, x8
	fadd	v0.4s, v1.4s, v0.4s
	faddp	v0.4s, v0.4s, v0.4s
	faddp	s0, v0.2s
	b.eq	.LBB8_13
	tbz	w2, #3, .LBB8_14
.LBB8_10:
	movi	v1.2d, #0000000000000000
	lsl	x10, x8, #2
	sub	x8, x2, x8
	add	x9, x3, x10
	add	x10, x1, x10
	mov	v1.s[0], v0.s[0]
.LBB8_11:
	ldr	q0, [x10], #16
	subs	x8, x8, #4
	ldr	q2, [x9], #16
	fmla	v1.4s, v2.4s, v0.4s
	b.ne	.LBB8_11
	faddp	v0.4s, v1.4s, v1.4s
	faddp	s0, v0.2s
.LBB8_13:
	str	s0, [x5]
	.cfi_def_cfa wsp, 48
	ldp	x29, x30, [sp, #32]
	add	sp, sp, #48
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB8_14:
	.cfi_restore_state
	lsl	x8, x2, #2
	and	x10, x8, #0xffffffffffffffc0
	add	x8, x3, x10
	add	x10, x1, x10
.LBB8_15:
	ldr	s1, [x10], #4
	subs	x9, x9, #1
	ldr	s2, [x8], #4
	fmadd	s0, s2, s1, s0
	b.ne	.LBB8_15
	b	.LBB8_13
.LBB8_16:
	adrp	x5, .Lalloc_1588954c8aea71d945a6f9763cc6cdfe
	add	x5, x5, :lo12:.Lalloc_1588954c8aea71d945a6f9763cc6cdfe
	add	x1, sp, #8
	sub	x2, x29, #8
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.LBB8_17:
	adrp	x5, .Lalloc_0f99138800b46912980550b9cb589245
	add	x5, x5, :lo12:.Lalloc_0f99138800b46912980550b9cb589245
	sub	x1, x29, #8
	add	x2, sp, #16
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.LBB8_18:
	adrp	x2, .Lalloc_53973d2fe29b4adba8bb7390b5678745
	add	x2, x2, :lo12:.Lalloc_53973d2fe29b4adba8bb7390b5678745
	adrp	x5, .Lalloc_37c45c22558f47ff37a5feb5b1439d82
	add	x5, x5, :lo12:.Lalloc_37c45c22558f47ff37a5feb5b1439d82
	sub	x1, x29, #8
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.LBB8_19:
	adrp	x2, .Lalloc_1958a884e0e4eb44cf896dc63a62cea9
	add	x2, x2, :lo12:.Lalloc_1958a884e0e4eb44cf896dc63a62cea9
	mov	x0, xzr
	mov	x1, xzr
	bl	_RNvNtCslpfLnFkgLmC_4core9panicking18panic_bounds_check
.Lfunc_end8:
