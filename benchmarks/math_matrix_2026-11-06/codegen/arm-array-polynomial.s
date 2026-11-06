# Source arm/math_matrix_study-107ee31fd7b8ef38.s:2272
# Exact symbol _RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32s9_0B5_
_RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32s9_0B5_:
	.cfi_startproc
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	str	d8, [sp, #16]
	stp	x29, x30, [sp, #32]
	add	x29, sp, #32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset b8, -32
	.cfi_remember_state
	cmp	x2, x4
	stur	x4, [x29, #-8]
	str	x2, [sp]
	b.ne	.LBB16_12
	cmp	x2, x6
	str	x2, [sp, #8]
	stur	x6, [x29, #-8]
	b.ne	.LBB16_13
	ands	x8, x2, #0x7
	stur	x8, [x29, #-8]
	b.ne	.LBB16_15
	cbz	x2, .LBB16_11
	lsr	x8, x2, #3
	cmp	x2, #31
	b.hi	.LBB16_6
	mov	x9, xzr
	b	.LBB16_9
.LBB16_6:
	mov	w9, #43691
	mov	x10, x5
	mov	x11, x1
	movk	w9, #15914, lsl #16
	and	x12, x8, #0x3fffffffffffffc
	dup	v0.4s, w9
	and	x9, x8, #0x3fffffffffffffc
.LBB16_7:
	movi	v17.4s, #63, lsl #24
	movi	v19.4s, #63, lsl #24
	subs	x12, x12, #4
	ldp	q16, q18, [x11, #96]
	movi	v5.4s, #63, lsl #24
	movi	v6.4s, #63, lsl #24
	movi	v20.4s, #63, lsl #24
	movi	v21.4s, #63, lsl #24
	movi	v23.4s, #63, lsl #24
	movi	v24.4s, #63, lsl #24
	fmov	v25.4s, #1.00000000
	fmla	v17.4s, v18.4s, v0.4s
	fmla	v19.4s, v16.4s, v0.4s
	fmov	v26.4s, #1.00000000
	ldp	q1, q2, [x11]
	fmov	v27.4s, #1.00000000
	ldp	q3, q4, [x11, #32]
	fmov	v28.4s, #1.00000000
	ldp	q7, q22, [x11, #64]
	fmov	v29.4s, #1.00000000
	fmla	v5.4s, v1.4s, v0.4s
	fmla	v6.4s, v2.4s, v0.4s
	fmov	v30.4s, #1.00000000
	fmla	v20.4s, v3.4s, v0.4s
	fmla	v23.4s, v4.4s, v0.4s
	fmov	v31.4s, #1.00000000
	fmla	v21.4s, v22.4s, v0.4s
	fmla	v24.4s, v7.4s, v0.4s
	fmov	v8.4s, #1.00000000
	fmla	v25.4s, v17.4s, v18.4s
	fmov	v17.4s, #1.00000000
	fmla	v26.4s, v19.4s, v16.4s
	fmov	v19.4s, #1.00000000
	add	x11, x11, #128
	fmla	v29.4s, v23.4s, v4.4s
	fmla	v30.4s, v20.4s, v3.4s
	fmla	v27.4s, v21.4s, v22.4s
	fmov	v21.4s, #1.00000000
	fmla	v28.4s, v24.4s, v7.4s
	fmov	v24.4s, #1.00000000
	fmla	v17.4s, v25.4s, v18.4s
	fmov	v18.4s, #1.00000000
	fmla	v19.4s, v26.4s, v16.4s
	fmov	v16.4s, #1.00000000
	fmla	v31.4s, v6.4s, v2.4s
	fmov	v6.4s, #1.00000000
	fmla	v8.4s, v5.4s, v1.4s
	fmov	v5.4s, #1.00000000
	fmla	v21.4s, v27.4s, v22.4s
	fmla	v24.4s, v28.4s, v7.4s
	fmla	v18.4s, v29.4s, v4.4s
	fmla	v16.4s, v30.4s, v3.4s
	stp	q19, q17, [x10, #96]
	fmla	v6.4s, v31.4s, v2.4s
	fmla	v5.4s, v8.4s, v1.4s
	stp	q24, q21, [x10, #64]
	stp	q16, q18, [x10, #32]
	stp	q5, q6, [x10], #128
	b.ne	.LBB16_7
	cmp	x8, x9
	b.eq	.LBB16_11
.LBB16_9:
	mov	w11, #43691
	mov	w10, #16
	sub	x8, x8, x9
	movk	w11, #15914, lsl #16
	orr	x12, x10, x9, lsl #5
	dup	v0.4s, w11
	add	x10, x1, x12
	add	x11, x5, x12
.LBB16_10:
	movi	v1.4s, #63, lsl #24
	movi	v2.4s, #63, lsl #24
	subs	x8, x8, #1
	ldp	q3, q4, [x10, #-16]
	fmov	v5.4s, #1.00000000
	fmov	v6.4s, #1.00000000
	add	x10, x10, #32
	fmla	v1.4s, v3.4s, v0.4s
	fmla	v2.4s, v4.4s, v0.4s
	fmla	v5.4s, v1.4s, v3.4s
	fmov	v1.4s, #1.00000000
	fmla	v6.4s, v2.4s, v4.4s
	fmov	v2.4s, #1.00000000
	fmla	v1.4s, v5.4s, v3.4s
	fmla	v2.4s, v6.4s, v4.4s
	stp	q1, q2, [x11, #-16]
	add	x11, x11, #32
	b.ne	.LBB16_10
.LBB16_11:
	.cfi_def_cfa wsp, 48
	ldp	x29, x30, [sp, #32]
	ldr	d8, [sp, #16]
	add	sp, sp, #48
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	ret
.LBB16_12:
	.cfi_restore_state
	adrp	x5, .Lalloc_5a34f9299c065b0902f76e5a7a205f93
	add	x5, x5, :lo12:.Lalloc_5a34f9299c065b0902f76e5a7a205f93
	mov	x1, sp
	b	.LBB16_14
.LBB16_13:
	adrp	x5, .Lalloc_ac4a105c14647a3841f29e521af59ec7
	add	x5, x5, :lo12:.Lalloc_ac4a105c14647a3841f29e521af59ec7
	add	x1, sp, #8
.LBB16_14:
	sub	x2, x29, #8
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.LBB16_15:
	adrp	x2, .Lalloc_53973d2fe29b4adba8bb7390b5678745
	add	x2, x2, :lo12:.Lalloc_53973d2fe29b4adba8bb7390b5678745
	adrp	x5, .Lalloc_809008a91022f932ff624f4d6cd759cf
	add	x5, x5, :lo12:.Lalloc_809008a91022f932ff624f4d6cd759cf
	sub	x1, x29, #8
	mov	w0, wzr
	mov	x3, xzr
	bl	_RINvNtCslpfLnFkgLmC_4core9panicking13assert_failedjjEB4_
.Lfunc_end16:
