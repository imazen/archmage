# Source wasm/math_matrix_study-ba178c59a0ecbf47.s:59672
# Exact symbol _RNCNvNtCsdrsRtign2uv_17math_matrix_study9algebraic9cases_f32sd_0B5_
_RNCNvNtCsdrsRtign2uv_17math_matrix_study9algebraic9cases_f32sd_0B5_:
	.functype	_RNCNvNtCsdrsRtign2uv_17math_matrix_study9algebraic9cases_f32sd_0B5_ (i32, i32, i32, i32, i32, i32, i32) -> ()
	.local  	i32, v128, v128
	global.get	__stack_pointer
	i32.const	16
	i32.sub 
	local.tee	7
	global.set	__stack_pointer
	local.get	7
	local.get	2
	i32.store	4
	local.get	7
	local.get	4
	i32.store	12
	block   	
	block   	
	block   	
	block   	
	local.get	2
	local.get	4
	i32.ne  
	br_if   	0
	local.get	7
	local.get	6
	i32.store	8
	local.get	7
	local.get	2
	i32.store	12
	local.get	2
	local.get	6
	i32.ne  
	br_if   	1
	local.get	7
	local.get	2
	i32.const	7
	i32.and 
	local.tee	4
	i32.store	12
	local.get	4
	br_if   	2
	local.get	2
	i32.eqz
	br_if   	3
	local.get	2
	i32.const	3
	i32.shr_u
	local.set	2
	v128.const	0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee	8
	local.set	9
.LBB95_5:
	loop    	
	local.get	3
	i32.const	16
	i32.add 
	v128.load	0:p2align=2
	local.get	1
	i32.const	16
	i32.add 
	v128.load	0:p2align=2
	f32x4.mul
	local.get	9
	f32x4.add
	local.set	9
	local.get	3
	v128.load	0:p2align=2
	local.get	1
	v128.load	0:p2align=2
	f32x4.mul
	local.get	8
	f32x4.add
	local.set	8
	local.get	1
	i32.const	32
	i32.add 
	local.set	1
	local.get	3
	i32.const	32
	i32.add 
	local.set	3
	local.get	2
	i32.const	-1
	i32.add 
	local.tee	2
	br_if   	0
	end_loop
	local.get	5
	local.get	8
	local.get	9
	f32x4.add
	local.tee	8
	local.get	8
	local.get	8
	i8x16.shuffle	4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 0, 1, 2, 3
	f32x4.add
	local.tee	8
	local.get	8
	local.get	8
	i8x16.shuffle	8, 9, 10, 11, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
	f32x4.add
	f32x4.extract_lane	0
	f32.store	0
	local.get	7
	i32.const	16
	i32.add 
	global.set	__stack_pointer
	return
.LBB95_7:
	end_block
	i32.const	0
	local.get	7
	i32.const	4
	i32.add 
	local.get	7
	i32.const	12
	i32.add 
	i32.const	0
	local.get	3
	i32.const	.Lalloc_6a1705bd31e826403e1495471f07f59e
	call	_RINvNtCsjyCYk6C1TV_4core9panicking13assert_failedjjEB4_
	unreachable
.LBB95_8:
	end_block
	i32.const	0
	local.get	7
	i32.const	12
	i32.add 
	local.get	7
	i32.const	8
	i32.add 
	i32.const	0
	local.get	3
	i32.const	.Lalloc_bb34989c33e6fea963cdcf7e7fa33c3f
	call	_RINvNtCsjyCYk6C1TV_4core9panicking13assert_failedjjEB4_
	unreachable
.LBB95_9:
	end_block
	i32.const	0
	local.get	7
	i32.const	12
	i32.add 
	i32.const	.Lalloc_83ea17bf0c4f4a5a5a13d3ae7955acd0
	i32.const	0
	local.get	3
	i32.const	.Lalloc_4dff0c623996dc0b3334d147f8fabd3b
	call	_RINvNtCsjyCYk6C1TV_4core9panicking13assert_failedjjEB4_
	unreachable
.LBB95_10:
	end_block
	i32.const	0
	i32.const	0
	i32.const	.Lalloc_3a08169e9ab56adfd1f18932e1d57146
	call	_RNvNtCsjyCYk6C1TV_4core9panicking18panic_bounds_check
	unreachable
	end_function
