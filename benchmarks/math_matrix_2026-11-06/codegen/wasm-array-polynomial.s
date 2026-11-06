# Source wasm/math_matrix_study-ba178c59a0ecbf47.s:58522
# Exact symbol _RNCNvNtCsdrsRtign2uv_17math_matrix_study9algebraic9cases_f32s9_0B5_
_RNCNvNtCsdrsRtign2uv_17math_matrix_study9algebraic9cases_f32s9_0B5_:
	.functype	_RNCNvNtCsdrsRtign2uv_17math_matrix_study9algebraic9cases_f32s9_0B5_ (i32, i32, i32, i32, i32, i32, i32) -> ()
	.local  	i32, v128, v128, v128, v128
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
	local.get	2
	local.get	4
	i32.ne  
	br_if   	0
	local.get	7
	local.get	2
	i32.store	8
	local.get	7
	local.get	6
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
	block   	
	local.get	2
	i32.eqz
	br_if   	0
	local.get	2
	i32.const	3
	i32.shr_u
	local.set	2
.LBB90_5:
	loop    	
	local.get	5
	local.get	1
	v128.load	0:p2align=2
	local.tee	8
	v128.const	0x1.555556p-3, 0x1.555556p-3, 0x1.555556p-3, 0x1.555556p-3
	local.tee	9
	f32x4.mul
	v128.const	0x1p-1, 0x1p-1, 0x1p-1, 0x1p-1
	local.tee	10
	f32x4.add
	local.get	8
	f32x4.mul
	v128.const	0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee	11
	f32x4.add
	local.get	8
	f32x4.mul
	local.get	11
	f32x4.add
	v128.store	0:p2align=2
	local.get	5
	i32.const	16
	i32.add 
	local.get	1
	i32.const	16
	i32.add 
	v128.load	0:p2align=2
	local.tee	8
	local.get	9
	f32x4.mul
	local.get	10
	f32x4.add
	local.get	8
	f32x4.mul
	local.get	11
	f32x4.add
	local.get	8
	f32x4.mul
	local.get	11
	f32x4.add
	v128.store	0:p2align=2
	local.get	1
	i32.const	32
	i32.add 
	local.set	1
	local.get	5
	i32.const	32
	i32.add 
	local.set	5
	local.get	2
	i32.const	-1
	i32.add 
	local.tee	2
	br_if   	0
.LBB90_6:
	end_loop
	end_block
	local.get	7
	i32.const	16
	i32.add 
	global.set	__stack_pointer
	return
.LBB90_7:
	end_block
	i32.const	0
	local.get	7
	i32.const	4
	i32.add 
	local.get	7
	i32.const	12
	i32.add 
	i32.const	0
	local.get	5
	i32.const	.Lalloc_3513d8af39f78be3b5d08186db0adb84
	call	_RINvNtCsjyCYk6C1TV_4core9panicking13assert_failedjjEB4_
	unreachable
.LBB90_8:
	end_block
	i32.const	0
	local.get	7
	i32.const	8
	i32.add 
	local.get	7
	i32.const	12
	i32.add 
	i32.const	0
	local.get	5
	i32.const	.Lalloc_5b30c4e8771efbffae2d2aba45c50ba6
	call	_RINvNtCsjyCYk6C1TV_4core9panicking13assert_failedjjEB4_
	unreachable
.LBB90_9:
	end_block
	i32.const	0
	local.get	7
	i32.const	12
	i32.add 
	i32.const	.Lalloc_83ea17bf0c4f4a5a5a13d3ae7955acd0
	i32.const	0
	local.get	5
	i32.const	.Lalloc_c23cb066a92e4facc1b4eb78c91edb30
	call	_RINvNtCsjyCYk6C1TV_4core9panicking13assert_failedjjEB4_
	unreachable
	end_function
