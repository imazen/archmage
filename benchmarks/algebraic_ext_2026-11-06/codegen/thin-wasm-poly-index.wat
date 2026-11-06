;; wasm/timed-codegen/thin.wat:5970
;; Module hash: 7c21873f5b0db82ec579b4ec3240293b5fbbb811f94942d4ecc4cfadc1beb434  wasm-bin/wasm-kernel-thin.wasm
;; 
  (func $_RNvNtCsjbQoMZwQfxY_22algebraic_ext_consumer9generated31___attune_poly_index_f32_wasm128B3_ (;157;) (type 24) (param i32 i32 i32 i32 i32 i32)
    (local i32 v128 v128 v128 v128)
    global.get $__stack_pointer
    i32.const 16
    i32.sub
    local.tee 6
    global.set $__stack_pointer
    local.get 6
    local.get 1
    i32.store offset=4
    local.get 6
    local.get 3
    i32.store offset=12
    block ;; label = @1
      block ;; label = @2
        block ;; label = @3
          local.get 1
          local.get 3
          i32.ne
          br_if 0 (;@3;)
          local.get 6
          local.get 1
          i32.store offset=8
          local.get 6
          local.get 5
          i32.store offset=12
          local.get 1
          local.get 5
          i32.ne
          br_if 1 (;@2;)
          local.get 6
          local.get 1
          i32.const 7
          i32.and
          local.tee 3
          i32.store offset=12
          local.get 3
          br_if 2 (;@1;)
          block ;; label = @4
            local.get 1
            i32.eqz
            br_if 0 (;@4;)
            local.get 1
            i32.const 3
            i32.shr_u
            local.set 1
            loop ;; label = @5
              local.get 4
              local.get 0
              v128.load align=4
              local.tee 7
              v128.const i32x4 0x3e2aaaab 0x3e2aaaab 0x3e2aaaab 0x3e2aaaab
              local.tee 8
              f32x4.mul
              v128.const i32x4 0x3f000000 0x3f000000 0x3f000000 0x3f000000
              local.tee 9
              f32x4.add
              local.get 7
              f32x4.mul
              v128.const i32x4 0x3f800000 0x3f800000 0x3f800000 0x3f800000
              local.tee 10
              f32x4.add
              local.get 7
              f32x4.mul
              local.get 10
              f32x4.add
              v128.store align=4
              local.get 4
              i32.const 16
              i32.add
              local.get 0
              i32.const 16
              i32.add
              v128.load align=4
              local.tee 7
              local.get 8
              f32x4.mul
              local.get 9
              f32x4.add
              local.get 7
              f32x4.mul
              local.get 10
              f32x4.add
              local.get 7
              f32x4.mul
              local.get 10
              f32x4.add
              v128.store align=4
              local.get 4
              i32.const 32
              i32.add
              local.set 4
              local.get 0
              i32.const 32
              i32.add
              local.set 0
              local.get 1
              i32.const -1
              i32.add
              local.tee 1
              br_if 0 (;@5;)
            end
          end
          local.get 6
          i32.const 16
          i32.add
          global.set $__stack_pointer
          return
        end
        i32.const 0
        local.get 6
        i32.const 4
        i32.add
        local.get 6
        i32.const 12
        i32.add
        i32.const 0
        local.get 4
        i32.const 1053708
        call $_RINvNtCsjyCYk6C1TV_4core9panicking13assert_failedjjEB4_
        unreachable
      end
      i32.const 0
      local.get 6
      i32.const 8
      i32.add
      local.get 6
      i32.const 12
      i32.add
      i32.const 0
      local.get 4
      i32.const 1053724
      call $_RINvNtCsjyCYk6C1TV_4core9panicking13assert_failedjjEB4_
      unreachable
    end
    i32.const 0
    local.get 6
    i32.const 12
    i32.add
    i32.const 1052040
    i32.const 0
    local.get 4
    i32.const 1053740
    call $_RINvNtCsjyCYk6C1TV_4core9panicking13assert_failedjjEB4_
    unreachable
  )
