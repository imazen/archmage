//! Timer-free guest; symbol discovery/export is handled by build_guest.py.
#![forbid(unsafe_code)]
#[cfg(target_arch = "wasm32")]
#[path = "../policy_check.rs"]
mod check;
#[cfg(target_arch = "wasm32")]
#[path = "../policy_workload.rs"]
mod workload;
#[cfg(target_arch = "wasm32")]
mod guest {
    use crate::workload::*;
    use archmage::SimdToken;
    use std::sync::{Mutex, OnceLock};
    struct State {
        a32: [f32; 4096],
        b32: [f32; 4096],
        o32: [f32; 4096],
        a64: [f64; 4096],
        b64: [f64; 4096],
        o64: [f64; 4096],
    }
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    pub extern "C" fn guest_prepare(op: i32, kind: i32) {
        assert!((0..6).contains(&op) && (0..4).contains(&kind));
        assert!(kind != 3 || op < 4, "zero clamps are min/max only");
        let mut s = STATE
            .get_or_init(|| {
                Mutex::new(State {
                    a32: [0.; 4096],
                    b32: [0.; 4096],
                    o32: [0.; 4096],
                    a64: [0.; 4096],
                    b64: [0.; 4096],
                    o64: [0.; 4096],
                })
            })
            .lock()
            .unwrap();
        let n = if kind == 1 { 4096 } else { 64 };
        for i in 0..n {
            if op == 2 || op == 3 {
                let (a, b) = input64(i, kind == 3);
                s.a64[i] = a;
                s.b64[i] = b;
            } else {
                let (a, b) = input(i, kind == 3, op >= 4, op == 5);
                s.a32[i] = a;
                s.b32[i] = b;
            }
        }
    }
    pub extern "C" fn guest_probe() -> i32 {
        use magetypes::simd::generic::f32x4 as V;
        let t = archmage::Wasm128Token::summon().unwrap();
        let a = V::splat_t(t, std::hint::black_box(1.0 + 1.0 / 4096.0));
        let c = V::splat_t(t, std::hint::black_box(-(1.0 + 1.0 / 2048.0)));
        i32::from(a.mul_add_relaxed(a, c).to_array()[0] == 1.0 / 16777216.0)
    }
    macro_rules! functions {
        ($exec:ident,$verify:ident,$select:ident,$dispatch:ident,$stream:ident,$chain:ident,$f:ty,$w:literal,$a:ident,$b:ident,$out:ident,$rc:literal) => {
            fn $exec<
                const O: u32,
                const OP: bool,
                const L: bool,
                const N: usize,
                const CHAIN: bool,
            >(
                s: &mut State,
                reps: i32,
            ) {
                let t = archmage::Wasm128Token::summon().unwrap();
                let State {
                    $a: a,
                    $b: b,
                    $out: out,
                    ..
                } = s;
                let a: &[$f; N] = a[..N].try_into().unwrap();
                let b: &[$f; N] = b[..N].try_into().unwrap();
                let out: &mut [$f; N] = (&mut out[..N]).try_into().unwrap();
                for _ in 0..reps {
                    if CHAIN {
                        let y = $chain::<O, OP, L>(t, std::hint::black_box(0.75));
                        out[..$w].copy_from_slice(&std::hint::black_box(y));
                    } else {
                        $stream::<O, OP, L, N>(
                            t,
                            std::hint::black_box(a),
                            std::hint::black_box(b),
                            std::hint::black_box(&mut *out),
                        );
                        std::hint::black_box(&*out);
                    }
                }
            }
            fn $verify<
                const O: u32,
                const OP: bool,
                const L: bool,
                const N: usize,
                const CHAIN: bool,
            >(
                s: &State,
            ) -> f64 {
                let t = archmage::Wasm128Token::summon().unwrap();
                let a: [$f; N] = s.$a[..N].try_into().unwrap();
                let b: [$f; N] = s.$b[..N].try_into().unwrap();
                crate::check::verify::<$f, N, $w>(
                    O,
                    L,
                    $rc,
                    OP,
                    CHAIN,
                    &(a, b),
                    |a, b, out| $stream::<O, OP, L, N>(t, a, b, out),
                    |a, b, out| $stream::<O, OP, L, $w>(t, a, b, out),
                    |x| $chain::<O, OP, L>(t, x),
                )
            }
            fn $select<const O: u32, const L: bool>(
                s: &mut State,
                op: bool,
                kind: i32,
                reps: i32,
                verify: bool,
            ) -> f64 {
                macro_rules! call {
                    ($op:literal,$n:literal,$dep:literal) => {{
                        if verify {
                            $verify::<O, $op, L, $n, $dep>(s)
                        } else {
                            $exec::<O, $op, L, $n, $dep>(s, reps);
                            0.0
                        }
                    }};
                }
                match (op, kind) {
                    (false, 0 | 3) => call!(false, 64, false),
                    (true, 0 | 3) => call!(true, 64, false),
                    (false, 1) => call!(false, 4096, false),
                    (true, 1) => call!(true, 4096, false),
                    (false, 2) => call!(false, 64, true),
                    (true, 2) => call!(true, 64, true),
                    _ => panic!("workload"),
                }
            }
            // Dispatcher emitted below, separately per width to bound type checking.
        };
    }
    functions!(
        exec_mm4,
        verify_mm4,
        select_mm4,
        dispatch_mm4,
        stream_mm4_wasm128,
        chain_mm4_wasm128,
        f32,
        4,
        a32,
        b32,
        o32,
        false
    );
    functions!(
        exec_mm8,
        verify_mm8,
        select_mm8,
        dispatch_mm8,
        stream_mm8_wasm128,
        chain_mm8_wasm128,
        f32,
        8,
        a32,
        b32,
        o32,
        false
    );
    functions!(
        exec_mm16,
        verify_mm16,
        select_mm16,
        dispatch_mm16,
        stream_mm16_wasm128,
        chain_mm16_wasm128,
        f32,
        16,
        a32,
        b32,
        o32,
        false
    );
    functions!(
        exec_mm64x2,
        verify_mm64x2,
        select_mm64x2,
        dispatch_mm64x2,
        stream_mm64x2_wasm128,
        chain_mm64x2_wasm128,
        f64,
        2,
        a64,
        b64,
        o64,
        false
    );
    functions!(
        exec_mm64x4,
        verify_mm64x4,
        select_mm64x4,
        dispatch_mm64x4,
        stream_mm64x4_wasm128,
        chain_mm64x4_wasm128,
        f64,
        4,
        a64,
        b64,
        o64,
        false
    );
    functions!(
        exec_mm64x8,
        verify_mm64x8,
        select_mm64x8,
        dispatch_mm64x8,
        stream_mm64x8_wasm128,
        chain_mm64x8_wasm128,
        f64,
        8,
        a64,
        b64,
        o64,
        false
    );
    functions!(
        exec_rc4,
        verify_rc4,
        select_rc4,
        dispatch_rc4,
        stream_rc4_wasm128,
        chain_rc4_wasm128,
        f32,
        4,
        a32,
        b32,
        o32,
        true
    );
    functions!(
        exec_rc8,
        verify_rc8,
        select_rc8,
        dispatch_rc8,
        stream_rc8_wasm128,
        chain_rc8_wasm128,
        f32,
        8,
        a32,
        b32,
        o32,
        true
    );
    functions!(
        exec_rc16,
        verify_rc16,
        select_rc16,
        dispatch_rc16,
        stream_rc16_wasm128,
        chain_rc16_wasm128,
        f32,
        16,
        a32,
        b32,
        o32,
        true
    );
    fn dispatch_mm4(
        s: &mut State,
        flags: i32,
        op: bool,
        kind: i32,
        reps: i32,
        verify: bool,
    ) -> f64 {
        match flags {
            -1 => select_mm4::<0, true>(s, op, kind, reps, verify),
            0 => select_mm4::<0, false>(s, op, kind, reps, verify),
            1 => select_mm4::<1, false>(s, op, kind, reps, verify),
            2 => select_mm4::<2, false>(s, op, kind, reps, verify),
            4 => select_mm4::<4, false>(s, op, kind, reps, verify),
            8 => select_mm4::<8, false>(s, op, kind, reps, verify),
            16 => select_mm4::<16, false>(s, op, kind, reps, verify),
            17 => select_mm4::<17, false>(s, op, kind, reps, verify),
            20 => select_mm4::<20, false>(s, op, kind, reps, verify),
            24 => select_mm4::<24, false>(s, op, kind, reps, verify),
            33 => select_mm4::<33, false>(s, op, kind, reps, verify),
            34 => select_mm4::<34, false>(s, op, kind, reps, verify),
            49 => select_mm4::<49, false>(s, op, kind, reps, verify),
            64 => select_mm4::<64, false>(s, op, kind, reps, verify),
            65 => select_mm4::<65, false>(s, op, kind, reps, verify),
            66 => select_mm4::<66, false>(s, op, kind, reps, verify),
            68 => select_mm4::<68, false>(s, op, kind, reps, verify),
            72 => select_mm4::<72, false>(s, op, kind, reps, verify),
            80 => select_mm4::<80, false>(s, op, kind, reps, verify),
            81 => select_mm4::<81, false>(s, op, kind, reps, verify),
            84 => select_mm4::<84, false>(s, op, kind, reps, verify),
            88 => select_mm4::<88, false>(s, op, kind, reps, verify),
            97 => select_mm4::<97, false>(s, op, kind, reps, verify),
            98 => select_mm4::<98, false>(s, op, kind, reps, verify),
            113 => select_mm4::<113, false>(s, op, kind, reps, verify),
            _ => panic!("unsupported flags"),
        }
    }
    fn dispatch_mm8(
        s: &mut State,
        flags: i32,
        op: bool,
        kind: i32,
        reps: i32,
        verify: bool,
    ) -> f64 {
        match flags {
            -1 => select_mm8::<0, true>(s, op, kind, reps, verify),
            0 => select_mm8::<0, false>(s, op, kind, reps, verify),
            1 => select_mm8::<1, false>(s, op, kind, reps, verify),
            2 => select_mm8::<2, false>(s, op, kind, reps, verify),
            4 => select_mm8::<4, false>(s, op, kind, reps, verify),
            8 => select_mm8::<8, false>(s, op, kind, reps, verify),
            16 => select_mm8::<16, false>(s, op, kind, reps, verify),
            17 => select_mm8::<17, false>(s, op, kind, reps, verify),
            20 => select_mm8::<20, false>(s, op, kind, reps, verify),
            24 => select_mm8::<24, false>(s, op, kind, reps, verify),
            33 => select_mm8::<33, false>(s, op, kind, reps, verify),
            34 => select_mm8::<34, false>(s, op, kind, reps, verify),
            49 => select_mm8::<49, false>(s, op, kind, reps, verify),
            64 => select_mm8::<64, false>(s, op, kind, reps, verify),
            65 => select_mm8::<65, false>(s, op, kind, reps, verify),
            66 => select_mm8::<66, false>(s, op, kind, reps, verify),
            68 => select_mm8::<68, false>(s, op, kind, reps, verify),
            72 => select_mm8::<72, false>(s, op, kind, reps, verify),
            80 => select_mm8::<80, false>(s, op, kind, reps, verify),
            81 => select_mm8::<81, false>(s, op, kind, reps, verify),
            84 => select_mm8::<84, false>(s, op, kind, reps, verify),
            88 => select_mm8::<88, false>(s, op, kind, reps, verify),
            97 => select_mm8::<97, false>(s, op, kind, reps, verify),
            98 => select_mm8::<98, false>(s, op, kind, reps, verify),
            113 => select_mm8::<113, false>(s, op, kind, reps, verify),
            _ => panic!("unsupported flags"),
        }
    }
    fn dispatch_mm16(
        s: &mut State,
        flags: i32,
        op: bool,
        kind: i32,
        reps: i32,
        verify: bool,
    ) -> f64 {
        match flags {
            -1 => select_mm16::<0, true>(s, op, kind, reps, verify),
            0 => select_mm16::<0, false>(s, op, kind, reps, verify),
            1 => select_mm16::<1, false>(s, op, kind, reps, verify),
            2 => select_mm16::<2, false>(s, op, kind, reps, verify),
            4 => select_mm16::<4, false>(s, op, kind, reps, verify),
            8 => select_mm16::<8, false>(s, op, kind, reps, verify),
            16 => select_mm16::<16, false>(s, op, kind, reps, verify),
            17 => select_mm16::<17, false>(s, op, kind, reps, verify),
            20 => select_mm16::<20, false>(s, op, kind, reps, verify),
            24 => select_mm16::<24, false>(s, op, kind, reps, verify),
            33 => select_mm16::<33, false>(s, op, kind, reps, verify),
            34 => select_mm16::<34, false>(s, op, kind, reps, verify),
            49 => select_mm16::<49, false>(s, op, kind, reps, verify),
            64 => select_mm16::<64, false>(s, op, kind, reps, verify),
            65 => select_mm16::<65, false>(s, op, kind, reps, verify),
            66 => select_mm16::<66, false>(s, op, kind, reps, verify),
            68 => select_mm16::<68, false>(s, op, kind, reps, verify),
            72 => select_mm16::<72, false>(s, op, kind, reps, verify),
            80 => select_mm16::<80, false>(s, op, kind, reps, verify),
            81 => select_mm16::<81, false>(s, op, kind, reps, verify),
            84 => select_mm16::<84, false>(s, op, kind, reps, verify),
            88 => select_mm16::<88, false>(s, op, kind, reps, verify),
            97 => select_mm16::<97, false>(s, op, kind, reps, verify),
            98 => select_mm16::<98, false>(s, op, kind, reps, verify),
            113 => select_mm16::<113, false>(s, op, kind, reps, verify),
            _ => panic!("unsupported flags"),
        }
    }
    fn dispatch_mm64x2(
        s: &mut State,
        flags: i32,
        op: bool,
        kind: i32,
        reps: i32,
        verify: bool,
    ) -> f64 {
        match flags {
            -1 => select_mm64x2::<0, true>(s, op, kind, reps, verify),
            0 => select_mm64x2::<0, false>(s, op, kind, reps, verify),
            1 => select_mm64x2::<1, false>(s, op, kind, reps, verify),
            2 => select_mm64x2::<2, false>(s, op, kind, reps, verify),
            4 => select_mm64x2::<4, false>(s, op, kind, reps, verify),
            8 => select_mm64x2::<8, false>(s, op, kind, reps, verify),
            16 => select_mm64x2::<16, false>(s, op, kind, reps, verify),
            17 => select_mm64x2::<17, false>(s, op, kind, reps, verify),
            20 => select_mm64x2::<20, false>(s, op, kind, reps, verify),
            24 => select_mm64x2::<24, false>(s, op, kind, reps, verify),
            33 => select_mm64x2::<33, false>(s, op, kind, reps, verify),
            34 => select_mm64x2::<34, false>(s, op, kind, reps, verify),
            49 => select_mm64x2::<49, false>(s, op, kind, reps, verify),
            64 => select_mm64x2::<64, false>(s, op, kind, reps, verify),
            65 => select_mm64x2::<65, false>(s, op, kind, reps, verify),
            66 => select_mm64x2::<66, false>(s, op, kind, reps, verify),
            68 => select_mm64x2::<68, false>(s, op, kind, reps, verify),
            72 => select_mm64x2::<72, false>(s, op, kind, reps, verify),
            80 => select_mm64x2::<80, false>(s, op, kind, reps, verify),
            81 => select_mm64x2::<81, false>(s, op, kind, reps, verify),
            84 => select_mm64x2::<84, false>(s, op, kind, reps, verify),
            88 => select_mm64x2::<88, false>(s, op, kind, reps, verify),
            97 => select_mm64x2::<97, false>(s, op, kind, reps, verify),
            98 => select_mm64x2::<98, false>(s, op, kind, reps, verify),
            113 => select_mm64x2::<113, false>(s, op, kind, reps, verify),
            _ => panic!("unsupported flags"),
        }
    }
    fn dispatch_mm64x4(
        s: &mut State,
        flags: i32,
        op: bool,
        kind: i32,
        reps: i32,
        verify: bool,
    ) -> f64 {
        match flags {
            -1 => select_mm64x4::<0, true>(s, op, kind, reps, verify),
            0 => select_mm64x4::<0, false>(s, op, kind, reps, verify),
            1 => select_mm64x4::<1, false>(s, op, kind, reps, verify),
            2 => select_mm64x4::<2, false>(s, op, kind, reps, verify),
            4 => select_mm64x4::<4, false>(s, op, kind, reps, verify),
            8 => select_mm64x4::<8, false>(s, op, kind, reps, verify),
            16 => select_mm64x4::<16, false>(s, op, kind, reps, verify),
            17 => select_mm64x4::<17, false>(s, op, kind, reps, verify),
            20 => select_mm64x4::<20, false>(s, op, kind, reps, verify),
            24 => select_mm64x4::<24, false>(s, op, kind, reps, verify),
            33 => select_mm64x4::<33, false>(s, op, kind, reps, verify),
            34 => select_mm64x4::<34, false>(s, op, kind, reps, verify),
            49 => select_mm64x4::<49, false>(s, op, kind, reps, verify),
            64 => select_mm64x4::<64, false>(s, op, kind, reps, verify),
            65 => select_mm64x4::<65, false>(s, op, kind, reps, verify),
            66 => select_mm64x4::<66, false>(s, op, kind, reps, verify),
            68 => select_mm64x4::<68, false>(s, op, kind, reps, verify),
            72 => select_mm64x4::<72, false>(s, op, kind, reps, verify),
            80 => select_mm64x4::<80, false>(s, op, kind, reps, verify),
            81 => select_mm64x4::<81, false>(s, op, kind, reps, verify),
            84 => select_mm64x4::<84, false>(s, op, kind, reps, verify),
            88 => select_mm64x4::<88, false>(s, op, kind, reps, verify),
            97 => select_mm64x4::<97, false>(s, op, kind, reps, verify),
            98 => select_mm64x4::<98, false>(s, op, kind, reps, verify),
            113 => select_mm64x4::<113, false>(s, op, kind, reps, verify),
            _ => panic!("unsupported flags"),
        }
    }
    fn dispatch_mm64x8(
        s: &mut State,
        flags: i32,
        op: bool,
        kind: i32,
        reps: i32,
        verify: bool,
    ) -> f64 {
        match flags {
            -1 => select_mm64x8::<0, true>(s, op, kind, reps, verify),
            0 => select_mm64x8::<0, false>(s, op, kind, reps, verify),
            1 => select_mm64x8::<1, false>(s, op, kind, reps, verify),
            2 => select_mm64x8::<2, false>(s, op, kind, reps, verify),
            4 => select_mm64x8::<4, false>(s, op, kind, reps, verify),
            8 => select_mm64x8::<8, false>(s, op, kind, reps, verify),
            16 => select_mm64x8::<16, false>(s, op, kind, reps, verify),
            17 => select_mm64x8::<17, false>(s, op, kind, reps, verify),
            20 => select_mm64x8::<20, false>(s, op, kind, reps, verify),
            24 => select_mm64x8::<24, false>(s, op, kind, reps, verify),
            33 => select_mm64x8::<33, false>(s, op, kind, reps, verify),
            34 => select_mm64x8::<34, false>(s, op, kind, reps, verify),
            49 => select_mm64x8::<49, false>(s, op, kind, reps, verify),
            64 => select_mm64x8::<64, false>(s, op, kind, reps, verify),
            65 => select_mm64x8::<65, false>(s, op, kind, reps, verify),
            66 => select_mm64x8::<66, false>(s, op, kind, reps, verify),
            68 => select_mm64x8::<68, false>(s, op, kind, reps, verify),
            72 => select_mm64x8::<72, false>(s, op, kind, reps, verify),
            80 => select_mm64x8::<80, false>(s, op, kind, reps, verify),
            81 => select_mm64x8::<81, false>(s, op, kind, reps, verify),
            84 => select_mm64x8::<84, false>(s, op, kind, reps, verify),
            88 => select_mm64x8::<88, false>(s, op, kind, reps, verify),
            97 => select_mm64x8::<97, false>(s, op, kind, reps, verify),
            98 => select_mm64x8::<98, false>(s, op, kind, reps, verify),
            113 => select_mm64x8::<113, false>(s, op, kind, reps, verify),
            _ => panic!("unsupported flags"),
        }
    }
    fn dispatch_rc4(
        s: &mut State,
        flags: i32,
        op: bool,
        kind: i32,
        reps: i32,
        verify: bool,
    ) -> f64 {
        match flags {
            -1 => select_rc4::<0, true>(s, op, kind, reps, verify),
            0 => select_rc4::<0, false>(s, op, kind, reps, verify),
            1 => select_rc4::<1, false>(s, op, kind, reps, verify),
            3 => select_rc4::<3, false>(s, op, kind, reps, verify),
            5 => select_rc4::<5, false>(s, op, kind, reps, verify),
            11 => select_rc4::<11, false>(s, op, kind, reps, verify),
            13 => select_rc4::<13, false>(s, op, kind, reps, verify),
            17 => select_rc4::<17, false>(s, op, kind, reps, verify),
            19 => select_rc4::<19, false>(s, op, kind, reps, verify),
            21 => select_rc4::<21, false>(s, op, kind, reps, verify),
            27 => select_rc4::<27, false>(s, op, kind, reps, verify),
            29 => select_rc4::<29, false>(s, op, kind, reps, verify),
            32 => select_rc4::<32, false>(s, op, kind, reps, verify),
            33 => select_rc4::<33, false>(s, op, kind, reps, verify),
            35 => select_rc4::<35, false>(s, op, kind, reps, verify),
            37 => select_rc4::<37, false>(s, op, kind, reps, verify),
            43 => select_rc4::<43, false>(s, op, kind, reps, verify),
            45 => select_rc4::<45, false>(s, op, kind, reps, verify),
            49 => select_rc4::<49, false>(s, op, kind, reps, verify),
            51 => select_rc4::<51, false>(s, op, kind, reps, verify),
            53 => select_rc4::<53, false>(s, op, kind, reps, verify),
            59 => select_rc4::<59, false>(s, op, kind, reps, verify),
            61 => select_rc4::<61, false>(s, op, kind, reps, verify),
            _ => panic!("unsupported flags"),
        }
    }
    fn dispatch_rc8(
        s: &mut State,
        flags: i32,
        op: bool,
        kind: i32,
        reps: i32,
        verify: bool,
    ) -> f64 {
        match flags {
            -1 => select_rc8::<0, true>(s, op, kind, reps, verify),
            0 => select_rc8::<0, false>(s, op, kind, reps, verify),
            1 => select_rc8::<1, false>(s, op, kind, reps, verify),
            3 => select_rc8::<3, false>(s, op, kind, reps, verify),
            5 => select_rc8::<5, false>(s, op, kind, reps, verify),
            11 => select_rc8::<11, false>(s, op, kind, reps, verify),
            13 => select_rc8::<13, false>(s, op, kind, reps, verify),
            17 => select_rc8::<17, false>(s, op, kind, reps, verify),
            19 => select_rc8::<19, false>(s, op, kind, reps, verify),
            21 => select_rc8::<21, false>(s, op, kind, reps, verify),
            27 => select_rc8::<27, false>(s, op, kind, reps, verify),
            29 => select_rc8::<29, false>(s, op, kind, reps, verify),
            32 => select_rc8::<32, false>(s, op, kind, reps, verify),
            33 => select_rc8::<33, false>(s, op, kind, reps, verify),
            35 => select_rc8::<35, false>(s, op, kind, reps, verify),
            37 => select_rc8::<37, false>(s, op, kind, reps, verify),
            43 => select_rc8::<43, false>(s, op, kind, reps, verify),
            45 => select_rc8::<45, false>(s, op, kind, reps, verify),
            49 => select_rc8::<49, false>(s, op, kind, reps, verify),
            51 => select_rc8::<51, false>(s, op, kind, reps, verify),
            53 => select_rc8::<53, false>(s, op, kind, reps, verify),
            59 => select_rc8::<59, false>(s, op, kind, reps, verify),
            61 => select_rc8::<61, false>(s, op, kind, reps, verify),
            _ => panic!("unsupported flags"),
        }
    }
    fn dispatch_rc16(
        s: &mut State,
        flags: i32,
        op: bool,
        kind: i32,
        reps: i32,
        verify: bool,
    ) -> f64 {
        match flags {
            -1 => select_rc16::<0, true>(s, op, kind, reps, verify),
            0 => select_rc16::<0, false>(s, op, kind, reps, verify),
            1 => select_rc16::<1, false>(s, op, kind, reps, verify),
            3 => select_rc16::<3, false>(s, op, kind, reps, verify),
            5 => select_rc16::<5, false>(s, op, kind, reps, verify),
            11 => select_rc16::<11, false>(s, op, kind, reps, verify),
            13 => select_rc16::<13, false>(s, op, kind, reps, verify),
            17 => select_rc16::<17, false>(s, op, kind, reps, verify),
            19 => select_rc16::<19, false>(s, op, kind, reps, verify),
            21 => select_rc16::<21, false>(s, op, kind, reps, verify),
            27 => select_rc16::<27, false>(s, op, kind, reps, verify),
            29 => select_rc16::<29, false>(s, op, kind, reps, verify),
            32 => select_rc16::<32, false>(s, op, kind, reps, verify),
            33 => select_rc16::<33, false>(s, op, kind, reps, verify),
            35 => select_rc16::<35, false>(s, op, kind, reps, verify),
            37 => select_rc16::<37, false>(s, op, kind, reps, verify),
            43 => select_rc16::<43, false>(s, op, kind, reps, verify),
            45 => select_rc16::<45, false>(s, op, kind, reps, verify),
            49 => select_rc16::<49, false>(s, op, kind, reps, verify),
            51 => select_rc16::<51, false>(s, op, kind, reps, verify),
            53 => select_rc16::<53, false>(s, op, kind, reps, verify),
            59 => select_rc16::<59, false>(s, op, kind, reps, verify),
            61 => select_rc16::<61, false>(s, op, kind, reps, verify),
            _ => panic!("unsupported flags"),
        }
    }
    fn invoke(width: i32, flags: i32, op: i32, kind: i32, reps: i32, verify: bool) -> f64 {
        assert!((0..6).contains(&op) && (0..4).contains(&kind) && reps > 0);
        assert!(
            kind != 3 || (op < 4 && (flags < 0 || flags & 3 != 2)),
            "unsupported zero-clamp input domain"
        );
        let mut s = STATE.get().expect("prepare first").lock().unwrap();
        match (op, width) {
            (0, 4) => dispatch_mm4(&mut s, flags, false, kind, reps, verify),
            (1, 4) => dispatch_mm4(&mut s, flags, true, kind, reps, verify),
            (0, 8) => dispatch_mm8(&mut s, flags, false, kind, reps, verify),
            (1, 8) => dispatch_mm8(&mut s, flags, true, kind, reps, verify),
            (0, 16) => dispatch_mm16(&mut s, flags, false, kind, reps, verify),
            (1, 16) => dispatch_mm16(&mut s, flags, true, kind, reps, verify),
            (2, 2) => dispatch_mm64x2(&mut s, flags, false, kind, reps, verify),
            (3, 2) => dispatch_mm64x2(&mut s, flags, true, kind, reps, verify),
            (2, 4) => dispatch_mm64x4(&mut s, flags, false, kind, reps, verify),
            (3, 4) => dispatch_mm64x4(&mut s, flags, true, kind, reps, verify),
            (2, 8) => dispatch_mm64x8(&mut s, flags, false, kind, reps, verify),
            (3, 8) => dispatch_mm64x8(&mut s, flags, true, kind, reps, verify),
            (4, 4) => dispatch_rc4(&mut s, flags, false, kind, reps, verify),
            (5, 4) => dispatch_rc4(&mut s, flags, true, kind, reps, verify),
            (4, 8) => dispatch_rc8(&mut s, flags, false, kind, reps, verify),
            (5, 8) => dispatch_rc8(&mut s, flags, true, kind, reps, verify),
            (4, 16) => dispatch_rc16(&mut s, flags, false, kind, reps, verify),
            (5, 16) => dispatch_rc16(&mut s, flags, true, kind, reps, verify),
            _ => panic!("operation/width"),
        }
    }
    pub extern "C" fn guest_run(width: i32, flags: i32, op: i32, kind: i32, reps: i32) -> i64 {
        invoke(width, flags, op, kind, reps, false);
        let s = STATE.get().unwrap().lock().unwrap();
        if op == 2 || op == 3 {
            s.o64[0].to_bits() as i64
        } else {
            i64::from(s.o32[0].to_bits())
        }
    }
    pub extern "C" fn guest_verify(width: i32, flags: i32, op: i32, kind: i32) -> f64 {
        invoke(width, flags, op, kind, 1, true)
    }
    pub fn entry() {
        std::hint::black_box((
            guest_prepare as extern "C" fn(i32, i32),
            guest_probe as extern "C" fn() -> i32,
            guest_run as extern "C" fn(i32, i32, i32, i32, i32) -> i64,
            guest_verify as extern "C" fn(i32, i32, i32, i32) -> f64,
        ));
    }
}
fn main() {
    #[cfg(target_arch = "wasm32")]
    guest::entry();
    #[cfg(not(target_arch = "wasm32"))]
    panic!("compile the WASI guest target");
}
