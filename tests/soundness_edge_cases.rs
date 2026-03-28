//! Soundness edge case tests.
//!
//! These test combinations and scenarios NOT covered by the expansion snapshots:
//! threading, function pointers, callbacks, Send/Sync, async, closures, etc.

#![cfg(target_arch = "x86_64")]

use archmage::{ScalarToken, SimdToken, X64V3Token, X64V4Token, arcane, autoversion, rite};

// ============================================================================
// 1. Token + Send/Sync: tokens should be safely sendable across threads
// ============================================================================

#[test]
fn token_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<X64V3Token>();
    assert_send_sync::<X64V4Token>();
    assert_send_sync::<ScalarToken>();
}

#[test]
fn token_works_across_threads() {
    if let Some(token) = X64V3Token::summon() {
        let handle = std::thread::spawn(move || {
            // Token moved to another thread — should work (all cores have same features)
            process_with_token(token, &[1.0, 2.0, 3.0, 4.0])
        });
        let result = handle.join().unwrap();
        assert_eq!(result, 10.0);
    }
}

#[arcane]
fn process_with_token(token: X64V3Token, data: &[f32; 4]) -> f32 {
    let _ = token;
    data.iter().sum()
}

// ============================================================================
// 2. Function pointer to trampoline: can't bypass token requirement
// ============================================================================

#[test]
fn fn_pointer_to_trampoline_requires_token() {
    // Taking a fn pointer to the trampoline is fine — it still requires a token
    let fp: fn(X64V3Token, &[f32; 4]) -> f32 = process_with_token;

    // Can only call it with a real token
    if let Some(token) = X64V3Token::summon() {
        assert_eq!(fp(token, &[1.0, 2.0, 3.0, 4.0]), 10.0);
    }
    // Without a token, fp is uncallable — no way to construct X64V3Token safely
}

// ============================================================================
// 3. Closures inside #[arcane] inherit target_feature
// ============================================================================

#[arcane]
fn process_with_closure(token: X64V3Token, data: &[f32; 4], transform: impl Fn(f32) -> f32) -> f32 {
    let _ = token;
    // The closure `transform` does NOT have target_feature — that's fine.
    // Calling a non-target_feature fn from target_feature context is always safe.
    data.iter().map(|&x| transform(x)).sum()
}

#[test]
fn callback_without_target_feature_is_safe() {
    if let Some(token) = X64V3Token::summon() {
        let result = process_with_closure(token, &[1.0, 2.0, 3.0, 4.0], |x| x * 2.0);
        assert_eq!(result, 20.0);
    }
}

// ============================================================================
// 4. Autoversion dispatcher is thread-safe
// ============================================================================

#[autoversion]
fn sum_squares(data: &[f32; 4]) -> f32 {
    let mut s = 0.0f32;
    for &x in data {
        s += x * x;
    }
    s
}

#[test]
fn autoversion_from_multiple_threads() {
    let handles: Vec<_> = (0..4)
        .map(|i| {
            std::thread::spawn(move || {
                let data = [i as f32; 4];
                sum_squares(&data)
            })
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        let expected = 4.0 * (i as f32) * (i as f32);
        assert_eq!(h.join().unwrap(), expected);
    }
}

// ============================================================================
// 5. Token downgrade in threaded context
// ============================================================================

#[arcane]
fn v3_work(_token: X64V3Token, x: f32) -> f32 {
    x * 2.0
}

#[arcane]
fn v4_calls_v3(token: X64V4Token, x: f32) -> f32 {
    // Downgrade V4 → V3 and call. This should be sound even across threads.
    v3_work(token.v3(), x) + 1.0
}

#[test]
fn downgrade_in_thread() {
    if let Some(token) = X64V4Token::summon() {
        let handle = std::thread::spawn(move || v4_calls_v3(token, 5.0));
        assert_eq!(handle.join().unwrap(), 11.0);
    }
}

// ============================================================================
// 6. Summon cache is safe under contention
// ============================================================================

#[test]
fn concurrent_summon_is_safe() {
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                // All threads race to summon — should all get the same result
                X64V3Token::summon()
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // All results should be identical
    let first = results[0].is_some();
    for r in &results {
        assert_eq!(r.is_some(), first);
    }
}

// ============================================================================
// 7. Token stored in struct, used later
// ============================================================================

struct SimdProcessor {
    token: X64V3Token,
}

impl SimdProcessor {
    fn new() -> Option<Self> {
        Some(Self {
            token: X64V3Token::summon()?,
        })
    }

    fn process(&self, data: &[f32; 4]) -> f32 {
        process_with_token(self.token, data)
    }
}

#[test]
fn token_stored_in_struct() {
    if let Some(proc) = SimdProcessor::new() {
        assert_eq!(proc.process(&[1.0, 2.0, 3.0, 4.0]), 10.0);
    }
}

// ============================================================================
// 8. Token in Arc across threads
// ============================================================================

#[test]
fn token_in_arc_across_threads() {
    use std::sync::Arc;

    if let Some(token) = X64V3Token::summon() {
        // Token is Copy, but test Arc<Struct> pattern too
        let proc = Arc::new(SimdProcessor { token });

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let proc = Arc::clone(&proc);
                std::thread::spawn(move || proc.process(&[1.0, 2.0, 3.0, 4.0]))
            })
            .collect();

        for h in handles {
            assert_eq!(h.join().unwrap(), 10.0);
        }
    }
}

// ============================================================================
// 9. Rite function called from arcane context (mixed macros)
// ============================================================================

#[rite(v3)]
fn rite_helper(a: f32, b: f32) -> f32 {
    a * b
}

#[arcane]
fn arcane_calls_rite(_token: X64V3Token, data: &[f32; 4]) -> f32 {
    // rite_helper has #[target_feature] matching our context — safe direct call
    rite_helper(data[0], data[1]) + rite_helper(data[2], data[3])
}

#[test]
fn mixed_macro_call() {
    if let Some(token) = X64V3Token::summon() {
        assert_eq!(arcane_calls_rite(token, &[2.0, 3.0, 4.0, 5.0]), 26.0);
    }
}

// ============================================================================
// 10. Recursive arcane function (does the trampoline handle recursion?)
// ============================================================================

#[arcane]
fn recursive_sum(token: X64V3Token, data: &[f32], acc: f32) -> f32 {
    if data.is_empty() {
        acc
    } else {
        recursive_sum(token, &data[1..], acc + data[0])
    }
}

#[test]
fn recursive_arcane() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(recursive_sum(token, &data, 0.0), 15.0);
    }
}
