//! A cached bug screen for std's specified once-rounded FMA operation.
//! Passing fixtures is not an exhaustive proof of an implementation.

#[cfg(all(feature = "std", not(miri)))]
mod implementation {
    use core::hint::black_box;
    use core::sync::atomic::{AtomicU8, Ordering};
    mod vectors {
        include!("std_probe_vectors.rs");
    }
    static STATE: AtomicU8 = AtomicU8::new(0);

    #[inline]
    pub(super) fn available() -> bool {
        match STATE.load(Ordering::Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let accepted = screen();
                STATE.store(if accepted { 2 } else { 1 }, Ordering::Relaxed);
                accepted
            }
        }
    }

    pub(super) fn screen() -> bool {
        screen_with(|a, b, c| black_box(a).mul_add(black_box(b), black_box(c)))
    }

    fn screen_with(mut operation: impl FnMut(f32, f32, f32) -> f32) -> bool {
        vectors::F32.iter().all(|row| {
            let [a, b, c, expected] = row.map(f32::from_bits);
            let result = operation(a, b, c);
            if expected.is_nan() {
                result.is_nan()
            } else {
                result.to_bits() == expected.to_bits()
            }
        })
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn admits_exact_software_and_rejects_twice_rounded_provider() {
            assert!(super::screen_with(crate::nostd_math::fmaf_soft));
            assert!(!super::screen_with(|a, b, c| a * b + c));
        }

        #[test]
        fn cached_std_answer_matches_independent_corpus() {
            let expected = super::screen();
            for _ in 0..8 {
                assert_eq!(super::available(), expected);
            }
        }
    }
}

#[cfg(all(feature = "std", not(miri)))]
pub(super) fn available() -> bool {
    implementation::available()
}
