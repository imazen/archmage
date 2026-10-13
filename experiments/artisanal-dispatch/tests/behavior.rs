#![forbid(unsafe_code)]
use artisanal_dispatch::{Scalar, bridge, dispatch, tier};

#[bridge(scalar)]
fn identify_scalar(x: String) -> (String, &'static str) {
    (x, "scalar")
}
#[bridge(v3)]
fn identify_v3(x: String) -> (String, &'static str) {
    (x, "v3")
}
#[bridge(neon)]
fn identify_neon(x: String) -> (String, &'static str) {
    (x, "neon")
}
#[bridge(wasm128)]
fn identify_wasm128(x: String) -> (String, &'static str) {
    (x, "wasm128")
}

#[test]
fn dispatch_order_fallback_and_argument_evaluation() {
    let mut calls = 0;
    let answer =
        dispatch!(identify({ calls += 1; String::from("owned") }); v3, neon, wasm128, scalar);
    assert_eq!(calls, 1);
    assert_eq!(answer.0, "owned");
    let expected = if artisanal_dispatch::V3::summon().is_some() {
        "v3"
    } else if artisanal_dispatch::Neon::summon().is_some() {
        "neon"
    } else if artisanal_dispatch::Wasm128::summon().is_some() {
        "wasm128"
    } else {
        "scalar"
    };
    assert_eq!(answer.1, expected);
    assert_eq!(dispatch!(identify(String::new()); scalar).1, "scalar");
    // Callers can request a concrete branch from an external test runner.
    if let Ok(required) = std::env::var("ARTISANAL_EXPECT_TIER") {
        assert_eq!(expected, required);
    }
}

#[bridge(scalar)]
#[allow(clippy::needless_lifetimes)] // Exercise explicit lifetime forwarding.
fn borrowed_scalar<'a, T, const N: usize>(values: &'a mut [T; N], value: T) -> &'a T
where
    T: Copy,
{
    values[0] = value;
    &values[0]
}
#[test]
fn generics_lifetimes_where_and_mutable_arguments() {
    let mut values = [0u32; 4];
    assert_eq!(*dispatch!(borrowed::<u32, 4>(&mut values, 9); scalar), 9);
    assert_eq!(values[0], 9);
}

#[bridge(scalar)]
fn opaque_scalar(x: impl Iterator<Item = u32>) -> impl Iterator<Item = u32> {
    x.map(|x| x + 1)
}
#[bridge(scalar)]
#[cfg(any())]
fn erased(_: DoesNotExist) -> DoesNotExist {
    unimplemented!()
}
#[bridge(v3)]
fn context_v3() -> u32 {
    let t = artisanal_dispatch::V3::from_context();
    let _: artisanal_dispatch::V2 = t.into();
    let _: Scalar = t.into();
    leaf_v3(41)
}
#[tier(v3)]
fn leaf_v3(x: u32) -> u32 {
    x + 1
}
#[test]
fn opaque_arguments_nested_functions_and_context() {
    assert_eq!(
        dispatch!(opaque(0..3); scalar).collect::<Vec<_>>(),
        [1, 2, 3]
    );
    #[bridge(scalar)]
    fn nested(x: u32) -> u32 {
        x + 1
    }
    assert_eq!(nested_t(Scalar::from_context(), 4), 5);
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = artisanal_dispatch::V3::summon() {
        assert_eq!(context_v3_t(t), 42);
    }
}

#[test]
fn build_time_only_without_std() {
    #[cfg(all(
        not(feature = "std"),
        target_arch = "x86_64",
        not(target_feature = "avx2")
    ))]
    assert!(artisanal_dispatch::V3::summon().is_none());
    #[cfg(not(target_arch = "x86_64"))]
    assert!(artisanal_dispatch::V3::summon().is_none());
    assert!(Scalar::summon().is_some());
}

#[bridge(scalar)]
fn identity_scalar(x: u32) -> u32 {
    x
}
#[bridge(scalar)]
fn unit_scalar() {}
#[bridge(scalar)]
#[track_caller]
fn caller_scalar() -> u32 {
    core::panic::Location::caller().line()
}
#[test]
fn argument_control_flow_and_hygiene() {
    fn early(value: Option<u32>) -> Option<u32> {
        Some(dispatch!(identity(value?); scalar))
    }
    assert_eq!(early(None), None);
    assert_eq!(early(Some(5)), Some(5));
    let mut n = 0;
    let out = loop {
        n += 1;
        let _: u32 = dispatch!(identity(if n == 2 { break 9; } else if n == 1 { continue; } else { 0 }); scalar);
    };
    assert_eq!(out, 9);
    let (__arg0, __proof) = (3, 7);
    assert_eq!(dispatch!(identity(__arg0 + __proof); scalar), 10);
    dispatch!(unit(); scalar);
    let line = line!() + 1;
    assert_eq!(dispatch!(caller(); scalar), line);
}

mod family {
    #[artisanal_dispatch::bridge(scalar)]
    pub fn echo_scalar<T>(x: T) -> T {
        x
    }
}
#[bridge(scalar)]
fn apply_scalar(f: impl FnOnce(u32) -> u32) -> u32 {
    f(7)
}
#[test]
fn qualified_paths_closures_and_cross_crate_proofs() {
    let owned = String::from("forwarded");
    assert_eq!(
        dispatch!(family::echo::<String>(owned); scalar),
        "forwarded"
    );
    assert_eq!(dispatch!(apply(|x: u32| x.count_ones()); scalar), 3);
}
