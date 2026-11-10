#![forbid(unsafe_code)]
macro_rules! cases {
    ($check:ident,$token:expr) => {
        $check::<8>($token);
        $check::<9>($token);
        $check::<10>($token);
        $check::<12>($token);
        #[cfg(feature = "_exhaustive-tests")]
        $check::<13>($token);
        #[cfg(feature = "_exhaustive-tests")]
        $check::<14>($token);
        #[cfg(feature = "_exhaustive-tests")]
        $check::<40>($token);
        #[cfg(feature = "_exhaustive-tests")]
        $check::<41>($token);
        #[cfg(feature = "_exhaustive-tests")]
        $check::<42>($token);
        #[cfg(feature = "_exhaustive-tests")]
        $check::<44>($token);
        #[cfg(feature = "_exhaustive-tests")]
        $check::<45>($token);
        #[cfg(feature = "_exhaustive-tests")]
        $check::<46>($token);
        $check::<138>($token);
        #[cfg(feature = "_exhaustive-tests")]
        $check::<142>($token);
        #[cfg(feature = "_exhaustive-tests")]
        $check::<170>($token);
        $check::<174>($token);
    };
}
#[path = "common/powf_tails.rs"]
mod verifier;
