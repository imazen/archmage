#![forbid(unsafe_code)]
macro_rules! cases {
    ($check:ident,$token:expr) => {
        $check::<16>($token);
        $check::<17>($token);
        $check::<18>($token);
        $check::<20>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<21>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<22>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<48>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<49>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<50>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<52>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<53>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<54>($token);
        $check::<146>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<150>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<178>($token);
        $check::<182>($token);
    };
}
#[path = "common/powf_tails.rs"]
mod verifier;
