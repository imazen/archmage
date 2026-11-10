#![forbid(unsafe_code)]
macro_rules! cases {
    ($check:ident,$token:expr) => {
        $check::<0>($token);
        $check::<1>($token);
        $check::<2>($token);
        $check::<4>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<5>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<6>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<32>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<33>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<34>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<36>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<37>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<38>($token);
        $check::<130>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<134>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<162>($token);
        $check::<166>($token);
    };
}
#[path = "common/powf_tails.rs"]
mod verifier;
