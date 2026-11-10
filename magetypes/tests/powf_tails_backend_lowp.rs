#![forbid(unsafe_code)]
macro_rules! cases {
    ($check:ident,$token:expr) => {
        $check::<24>($token);
        $check::<25>($token);
        $check::<26>($token);
        $check::<28>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<29>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<30>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<56>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<57>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<58>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<60>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<61>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<62>($token);
        $check::<154>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<158>($token);
        #[cfg(feature = "exhaustive-tests")]
        $check::<186>($token);
        $check::<190>($token);
    };
}
#[path = "common/powf_tails.rs"]
mod verifier;
