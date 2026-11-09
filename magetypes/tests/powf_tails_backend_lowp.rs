#![forbid(unsafe_code)]
macro_rules! cases {
    ($check:ident,$token:expr) => {
        $check::<24>($token);
        $check::<25>($token);
        $check::<26>($token);
        $check::<28>($token);
        $check::<29>($token);
        $check::<30>($token);
        $check::<56>($token);
        $check::<57>($token);
        $check::<58>($token);
        $check::<60>($token);
        $check::<61>($token);
        $check::<62>($token);
        $check::<154>($token);
        $check::<158>($token);
        $check::<186>($token);
        $check::<190>($token);
    };
}
#[path = "common/powf_tails.rs"]
mod verifier;
