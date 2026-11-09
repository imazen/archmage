#![forbid(unsafe_code)]
macro_rules! cases {
    ($check:ident,$token:expr) => {
        $check::<0>($token);
        $check::<1>($token);
        $check::<2>($token);
        $check::<4>($token);
        $check::<5>($token);
        $check::<6>($token);
        $check::<32>($token);
        $check::<33>($token);
        $check::<34>($token);
        $check::<36>($token);
        $check::<37>($token);
        $check::<38>($token);
        $check::<130>($token);
        $check::<134>($token);
        $check::<162>($token);
        $check::<166>($token);
    };
}
#[path = "common/powf_tails.rs"]
mod verifier;
