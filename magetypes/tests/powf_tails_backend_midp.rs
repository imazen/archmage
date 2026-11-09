#![forbid(unsafe_code)]
macro_rules! cases {
    ($check:ident,$token:expr) => {
        $check::<16>($token);
        $check::<17>($token);
        $check::<18>($token);
        $check::<20>($token);
        $check::<21>($token);
        $check::<22>($token);
        $check::<48>($token);
        $check::<49>($token);
        $check::<50>($token);
        $check::<52>($token);
        $check::<53>($token);
        $check::<54>($token);
        $check::<146>($token);
        $check::<150>($token);
        $check::<178>($token);
        $check::<182>($token);
    };
}
#[path = "common/powf_tails.rs"]
mod verifier;
