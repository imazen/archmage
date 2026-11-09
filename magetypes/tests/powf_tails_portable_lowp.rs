#![forbid(unsafe_code)]
macro_rules! cases {
    ($check:ident,$token:expr) => {
        $check::<8>($token);
        $check::<9>($token);
        $check::<10>($token);
        $check::<12>($token);
        $check::<13>($token);
        $check::<14>($token);
        $check::<40>($token);
        $check::<41>($token);
        $check::<42>($token);
        $check::<44>($token);
        $check::<45>($token);
        $check::<46>($token);
        $check::<138>($token);
        $check::<142>($token);
        $check::<170>($token);
        $check::<174>($token);
    };
}
#[path = "common/powf_tails.rs"]
mod verifier;
