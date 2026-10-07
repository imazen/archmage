#![forbid(unsafe_code)]

#[diagnostic_probe::legacy_attr(always)]
fn visible() -> usize {
    7
}

#[allow(deprecated)]
#[diagnostic_probe::legacy_attr]
fn allowed() -> usize {
    8
}

#[expect(deprecated)]
#[diagnostic_probe::legacy_attr]
fn expected() -> usize {
    9
}

#[diagnostic_probe::statically_deprecated]
fn static_note() -> usize {
    10
}

fn main() {
    let mut evaluations = 0;
    let x = diagnostic_probe::legacy_expr!({
        evaluations += 1;
        11
    });
    assert_eq!(x, 11);
    assert_eq!(evaluations, 1);
    assert_eq!(
        (visible(), allowed(), expected(), static_note()),
        (7, 8, 9, 10)
    );
}
