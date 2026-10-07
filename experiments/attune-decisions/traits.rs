#![forbid(unsafe_code)]

use archmage::intrinsics::x86_64::*;
use archmage::{X64V3Token, arcane, rite};
use dependency_renamed::ForeignKernel;

// Handwritten candidate lowerings, not an implementation of the attune macro.
pub struct Processor<'a, T, const N: usize> {
    bias: T,
    label: &'a str,
    calls: usize,
}

pub trait Kernel {
    type Output;
    fn apply(&mut self, proof: X64V3Token, x: &[f32; 4]) -> Self::Output;
    fn label<'s>(&'s self, proof: X64V3Token, fallback: &'s str) -> &'s str;
}

impl<'a, T: Copy + Into<f32>, const N: usize> Kernel for Processor<'a, T, N> {
    type Output = [f32; N];
    fn apply(&mut self, proof: X64V3Token, x: &[f32; 4]) -> Self::Output {
        entry(proof, self, x)
    }
    fn label<'s>(&'s self, proof: X64V3Token, fallback: &'s str) -> &'s str {
        // Inner function explicitly declares the outer generics it needs.
        #[arcane]
        fn inner<'a, 's, T: Copy + Into<f32>, const N: usize>(
            _proof: X64V3Token,
            this: &'s Processor<'a, T, N>,
            fallback: &'s str,
        ) -> &'s str {
            if this.label.is_empty() {
                fallback
            } else {
                this.label
            }
        }
        inner(proof, self, fallback)
    }
}

#[arcane]
fn entry<'a, T: Copy + Into<f32>, const N: usize>(
    _proof: X64V3Token,
    this: &mut Processor<'a, T, N>,
    x: &[f32; 4],
) -> <Processor<'a, T, N> as Kernel>::Output {
    direct(this, x)
}

#[rite(v3)]
fn direct<'a, T: Copy + Into<f32>, const N: usize>(
    this: &mut Processor<'a, T, N>,
    x: &[f32; 4],
) -> <Processor<'a, T, N> as Kernel>::Output {
    let v = _mm_add_ps(_mm_loadu_ps(x), _mm_set1_ps(this.bias.into()));
    let mut lanes = [0.0; 4];
    _mm_storeu_ps(&mut lanes, v);
    this.calls += 1;
    std::array::from_fn(|i| lanes[i % 4])
}

pub trait DefaultKernel {
    fn bias(&self) -> f32;
    fn apply(&self, proof: X64V3Token, data: &[f32; 4]) -> f32 {
        default_entry(proof, self, data)
    }
}

#[arcane]
fn default_entry<S: DefaultKernel + ?Sized>(_proof: X64V3Token, this: &S, data: &[f32; 4]) -> f32 {
    let value = _mm_add_ps(_mm_loadu_ps(data), _mm_set1_ps(this.bias()));
    let mut lanes = [0.0; 4];
    _mm_storeu_ps(&mut lanes, value);
    lanes.into_iter().sum()
}

impl<'a, T: Copy + Into<f32>, const N: usize> DefaultKernel for Processor<'a, T, N> {
    fn bias(&self) -> f32 {
        self.bias.into()
    }
}

pub struct LocalTag;
// Foreign trait + foreign self type is legal here because LocalTag is local.
// A generator cannot always put the implementation in an inherent impl.
impl ForeignKernel<LocalTag, X64V3Token> for Vec<f32> {
    type Output = f32;
    fn apply(&self, proof: X64V3Token, data: &[f32; 4]) -> f32 {
        foreign_entry(proof, self, data)
    }
}

#[arcane]
fn foreign_entry(_proof: X64V3Token, this: &[f32], data: &[f32; 4]) -> f32 {
    let value = _mm_add_ps(_mm_loadu_ps(data), _mm_set1_ps(this[0]));
    let mut lanes = [0.0; 4];
    _mm_storeu_ps(&mut lanes, value);
    lanes.into_iter().sum()
}

pub trait Make {
    fn remake(&self, proof: X64V3Token) -> Self
    where
        Self: Sized;
}

impl<T: Copy> Make for Wrap<T> {
    fn remake(&self, proof: X64V3Token) -> Self {
        remake_entry(proof, self)
    }
}
pub struct Wrap<T>(T);
#[arcane]
fn remake_entry<T: Copy>(_proof: X64V3Token, this: &Wrap<T>) -> Wrap<T> {
    Wrap(this.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::SimdToken;

    fn proof() -> X64V3Token {
        X64V3Token::summon().expect("this SIMD experiment requires a V3 host")
    }

    #[test]
    fn associated_output_const_generic_mut_receiver_and_dyn() {
        let mut p = Processor::<u8, 6> {
            bias: 2,
            label: "borrowed",
            calls: 0,
        };
        let obj: &mut dyn Kernel<Output = [f32; 6]> = &mut p;
        assert_eq!(
            obj.apply(proof(), &[1., 2., 3., 4.]),
            [3., 4., 5., 6., 3., 4.]
        );
        assert_eq!(obj.label(proof(), "fallback"), "borrowed");
        assert_eq!(p.calls, 1);
    }

    #[test]
    fn default_trait_body_accepts_unsized_dyn_receiver() {
        let p = Processor::<u8, 1> {
            bias: 2,
            label: "default",
            calls: 0,
        };
        let obj: &dyn DefaultKernel = &p;
        assert_eq!(obj.apply(proof(), &[1., 2., 3., 4.]), 18.);
    }

    #[test]
    fn foreign_self_type_uses_free_helper() {
        let v = vec![2.];
        assert_eq!(
            ForeignKernel::<LocalTag, _>::apply(&v, proof(), &[1., 2., 3., 4.]),
            18.
        );
    }

    #[test]
    fn existing_sized_self_return_contract_is_preserved() {
        assert_eq!(Wrap(7u8).remake(proof()).0, 7);
    }
}
