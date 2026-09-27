//! Design probe only: distinct constructor modes, using existing magetypes storage.
//! This does not implement the proposed `#[magetypes(use(...))]` syntax.
#![forbid(unsafe_code)]

use archmage::ScalarToken;
use core::marker::PhantomData;
use core::ops::Add;
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::f32x8 as Existing;

pub enum Explicit {}
pub enum Context {}

#[repr(transparent)]
pub struct Vector<T: F32x8Backend, Mode = Explicit> {
    inner: Existing<T>,
    mode: PhantomData<Mode>,
}

impl<T: F32x8Backend, Mode> Vector<T, Mode> {
    fn wrap(inner: Existing<T>) -> Self {
        Self {
            inner,
            mode: PhantomData,
        }
    }

    pub fn into_existing(self) -> Existing<T> {
        self.inner
    }

    pub fn into_explicit(self) -> Vector<T> {
        Vector::wrap(self.inner)
    }
}

impl<T: F32x8Backend> Vector<T> {
    pub fn zero(token: T) -> Self {
        Self::wrap(Existing::zero(token))
    }
}

impl<T: F32x8Backend, Mode> Add for Vector<T, Mode> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::wrap(self.inner + other.inner)
    }
}

#[cfg(target_arch = "x86_64")]
impl Vector<archmage::X64V3Token, Context> {
    #[archmage::rite(v3)]
    pub fn zero() -> Self {
        Self::wrap(Existing::zero(archmage::X64V3Token::from_context()))
    }
}

impl Vector<ScalarToken, Context> {
    pub fn zero() -> Self {
        Self::wrap(Existing::zero(ScalarToken))
    }
}

// Existing generic token-taking code remains valid without feature attributes.
pub fn existing_generic<T: F32x8Backend>(token: T) -> Vector<T> {
    Vector::<T>::zero(token)
}

pub fn accepts_existing_generic<T: F32x8Backend>(value: Existing<T>) -> [f32; 8] {
    value.to_array()
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn context_variant(token: archmage::X64V3Token) -> [f32; 8] {
    // Equivalent to the proposed use(f32x8) alias for a V3 expansion.
    #[allow(non_camel_case_types)]
    type f32x8 = Vector<archmage::X64V3Token, Context>;
    let a = f32x8::zero();
    let b = f32x8::zero();
    let old_mode: Vector<archmage::X64V3Token> = (a + b).into_explicit();
    let old_vector = old_mode.into_existing();
    let _legacy = existing_generic(token);
    accepts_existing_generic(old_vector)
}

pub fn scalar_variant() -> [f32; 8] {
    #[allow(non_camel_case_types)]
    type f32x8 = Vector<ScalarToken, Context>;
    accepts_existing_generic((f32x8::zero() + f32x8::zero()).into_existing())
}

#[cfg(all(target_arch = "x86_64", feature = "reject-missing-context"))]
pub fn missing_context() {
    let _ = Vector::<archmage::X64V3Token, Context>::zero();
}

#[cfg(all(target_arch = "x86_64", feature = "reject-weaker-context"))]
#[target_feature(enable = "sse2")]
pub fn weaker_context() {
    let _ = Vector::<archmage::X64V3Token, Context>::zero();
}

#[cfg(all(target_arch = "x86_64", feature = "reject-implicit-conversion"))]
#[archmage::rite(v3)]
pub fn implicit_conversion() {
    let _old: Vector<archmage::X64V3Token> = Vector::<archmage::X64V3Token, Context>::zero();
}

#[cfg(feature = "reject-legacy-inference")]
pub fn existing_inferred<T: F32x8Backend>(token: T) -> Vector<T> {
    Vector::zero(token)
}

// A separate generic family leaves the existing type's method lookup unchanged.
#[repr(transparent)]
pub struct LocalVector<T: F32x8Backend>(Existing<T>);

impl<T: F32x8Backend> LocalVector<T> {
    pub fn into_existing(self) -> Existing<T> {
        self.0
    }
}

impl<T: F32x8Backend> From<Existing<T>> for LocalVector<T> {
    fn from(value: Existing<T>) -> Self {
        Self(value)
    }
}

impl<T: F32x8Backend> From<LocalVector<T>> for Existing<T> {
    fn from(value: LocalVector<T>) -> Self {
        value.0
    }
}

impl<T: F32x8Backend> Add for LocalVector<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

#[cfg(target_arch = "x86_64")]
impl LocalVector<archmage::X64V3Token> {
    #[archmage::rite(v3)]
    pub fn zero() -> Self {
        Self(Existing::zero(archmage::X64V3Token::from_context()))
    }
}

impl LocalVector<ScalarToken> {
    pub fn zero() -> Self {
        Self(Existing::zero(ScalarToken))
    }
}

pub fn unchanged_existing_inference<T: F32x8Backend>(token: T) -> Existing<T> {
    Existing::zero(token)
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn separate_family_variant(token: archmage::X64V3Token) -> [f32; 8] {
    #[allow(non_camel_case_types)]
    type f32x8 = LocalVector<archmage::X64V3Token>;
    let local: f32x8 = f32x8::zero() + f32x8::zero();
    let old = unchanged_existing_inference(token);
    let old_from_local: Existing<archmage::X64V3Token> = local.into();
    let local_from_old: f32x8 = old.into();
    accepts_existing_generic(old_from_local + local_from_old.into_existing())
}

#[cfg(all(target_arch = "x86_64", feature = "reject-local-missing-context"))]
pub fn local_missing_context() {
    let _ = LocalVector::<archmage::X64V3Token>::zero();
}
