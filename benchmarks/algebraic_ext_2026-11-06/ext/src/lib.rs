//! Experimental external extension traits for Magetypes generic vectors.
//!
//! These operations opt into Rust's algebraic floating-point contract: LLVM
//! may reassociate, contract, use reciprocals and disregard signed zero. There
//! is no precision bound or reproducibility promise. Ordinary vector operators
//! remain unchanged. This library uses public APIs only, without CPU probing.
#![no_std]
#![forbid(unsafe_code)]

/// Tokenless algebraic arithmetic, preserving the proof stored in `self`.
pub trait AlgebraicOps: Sized {
    /// Algebraic addition, with the scalar operation applied to each lane.
    fn algebraic_add(self, rhs: Self) -> Self;
    /// Algebraic subtraction.
    fn algebraic_sub(self, rhs: Self) -> Self;
    /// Algebraic multiplication.
    fn algebraic_mul(self, rhs: Self) -> Self;
    /// Algebraic division.
    fn algebraic_div(self, rhs: Self) -> Self;
    /// Algebraic remainder (not Euclidean remainder).
    fn algebraic_rem(self, rhs: Self) -> Self;
}

/// Reconstruction control accepting an already-proven backend token.
pub trait AlgebraicOpsWithToken<T>: Sized {
    /// Algebraic addition using public array conversion and reconstruction.
    fn algebraic_add_t(self, token: T, rhs: Self) -> Self;
    /// Algebraic subtraction.
    fn algebraic_sub_t(self, token: T, rhs: Self) -> Self;
    /// Algebraic multiplication.
    fn algebraic_mul_t(self, token: T, rhs: Self) -> Self;
    /// Algebraic division.
    fn algebraic_div_t(self, token: T, rhs: Self) -> Self;
    /// Algebraic remainder.
    fn algebraic_rem_t(self, token: T, rhs: Self) -> Self;
}

/// Separately named horizontal algebraic addition; ordinary reduce_add is unchanged.
pub trait AlgebraicReduce {
    /// Scalar lane type.
    type Scalar;
    /// An algebraic fold across the lanes; evaluation order may change.
    fn algebraic_reduce_add(self) -> Self::Scalar;
}

macro_rules! method {
    ($name:ident, $ty:ty, $n:expr) => {
        #[inline(always)]
        fn $name(mut self, rhs: Self) -> Self {
            for i in 0..$n {
                self[i] = self[i].$name(rhs[i]);
            }
            self
        }
    };
}
macro_rules! token_method {
    ($name:ident, $op:ident, $n:expr) => {
        #[inline(always)]
        fn $name(self, token: T, rhs: Self) -> Self {
            let a = self.to_array();
            let b = rhs.to_array();
            Self::from_array_t(token, core::array::from_fn(|i| a[i].$op(b[i])))
        }
    };
}
macro_rules! vector_impl {
    ($vec:ident, $backend:ident, $ty:ty, $n:expr) => {
        impl<T: magetypes::simd::backends::$backend> AlgebraicOps
            for magetypes::simd::generic::$vec<T>
        {
            method!(algebraic_add, $ty, $n);
            method!(algebraic_sub, $ty, $n);
            method!(algebraic_mul, $ty, $n);
            method!(algebraic_div, $ty, $n);
            method!(algebraic_rem, $ty, $n);
        }
        impl<T: magetypes::simd::backends::$backend> AlgebraicOpsWithToken<T>
            for magetypes::simd::generic::$vec<T>
        {
            token_method!(algebraic_add_t, algebraic_add, $n);
            token_method!(algebraic_sub_t, algebraic_sub, $n);
            token_method!(algebraic_mul_t, algebraic_mul, $n);
            token_method!(algebraic_div_t, algebraic_div, $n);
            token_method!(algebraic_rem_t, algebraic_rem, $n);
        }
        impl<T: magetypes::simd::backends::$backend> AlgebraicReduce
            for magetypes::simd::generic::$vec<T>
        {
            type Scalar = $ty;
            #[inline(always)]
            fn algebraic_reduce_add(self) -> $ty {
                self.to_array()
                    .into_iter()
                    .fold(0.0 as $ty, |s, a| s.algebraic_add(a))
            }
        }
    };
}
vector_impl!(f32x4, F32x4Backend, f32, 4);
vector_impl!(f32x8, F32x8Backend, f32, 8);
vector_impl!(f32x16, F32x16Backend, f32, 16);
vector_impl!(f64x2, F64x2Backend, f64, 2);
vector_impl!(f64x4, F64x4Backend, f64, 4);
vector_impl!(f64x8, F64x8Backend, f64, 8);
