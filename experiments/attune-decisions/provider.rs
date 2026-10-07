#![forbid(unsafe_code)]
#![deny(warnings)]

// Namespace/cfg probes only. These tier labels do not enable SIMD features.
pub mod macro_api {
    pub fn work(x: u32) -> u32 {
        selected(x)
    }
    pub fn base(x: u32) -> u32 {
        x + 3
    }
    #[cfg(feature = "fast")]
    pub fn fast(x: u32) -> u32 {
        x + 4
    }
    #[cfg(feature = "fast")]
    fn selected(x: u32) -> u32 {
        fast(x)
    }
    #[cfg(not(feature = "fast"))]
    fn selected(x: u32) -> u32 {
        base(x)
    }
    pub use crate::__family_macro_work as work;
}

// The prototype assigns this unique root name by hand. A generator must solve
// name allocation; it cannot assume two modules never use the same fn name.
#[cfg(feature = "fast")]
#[macro_export]
macro_rules! __family_macro_work {
    ($x:expr) => {
        $crate::macro_api::fast($x)
    };
}
#[cfg(not(feature = "fast"))]
#[macro_export]
macro_rules! __family_macro_work {
    ($x:expr) => {
        $crate::macro_api::base($x)
    };
}

pub mod type_api {
    #[doc(hidden)]
    #[allow(non_camel_case_types)]
    pub enum work {}
    pub fn work(x: u32) -> u32 {
        work::selected(x)
    }
    impl work {
        #[inline]
        pub fn selected(x: u32) -> u32 {
            #[cfg(feature = "fast")]
            {
                x + 4
            }
            #[cfg(not(feature = "fast"))]
            {
                x + 3
            }
        }
        #[inline]
        pub fn generic<T: Copy + Into<u32>, const N: usize>(x: [T; N]) -> u32 {
            x.into_iter().map(Into::into).sum()
        }
    }
}

pub use macro_api::work as macro_alias;
pub use type_api::work as type_alias;

// Deliberately wrong: this cfg is interpreted in the consuming crate.
#[macro_export]
macro_rules! consumer_cfg_leak {
    ($x:expr) => {{
        #[cfg(feature = "fast")]
        {
            $crate::macro_api::fast($x)
        }
        #[cfg(not(feature = "fast"))]
        {
            $crate::macro_api::base($x)
        }
    }};
}

pub trait ForeignKernel<Tag, Proof> {
    type Output;
    fn apply(&self, proof: Proof, data: &[f32; 4]) -> Self::Output;
}
