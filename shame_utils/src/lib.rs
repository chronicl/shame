use any::layout::{self, Repr};
use shame::{
    any, vec, Array, ArrayLen, GpuSized, GpuType, Len, Len2, NoBools, ScalarType, ScalarTypeFp, ScalarTypeInteger,
    SizedFields, Struct,
};

mod const_str;
mod no_padding;
mod to_glam;

pub use const_str::*;
pub use no_padding::*;
pub use to_glam::*;
pub use shame_utils_macros::*;
