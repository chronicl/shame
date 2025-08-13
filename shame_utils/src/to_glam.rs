use shame::{aliases::*, Array, GpuSized, GpuType, SizedFields, Struct};

use crate::NoPadding;

pub trait ToGlam: NoPadding {
    type GlamType;
}

macro_rules! impl_to_glam {
    ($($shame_ty:ty => $glam_ty:ty),*) => {
        $(
            impl ToGlam for $shame_ty {
                type GlamType = $glam_ty;
            }
        )*
    };
}

impl_to_glam!(
    f32x1 => f32,
    f32x2 => glam::Vec2,
    f32x3 => glam::Vec3,
    f32x4 => glam::Vec4,
    f32x2x2 => glam::Mat2,
    f32x3x3 => glam::Mat3A,
    f32x4x4 => glam::Mat4,
    i32x1 => i32,
    i32x2 => glam::IVec2,
    i32x3 => glam::IVec3,
    i32x4 => glam::IVec4,
    u32x1 => u32,
    u32x2 => glam::UVec2,
    u32x3 => glam::UVec3,
    u32x4 => glam::UVec4,
    shame::Atomic<i32> => i32,
    shame::Atomic<u32> => u32
);

impl<T: GpuType + GpuSized + NoPadding + ToGlam, const N: usize> ToGlam for Array<T, shame::Size<N>> {
    type GlamType = [T::GlamType; N];
}

impl<T: GpuType + GpuSized + NoPadding + ToGlam> ToGlam for Array<T> {
    type GlamType = Vec<T::GlamType>;
}

impl<T: SizedFields + NoPadding + ToGlam> ToGlam for Struct<T> {
    type GlamType = T::GlamType;
}
