#![cfg(feature = "glam")]
use core::mem::{align_of, size_of};
use super::{
    aliases::rust_simd::*,
    layout_traits::{gpu_layout, CpuLayout, GpuLayout},
    type_layout::TypeLayout,
};
use crate::any::U32PowerOf2;

fn gpu_layout_with_cpu_align_and_size<TCpu, TGpu: GpuLayout>() -> TypeLayout {
    let mut layout = gpu_layout::<TGpu>();
    let cpu_align = U32PowerOf2::try_from_usize(align_of::<TCpu>()).expect("alignment of types is always a power of 2");
    layout.set_align(cpu_align);
    layout.set_byte_size(Some(size_of::<TCpu>() as u64));
    layout
}

macro_rules! impl_layout {
    ($($cpu:ty => $gpu:ty;)*) => {
        $(
            impl CpuLayout for $cpu {
                fn cpu_layout() -> TypeLayout { gpu_layout_with_cpu_align_and_size::<$cpu, $gpu>() }
            }
        )*

    };
}

impl_layout!(
    glam::Vec2 => f32x2;
    glam::Vec3 => f32x3;
    glam::Vec4 => f32x4;
    glam::Vec3A => f32x3;
    glam::IVec2 => i32x2;
    glam::IVec3 => i32x3;
    glam::IVec4 => i32x4;
    glam::UVec2 => u32x2;
    glam::UVec3 => u32x3;
    glam::UVec4 => u32x4;
    glam::DVec2 => f64x2;
    glam::DVec3 => f64x3;
    glam::DVec4 => f64x4;
    glam::Mat2 => f32x2x2;
    glam::Mat3 => f32x3x3;
    glam::Mat4 => f32x4x4;
    glam::Mat3A => f32x3x3;
    glam::DMat2 => f64x2x2;
    glam::DMat3 => f64x3x3;
    glam::DMat4 => f64x4x4;
    glam::Quat => f32x4;
    glam::DQuat => f64x4;
);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_glam_cpu_layout_impl() {
        let layout = <glam::Vec3 as CpuLayout>::cpu_layout();
        assert_eq!(layout.byte_size(), Some(12));
        assert_eq!(layout.align().as_u32(), 4); // would be 16 for f32x3
        let layout = <glam::Vec3A as CpuLayout>::cpu_layout();
        assert_eq!(layout.byte_size(), Some(16));
        assert_eq!(layout.align().as_u32(), 16);
        assert_ne!(layout, gpu_layout::<f32x3>());
    }
}
