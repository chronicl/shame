use any::layout::{self, Repr};
use shame::{
    any, vec, Array, ArrayLen, GpuSized, GpuType, Len, Len2, NoBools, ScalarType, ScalarTypeFp, ScalarTypeInteger,
    SizedFields, Struct,
};
pub use shame_utils_macros::*;

pub trait NoPadding {
    const LAYOUT: Layout;
}

pub use const_str;

#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub size: Option<usize>,
    pub align: usize,
    pub offset: usize,
}

impl Layout {
    pub const fn from_align_size(align: usize, size: Option<usize>) -> Self {
        Self::from_align_size_offset(align, size, 0)
    }

    pub const fn from_align_size_offset(align: usize, size: Option<usize>, offset: usize) -> Self {
        assert!(align.is_power_of_two(), "Alignment must be a power of two");
        Self { size, align, offset }
    }

    pub const fn extend(self, layout: Layout) -> Self {
        let self_size = match self.size {
            Some(size) => size,
            None => panic!("Only the last field of a struct may be unsized"),
        };
        let offset = round_up(layout.align, self_size);
        let new_size = match layout.size {
            Some(s) => Some(offset + s),
            None => None,
        };
        let new_align = if self.align > layout.align {
            // const max
            self.align
        } else {
            layout.align
        };
        Self::from_align_size_offset(new_align, new_size, offset)
    }

    pub const fn size_rounded_to_align(self) -> Option<usize> {
        match self.size {
            Some(size) => Some(round_up(self.align, size)),
            None => None,
        }
    }
}

const fn round_up(align: usize, size: usize) -> usize { size.next_multiple_of(align) }

impl<T: ScalarType, L: Len> NoPadding for vec<T, L>
where
    vec<T, L>: NoBools,
{
    const LAYOUT: Layout = {
        let v = layout::Vector::new(
            match T::SCALAR_TYPE {
                any::ScalarType::F16 => layout::ScalarType::F16,
                any::ScalarType::F32 => layout::ScalarType::F32,
                any::ScalarType::F64 => layout::ScalarType::F64,
                any::ScalarType::I32 => layout::ScalarType::I32,
                any::ScalarType::U32 => layout::ScalarType::U32,
                any::ScalarType::Bool => panic!("NoBools checked above"),
            },
            L::LEN,
        );
        Layout::from_align_size(
            v.align(Repr::Wgsl).as_u32() as usize,
            Some(v.byte_size(Repr::Wgsl) as usize),
        )
    };
}

impl<T: ScalarTypeFp, C: Len2, R: Len2> NoPadding for shame::mat<T, C, R> {
    const LAYOUT: Layout = {
        let m = layout::Matrix {
            columns: C::LEN2,
            rows: R::LEN2,
            scalar: T::SCALAR_TYPE_FP,
        };
        Layout::from_align_size(
            m.align(Repr::Wgsl).as_u32() as usize,
            Some(m.byte_size(Repr::Wgsl) as usize),
        )
    };
}

impl<T: ScalarTypeInteger> NoPadding for shame::Atomic<T> {
    const LAYOUT: Layout = {
        let a = layout::Atomic {
            scalar: T::SCALAR_TYPE_INTEGER,
        };
        Layout::from_align_size(a.align(Repr::Wgsl).as_u32() as usize, Some(a.byte_size() as usize))
    };
}

impl<T: GpuType + GpuSized + NoPadding, N: ArrayLen> NoPadding for Array<T, N> {
    const LAYOUT: Layout = {
        let stride = match T::LAYOUT.size_rounded_to_align() {
            Some(size) => size,
            None => panic!("Array element type must not be unsized"),
        };
        let size = match N::LEN {
            Some(n) => Some(n.get() as usize * stride),
            None => None,
        };
        Layout::from_align_size(T::LAYOUT.align, size)
    };
}

impl<T: GpuType + SizedFields + NoPadding> NoPadding for Struct<T> {
    const LAYOUT: Layout = T::LAYOUT;
}


pub const fn padding_to_padding_field_count<const N: usize>() -> usize {
    match N {
        0 => 0,
        2 | 4 | 8 => 1,
        12 => 3,
        _ => panic!("Unsupported padding size"),
    }
}



// Re-export the padding_to_fields macro for use in the derive macro
#[macro_export]
macro_rules! padding_to_fields {
    ($padding:ident, $field_counter:ident) => {
        match $padding {
            0 => "\n",
            2 => $crate::const_str::format!("  _{}: f16x1,\n", $field_counter),
            4 => $crate::const_str::format!("  _{}: f32x1,\n", $field_counter),
            8 => $crate::const_str::format!("  _{}: f32x2,\n", $field_counter),
            12 => $crate::const_str::format!(
                "  _{}: f32x1, _{}: f32x1, _{}: f32x1, // or _{}: Array<f32x1, 3> if not used in uniform buffer\n",
                $field_counter, $field_counter + 1, $field_counter + 2, $field_counter
            ),
            _ => panic!("Unsupported padding size"),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate as shame_utils;

    // Test types that implement NoPadding
    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    #[repr(C)]
    pub struct F32x1(f32);

    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    #[repr(C)]
    pub struct F32x3([f32; 3]);

    impl NoPadding for F32x1 {
        const LAYOUT: Layout = Layout::from_align_size(4, Some(4));
    }

    impl NoPadding for F32x3 {
        const LAYOUT: Layout = Layout::from_align_size(16, Some(12));
    }

    // Test struct with proper alignment (no padding)
    #[derive(NoPadding)]
    #[repr(C)]
    struct AlignedStruct {
        a: F32x1,
        b: F32x1,
    }

    // Test that the macro generates working code
    #[test]
    fn test_no_padding_derive() {
        // This test will compile if the derive macro works correctly
        let layout = AlignedStruct::LAYOUT;
        assert_eq!(layout.size, Some(8));
        assert_eq!(layout.align, 4);
    }

    // This should cause a compile-time error due to padding
    // Uncomment to test padding detection:
    /*
    #[derive(NoPadding)]
    #[repr(C)]
    struct PaddedStruct {
        a: F32x1,  // 4 bytes, align 4
        b: F32x3,  // 12 bytes, align 16 - will cause 12 bytes padding after `a`
    }
    */
}
