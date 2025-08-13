use layout::Repr;
use shame::{
    any::{self, layout},
    ArrayLen, GpuSized, GpuType, Len, Len2, NoBools, ScalarType, ScalarTypeFp, ScalarTypeInteger, SizedFields,
};

use crate::{const_len, const_write, StrBuf, ToStr};

pub trait NoPadding {
    const LAYOUT: Layout;
}

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

impl<T: ScalarType, L: Len> NoPadding for shame::vec<T, L>
where
    shame::vec<T, L>: NoBools,
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

impl<T: GpuType + GpuSized + NoPadding, N: ArrayLen> NoPadding for shame::Array<T, N> {
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

impl<T: SizedFields + NoPadding> NoPadding for shame::Struct<T> {
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

// This uses a specialized ToStr type: PaddingToFields. This keeps the size of the generated code in check.
#[macro_export]
macro_rules! padding_to_fields {
    ($padding:ident, $field_counter:ident) => {{
        $crate::const_to_str!($crate::PaddingToFields {
            padding: $padding,
            field_counter: $field_counter
        })
    }};
}

pub struct PaddingToFields {
    pub padding: usize,
    pub field_counter: usize,
}

impl ToStr<PaddingToFields> {
    pub const fn bytes_len(&self) -> usize {
        let this = &self.0;
        let c = self.0.field_counter;
        match this.padding {
            0 => const_len!("\n"),
            2 => const_len!("  _", c, ": f16x1,\n"),
            4 => const_len!("  _", c, ": f32x1,\n"),
            8 => const_len!("  _", c, ": f32x2,\n"),
            12 => const_len!("  _", c, ": f32x1, _", c + 1, ": f32x1, _", c + 2, ": f32x1,\n"),
            _ => panic!("Unsupported padding size"),
        }
    }

    pub const fn to_buf<const N: usize>(&self) -> StrBuf<N> {
        let c = self.0.field_counter;

        let mut b = [0u8; N];
        let mut p = 0;
        match self.0.padding {
            0 => return StrBuf::Str("\n"),
            2 => const_write!(p, b <- "  _", c, ": f16x1,\n"),
            4 => const_write!(p, b <- "  _", c, ": f32x1,\n"),
            8 => const_write!(p, b <- "  _", c, ": f32x2,\n"),
            12 => const_write!(p, b <- "  _", c, ": f32x1, _", c + 1, ": f32x1, _", c + 2, ": f32x1,\n"),
            _ => panic!("Unsupported padding size"),
        };
        let _ = p; // silence warning

        StrBuf::Bytes(b)
    }
}
