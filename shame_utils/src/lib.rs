use any::layout::{self, Repr};
use shame::{
    any, vec, Array, ArrayLen, GpuSized, GpuType, Len, Len2, NoBools, ScalarType, ScalarTypeFp, ScalarTypeInteger,
    SizedFields, Struct,
};
pub use shame_utils_macros::*;

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

impl<T: SizedFields + NoPadding> NoPadding for Struct<T> {
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
            2 => $crate::const_concat!("  _", $field_counter, ": f16x1,\n"),
            4 => $crate::const_concat!("  _", $field_counter, ": f32x1,\n"),
            8 => $crate::const_concat!("  _", $field_counter, ": f32x2,\n"),
            12 => $crate::const_concat!(
                "  _", $field_counter, ": f32x1, _", $field_counter + 1, ": f32x1, _", $field_counter + 2, ": f32x1, // or _", $field_counter, ": Array<f32x1, 3> if not used in uniform buffer\n"
            ),
            _ => panic!("Unsupported padding size"),
        }
    };
}

pub trait ToGlam: NoPadding {
    type GlamType;
}

use shame::aliases::*;

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


pub enum StrBuf<'a, const N: usize> {
    Bytes([u8; N]),
    Str(&'a str),
}

impl<const N: usize> StrBuf<'_, N> {
    pub const fn as_str(&self) -> &str {
        match self {
            StrBuf::Bytes(b) => match str::from_utf8(b) {
                Ok(s) => s,
                Err(_) => panic!("Invalid UTF-8 sequence in concatenated string"),
            },
            StrBuf::Str(s) => s,
        }
    }
}

pub struct ToStr<T>(pub T);

impl ToStr<&str> {
    pub const fn bytes_len(&self) -> usize { self.0.len() }

    pub const fn as_str<const N: usize>(&self) -> StrBuf<'_, N> { StrBuf::Str(self.0) }
}

impl ToStr<usize> {
    pub const fn bytes_len(&self) -> usize {
        let mut n = self.0;

        let mut digits = 1;
        while n > 9 {
            n /= 10;
            digits += 1;
        }
        digits
    }

    /// Must be called with N = Self.output_len()
    pub const fn as_str<const N: usize>(&self) -> StrBuf<'_, N> {
        let mut buf = [0; N];
        let mut i = N - 1; // N is at least one method is used correctly
        let mut n = self.0;
        loop {
            buf[i] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
            i -= 1;
        }
        assert!(i == 0);
        StrBuf::Bytes(buf)
    }
}

impl ToStr<&[&str]> {
    pub const fn bytes_len(&self) -> usize {
        let mut len = 0;
        let mut i = 0;
        while i < self.0.len() {
            len += self.0[i].len();
            i += 1;
        }
        len
    }

    pub const fn as_str<const N: usize>(&self) -> StrBuf<'_, N> {
        let mut buf = [0u8; N];
        let mut pos = 0;
        let mut str_idx = 0;

        while str_idx < self.0.len() {
            let bytes = self.0[str_idx].as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                buf[pos] = bytes[i];
                pos += 1;
                i += 1;
            }

            str_idx += 1;
        }
        StrBuf::Bytes(buf)
    }
}

#[macro_export]
macro_rules! const_concat {
    ($($s:expr),*) => {{
        const STRINGS: &[&str] = &[$($crate::const_to_str!($s)),*];
        $crate::const_to_str!(STRINGS)
    }};
}

#[macro_export]
macro_rules! const_to_str {
    ($x:expr) => {{
        const LEN: usize = $crate::ToStr($x).bytes_len();
        const S: $crate::StrBuf<LEN> = $crate::ToStr($x).as_str();
        S.as_str()
    }};
}

#[cfg(test)]
mod tests {}
