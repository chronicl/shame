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
            2 => $crate::const_concat!("  _", $crate::const_100_to_str($field_counter), ": f16x1,\n"),
            4 => $crate::const_concat!("  _", $crate::const_100_to_str($field_counter), ": f32x1,\n"),
            8 => $crate::const_concat!("  _", $crate::const_100_to_str($field_counter), ": f32x2,\n"),
            12 => $crate::const_concat!(
                "  _", $crate::const_100_to_str($field_counter), ": f32x1, _", $crate::const_100_to_str($field_counter + 1), ": f32x1, _", $crate::const_100_to_str($field_counter + 2), ": f32x1, // or _", $crate::const_100_to_str($field_counter), ": Array<f32x1, 3> if not used in uniform buffer\n"
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

#[macro_export]
macro_rules! const_concat {
    ($($strs:expr),+) => {{
        const STRINGS: &[&str] = &[$($strs),+];
        const TOTAL_LEN: usize = {
            let mut len = 0;
            let mut i = 0;
            while i < STRINGS.len() {
                len += STRINGS[i].len();
                i += 1;
            }
            len
        };

        const RESULT_BYTES: [u8; TOTAL_LEN] = const {
            let mut result = [0u8; TOTAL_LEN];
            let mut pos = 0;
            let mut str_idx = 0;

            while str_idx < STRINGS.len() {
                let current_str = STRINGS[str_idx];
                let current_bytes = current_str.as_bytes();
                let mut byte_idx = 0;

                while byte_idx < current_bytes.len() {
                    result[pos] = current_bytes[byte_idx];
                    pos += 1;
                    byte_idx += 1;
                }

                str_idx += 1;
            }

            result
        };

        match str::from_utf8(&RESULT_BYTES) {
            Ok(s) => s,
            Err(_) => panic!("Invalid UTF-8 sequence in concatenated string"),
        }
    }};
}

#[rustfmt::skip]
pub const fn const_100_to_str(n: usize) -> &'static str {
    match n {
        0 => "0", 1 => "1", 2 => "2", 3 => "3", 4 => "4", 5 => "5", 6 => "6", 7 => "7", 8 => "8", 9 => "9",
        10 => "10", 11 => "11", 12 => "12", 13 => "13", 14 => "14", 15 => "15", 16 => "16", 17 => "17", 18 => "18", 19 => "19",
        20 => "20", 21 => "21", 22 => "22", 23 => "23", 24 => "24", 25 => "25", 26 => "26", 27 => "27", 28 => "28", 29 => "29",
        30 => "30", 31 => "31", 32 => "32", 33 => "33", 34 => "34", 35 => "35", 36 => "36", 37 => "37", 38 => "38", 39 => "39",
        40 => "40", 41 => "41", 42 => "42", 43 => "43", 44 => "44", 45 => "45", 46 => "46", 47 => "47", 48 => "48", 49 => "49",
        50 => "50", 51 => "51", 52 => "52", 53 => "53", 54 => "54", 55 => "55", 56 => "56", 57 => "57", 58 => "58", 59 => "59",
        60 => "60", 61 => "61", 62 => "62", 63 => "63", 64 => "64", 65 => "65", 66 => "66", 67 => "67", 68 => "68", 69 => "69",
        70 => "70", 71 => "71", 72 => "72", 73 => "73", 74 => "74", 75 => "75", 76 => "76", 77 => "77", 78 => "78", 79 => "79",
        80 => "80", 81 => "81", 82 => "82", 83 => "83", 84 => "84", 85 => "85", 86 => "86", 87 => "87", 88 => "88", 89 => "89",
        90 => "90", 91 => "91", 92 => "92", 93 => "93", 94 => "94", 95 => "95", 96 => "96", 97 => "97", 98 => "98", 99 => "99",
        _ => panic!("Number out of range, create an issue on github and we can raise this limit.")
    }
}

#[cfg(test)]
mod tests {}
