use std::fmt::Display;

use crate::{
    U32PowerOf2,
    ir::{Len, LenEven, Repr, ScalarType, SizedType, Vector, ir_type::layout_type::align_size::PACKED_ALIGN},
};

/// (no documentation yet)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedBitsPerComponent {
    /// (no documentation yet)
    _8,
    /// (no documentation yet)
    _16,
}

impl From<PackedBitsPerComponent> for u8 {
    fn from(value: PackedBitsPerComponent) -> Self {
        match value {
            PackedBitsPerComponent::_8 => 8,
            PackedBitsPerComponent::_16 => 16,
        }
    }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedFloat {
    /// components are `u8` or `u16` depending on `PackedBitsPerComponent`.
    /// - `u8`: `[0, 255]`
    /// - `u16`: `[0, 65535]`
    ///
    /// ranges are converted to float `[0.0, 1.0]` `f32` in shaders.
    Unorm,
    /// components are `i8` or `i16`  depending on `PackedBitsPerComponent`.
    /// - `i8`: `[-127, 127]`
    /// - `i16` `[-32767, 32767]`
    ///
    /// ranges are converted to float `[-1.0, 1.0]` `f32` in shaders.
    /// - an `i8` value of `-128` is converted to `-1.0`
    /// - an `i16` value of `-32768` is converted to `-1.0`
    Snorm,
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedScalarType {
    Float(PackedFloat),
    Int,
    Uint,
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedVector {
    pub len: LenEven,
    pub bits_per_component: PackedBitsPerComponent,
    pub scalar_type: PackedScalarType,
}

/// exhaustive list of all byte sizes a `packed_vec` can have
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedVectorByteSize {
    _2,
    _4,
    _8,
}

impl From<PackedVectorByteSize> for u8 {
    fn from(value: PackedVectorByteSize) -> Self {
        match value {
            PackedVectorByteSize::_2 => 2,
            PackedVectorByteSize::_4 => 4,
            PackedVectorByteSize::_8 => 8,
        }
    }
}

impl PackedVectorByteSize {
    pub fn as_u32(self) -> u32 { u8::from(self) as u32 }

    pub fn as_u64(self) -> u64 { u8::from(self) as u64 }
}

impl Display for PackedVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stype = match self.scalar_type {
            PackedScalarType::Float(x) => match x {
                PackedFloat::Unorm => "unorm",
                PackedFloat::Snorm => "snorm",
            },
            PackedScalarType::Int => "i",
            PackedScalarType::Uint => "u",
        };
        let bits_per_c = u8::from(self.bits_per_component);
        let n = u64::from(self.len);
        write!(f, "{stype}{bits_per_c}x{n}")
    }
}

impl PackedVector {
    #[rustfmt::skip]
    pub fn byte_size(&self) -> PackedVectorByteSize {
        match (self.len, self.bits_per_component) {
            (LenEven::X2, PackedBitsPerComponent:: _8) => PackedVectorByteSize::_2, // 2 * 1
            (LenEven::X2, PackedBitsPerComponent::_16) => PackedVectorByteSize::_4, // 2 * 2
            (LenEven::X4, PackedBitsPerComponent:: _8) => PackedVectorByteSize::_4, // 4 * 1
            (LenEven::X4, PackedBitsPerComponent::_16) => PackedVectorByteSize::_8, // 4 * 2
        }
    }

    pub fn align(&self, repr: Repr) -> U32PowerOf2 {
        match repr {
            Repr::Packed => PACKED_ALIGN,
            Repr::Wgsl | Repr::WgslUniform => {
                // Treating WgslUniform as Wgsl, because packed vectors aren't supported in uniform buffers.
                let align = match self.byte_size() {
                    PackedVectorByteSize::_2 => Vector::new(ScalarType::F16, Len::X1).align(Repr::Wgsl),
                    PackedVectorByteSize::_4 => Vector::new(ScalarType::U32, Len::X1).align(Repr::Wgsl),
                    PackedVectorByteSize::_8 => Vector::new(ScalarType::U32, Len::X2).align(Repr::Wgsl),
                };
                U32PowerOf2::try_from(align as u32).expect("the above all have power of 2 align")
            }
        }
    }
}

impl PackedScalarType {
    pub fn decompressed_ty(&self) -> ScalarType {
        match self {
            PackedScalarType::Float(_) => ScalarType::F32,
            PackedScalarType::Int => ScalarType::I32,
            PackedScalarType::Uint => ScalarType::U32,
        }
    }
}

impl PackedVector {
    pub fn decompressed_ty(&self) -> SizedType {
        Vector::new(self.scalar_type.decompressed_ty(), self.len.into()).into()
    }
}
