use crate::{
    frontend::any::shared_io,
    ir::{LayoutType, Repr, SizedType, recording::MemoryRegion},
};

use super::*;
use std::{
    fmt::Display,
    num::{NonZeroU32, NonZeroU64},
    rc::Rc,
};

#[derive(Clone, PartialEq, Eq)]
// Types according to the WebGPU type system,
// slightly modified to include
// - `StoreType::BufferBlock` for potential OpenGL/GLSL compatibility
// - `Rc<Allocation>` for tracking memory cell accesses (useful for the stage solver)
#[doc(hidden)] // runtime api
pub enum Type {
    Unit,
    Ptr(Rc<MemoryRegion>, StoreType, AccessMode),
    Ref(Rc<MemoryRegion>, StoreType, AccessMode),
    Store(StoreType),
}

/// types that pointers/reference can point to.
#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StoreType {
    /// WGSL "creation-fixed-footprint"
    Layout(LayoutType),
    Handle(HandleType),
    BindingArray(Rc<StoreType>, Option<NonZeroU32>),
}

impl<T> From<T> for StoreType
where
    LayoutType: From<T>,
{
    fn from(value: T) -> Self { StoreType::Layout(value.into()) }
}

impl<T> From<T> for Type
where
    StoreType: From<T>,
{
    fn from(value: T) -> Self { Type::Store(value.into()) }
}

/// types that represent handles to resources (Textures and Samplers).
#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HandleType {
    SampledTexture(TextureShape, TextureSampleUsageType, SamplesPerPixel),
    StorageTexture(TextureShape, TextureFormatWrapper, AccessMode),
    Sampler(shared_io::SamplingMethod),
}

impl Type {
    pub fn is_ref(&self) -> bool { matches!(self, Type::Ref { .. }) }
}

impl std::fmt::Debug for Type {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// custom debug implementation to hide details of the allocation, and instead print refs and ptrs
        /// as expected from wgsl
        match self {
            Self::Unit => write!(f, "Unit"),
            Self::Ptr(arg0, arg1, arg2) => f.debug_tuple("Ptr").field(&arg0.address_space).field(arg1).field(arg2).finish(),
            Self::Ref(arg0, arg1, arg2) => f.debug_tuple("Ref").field(&arg0.address_space).field(arg1).field(arg2).finish(),
            Self::Store(arg0) => f.debug_tuple("Store").field(arg0).finish(),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_view_str = match self {
            Type::Ptr(..) => "Ptr",
            Type::Ref(..) => "Ref",
            _ => "",
        };
        match self {
            Type::Unit => f.write_str("()"),
            Type::Ptr(a, s, am) | Type::Ref(a, s, am) => write!(f, "{mem_view_str}<{s}, {}, {am}>", a.address_space),
            Type::Store(s) => write!(f, "{s}"),
        }
    }
}

impl Display for StoreType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreType::Layout(layout_type) => match layout_type {
                LayoutType::Sized(s) => write!(f, "{s}"),
                LayoutType::UnsizedStruct(s) => write!(f, "{}", s.name),
                LayoutType::RuntimeSizedArray(a) => write!(f, "Array<{}>", a.element),
            },
            StoreType::Handle(x) => write!(f, "{x}"),
            StoreType::BindingArray(x, _) => write!(f, "BindingArray<{}>", x),
        }
    }
}

impl Display for HandleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandleType::SampledTexture(shape, fmt, spp) => write!(f, "Texture<{fmt}, {shape}, {spp:?}>"),
            HandleType::StorageTexture(shape, fmt, access) => write!(f, "StorageTexture<{fmt:?}, {shape}, {access}>"),
            HandleType::Sampler(s) => write!(f, "Sampler<{s}>"),
        }
    }
}

impl StoreType {
    pub fn min_byte_size(&self) -> Option<NonZeroU64> {
        match self {
            // TODO(chronicl) repr
            StoreType::Layout(layout_type) => match layout_type {
                LayoutType::Sized(s) => NonZeroU64::new(s.byte_size(Repr::Wgsl)),
                LayoutType::UnsizedStruct(s) => todo!(),
                LayoutType::RuntimeSizedArray(a) => NonZeroU64::new(a.byte_stride(Repr::Wgsl)),
            },
            StoreType::Handle(handle_type) => None,
            // TODO(chronicl) check correct
            StoreType::BindingArray(binding_type, _) => binding_type.min_byte_size(),
        }
    }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AlignedType {
    Sized(SizedType),
    RuntimeSizedArray(SizedType),
}

impl AlignedType {
    pub fn align(&self) -> u64 {
        // TODO(chronicl)
        match self {
            AlignedType::Sized(sized) => sized.align(Repr::Wgsl) as u64,
            AlignedType::RuntimeSizedArray(a) => a.align(Repr::Wgsl) as u64,
        }
    }
}
