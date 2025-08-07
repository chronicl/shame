use crate::frontend::any::Any;
use crate::frontend::any::{shared_io::BufferBindingType, InvalidReason};
use crate::frontend::rust_types::layout_traits::GpuLayout;
use crate::frontend::rust_types::reference::Ref;
use crate::frontend::rust_types::reference::{AccessMode, AccessModeReadable, Read};
use crate::frontend::rust_types::type_traits::BindingArgs;
use crate::frontend::rust_types::{reference::ReadWrite, struct_::SizedFields, type_traits::NoBools};
use crate::frontend::texture::storage_texture::StorageTexture;
use crate::frontend::texture::texture_array::{StorageTextureArray, TextureArray};
use crate::frontend::texture::texture_traits::{SamplingFormat, Spp, StorageTextureFormat};
use crate::frontend::texture::{Sampler, Texture};
use crate::ir::pipeline::StageMask;
use crate::ir::HandleType;
use crate::{
    frontend::any::{shared_io::BindPath, shared_io::BindingType},
    frontend::{
        rust_types::reference::AccessModeWritable,
        texture::{
            texture_traits::{
                LayerCoords, SamplingMethod, StorageTextureCoords, SupportsCoords, SupportsSpp, TextureCoords,
            },
            TextureKind,
        },
    },
    ir::{self, TextureFormatWrapper},
    mem::{self, AddressSpace, SupportsAccess},
};
use std::marker::PhantomData;
use std::num::NonZeroU32;

/// Types of resources that can be bound as part of a bind-group and accessed in Gpu pipelines.
///
/// [`Binding`] types include
/// - [`Buffer`] (readonly)
/// - [`BufferRef`] (supports readwrite and atomics)
/// - [`Sampler`]
/// - [`Texture`] (for sampling)
/// - [`StorageTexture`] (for writing)
///
/// [`Buffer`]: crate::Buffer
/// [`BufferRef`]: crate::BufferRef
#[diagnostic::on_unimplemented(message = "`{Self}` is not a valid type for a bind-group binding")]
pub trait Binding {
    /// runtime representation of `Self`
    fn binding_type() -> BindingType;
    /// Is None if the binding is not a binding array.
    fn binding_array_len() -> Option<Option<NonZeroU32>> { None }
    /// shader type of `Self`
    fn store_ty() -> ir::StoreType;

    #[doc(hidden)]
    fn new_invalid(reason: InvalidReason) -> Self;
    #[doc(hidden)]
    #[track_caller]
    fn new_binding(args: BindingArgs) -> Self;
}

impl<T: TextureHandle> Binding for T {
    fn binding_type() -> BindingType {
        match T::texture_type() {
            HandleType::SampledTexture(shape, sample_type, spp) => BindingType::SampledTexture {
                shape,
                sample_type,
                samples_per_pixel: spp,
            },
            HandleType::StorageTexture(shape, format, access) => BindingType::StorageTexture { shape, format, access },
            HandleType::Sampler(sampling_method) => BindingType::Sampler(sampling_method),
        }
    }

    fn new_invalid(reason: InvalidReason) -> Self { T::from_any(Any::new_invalid(reason)) }
    fn store_ty() -> ir::StoreType { ir::StoreType::Handle(T::texture_type()) }

    fn new_binding(args: BindingArgs) -> Self {
        let handle_type = T::texture_type();
        let any = Any::texture_binding(args.path, args.visibility, handle_type);
        T::from_any(any)
    }
}

pub trait TextureHandle {
    fn texture_type() -> HandleType;
    fn from_any(any: Any) -> Self;
}

impl<Format, Coords, SPP> TextureHandle for Texture<Format, Coords, SPP>
where
    Coords: TextureCoords + SupportsSpp<SPP>,
    Format: SamplingFormat + SupportsSpp<SPP> + SupportsCoords<Coords>,
    SPP: Spp,
{
    fn texture_type() -> HandleType {
        HandleType::SampledTexture(
            Coords::SHAPE,
            Format::SAMPLE_TYPE.restrict_with_spp(SPP::SAMPLES_PER_PIXEL),
            SPP::SAMPLES_PER_PIXEL,
        )
    }
    fn from_any(any: Any) -> Self { Texture::from_inner(TextureKind::Standalone(any)) }
}

impl<Access: AccessMode, Format: StorageTextureFormat<Access> + SupportsCoords<Coords>, Coords: StorageTextureCoords>
    TextureHandle for StorageTexture<Format, Coords, Access>
{
    fn texture_type() -> HandleType {
        HandleType::StorageTexture(Coords::SHAPE, TextureFormatWrapper::new(Format::id()), Access::ACCESS)
    }
    fn from_any(any: Any) -> Self { StorageTexture::from_inner(TextureKind::Standalone(any)) }
}

impl<Format, const N: u32, Coords> TextureHandle for TextureArray<Format, N, Coords>
where
    Format: SamplingFormat + SupportsCoords<Coords>,
    Coords: TextureCoords + LayerCoords,
{
    fn texture_type() -> HandleType {
        HandleType::SampledTexture(
            Coords::ARRAY_SHAPE(Self::NONZERO_N),
            Format::SAMPLE_TYPE.restrict_with_spp(ir::SamplesPerPixel::Single),
            ir::SamplesPerPixel::Single,
        )
    }
    fn from_any(any: Any) -> Self { TextureArray::from_inner(any) }
}

impl<
    Access: AccessMode,
    Format: StorageTextureFormat<Access> + SupportsCoords<Coords>,
    const N: u32,
    Coords: StorageTextureCoords + LayerCoords,
> TextureHandle for StorageTextureArray<Format, N, Coords, Access>
{
    fn texture_type() -> HandleType {
        HandleType::StorageTexture(
            Coords::ARRAY_SHAPE(Self::NONZERO_N),
            TextureFormatWrapper::new(Format::id()),
            Access::ACCESS,
        )
    }
    fn from_any(any: Any) -> Self { StorageTextureArray::from_inner(any) }
}

impl<M: SamplingMethod> TextureHandle for Sampler<M> {
    fn texture_type() -> HandleType { HandleType::Sampler(M::SAMPLING_METHOD) }
    fn from_any(any: Any) -> Self { Sampler::from_inner(any) }
}
