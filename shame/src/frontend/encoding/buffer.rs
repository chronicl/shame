use crate::any::layout::{repr, GpuTypeLayout, LayoutableSized, LayoutableType};
use crate::common::proc_macro_reexports::GpuStoreImplCategory;
use crate::frontend::any::shared_io::{BindPath, BindingType, BufferBindingType};
use crate::frontend::any::{Any, InvalidReason};
use crate::frontend::rust_types::array::{Array, ArrayLen, RuntimeSize, Size};
use crate::frontend::rust_types::atomic::Atomic;
use crate::frontend::rust_types::layout_traits::{
    get_layout_compare_with_cpu_push_error, gpu_type_layout, FromAnys, GetAllFields,
};
use crate::frontend::rust_types::len::{Len, Len2};
use crate::frontend::rust_types::mem::{self, AddressSpace, SupportsAccess};
use crate::frontend::rust_types::reference::Ref;
use crate::frontend::rust_types::reference::{AccessMode, Read, ReadWrite};
use crate::frontend::rust_types::scalar_type::{ScalarType, ScalarTypeFp, ScalarTypeNumber};
use crate::frontend::rust_types::struct_::{BufferFields, SizedFields, Struct};
use crate::frontend::rust_types::type_layout::layoutable;
use crate::frontend::rust_types::type_traits::{BindingArgs, GpuSized, GpuStore, NoAtomics, NoBools, NoHandles};
use crate::frontend::rust_types::vec::vec;
use crate::frontend::rust_types::{mat::mat, GpuType};
use crate::frontend::rust_types::{reference::AccessModeReadable, scalar_type::ScalarTypeInteger};
use crate::ir::pipeline::StageMask;
use crate::ir::recording::{Context, MemoryRegion};
use crate::ir::{Type};
use crate::{self as shame, call_info, ir, GpuLayout};

use std::borrow::Borrow;
use std::marker::PhantomData;
use std::ops::{Deref, Mul};

use super::binding::Binding;
use super::EncodingErrorKind;

/// Address spaces used for [`Buffer`] and [`BufferRef`] bindings.
///
/// Implemented by the marker types
/// - [`mem::Uniform`]
/// - [`mem::Storage`]
pub trait BufferAddressSpace: AddressSpace + SupportsAccess<Read> {
    /// Either Storage or Uniform address space.
    const BUFFER_ADDRESS_SPACE: BufferAddressSpaceEnum;
}
/// Either Storage or Uniform address space.
#[derive(Debug)]
pub enum BufferAddressSpaceEnum {
    /// Storage address space
    Storage,
    /// Uniform address space
    Uniform,
}
impl BufferAddressSpace for mem::Uniform {
    const BUFFER_ADDRESS_SPACE: BufferAddressSpaceEnum = BufferAddressSpaceEnum::Uniform;
}
impl BufferAddressSpace for mem::Storage {
    const BUFFER_ADDRESS_SPACE: BufferAddressSpaceEnum = BufferAddressSpaceEnum::Storage;
}

#[diagnostic::on_unimplemented(message = "The uniform address space is read only. Change `ReadWrite` to `Read`.")]
/// TODO(chronicl)
#[allow(missing_docs)]
pub trait UniformIsRead {}
impl<AM: AccessModeReadable> UniformIsRead for (mem::Storage, AM) {}
impl UniformIsRead for (mem::Uniform, Read) {}

// Diagnostics carry over from `NoAtomics`.
/// TODO(chronicl)
#[allow(missing_docs)]
pub trait AtomicInStorageOnly {}
impl<T> AtomicInStorageOnly for (mem::Storage, T) {}
impl<T: NoAtomics> AtomicInStorageOnly for (mem::Uniform, T) {}

#[diagnostic::on_unimplemented(message = "One of two things happened.\n\
    1. Access mode `ReadWrite` was used and the buffer content was not wrapped in `Ref`. Change `Buffer<T, ...>` to `Buffer<shame::Ref<T>, ...>`.\n\
    2. A runtime sized array was used without `Ref`. Change `Buffer<Array<T>, ...>` to `Buffer<shame::Ref<Array<T>>, ...>`.")]
/// TODO(chronicl)
#[allow(missing_docs)]
pub trait WriteRequiresRef {}
impl<T: GpuStore, AS: AddressSpace, AM: AccessMode> WriteRequiresRef for (ReadWrite, Ref<T, AS, AM>) {}
impl<T: GpuStore> WriteRequiresRef for (Read, T) {}
// This includes the requirement of
impl<T: GpuStore + GpuType + GpuSized + GpuLayout + LayoutableSized, AS: AddressSpace, AM: AccessMode> WriteRequiresRef
    for (Read, Ref<Array<T>, AS, AM>)
{
}

/// A buffer binding.
///
/// Buffer contents are accessible via [`std::ops::Deref`] `*`.
///
/// Only the [`mem::Storage`] address space supports [`ReadWrite`] access.
///
/// ## Generic arguments
/// - `Content`: the buffer content. May not contain bools
/// - `DYNAMIC_OFFSET`: whether an offset into the bound buffer can be specified when binding its bind-group in the graphics api.
/// - `AS`: the address space can be either of
///     - `mem::Uniform`
///         - read only access
///         - has special memory layout requirements, see https://www.w3.org/TR/WGSL/#address-space-layout-constraints
///         - check the uniform buffer size limitations via the graphics api
///     - `mem::Storage`
///         - readwrite or read-only access
///         - for large buffers
/// # Example
/// ```
/// use shame as sm;
/// use sm::f32x4x4;
/// use sm::f32x4;
/// use sm::Ref;
///
/// // storage buffers
/// let buffer: sm::Buffer<Ref<f32x4>, sm::mem::Storage, sm::ReadWrite> = bind_group_iter.next();
/// // same as above but only readable. Storage and Read are the default.
/// // let buffer: sm::Buffer<f32x4>> = bind_group_iter.next();
///
/// // read access via `.get()`
/// let value = buffer.get() + sm::vec!(1.0, 2.0, 3.0, 4.0);
///
/// // write access via `.set()`
/// buffer.set(sm::vec!(1.0, 2.0, 3.0, 4.0));
///
/// // uniform buffers
/// let buffer: sm::Buffer<f32x4, sm::mem::Uniform> = bind_group_iter.next();
///
/// // fixed size array buffer
/// let buffer: sm::Buffer<sm::Array<f32x4, sm::Size<512>>> = bind_group_iter.next();
///
/// // runtime-sized array buffer
/// let buffer: sm::Buffer<Ref<sm::Array<f32x4>>, _, ReadWrite> = bind_group_iter.next();
///
/// // array lookup returns reference
/// let element: sm::Ref<f32x4> = buffer.at(4u32);
/// buffer.at(8u32).set(sm::vec!(1.0, 2.0, 3.0, 4.0));
///
/// // custom struct type buffer
/// #[derive(sm::GpuLayout)]
/// struct Transforms {
///     world: f32x4x4,
///     view: f32x4x4,
///     proj: f32x4x4,
/// }
/// let buffer: sm::Buffer<Ref<Transforms>, _, ReadWrite> = bind_group_iter.next();
///
/// // field access returns references
/// let world: sm::Ref<f32x4x4> = buffer.world;
///
/// // get fields via `.get()`
/// let matrix: f32x4x4 = buffer.world.get();
///
/// // write to fields via `.set(_)`
/// buffer.world.set(mat::zero())
///
/// // array of structs
/// let buffer: sm::Buffer<sm::Array<sm::Struct<Transforms>>> = bind_group_iter.next();
///
/// ```
pub struct Buffer<Content, AS = mem::Storage, AM = Read, const DYNAMIC_OFFSET: bool = false>
where
    Content: BufferContent + GpuLayout<GpuRepr = repr::Storage>,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
    (AS, AM): UniformIsRead,
    (AS, Content): AtomicInStorageOnly,
    (AM, Content): WriteRequiresRef,
{
    content: Content::BufferContent<AS, AM>,
    _phantom: PhantomData<(AS, AM)>,
}

/// TODO(chronicl)
#[allow(missing_docs)]
pub trait BufferContent {
    const IS_REF: bool = false;
    type BufferContent<AS: AddressSpace, AM: AccessMode>;
    fn from_any<AS: AddressSpace, AM: AccessMode>(any: Any) -> Self::BufferContent<AS, AM>;
}

impl<Content, AS, AM, const DYNAMIC_OFFSET: bool> std::ops::Deref for Buffer<Content, AS, AM, DYNAMIC_OFFSET>
where
    Content: BufferContent + GpuLayout<GpuRepr = repr::Storage>,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
    (AS, AM): UniformIsRead,
    (AS, Content): AtomicInStorageOnly,
    (AM, Content): WriteRequiresRef,
{
    type Target = Content::BufferContent<AS, AM>;

    fn deref(&self) -> &Self::Target { &self.content }
}

impl<Content, AS, AM, const DYNAMIC_OFFSET: bool> Buffer<Content, AS, AM, DYNAMIC_OFFSET>
where
    Content: BufferContent + GpuLayout<GpuRepr = repr::Storage>,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
    (AS, AM): UniformIsRead,
    (AS, Content): AtomicInStorageOnly,
    (AM, Content): WriteRequiresRef,
{
    /// TODO(chronicl)
    pub fn new(args: Result<BindingArgs, InvalidReason>) -> Self {
        let bind_ty = match AS::BUFFER_ADDRESS_SPACE {
            BufferAddressSpaceEnum::Storage => BufferBindingType::Storage(AM::ACCESS_MODE_READABLE),
            BufferAddressSpaceEnum::Uniform => {
                // TODO(chronicl) check AM is readable
                BufferBindingType::Uniform
            }
        };

        let any = create_buffer_any::<Content>(args, bind_ty, DYNAMIC_OFFSET, Content::IS_REF);

        Self {
            content: Content::from_any(any),
            _phantom: PhantomData,
        }
    }
}

#[track_caller]
fn create_buffer_any<T: GpuLayout<GpuRepr = repr::Storage>>(
    args: Result<BindingArgs, InvalidReason>,
    bind_ty: BufferBindingType,
    has_dynamic_offset: bool,
    as_ref: bool,
) -> Any {
    let BindingArgs { path, visibility } = match args {
        Ok(a) => a,
        Err(e) => return Any::new_invalid(e),
    };

    let storage = gpu_type_layout::<T>();
    match bind_ty {
        BufferBindingType::Uniform => {
            let uniform = match GpuTypeLayout::<repr::Uniform>::try_from(storage) {
                Ok(u) => u,
                Err(e) => {
                    let invalid = Context::try_with(call_info!(), |ctx| {
                        ctx.push_error(e.into());
                        InvalidReason::ErrorThatWasPushed
                    })
                    .unwrap_or(InvalidReason::CreatedWithNoActiveEncoding);

                    return Any::new_invalid(invalid);
                }
            };
            // TODO(chronicl) throw error or panic if as_ref == true here
            Any::uniform_buffer_binding(path, visibility, uniform, has_dynamic_offset)
        }
        BufferBindingType::Storage(access) => {
            Any::storage_buffer_binding(path, visibility, storage, access, as_ref, has_dynamic_offset)
        }
    }
}

impl<Content, AS, AM, const DYNAMIC_OFFSET: bool> Binding for Buffer<Content, AS, AM, DYNAMIC_OFFSET>
where
    Content: BufferContent + GpuLayout<GpuRepr = repr::Storage>,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
    (AS, AM): UniformIsRead,
    (AS, Content): AtomicInStorageOnly,
    (AM, Content): WriteRequiresRef,
{
    fn binding_type() -> BindingType {
        BindingType::Buffer {
            ty: match AS::BUFFER_ADDRESS_SPACE {
                BufferAddressSpaceEnum::Storage => BufferBindingType::Storage(AM::ACCESS_MODE_READABLE),
                BufferAddressSpaceEnum::Uniform => {
                    // TODO(chronicl) check AM is readable
                    BufferBindingType::Uniform
                }
            },
            has_dynamic_offset: DYNAMIC_OFFSET,
        }
    }

    // TODO(chronicl) probably remove entirely
    fn store_ty() -> ir::StoreType { Content::layoutable_type().try_into().unwrap() }

    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self { Self::new(args) }
}

trait BufferContentPlain: From<Any> {}

// Trivial implementations of BufferContent.
macro_rules! impl_buffer_content_plain {
    ($(
        impl<$($t:ident : $bound:tt),*> BufferContent for $type:ty;
    )*) => {
        $(
        impl<$($t: $bound),*> BufferContent for $type {
            const IS_REF: bool = false;
            type BufferContent<AS: AddressSpace, AM: AccessMode> = Self;
            fn from_any<AS: AddressSpace, AM: AccessMode>(any: Any) -> Self::BufferContent<AS, AM> { any.into() }
        }

        impl<$($t: $bound),*> BufferContent for $crate::Ref<$type> {
            const IS_REF: bool = true;
            type BufferContent<AS: AddressSpace, AM: AccessMode> = $crate::Ref<$type, AS, AM>;
            fn from_any<AS: AddressSpace, AM: AccessMode>(any: Any) -> Self::BufferContent<AS, AM> { any.into() }
        }
        )*
    };
}
impl_buffer_content_plain!(
    impl<T: ScalarTypeFp, C: Len2, R: Len2> BufferContent for mat<T, C, R>;
    impl<T: SizedFields>                    BufferContent for Struct<T>   ;
    impl<T: ScalarType, L: Len>             BufferContent for vec<T, L>   ;
    impl<T: ScalarTypeInteger>              BufferContent for Atomic<T>   ;
);

// Implementations of BufferContent for structs that aren't wrapped in shame::Struct.
impl<T: BufferFields> BufferContent for T {
    const IS_REF: bool = false;
    type BufferContent<AS: AddressSpace, AM: AccessMode> = T;

    fn from_any<AS: AddressSpace, AM: AccessMode>(any: Any) -> Self::BufferContent<AS, AM> {
        let fields_anys = <T as GetAllFields>::fields_as_anys_unchecked(any);
        let fields_anys = (fields_anys.borrow() as &[Any]).iter().cloned();
        <T as FromAnys>::from_anys(fields_anys)
    }
}
impl<T: BufferFields> BufferContent for Ref<T> {
    const IS_REF: bool = true;
    type BufferContent<AS: AddressSpace, AM: AccessMode> = Ref<T, AS, AM>;

    fn from_any<AS: AddressSpace, AM: AccessMode>(any: Any) -> Self::BufferContent<AS, AM> {
        // TODO, ideally this could just use From::from, but the GpuType requirement
        // on GpuStore::store_type() doesn't allow it.
        Ref::<T, AS, AM>::from_any_unchecked(any)
    }
}

// Implementations of BufferContent for arrays. Starting with fixed size.
impl<T: GpuType + GpuSized + GpuLayout + LayoutableSized, const N: usize> BufferContent for Array<T, Size<N>> {
    const IS_REF: bool = false;
    type BufferContent<AS: AddressSpace, AM: AccessMode> = T;

    fn from_any<AS: AddressSpace, AM: AccessMode>(any: Any) -> Self::BufferContent<AS, AM> { any.into() }
}
impl<T: GpuType + GpuStore + GpuSized + GpuLayout + LayoutableSized, const N: usize> BufferContent
    for Ref<Array<T, Size<N>>>
{
    const IS_REF: bool = true;
    type BufferContent<AS: AddressSpace, AM: AccessMode> = Ref<Array<T, Size<N>>, AS, AM>;

    fn from_any<AS: AddressSpace, AM: AccessMode>(any: Any) -> Self::BufferContent<AS, AM> { any.into() }
}
// Runtime sized only with Ref.
impl<T: GpuType + GpuStore + GpuSized + GpuLayout + LayoutableSized> BufferContent for Ref<Array<T>> {
    const IS_REF: bool = true;
    type BufferContent<AS: AddressSpace, AM: AccessMode> = Ref<Array<T>, AS, AM>;

    fn from_any<AS: AddressSpace, AM: AccessMode>(any: Any) -> Self::BufferContent<AS, AM> { any.into() }
}


#[cfg(test)]
mod buffer_v2_tests {
    use crate::BufferAddressSpace;
    use crate::Ref;
    use crate as shame;
    use shame as sm;
    use shame::prelude::*;
    use shame::aliases::*;


    #[derive(sm::GpuLayout)]
    struct Transforms {
        world: f32x4x4,
        view: f32x4x4,
        proj: f32x4x4,
    }

    #[test]
    fn buffer_v2_test() -> Result<(), sm::EncodingErrors> {
        let mut encoder = sm::start_encoding(sm::Settings::default())?;
        let mut drawcall = encoder.new_render_pipeline(sm::Indexing::BufferU16);

        let mut group = drawcall.bind_groups.next();

        let xforms_sto: sm::Buffer<Transforms, sm::mem::Storage> = group.next();
        let xforms_uni: sm::Buffer<Transforms, sm::mem::Uniform> = group.next();

        let _: sm::Buffer<Ref<sm::f32x1>, sm::mem::Storage, sm::ReadWrite> = group.next();
        let a: sm::Buffer<Ref<sm::Array<f32x1>>, sm::mem::Storage, sm::ReadWrite> = group.next();
        // println!(
        //     "{}",
        //     sm::BufferV2::<shame::f32x1, sm::mem::Uniform, sm::ReadWrite>::UNIFORM_IS_READ
        // );
        // println!("{:?}", <sm::mem::Uniform as BufferAddressSpace>::BUFFER_ADDRESS_SPACE);

        Ok(())
    }
}
