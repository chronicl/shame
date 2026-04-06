use std::{marker::PhantomData, ops::Deref, borrow::Borrow};

use crate::{
    GpuLayout,
    any::{AsAny, TypeLayoutCompatibleWith},
    call_info,
    frontend::{
        any::{
            Any, InvalidReason,
            shared_io::{BindingType, BufferBindingType},
        },
        rust_types::{
            GpuType,
            array::{Array, ArrayRef, RuntimeSize, Size},
            atomic::Atomic,
            layout_traits::{FromAnys, get_layout_compare_with_cpu_push_error},
            len::{Len, Len2, LenEven},
            mat::mat,
            mem::{self, AddressSpace, SupportsAccess},
            packed_vec::PackedScalarType,
            reference::{AccessModeReadable, Read, ReadWrite, Ref},
            scalar_type::{ScalarType, ScalarTypeFp, ScalarTypeInteger},
            type_traits::{BindingArgs, GpuSized, GpuStore, NoAtomics, NoBools, NoHandles},
            vec::vec,
        },
    },
    ir::{self, LayoutType, recording::Context},
    packed::PackedVec,
};

use super::binding::Binding;

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
#[derive(Debug, Clone, Copy)]
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
impl std::fmt::Display for BufferAddressSpaceEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferAddressSpaceEnum::Storage => write!(f, "storage"),
            BufferAddressSpaceEnum::Uniform => write!(f, "uniform"),
        }
    }
}

/// A read-only buffer binding, for writeable buffers and atomics use [`BufferRef`] instead.
///
/// Buffer contents are accessible via [`std::ops::Deref`] `*`.
///
/// ## Generic arguments
/// - `Content`: the buffer content. May not contain bools and if access mode is `Read` may not contain atomics.
/// - `AS`: the address space can be either of
///     - `mem::Uniform`
///         - has special memory layout requirements, see https://www.w3.org/TR/WGSL/#address-space-layout-constraints
///         - check the uniform buffer size limitations via the graphics api
///     - `mem::Storage`
///         - for large buffers
/// - `AM`: the access mode, either `Read` or `ReadWrite`.
/// - `DYNAMIC_OFFSET`: whether an offset into the bound buffer can be specified when binding its bind-group in the graphics api.
///
/// ## Example read buffer usage
/// ```
/// use shame as sm;
///
/// // storage buffers
/// let buffer: sm::Buffer<sm::f32x4, sm::mem::Storage> = bind_group_iter.next();
/// // same as above, since `mem::Storage` is the default
/// let buffer: sm::Buffer<sm::f32x4> = bind_group_iter.next();
///
/// // access via `std::ops::Deref` `*`
/// let value = *buffer + sm::vec!(1.0, 2.0, 3.0, 4.0);
///
/// // uniform buffers
/// let buffer: sm::Buffer<sm::f32x4, sm::mem::Uniform> = bind_group_iter.next();
///
/// // runtime-sized array buffer
/// let buffer: sm::Buffer<sm::Array<sm::f32x4>> = bind_group_iter.next();
///
/// // fixed size array buffer
/// let buffer: sm::Buffer<sm::Array<sm::f32x4, sm::Size<512>>> = bind_group_iter.next();
///
/// // custom struct type buffer
/// #[derive(sm::GpuLayout)]
/// struct Transforms {
///     world: f32x4x4,
///     view: f32x4x4,
///     proj: f32x4x4,
/// }
/// let buffer: sm::Buffer<Transforms> = bind_group_iter.next();
/// // equivalent to
/// let buffer: sm::Buffer<sm::Struct<Transforms>> = bind_group_iter.next();
///
/// // array of structs
/// let buffer: sm::Buffer<sm::Array<sm::Struct<Transforms>>> = bind_group_iter.next();
/// ```
///
/// /// # Example read-write buffer usage
/// ```
/// use shame as sm;
/// use sm::f32x4x4;
/// use sm::f32x4;
///
/// // storage buffers
/// let buffer: sm::Buffer<f32x4, sm::mem::Storage, sm::ReadWrite> = bind_group_iter.next();
/// // same as above, since `mem::Storage` and `ReadWrite` is the default
/// let buffer: sm::Buffer<f32x4> = bind_group_iter.next();
///
/// // read access via `.get()`
/// let value = buffer.get() + sm::vec!(1.0, 2.0, 3.0, 4.0);
///
/// // write access via `.set()`
/// buffer.set(sm::vec!(1.0, 2.0, 3.0, 4.0));
///
/// // uniform buffers
/// let buffer: sm::Buffer<f32x4, sm::mem::Uniform, sm::Read> = bind_group_iter.next();
///
/// // fixed size array buffer
/// let buffer: sm::Buffer<sm::Array<f32x4, sm::Size<512>>> = bind_group_iter.next();
///
/// // runtime-sized array buffer
/// let buffer: sm::Buffer<sm::Array<f32x4>> = bind_group_iter.next();
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
/// let buffer: sm::Buffer<Transforms> = bind_group_iter.next();
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
///
/// > maintainer note:
/// > the precise trait bounds of buffer bindings are found in the `Binding` impl blocks.
pub struct Buffer<T, AS = mem::Storage, AM = Read, const DYNAMIC_OFFSET: bool = false>
where
    T: GpuLayout + NoBools + NoHandles,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
{
    content: Ref<T, AS, AM>,
    _phantom: PhantomData<(T, AS, AM)>,
}

impl<T, AS, AM, const DYNAMIC_OFFSET: bool> Deref for Buffer<T, AS, AM, DYNAMIC_OFFSET>
where
    T: GpuLayout + NoBools + NoHandles,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
{
    type Target = Ref<T, AS, AM>;
    fn deref(&self) -> &Self::Target { &self.content }
}

impl<T, AS, AM, const DYNAMIC_OFFSET: bool> Buffer<T, AS, AM, DYNAMIC_OFFSET>
where
    T: GpuLayout + NoBools + NoHandles,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
{
    fn new(args: BindingArgs) -> Self { Self::from_ref(create_ref_for_buffer_binding(args, DYNAMIC_OFFSET)) }

    fn new_invalid(reason: InvalidReason) -> Self { Self::from_ref(Ref::from(Any::new_invalid(reason))) }

    /// TODO(chronicl)
    pub fn from_ref(r: Ref<T, AS, AM>) -> Self {
        Self {
            content: r,
            _phantom: PhantomData,
        }
    }
}

/// Bind a new buffer with the binding arguments provided. This skips a lot of static type checks
/// that [`Buffer`] performs. Those type checks instead are runtime errors.
pub fn create_ref_for_buffer_binding<T, AS, AM>(args: BindingArgs, has_dynamic_offset: bool) -> Ref<T, AS, AM>
where
    T: GpuLayout + GpuStore,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
{
    Context::try_with(call_info!(), |ctx| {
        get_layout_compare_with_cpu_push_error::<T>(ctx, None);
    });

    let ty = T::layout_type_owned();
    Ref::from(create_ref_any_for_buffer_binding(
        args,
        has_dynamic_offset,
        ty,
        AS::BUFFER_ADDRESS_SPACE,
        AM::ACCESS_MODE_READABLE,
    ))
}

/// Bind a new buffer with the binding arguments provided. Returns an Any, which is Ref<T, AS, AM>,
/// where T corresponds to `ty`, AS to `address_space` and AM to `access`.
/// This skips a lot of static type checks that [`Buffer`] performs.
/// Those type checks instead are runtime errors.
pub fn create_ref_any_for_buffer_binding(
    args: BindingArgs,
    has_dynamic_offset: bool,
    ty: LayoutType,
    address_space: BufferAddressSpaceEnum,
    access: crate::ir::AccessModeReadable,
) -> crate::any::Any {
    Context::try_with(call_info!(), |ctx| {
        let bind_ty = BindingType::Buffer {
            ty: match address_space {
                BufferAddressSpaceEnum::Storage => BufferBindingType::Storage(access),
                BufferAddressSpaceEnum::Uniform => BufferBindingType::Uniform,
            },
            has_dynamic_offset,
        };

        let vert_write_storage = ctx.settings().vertex_writable_storage_by_default;
        let vis = bind_ty.max_supported_stage_visibility(vert_write_storage);

        // Check that the layout of `T` is compatible with the address space
        // and if it is, create the binding.
        match address_space {
            // Bad duplication in match arms, but not worth abstracting away
            BufferAddressSpaceEnum::Uniform => {
                match TypeLayoutCompatibleWith::<mem::Uniform>::try_from(crate::Language::Wgsl, ty) {
                    Ok(l) => Any::buffer_binding(args.path, vis, l, access, has_dynamic_offset),
                    Err(e) => {
                        ctx.push_error(e.into());
                        Any::new_invalid(InvalidReason::ErrorThatWasPushed)
                    }
                }
            }
            BufferAddressSpaceEnum::Storage => {
                match TypeLayoutCompatibleWith::<mem::Storage>::try_from(crate::Language::Wgsl, ty) {
                    Ok(layout) => Any::buffer_binding(args.path, vis, layout, access, has_dynamic_offset),
                    Err(e) => {
                        ctx.push_error(e.into());
                        Any::new_invalid(InvalidReason::ErrorThatWasPushed)
                    }
                }
            }
        }
    })
    .unwrap_or_else(|| Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding))
}

impl<T, AS, AM, const DYNAMIC_OFFSET: bool> Binding for Buffer<T, AS, AM, DYNAMIC_OFFSET>
where
    T: GpuLayout + GpuStore + NoBools + NoHandles,
    AS: BufferAddressSpace + SupportsAccess<AM>,
    AM: AccessModeReadable,
    (AS, T): AtomicsInStorageOnly,
    (AM, T): AtomicsRequireWriteable,
{
    fn binding_type() -> BindingType {
        let access = AM::ACCESS_MODE_READABLE;
        BindingType::Buffer {
            ty: match AS::BUFFER_ADDRESS_SPACE {
                BufferAddressSpaceEnum::Storage => BufferBindingType::Storage(access),
                BufferAddressSpaceEnum::Uniform => BufferBindingType::Uniform,
            },
            has_dynamic_offset: DYNAMIC_OFFSET,
        }
    }
    fn new_invalid(reason: InvalidReason) -> Self { Self::new_invalid(reason) }
    #[track_caller]
    fn new_binding(args: BindingArgs) -> Self { Self::new(args) }
    fn store_ty() -> ir::StoreType { <T as GpuStore>::store_ty() }
}

#[diagnostic::on_unimplemented(message = "atomics can only be used in read-write storage buffers`.")]
pub trait AtomicsInStorageOnly {}
impl<T> AtomicsInStorageOnly for (mem::Storage, T) {}
impl<T: NoAtomics> AtomicsInStorageOnly for (mem::Uniform, T) {}

#[diagnostic::on_unimplemented(
    message = "atomics can only be used in read-write storage buffers. Use `ReadWrite` instead of `Read`."
)]
pub trait AtomicsRequireWriteable {}
impl<T> AtomicsRequireWriteable for (ReadWrite, T) {}
impl<T: NoAtomics> AtomicsRequireWriteable for (Read, T) {}

#[cfg(test)]
mod tests {
    use crate::{
        self as shame, aliases::*, frontend::rust_types::array::ArrayRef, Array, Buffer, GpuLayout, Read, Ref,
        RuntimeSize,
    };
    use shame as sm;
    use sm::{mem::Storage, ReadWrite};

    #[test]
    fn test_buffer_deref() {
        let mut encoder = sm::start_encoding(sm::Settings::default()).unwrap();
        let mut drawcall = encoder.new_render_pipeline(sm::Indexing::BufferU16);
        let mut group = drawcall.bind_groups.next();

        let f: Ref<f32x1, _, _> = *group.next::<Buffer<f32x1>>();

        #[derive(GpuLayout)]
        struct A {
            a: f32x4x4,
        }
        let a: &A = &group.next::<Buffer<A>>().get();
        let a: &Ref<A, Storage, ReadWrite> = &group.next::<Buffer<A, Storage, ReadWrite>>();

        #[derive(GpuLayout)]
        struct AUnsized {
            a: f32x4x4,
            b: Array<f32x1>,
        }
        let a_unsized: &Ref<AUnsized, Storage, Read> = &group.next::<Buffer<AUnsized>>();
        let a_unsized: &Ref<AUnsized, Storage, ReadWrite> = &group.next::<Buffer<AUnsized, Storage, ReadWrite>>();

        let array: Array<f32x1, sm::Size<4>> = group.next::<Buffer<Array<f32x1, sm::Size<4>>>>().get();
        let array: &Ref<Array<f32x1, sm::Size<4>>, Storage, ReadWrite> =
            &group.next::<Buffer<Array<f32x1, sm::Size<4>>, Storage, ReadWrite>>();

        let unsized_array: Ref<Array<f32x1>, Storage, Read> = *group.next::<Buffer<Array<f32x1>>>();
        let f: f32x1 = group.next::<Buffer<Array<f32x1>>>().at(0).get();
        let unsized_array: &Ref<Array<f32x1>, Storage, ReadWrite> =
            &group.next::<Buffer<Array<f32x1>, Storage, ReadWrite>>();
        let f: Ref<f32x1, Storage, ReadWrite> = group.next::<Buffer<Array<f32x1>, Storage, ReadWrite>>().at(0);

        // this commented line should compile fail
        // let atomic: Buffer<sm::AtomicU32, Storage, Read> = group.next();
        let atomic: Buffer<sm::AtomicU32, Storage, ReadWrite> = group.next();

        #[derive(GpuLayout)]
        struct AAtomic {
            a: sm::AtomicU32,
        }
        // this commented line should compile fail
        // let a_atomic: &AAtomic = &group.next::<Buffer<AAtomic>>();
        let a_atomic: &Ref<AAtomic, Storage, ReadWrite> = &group.next::<Buffer<AAtomic, Storage, ReadWrite>>();
    }
}
