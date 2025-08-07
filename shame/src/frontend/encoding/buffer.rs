use crate::common::proc_macro_reexports::{GpuLayoutField, GpuStoreImplCategory};
use crate::frontend::any::shared_io::{BindPath, BindingType, BufferBindingType};
use crate::frontend::any::{Any, InvalidReason};
use crate::frontend::rust_types::array::{Array, ArrayLen, ArrayRef, RuntimeSize, Size};
use crate::frontend::rust_types::atomic::Atomic;
use crate::frontend::rust_types::layout_traits::{get_layout_compare_with_cpu_push_error, FromAnys, GetAllFields};
use crate::frontend::rust_types::len::{Len, Len2, LenEven};
use crate::frontend::rust_types::mem::{self, AddressSpace, SupportsAccess};
use crate::frontend::rust_types::packed_vec::PackedScalarType;
use crate::frontend::rust_types::reference::{Ref};
use crate::frontend::rust_types::reference::{AccessMode, Read, ReadWrite};
use crate::frontend::rust_types::scalar_type::{ScalarType, ScalarTypeFp, ScalarTypeNumber};
use crate::frontend::rust_types::struct_::{BufferFields, SizedFields, Struct};
use crate::frontend::rust_types::type_layout::compatible_with::{AddressSpaceError, TypeLayoutCompatibleWith};
use crate::frontend::rust_types::type_traits::{BindingArgs, GpuSized, GpuStore, NoAtomics, NoBools, NoHandles};
use crate::frontend::rust_types::AsAny;
use crate::frontend::rust_types::vec::vec;
use crate::frontend::rust_types::{mat::mat, GpuType};
use crate::frontend::rust_types::{reference::AccessModeReadable, scalar_type::ScalarTypeInteger};
use crate::ir::pipeline::StageMask;
use crate::ir::recording::{Context, MemoryRegion};
use crate::ir::Type;
use crate::packed::PackedVec;
use crate::{self as shame, call_info, ir, GpuLayout};

use std::borrow::Borrow;
use std::marker::PhantomData;
use std::ops::{Deref, Mul};

use super::binding::Binding;

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
    T: BufferContent<AS, AM> + NoBools + NoHandles,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
{
    content: T::DerefTarget,
    _phantom: PhantomData<(T, AS, AM)>,
}

impl<T, AS, AM, const DYNAMIC_OFFSET: bool> Deref for Buffer<T, AS, AM, DYNAMIC_OFFSET>
where
    T: BufferContent<AS, AM> + NoBools + NoHandles,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
{
    type Target = T::DerefTarget;
    fn deref(&self) -> &Self::Target { &self.content }
}

impl<T, AS, AM, const DYNAMIC_OFFSET: bool> Buffer<T, AS, AM, DYNAMIC_OFFSET>
where
    T: BufferContent<AS, AM> + NoBools + NoHandles,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
{
    fn new(args: BindingArgs) -> Self
    where
        T: GpuLayout,
    {
        Self::from_ref(Self::create_ref_for_binding(args))
    }

    fn new_invalid(reason: InvalidReason) -> Self { Self::from_ref(Ref::from(Any::new_invalid(reason))) }

    /// TODO(chronicl)
    pub fn from_ref(r: Ref<T, AS, AM>) -> Self {
        Self {
            content: T::ref_to_deref_target(r),
            _phantom: PhantomData,
        }
    }

    fn create_ref_for_binding(args: BindingArgs) -> Ref<T, AS, AM>
    where
        T: GpuLayout,
    {
        let any = Context::try_with(call_info!(), |ctx| {
            let skip_stride_check = true; // not a vertex buffer
            get_layout_compare_with_cpu_push_error::<T>(ctx, skip_stride_check);

            let access = AM::ACCESS_MODE_READABLE;
            let bind_ty = BindingType::Buffer {
                ty: match AS::BUFFER_ADDRESS_SPACE {
                    BufferAddressSpaceEnum::Storage => BufferBindingType::Storage(access),
                    BufferAddressSpaceEnum::Uniform => BufferBindingType::Uniform,
                },
                has_dynamic_offset: DYNAMIC_OFFSET,
            };

            let vert_write_storage = ctx.settings().vertex_writable_storage_by_default;
            let vis = bind_ty.max_supported_stage_visibility(vert_write_storage);

            // Check that the layout of `T` is compatible with the address space
            // and if it is, create the binding.
            let recipe = T::layout_recipe();
            match AS::BUFFER_ADDRESS_SPACE {
                // Bad duplication in match arms, but not worth abstracting away
                BufferAddressSpaceEnum::Uniform => {
                    match TypeLayoutCompatibleWith::<mem::Uniform>::try_from(crate::Language::Wgsl, recipe) {
                        Ok(l) => Any::buffer_binding(args.path, vis, l, access, DYNAMIC_OFFSET),
                        Err(e) => {
                            ctx.push_error(e.into());
                            Any::new_invalid(InvalidReason::ErrorThatWasPushed)
                        }
                    }
                }
                BufferAddressSpaceEnum::Storage => {
                    match TypeLayoutCompatibleWith::<mem::Storage>::try_from(crate::Language::Wgsl, recipe) {
                        Ok(layout) => Any::buffer_binding(args.path, vis, layout, access, DYNAMIC_OFFSET),
                        Err(e) => {
                            ctx.push_error(e.into());
                            Any::new_invalid(InvalidReason::ErrorThatWasPushed)
                        }
                    }
                }
            }
        })
        .unwrap_or_else(|| Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding));

        Ref::from(any)
    }
}

/// TODO(chronicl)
pub trait BufferContent<AS: BufferAddressSpace, AM: AccessModeReadable>: GpuStore + Sized {
    /// TODO(chronicl)
    type DerefTarget;
    /// TODO(chronicl)
    fn ref_to_deref_target(r: Ref<Self, AS, AM>) -> Self::DerefTarget;
}

// Read-write buffers are simple, their deref target is always Ref<T>
impl<T: GpuStore, AS: BufferAddressSpace> BufferContent<AS, ReadWrite> for T {
    type DerefTarget = Ref<Self, AS, ReadWrite>;
    fn ref_to_deref_target(r: Ref<Self, AS, ReadWrite>) -> Self::DerefTarget { r }
}

// Construcible types T can be dereferenced to themselves from Buffer<T, Read>.
// This is simple to implement for all constructible T that aren't structs.
macro_rules! default_buffer_content_read {
    () => {
        type DerefTarget = Self;
        fn ref_to_deref_target(r: Ref<Self, AS, Read>) -> Self::DerefTarget {
            r.get()
        }
    };
}
impl<T: ScalarTypeFp, C: Len2, R: Len2, AS: BufferAddressSpace> BufferContent<AS, Read> for mat<T, C, R> {
    default_buffer_content_read!();
}
impl<T: ScalarType, L: Len, AS: BufferAddressSpace> BufferContent<AS, Read> for vec<T, L> {
    default_buffer_content_read!();
}
impl<T: GpuType + SizedFields + NoAtomics, AS: BufferAddressSpace> BufferContent<AS, Read> for Struct<T> {
    default_buffer_content_read!();
}
impl<T: GpuStore + GpuType + GpuSized + NoAtomics, const N: usize, AS: BufferAddressSpace> BufferContent<AS, Read>
    for Array<T, Size<N>>
{
    default_buffer_content_read!();
}

// Array<T> is not constructible, because it's unsized, so we can only deref to Ref<Array<T>>
impl<T: GpuType + GpuSized + GpuStore + NoAtomics + 'static, AS: BufferAddressSpace + 'static> BufferContent<AS, Read>
    for Array<T>
{
    type DerefTarget = ArrayRef<T, AS, Read>;
    fn ref_to_deref_target(r: Ref<Self, AS, Read>) -> Self::DerefTarget { ArrayRef::new(r) }
}

// Sized structs T should allow to deref Buffer<T, Read> to T, but
// unsized structs T must deref to Ref<T, AS, Read>, because they aren't constructible.
//
// Wether a struct is sized is determined by it's last field, so we implement a
// StructField trait for all types that can be used as struct fields and by
// accessing the type of the last field of the struct we can use StructField
// to determine whether Buffer<T, Read> should deref to T or Ref<T, AS, Read>.
//
// Note, that currently being constructible means to not be unsized nor contain
// an atomic type. Since Buffer<T, Read> can't contain atomics, we only need
// to handle unsizedness for structs here.
// impl for structs not wrapped in `shame::Struct`
impl<T, AS> BufferContent<AS, Read> for T
where
    T: GpuStore + BufferFields + NoAtomics,
    AS: BufferAddressSpace,
{
    type DerefTarget = <T::LastField as StructField>::StructDerefTarget<T, AS>;
    fn ref_to_deref_target(r: Ref<Self, AS, Read>) -> Self::DerefTarget {
        <T::LastField as StructField>::struct_ref_to_deref_target(r)
    }
}

pub trait StructField {
    type StructDerefTarget<T: GpuStore, AS: BufferAddressSpace>;
    fn struct_ref_to_deref_target<T, AS>(r: Ref<T, AS, Read>) -> Self::StructDerefTarget<T, AS>
    where
        T: GpuStore + BufferFields + GetAllFields + FromAnys,
        AS: BufferAddressSpace;
}

macro_rules! default_struct_field {
    () => {
        type StructDerefTarget<T_: GpuStore, AS: BufferAddressSpace> = T_;
        fn struct_ref_to_deref_target<T_, AS>(r: Ref<T_, AS, Read>) -> Self::StructDerefTarget<T_, AS>
        where
            T_: GpuStore + BufferFields + GetAllFields + FromAnys,
            AS: BufferAddressSpace,
        {
            let field_anys_refs = <T_ as GetAllFields>::fields_as_anys_unchecked(r.as_any());
            FromAnys::from_anys((field_anys_refs.borrow() as &[Any]).iter().map(|a| a.ref_load()))
        }
    };
}
impl<T: ScalarType, L: Len> StructField for vec<T, L> {
    default_struct_field!();
}
impl<T: ScalarTypeFp, C: Len2, R: Len2> StructField for mat<T, C, R> {
    default_struct_field!();
}
impl<T: SizedFields> StructField for Struct<T> {
    default_struct_field!();
}
impl<T: ScalarTypeInteger> StructField for Atomic<T> {
    default_struct_field!();
}
impl<T: PackedScalarType, L: LenEven> StructField for PackedVec<T, L> {
    default_struct_field!();
}
impl<T: GpuType + GpuSized, const N: usize> StructField for Array<T, Size<N>> {
    default_struct_field!();
}
impl<T: GpuType + GpuSized> StructField for Array<T, RuntimeSize> {
    type StructDerefTarget<T_: GpuStore, AS: BufferAddressSpace> = Ref<T_, AS, Read>;
    fn struct_ref_to_deref_target<T_, AS>(r: Ref<T_, AS, Read>) -> Self::StructDerefTarget<T_, AS>
    where
        T_: GpuStore + BufferFields + GetAllFields + FromAnys,
        AS: BufferAddressSpace,
    {
        r
    }
}

impl<T, AS, AM, const DYNAMIC_OFFSET: bool> Binding for Buffer<T, AS, AM, DYNAMIC_OFFSET>
where
    T: BufferContent<AS, AM> + GpuLayout + GpuStore + NoBools + NoHandles,
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
    fn store_ty() -> ir::StoreType { T::impl_category().to_store_ty() }
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
            BufferAddressSpaceEnum::Storage => write!(f, "storage address space"),
            BufferAddressSpaceEnum::Uniform => write!(f, "uniform address space"),
        }
    }
}

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

        let f: f32x1 = *group.next::<Buffer<f32x1>>();

        #[derive(GpuLayout)]
        struct A {
            a: f32x4x4,
        }
        let a: &A = &group.next::<Buffer<A>>();
        let a: &Ref<A, Storage, ReadWrite> = &group.next::<Buffer<A, Storage, ReadWrite>>();

        #[derive(GpuLayout)]
        struct AUnsized {
            a: f32x4x4,
            b: Array<f32x1>,
        }
        let a_unsized: &Ref<AUnsized, Storage, Read> = &group.next::<Buffer<AUnsized>>();
        let a_unsized: &Ref<AUnsized, Storage, ReadWrite> = &group.next::<Buffer<AUnsized, Storage, ReadWrite>>();

        let array: &Array<f32x1, sm::Size<4>> = &group.next::<Buffer<Array<f32x1, sm::Size<4>>>>();
        let array: &Ref<Array<f32x1, sm::Size<4>>, Storage, ReadWrite> =
            &group.next::<Buffer<Array<f32x1, sm::Size<4>>, Storage, ReadWrite>>();

        let unsized_array: &ArrayRef<f32x1, Storage, Read> = &group.next::<Buffer<Array<f32x1>>>();
        let f32x1 = group.next::<Buffer<Array<f32x1>>>().at(0);
        let unsized_array: &Ref<Array<f32x1>, Storage, ReadWrite> =
            &group.next::<Buffer<Array<f32x1>, Storage, ReadWrite>>();

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
