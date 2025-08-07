#![allow(missing_docs)]
use std::rc::Rc;

use crate::{
    any::{Any, BindingType},
    call_info,
    common::proc_macro_reexports::BindingArgs,
    frontend::{
        any::InvalidReason,
        encoding::{binding::TextureHandle, buffer::BufferAddressSpaceEnum},
        rust_types::{
            layout_traits::get_layout_compare_with_cpu_push_error, mem,
            type_layout::compatible_with::TypeLayoutCompatibleWith, vec::ToInteger,
        },
    },
    ir::{recording::Context, StoreType},
    mem::Storage,
    AccessModeReadable, ArrayLen, Binding, Buffer, BufferAddressSpace, BufferContent, GpuIndex, GpuLayout, GpuStore,
    NoAtomics, NoBools, NoHandles, RuntimeSize,
};

pub struct BindingArray<T, L = RuntimeSize> {
    pub bindings: Any,
    _phantom: std::marker::PhantomData<(T, L)>,
}

impl<T, L> BindingArray<T, L> {
    pub fn new_invalid(reason: InvalidReason) -> Self {
        Self {
            bindings: Any::new_invalid(reason),
            _phantom: std::marker::PhantomData,
        }
    }
}

// Binding arrays don't support dynamic offsets in wgpu currently,
// so it's only implemented for buffers with DYNAMIC_OFFSET = false
impl<T, L, AS, AM> Binding for BindingArray<Buffer<T, AS, AM>, L>
where
    T: BufferContent<AS, AM> + GpuStore + GpuLayout + NoHandles + NoBools,
    Buffer<T, AS, AM>: Binding,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
    L: ArrayLen,
{
    fn binding_type() -> BindingType { Buffer::<T, AS, AM>::binding_type() }
    fn binding_array_len() -> Option<Option<std::num::NonZeroU32>> { Some(L::LEN) }

    fn store_ty() -> StoreType { StoreType::BindingArray(Rc::new(<Buffer<T, AS, AM>>::store_ty()), L::LEN) }
    fn new_invalid(reason: InvalidReason) -> Self { Self::new_invalid(reason) }
    fn new_binding(args: BindingArgs) -> Self {
        let any = Context::try_with(call_info!(), |ctx| {
            let skip_stride_check = true; // not a vertex buffer
            get_layout_compare_with_cpu_push_error::<T>(ctx, skip_stride_check);

            let access = AM::ACCESS_MODE_READABLE;
            let bind_ty = Buffer::<T, AS, AM>::binding_type();

            let vert_write_storage = ctx.settings().vertex_writable_storage_by_default;
            let vis = bind_ty.max_supported_stage_visibility(vert_write_storage);

            // Check that the layout of `T` is compatible with the address space
            // and if it is, create the binding.
            let recipe = T::layout_recipe();
            match AS::BUFFER_ADDRESS_SPACE {
                // Bad duplication in match arms, but not worth abstracting away
                BufferAddressSpaceEnum::Uniform => {
                    match TypeLayoutCompatibleWith::<mem::Uniform>::try_from(crate::Language::Wgsl, recipe) {
                        Ok(l) => Any::buffer_binding_array(args.path, vis, l, access, false, L::LEN),
                        Err(e) => {
                            ctx.push_error(e.into());
                            Any::new_invalid(InvalidReason::ErrorThatWasPushed)
                        }
                    }
                }
                BufferAddressSpaceEnum::Storage => {
                    match TypeLayoutCompatibleWith::<mem::Storage>::try_from(crate::Language::Wgsl, recipe) {
                        Ok(layout) => Any::buffer_binding_array(args.path, vis, layout, access, false, L::LEN),
                        Err(e) => {
                            ctx.push_error(e.into());
                            Any::new_invalid(InvalidReason::ErrorThatWasPushed)
                        }
                    }
                }
            }
        })
        .unwrap_or_else(|| Any::new_invalid(InvalidReason::CreatedWithNoActiveEncoding));


        Self {
            bindings: any,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, AS, AM, L> BindingArray<Buffer<T, AS, AM>, L>
where
    T: BufferContent<AS, AM> + GpuStore + GpuLayout + NoHandles + NoBools,
    Buffer<T, AS, AM>: Binding,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
    L: ArrayLen,
{
    pub fn at<Idx: ToInteger>(&self, index: Idx) -> Buffer<T, AS, AM> {
        let ref_any = self.bindings.binding_array_index(index.to_any());
        Buffer::from_ref(ref_any.into())
    }
}

impl<Idx: ToInteger, T, AS, AM, L> GpuIndex<Idx> for BindingArray<Buffer<T, AS, AM>, L>
where
    T: BufferContent<AS, AM> + GpuStore + GpuLayout + NoHandles + NoBools,
    Buffer<T, AS, AM>: Binding,
    AS: BufferAddressSpace,
    AM: AccessModeReadable,
    L: ArrayLen,
{
    type Output = Buffer<T, AS, AM>;
    fn index(&self, index: Idx) -> Self::Output { self.at(index) }
}

impl<T: TextureHandle, L: ArrayLen> Binding for BindingArray<T, L> {
    fn binding_type() -> BindingType { T::binding_type() }
    fn binding_array_len() -> Option<Option<std::num::NonZeroU32>> { Some(L::LEN) }
    fn store_ty() -> StoreType { StoreType::BindingArray(Rc::new(T::store_ty()), L::LEN) }

    fn new_invalid(reason: InvalidReason) -> Self { Self::new_invalid(reason) }
    fn new_binding(args: BindingArgs) -> Self {
        let handle_type = T::texture_type();
        let any = Any::texture_binding_array(args.path, args.visibility, handle_type, L::LEN);
        Self {
            bindings: any,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: TextureHandle, L: ArrayLen> BindingArray<T, L> {
    pub fn at<Idx: ToInteger>(&self, index: Idx) -> T {
        let any = self.bindings.binding_array_index(index.to_any());
        T::from_any(any)
    }
}

impl<Idx: ToInteger, T: TextureHandle, L: ArrayLen> GpuIndex<Idx> for BindingArray<T, L> {
    type Output = T;
    fn index(&self, index: Idx) -> Self::Output { self.at(index) }
}
