#![allow(missing_docs)]
use std::rc::Rc;

use crate::{
    any::{Any, BindingType},
    call_info,
    common::proc_macro_reexports::BindingArgs,
    frontend::{
        any::InvalidReason,
        rust_types::{layout_traits::get_layout_compare_with_cpu_push_error, vec::ToInteger},
    },
    ir::{recording::Context, StoreType},
    mem::Storage,
    ArrayLen, Binding, Buffer, BufferAddressSpace, GpuIndex, GpuStore, NoAtomics, NoBools, NoHandles, RuntimeSize,
};

pub struct BindingArray<T, L = RuntimeSize> {
    pub bindings: Any,
    _phantom: std::marker::PhantomData<(T, L)>,
}

// Binding arrays don't support dynamic offsets in wgpu current,
// so it's only implemented for buffers with DYNAMIC_OFFSET = false
impl<T, L: ArrayLen> Binding for BindingArray<Buffer<T, Storage, false>, L>
where
    T: GpuStore + NoHandles + NoAtomics + NoBools,
    Buffer<T, Storage, false>: Binding,
{
    fn binding_type() -> BindingType {
        BindingType::BindingArray(Rc::new(<Buffer<T, Storage, false>>::binding_type()), L::LEN)
    }

    fn store_ty() -> StoreType { StoreType::BindingArray(Rc::new(<Buffer<T, Storage, false>>::store_ty()), L::LEN) }

    fn new_binding(args: Result<BindingArgs, InvalidReason>) -> Self {
        let skip_stride_check = true; // not a vertex buffer
        Context::try_with(call_info!(), |ctx| {
            get_layout_compare_with_cpu_push_error::<T>(ctx, skip_stride_check)
        });
        let as_ref = false;
        let any = match args {
            Err(reason) => Any::new_invalid(reason),
            Ok(BindingArgs { path, visibility }) => {
                Any::binding(path, visibility, Self::store_ty(), Self::binding_type(), as_ref)
            }
        };
        Self {
            bindings: any,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<Idx: ToInteger, T, L> GpuIndex<Idx> for BindingArray<T, L> {
    type Output = T;

    fn index(&self, index: Idx) -> Self::Output { todo!() }
}
