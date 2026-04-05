use crate::{
    call_info,
    frontend::{
        any::{Any, InvalidReason},
    },
    ir::{self, recording::Context, type_layout::TypeLayout},
};

use std::{
    borrow::{Borrow, Cow},
    ops::Deref,
};

use super::layout_traits::{GetAllFields, GpuLayout};
use super::type_traits::{GpuSized, GpuStore, GpuStoreImplCategory, NoBools};
use super::{
    layout_traits::{ArrayElementsUnsizedError, FromAnys},
    mem::AddressSpace,
    reference::AccessMode,
    type_traits::{NoAtomics, NoHandles},
    typecheck_downcast, AsAny,
};
use super::{GpuType, ToGpuType};

// the fields of a buffer binding, vertex buffer or struct (each of those options has their own additional bounds).
// May contain atomics, packed vectors, or a runtime-sized `Array<T>` at the last field
// TODO(release) consider renaming to StoreFields
/// (no documentation yet)
pub trait BufferFields: GpuStore + GpuLayout + NoHandles + FromAnys + GetAllFields {
    /// behaves just like `Clone::clone`.
    /// this exists only to not require #[derive(Clone)] on every #[derive(GpuLayout)]
    #[doc(hidden)]
    fn clone_fields(&self) -> Self;
}
