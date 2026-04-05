use super::*;
use crate::{
    frontend::rust_types::gpu_type2::align_size::{FieldOffsets, FieldOffsetsSized, FieldOffsetsUnsized},
    ir::ScalarType,
};

//            Size and align of layout recipe types             //
// https://www.w3.org/TR/WGSL/#address-space-layout-constraints //

pub(crate) const PACKED_ALIGN: U32PowerOf2 = U32PowerOf2::_1;

impl LayoutType {
    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the align.
    pub fn byte_size(&self, default_repr: Repr) -> Option<u64> { self.borrow_with(|l| l.byte_size(default_repr)) }

    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the size.
    pub fn align(&self, default_repr: Repr) -> U32PowerOf2 { self.borrow_with(|l| l.align(default_repr)) }

    /// This is expensive for structs as it calculates the byte size and align by traversing all fields recursively.
    pub fn byte_size_and_align(&self, default_repr: Repr) -> (Option<u64>, U32PowerOf2) {
        self.borrow_with(|l| l.byte_size_and_align(default_repr))
    }

    /// Returns a copy of self, but with all struct reprs changed to `repr`.
    pub fn to_unified_repr(&self, repr: Repr) -> Self {
        let mut this = self.clone();
        this.change_all_repr(repr);
        this
    }

    /// Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) {
        match self {
            LayoutType::Sized(s) => s.change_all_repr(repr),
            LayoutType::UnsizedStruct(s) => s.change_all_repr(repr),
            LayoutType::RuntimeSizedArray(a) => a.change_all_repr(repr),
        }
    }
}

impl SizedType {
    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the align.
    pub fn byte_size(&self, parent_repr: Repr) -> u64 { self.byte_size_and_align(parent_repr).0 }

    /// This is expensive for structs. Prefer `byte_size_and_align` if you also need the size.
    pub fn align(&self, parent_repr: Repr) -> U32PowerOf2 { self.byte_size_and_align(parent_repr).1 }

    /// This is expensive for structs as it calculates the byte size and align by traversing all fields recursively.
    pub fn byte_size_and_align(&self, parent_repr: Repr) -> (u64, U32PowerOf2) {
        self.borrow_with(|s| s.byte_size_and_align(parent_repr))
    }

    /// Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) {
        match self {
            SizedType::Struct(s) => s.change_all_repr(repr),
            SizedType::Array(s) => s.change_all_repr(repr),
            SizedType::Atomic(_) | SizedType::Vector(_) | SizedType::Matrix(_) => {
                // No repr to change for these types.
            }
        }
    }
}

impl SizedStruct {
    /// Returns [`FieldOffsetsSized`], which serves as an iterator over the offsets of the
    /// fields of this struct. `FieldOffsetsSized::struct_byte_size_and_align` can be
    /// used to efficiently obtain the byte_size and align.
    pub fn field_offsets<R>(&self, f: impl FnOnce(FieldOffsetsSized<'_>) -> R) -> R {
        self.borrow_with(|s| f(s.field_offsets()))
    }

    /// Returns (byte_size, align)
    ///
    /// This is expensive for structs as it calculates the byte size and align by traversing all fields recursively.
    pub fn byte_size_and_align(&self) -> (u64, U32PowerOf2) { self.borrow_with(|s| s.byte_size_and_align()) }

    /// Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) {
        self.repr = repr;
        for field in &mut self.fields {
            field.ty.change_all_repr(repr);
        }
    }
}

impl UnsizedStruct {
    /// Returns [`FieldOffsetsUnsized`].
    ///
    /// - Use [`FieldOffsetsUnsized::sized_field_offsets`] for an iterator over the sized field offsets.
    /// - Use [`FieldOffsetsUnsized::last_field_offset_and_struct_align`] for the last field's offset
    ///   and the struct's align
    pub fn field_offsets<R>(&self, f: impl FnOnce(FieldOffsetsUnsized<'_>) -> R) -> R {
        self.borrow_with(|s| f(s.field_offsets()))
    }

    /// Size of the struct ignoring the last unsized field
    pub fn min_byte_size(&self) -> u64 { self.borrow_with(|s| s.min_byte_size()) }

    /// This is expensive as it calculates the byte align by traversing all fields recursively.
    pub fn align(&self) -> U32PowerOf2 { self.borrow_with(|s| s.align()) }

    /// Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) {
        self.repr = repr;
        for field in &mut self.sized_fields {
            field.ty.change_all_repr(repr);
        }
        self.last_unsized.array.change_all_repr(repr);
    }
}



#[allow(missing_docs)]
impl SizedArray {
    pub fn byte_size(&self, repr: Repr) -> u64 { self.borrow_with(|a| a.byte_size(repr)) }

    pub fn align(&self, repr: Repr) -> U32PowerOf2 { self.borrow_with(|a| a.align(repr)) }

    pub fn byte_stride(&self, repr: Repr) -> u64 { self.borrow_with(|a| a.byte_stride(repr)) }

    // Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) {
        let mut element = (*self.element).clone();
        element.change_all_repr(repr);
        self.element = Rc::new(element);
    }
}

#[allow(missing_docs)]
impl RuntimeSizedArray {
    pub fn align(&self, parent_repr: Repr) -> U32PowerOf2 { self.borrow_with(|a| a.align(parent_repr)) }

    pub fn byte_stride(&self, parent_repr: Repr) -> u64 { self.borrow_with(|a| a.byte_stride(parent_repr)) }

    // Recursively changes all struct reprs to the given `repr`.
    pub fn change_all_repr(&mut self, repr: Repr) { self.element.change_all_repr(repr); }
}

#[allow(missing_docs)]
impl SizedField {
    pub fn byte_size(&self, repr: Repr) -> u64 { self.borrow_with(|f| f.byte_size(repr)) }
    pub fn align(&self, repr: Repr) -> U32PowerOf2 { self.borrow_with(|f| f.align(repr)) }
}

#[allow(missing_docs)]
impl RuntimeSizedArrayField {
    pub fn align(&self, repr: Repr) -> U32PowerOf2 { self.borrow_with(|f| f.align(repr)) }
}
