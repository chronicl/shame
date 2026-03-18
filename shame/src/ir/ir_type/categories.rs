use crate::{
    frontend::any::shared_io::BufferBindingType,
    ir::{Type, LayoutType, SizedType, ScalarType},
};

use super::{StoreType};

impl Type {
    #[allow(missing_docs)] // runtime api
    pub fn is_host_shareable(&self) -> bool {
        match self {
            Type::Store(store) => store.is_host_shareable(),
            _ => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_constructible(&self) -> bool {
        match self {
            Type::Store(store) => store.is_constructible(),
            _ => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_creation_fixed_footprint(&self) -> bool {
        match self {
            Type::Store(store) => store.is_creation_fixed_footprint(),
            _ => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_plain_and_fixed_footprint(&self) -> bool {
        match self {
            Type::Store(store) => store.is_plain_and_fixed_footprint(),
            _ => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn contains_atomics(&self) -> bool {
        match self {
            Type::Store(store) => store.contains_atomics(),
            _ => false,
        }
    }
}

impl StoreType {
    // https://www.w3.org/TR/WGSL/#host-shareable-types
    // every `SizedType` or `RuntimeSizedArray`
    // that doesn't contain any `bool`s.
    #[allow(missing_docs)] // runtime api
    pub fn is_host_shareable(&self) -> bool {
        use StoreType as T;
        match self {
            StoreType::Layout(l) => l.is_host_shareable(),
            T::Handle(_) => false,
            // TODO(chronicl) check if correct
            T::BindingArray(s, _) => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_creation_fixed_footprint(&self) -> bool {
        match self {
            StoreType::Layout(LayoutType::Sized(_)) => true,
            StoreType::Layout(LayoutType::UnsizedStruct(_)) |
            StoreType::Layout(LayoutType::RuntimeSizedArray(_)) |
            StoreType::Handle(_) |
            StoreType::BindingArray(_, _) => false,
        }
    }

    /// see https://www.w3.org/TR/WGSL/#fixed-footprint-types
    pub fn is_plain_and_fixed_footprint(&self) -> bool {
        match self {
            StoreType::Layout(LayoutType::Sized(_)) => true,
            StoreType::Layout(LayoutType::UnsizedStruct(_)) |
            StoreType::Layout(LayoutType::RuntimeSizedArray(_)) |
            StoreType::Handle(_) |
            StoreType::BindingArray(_, _) => false,
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn contains_atomics(&self) -> bool {
        match self {
            StoreType::Layout(l) => l.contains_atomics(),
            StoreType::Handle(_) => false,
            StoreType::BindingArray(s, _) => s.contains_atomics(),
        }
    }

    /// see https://www.w3.org/TR/WGSL/#constructible-types
    pub fn is_constructible(&self) -> bool {
        match self {
            StoreType::Layout(LayoutType::Sized(s)) => s.is_constructible(),
            StoreType::Layout(LayoutType::UnsizedStruct(_)) |
            StoreType::Layout(LayoutType::RuntimeSizedArray(_)) |
            StoreType::Handle(_) |
            StoreType::BindingArray(_, _) => false,
        }
    }
}

impl LayoutType {
    /// Whether LayoutType is host shareable (wgsl spec)
    pub fn is_host_shareable(&self) -> bool {
        match self {
            LayoutType::Sized(sized) => sized.is_host_shareable(),
            LayoutType::UnsizedStruct(s) => {
                s.sized_fields.iter().all(|t| t.ty.is_host_shareable()) &&
                    s.last_unsized.element_ty().is_host_shareable()
            }
            LayoutType::RuntimeSizedArray(a) => a.element.is_host_shareable(),
        }
    }

    /// Whether LayoutType contains atomics
    pub fn contains_atomics(&self) -> bool {
        match self {
            LayoutType::Sized(sized) => sized.contains_atomics(),
            LayoutType::UnsizedStruct(s) => {
                s.sized_fields.iter().any(|t| t.ty.contains_atomics()) || s.last_unsized.element_ty().contains_atomics()
            }
            LayoutType::RuntimeSizedArray(a) => a.element.contains_atomics(),
        }
    }
}

impl SizedType {
    #[allow(missing_docs)] // runtime api
    pub fn is_host_shareable(&self) -> bool {
        use SizedType as T;
        match self {
            T::Vector(v) => v.scalar.is_host_shareable(),
            T::Matrix(m) => ScalarType::from(m.scalar).is_host_shareable(),
            T::Atomic(a) => ScalarType::from(a.scalar).is_host_shareable(),
            T::Array(a) => a.element.is_host_shareable(),
            T::Struct(s) => s.fields.iter().all(|t| t.ty.is_host_shareable()),
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn contains_atomics(&self) -> bool {
        use SizedType as T;
        match self {
            T::Atomic(_) => true,
            T::Vector(_) | T::Matrix(_) => false,
            T::Array(a) => a.element.contains_atomics(),
            T::Struct(s) => s.fields.iter().any(|t| t.ty.contains_atomics()),
        }
    }

    #[allow(missing_docs)] // runtime api
    pub fn is_constructible(&self) -> bool {
        match self {
            SizedType::Vector(_) => true,
            SizedType::Matrix(_) => true,
            SizedType::Array(a) => a.element.is_constructible(),
            SizedType::Atomic(_) => false,
            SizedType::Struct(sized_struct) => sized_struct.fields.iter().all(|f| f.ty.is_constructible()),
        }
    }
}

impl ScalarType {
    #[allow(missing_docs)] // runtime api
    pub fn is_host_shareable(&self) -> bool {
        use ScalarType as T;
        match self {
            T::F16 | T::F32 | T::F64 | T::U32 | T::I32 => true,
            T::Bool => false,
        }
    }
}
