use crate::hs::{Major, SizedField, TypeLayoutRules, UnsizedStruct};

use super::*;
use TypeLayoutSemantics as TLS;

impl TypeLayout<constraint::Uniform> {
    pub fn from_host_shareable_unchecked(ty: impl Into<HostShareableType>) -> Self {
        match ty.into() {
            HostShareableType::Sized(ty) => Self::from_sized_type(ty),
            HostShareableType::UnsizedStruct(ty) => Self::from_unsized_struct(ty),
            HostShareableType::RuntimeSizedArray(ty) => Self::from_runtime_sized_array(ty),
        }
    }

    pub fn from_host_shareable(ty: impl Into<hs::SizedType>) -> Self { Self::from_sized_type(ty.into()) }

    fn from_sized_type(ty: hs::SizedType) -> Self {
        type_layout_internal::cast_unchecked(from_sized_type(ty, TypeLayoutRules::Uniform))
    }

    fn from_unsized_struct(ty: hs::UnsizedStruct) -> Self {
        type_layout_internal::cast_unchecked(from_unsized_struct(ty, TypeLayoutRules::Uniform))
    }

    fn from_runtime_sized_array(ty: hs::RuntimeSizedArray) -> Self {
        type_layout_internal::cast_unchecked(from_runtime_sized_array(ty, TypeLayoutRules::Uniform))
    }

    pub fn get_host_shareable(&self) -> &HostShareableType {
        self.host_shareable
            .as_ref()
            .expect("constraint::Uniform should always have a host-shareable")
    }
}

impl TypeLayout<constraint::Storage> {
    pub fn from_host_shareable(ty: impl Into<HostShareableType>) -> Self {
        match ty.into() {
            HostShareableType::Sized(ty) => Self::from_sized_type(ty),
            HostShareableType::UnsizedStruct(ty) => Self::from_unsized_struct(ty),
            HostShareableType::RuntimeSizedArray(ty) => Self::from_runtime_sized_array(ty),
        }
    }


    fn from_sized_type(ty: hs::SizedType) -> Self {
        type_layout_internal::cast_unchecked(from_sized_type(ty, TypeLayoutRules::Storage))
    }

    fn from_unsized_struct(ty: hs::UnsizedStruct) -> Self {
        type_layout_internal::cast_unchecked(from_unsized_struct(ty, TypeLayoutRules::Storage))
    }

    fn from_runtime_sized_array(ty: hs::RuntimeSizedArray) -> Self {
        type_layout_internal::cast_unchecked(from_runtime_sized_array(ty, TypeLayoutRules::Storage))
    }

    pub fn get_host_shareable(&self) -> &HostShareableType {
        self.host_shareable
            .as_ref()
            .expect("constraint::Storage should always have a host-shareable")
    }
}

fn from_sized_type(ty: hs::SizedType, rules: TypeLayoutRules) -> TypeLayout {
    let (size, align, tls) = match &ty {
        hs::SizedType::Vector(v) => (v.byte_size(), v.byte_align(), TLS::Vector(v.len, v.scalar)),
        hs::SizedType::Atomic(a) => (a.byte_size(), a.byte_align(), TLS::Vector(ir::Len::X1, a.scalar.into())),
        hs::SizedType::Matrix(m) => (
            m.byte_size(rules, Major::Row),
            m.byte_align(rules, Major::Row),
            TLS::Matrix(m.columns, m.rows, m.scalar),
        ),
        hs::SizedType::Array(a) => (
            a.byte_size(rules),
            a.byte_align(rules),
            TLS::Array(
                Rc::new(ElementLayout {
                    byte_stride: a.byte_stride(rules),
                    ty: from_sized_type((*a.element).clone(), rules),
                }),
                Some(a.len.get()),
            ),
        ),
        hs::SizedType::Struct(s) => {
            let mut field_offsets = s.field_offsets(rules);
            let fields = (&mut field_offsets)
                .zip(s.fields())
                .map(|(offset, field)| sized_field_to_field_layout(field, offset, rules))
                .collect();

            (
                field_offsets.byte_size(),
                field_offsets.byte_align(),
                TLS::Structure(Rc::new(StructLayout {
                    name: s.name.clone().into(),
                    fields,
                })),
            )
        }
    };

    TypeLayout::new(Some(size), align, tls, Some(ty.into()))
}

fn from_unsized_struct(s: UnsizedStruct, rules: TypeLayoutRules) -> TypeLayout {
    let mut field_offsets = s.sized_field_offsets(rules);
    let mut fields = (&mut field_offsets)
        .zip(s.sized_fields.iter())
        .map(|(offset, field)| sized_field_to_field_layout(field, offset, rules))
        .collect::<Vec<_>>();

    let (field_offset, align) = s.last_field_offset_and_struct_align(field_offsets);
    fields.push(FieldLayoutWithOffset {
        rel_byte_offset: field_offset,
        field: FieldLayout {
            name: s.last_unsized.name.clone(),
            custom_min_size: None.into(),
            custom_min_align: s.last_unsized.custom_min_align.into(),
            ty: from_runtime_sized_array(s.last_unsized.array.clone(), rules),
        },
    });


    TypeLayout::new(
        None,
        align,
        TLS::Structure(Rc::new(StructLayout {
            name: s.name.clone().into(),
            fields,
        })),
        Some(s.into()),
    )
}

fn sized_field_to_field_layout(field: &SizedField, offset: u64, rules: TypeLayoutRules) -> FieldLayoutWithOffset {
    FieldLayoutWithOffset {
        rel_byte_offset: offset,
        field: FieldLayout {
            name: field.name.clone(),
            custom_min_size: field.custom_min_size.into(),
            custom_min_align: field.custom_min_align.into(),
            ty: from_sized_type(field.ty.clone(), rules),
        },
    }
}


fn from_runtime_sized_array(ty: hs::RuntimeSizedArray, rules: TypeLayoutRules) -> TypeLayout {
    TypeLayout::new(
        None,
        ty.byte_align(rules),
        TLS::Array(
            Rc::new(ElementLayout {
                byte_stride: ty.byte_stride(rules),
                ty: from_sized_type(ty.element.clone(), rules),
            }),
            None,
        ),
        Some(ty.into()),
    )
}


impl<'a> TryFrom<&'a TypeLayout<constraint::Storage>> for TypeLayout<constraint::Uniform> {
    type Error = ();

    // TODO(chronicl) this should get a really nice error
    fn try_from(s_layout: &'a TypeLayout<constraint::Storage>) -> Result<Self, Self::Error> {
        // TODO(chronicl) consider whether to use unchecked or not here
        let u_layout =
            TypeLayout::<constraint::Uniform>::from_host_shareable_unchecked(s_layout.get_host_shareable().clone());

        (&u_layout == s_layout).then_some(u_layout).ok_or(())
    }
}
