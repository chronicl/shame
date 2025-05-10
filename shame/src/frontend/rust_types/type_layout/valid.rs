use std::rc::Rc;

use crate::ir::ir_type::{
    align_of_array_from_element_alignment, byte_size_of_array_from_stride_len, stride_of_array_from_element_align_size,
};

use super::*;

impl TypeLayout {
    /// Creates a new `TypeLayout` from the given struct options and fields.
    pub fn struct_from_parts(
        struct_options: impl Into<StructOptions>,
        mut fields: impl ExactSizeIterator<Item = (FieldOptions, TypeLayout)>,
    ) -> Result<TypeLayout, StructLayoutError> {
        let struct_options = struct_options.into();
        let num_fields = fields.len();

        let make_sized = |layout: TypeLayout, field_index: usize| {
            layout
                .try_into_sized()
                .map_err(|layout| StructLayoutError::UnsizedFieldMustBeLast {
                    struct_name: struct_options.name.clone(),
                    unsized_field_index: field_index,
                    num_fields,
                })
        };

        let (options, layout) = fields
            .next()
            .ok_or_else(|| StructLayoutError::HasNoFields(struct_options.name.clone()))?;
        let layout = make_sized(layout, 0)?;
        let mut builder = StructLayoutBuilder::new_struct(struct_options.clone(), options, layout);

        // This leaves the last field in field when finishing the for loop
        let mut field = None;
        let mut field_index = 1;
        for next_field in fields {
            let previous_field = field.replace(next_field);
            if let Some((options, layout)) = previous_field {
                builder = builder.extend(options, make_sized(layout, field_index)?);
                field_index += 1;
            }
        }

        // last field may be unsized
        if let Some((options, layout)) = field {
            return Ok(builder.extend(options, layout).finish());
        }
        Ok(builder.finish().into())
    }

    /// Attempts to convert this type layout into a sized type layout.
    pub fn try_into_sized(self) -> Result<TypeLayout<constraint::Sized>, TypeLayout> {
        if self.byte_size().is_some() {
            Ok(type_layout_internal::cast(self))
        } else {
            Err(self)
        }
    }

    /// Creates a new type layout from an intermediate representation type.
    pub fn from_ty(rules: TypeLayoutRules, ty: &ir::Type) -> Result<Self, TypeLayoutError> {
        match ty {
            Type::Unit | Type::Ptr(_, _, _) | Type::Ref(_, _, _) => {
                Err(TypeLayoutError::LayoutUndefined(ty.clone(), rules))
            }
            Type::Store(ty) => Self::from_store_ty(rules, ty),
        }
    }

    /// Creates a type layout for an array type from an intermediate representation.
    pub fn from_array_ir(rules: TypeLayoutRules, element: &ir::SizedType, len: Option<NonZeroU32>) -> Self {
        type_layout_internal::new_type_layout(
            len.map(|n| byte_size_of_array(element, n)),
            align_of_array(element),
            TypeLayoutSemantics::Array(
                Rc::new(ElementLayout {
                    byte_stride: match rules {
                        TypeLayoutRules::Wgsl => stride_of_array(element),
                    },
                    ty: TypeLayout::from_sized_ty(rules, element).into(),
                }),
                len.map(NonZeroU32::get),
            ),
        )
    }

    /// Creates a type layout for an array type from an element layout and a length.
    pub fn from_array(rules: TypeLayoutRules, element: TypeLayout<constraint::Sized>, len: Option<NonZeroU32>) -> Self {
        let byte_stride = match rules {
            TypeLayoutRules::Wgsl => {
                stride_of_array_from_element_align_size(element.align(), element.byte_size_sized())
            }
        };
        type_layout_internal::new_type_layout(
            len.map(|n| byte_size_of_array_from_stride_len(byte_stride, n.get().into())),
            align_of_array_from_element_alignment(element.align()),
            TypeLayoutSemantics::Array(
                Rc::new(ElementLayout {
                    byte_stride,
                    ty: element.into(),
                }),
                len.map(NonZeroU32::get),
            ),
        )
    }

    /// Creates a type layout for a struct type.
    pub fn from_struct(rules: TypeLayoutRules, s: &ir::Struct) -> Self {
        let sized_fields = s.sized_fields();

        let array_options =
            |field: &ir::RuntimeSizedArrayField| FieldOptions::new(field.name.clone(), field.custom_min_align, None);

        if sized_fields.is_empty() {
            // ir::Struct always has at least one field and if there are no sized_fields (checked above),
            // then there must be an unsized field.
            let array = s.last_unsized_field().as_ref().unwrap();
            let array_layout = TypeLayout::from_array_ir(rules, &array.element_ty, None);
            StructLayoutBuilder::new_struct(s.name().clone(), array_options(array), array_layout).finish()
        } else {
            // checked above that at least one sized field exists
            let first_field = &sized_fields[0];

            let options = |field: &ir::SizedField| {
                FieldOptions::new(field.name.clone(), field.custom_min_align, field.custom_min_size)
            };
            let layout = |field: &ir::SizedField| -> TypeLayout<constraint::Sized> {
                TypeLayout::from_sized_ty(rules, field.ty())
            };

            let mut builder = StructLayoutBuilder::<constraint::Sized>::new_struct(
                s.name().clone(),
                options(first_field),
                layout(first_field),
            );

            for field in &sized_fields[1..] {
                builder = builder.extend(options(field), layout(field));
            }

            if let Some(array) = s.last_unsized_field() {
                let array_layout = TypeLayout::from_array_ir(rules, &array.element_ty, None);
                let builder = builder.extend(array_options(array), array_layout);
                builder.finish()
            } else {
                builder.finish().into()
            }
        }
    }

    /// Creates a type layout from a store type.
    pub fn from_store_ty(rules: TypeLayoutRules, ty: &ir::StoreType) -> Result<Self, TypeLayoutError> {
        match ty {
            ir::StoreType::Sized(sized) => Ok(TypeLayout::from_sized_ty(rules, sized).into()),
            ir::StoreType::Handle(handle) => Err(TypeLayoutError::LayoutUndefined(ir::Type::Store(ty.clone()), rules)),
            ir::StoreType::RuntimeSizedArray(element) => Ok(TypeLayout::from_array_ir(rules, element, None)),
            ir::StoreType::BufferBlock(s) => Ok(TypeLayout::from_struct(rules, s)),
        }
    }

    /// Creates a type layout from an aligned type.
    pub fn from_aligned_type(rules: TypeLayoutRules, ty: &AlignedType) -> Self {
        match ty {
            AlignedType::Sized(sized) => TypeLayout::from_sized_ty(rules, sized).into(),
            AlignedType::RuntimeSizedArray(element) => Self::from_array_ir(rules, element, None),
        }
    }
}

impl StructLayoutBuilder<constraint::Basic> {
    pub(super) fn __new(
        struct_builder: StructLayoutBuilderErased,
        options: FieldOptions,
        last_field: TypeLayout<constraint::Basic>,
    ) -> Self {
        StructLayoutBuilder {
            struct_builder,
            last_maybe_unsized_field: Some((last_field, options)),
            _phantom: PhantomData,
        }
    }

    /// Finalizes the struct layout builder and returns the complete
    /// `TypeLayout<constraint::Basic>` for the (potentially unsized) struct.
    pub fn finish(mut self) -> TypeLayout {
        // TypeLayoutBuilder is only constructed through the `__new` method,
        // which guarantees this is Some
        let (layout, options) = self.last_maybe_unsized_field.take().unwrap();
        type_layout_internal::finish_maybe_unsized(self, layout, options)
    }
}

impl TryFrom<TypeLayout<constraint::Unconstraint>> for TypeLayout<constraint::Basic> {
    type Error = ();

    fn try_from(layout: TypeLayout<constraint::Unconstraint>) -> Result<Self, Self::Error> {
        layout.try_into_basic(false, TypeLayoutRules::Wgsl)
    }
}
