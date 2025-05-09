use super::*;

impl TypeLayout {
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

        if num_fields == 1 {
            return Ok(builder.finish().into());
        }
        for (i, (options, layout)) in fields.enumerate() {
            let field_index = i + 1;
            let is_last_field = field_index == num_fields - 1;
            if is_last_field {
                return Ok(builder.extend(options, layout).finish());
            } else {
                let layout = make_sized(layout, i)?;
                builder = builder.extend(options, layout);
            }
        }
        // TODO(chronicl) this is actually unreachable, but should replace this with different builder api anyway
        Ok(builder.finish().into())
    }


    pub fn try_into_sized(self) -> Result<TypeLayout<marker::Sized>, TypeLayout> {
        if self.byte_size().is_some() {
            Ok(unsafe_type_layout::cast(self))
        } else {
            Err(self)
        }
    }

    pub fn from_ty(rules: TypeLayoutRules, ty: &ir::Type) -> Result<Self, TypeLayoutError> {
        match ty {
            Type::Unit | Type::Ptr(_, _, _) | Type::Ref(_, _, _) => {
                Err(TypeLayoutError::LayoutUndefined(ty.clone(), rules))
            }
            Type::Store(ty) => Self::from_store_ty(rules, ty),
        }
    }

    pub fn from_array(rules: TypeLayoutRules, element: &ir::SizedType, len: Option<NonZeroU32>) -> Self {
        unsafe_type_layout::new_type_layout(
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

    pub fn from_struct(rules: TypeLayoutRules, s: &ir::Struct) -> Self {
        let sized_fields = s.sized_fields();

        let array_options =
            |field: &ir::RuntimeSizedArrayField| FieldOptions::new(field.name.clone(), field.custom_min_align, None);

        if sized_fields.is_empty() {
            // ir::Struct always has at least one field and if there are no sized_fields (checked above),
            // then there must be an unsized field.
            let array = s.last_unsized_field().as_ref().unwrap();
            let array_layout = TypeLayout::from_array(rules, &array.element_ty, None);
            StructLayoutBuilder::new_struct(s.name().clone(), array_options(array), array_layout).finish()
        } else {
            // checked above that at least one sized field exists
            let first_field = &sized_fields[0];

            let options = |field: &ir::SizedField| {
                FieldOptions::new(field.name.clone(), field.custom_min_align, field.custom_min_size)
            };
            let layout =
                |field: &ir::SizedField| -> TypeLayout<marker::Sized> { TypeLayout::from_sized_ty(rules, field.ty()) };

            let mut builder = StructLayoutBuilder::<marker::Sized>::new_struct(
                s.name().clone(),
                options(first_field),
                layout(first_field),
            );

            for field in &sized_fields[1..] {
                builder = builder.extend(options(field), layout(field));
            }

            if let Some(array) = s.last_unsized_field() {
                let array_layout = TypeLayout::from_array(rules, &array.element_ty, None);
                let builder = builder.extend(array_options(array), array_layout);
                builder.finish()
            } else {
                builder.finish().into()
            }
        }
    }

    pub fn from_store_ty(rules: TypeLayoutRules, ty: &ir::StoreType) -> Result<Self, TypeLayoutError> {
        match ty {
            ir::StoreType::Sized(sized) => Ok(TypeLayout::from_sized_ty(rules, sized).into()),
            ir::StoreType::Handle(handle) => Err(TypeLayoutError::LayoutUndefined(ir::Type::Store(ty.clone()), rules)),
            ir::StoreType::RuntimeSizedArray(element) => Ok(TypeLayout::from_array(rules, element, None)),
            ir::StoreType::BufferBlock(s) => Ok(TypeLayout::from_struct(rules, s)),
        }
    }


    pub fn from_aligned_type(rules: TypeLayoutRules, ty: &AlignedType) -> Self {
        match ty {
            AlignedType::Sized(sized) => TypeLayout::from_sized_ty(rules, sized).into(),
            AlignedType::RuntimeSizedArray(element) => Self::from_array(rules, element, None),
        }
    }
}

impl StructLayoutBuilder<marker::Valid> {
    pub(super) fn __new(
        struct_builder: StructLayoutBuilderErased,
        options: FieldOptions,
        last_field: TypeLayout<marker::Valid>,
    ) -> Self {
        StructLayoutBuilder {
            struct_builder,
            last_maybe_unsized_field: Some((last_field, options)),
            _phantom: PhantomData,
        }
    }

    pub fn finish(mut self) -> TypeLayout {
        // TypeLayoutBuilder is only constructed through the `__new` method,
        // which guarantees this is Some
        let (layout, options) = self.last_maybe_unsized_field.take().unwrap();
        unsafe_type_layout::finish_maybe_unsized(self, layout, options)
    }
}
