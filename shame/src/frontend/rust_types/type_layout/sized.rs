use crate::ir::{self, PackedVector};

use super::*;

impl TypeLayout<constraint::Sized> {
    /// Create a new `StructLayoutBuilder<T>`, which builds the type layout of a struct.
    ///
    /// It produces a `TypeLayout<restrict::Sized>` if only `TypeLayout<restrict::Sized>`
    /// fields are used, otherwise it produces a `TypeLayout<restrict::Basic>`.
    ///
    /// `layout` may either be `TypeLayout<restrict::Sized> or `TypeLayout<restrict::Basic>`.
    pub fn struct_builder<T: ValidOrSized>(
        struct_options: impl Into<StructOptions>,
        field_options: impl Into<FieldOptions>,
        layout: TypeLayout<T>,
    ) -> StructLayoutBuilder<T> {
        StructLayoutBuilder::new_struct(struct_options, field_options, layout)
    }

    /// Create a new `TypeLayout<restrict::Sized> from a sized types intermediate representation.
    pub fn from_sized_ty(rules: TypeLayoutRules, ty: &ir::SizedType) -> Self {
        use TypeLayoutSemantics as Sem;
        let size = ty.byte_size();
        let align = ty.align();
        match ty {
            ir::SizedType::Vector(l, t) =>
            // we treat bool as a type that has a layout to allow for an
            // `Eq` operator on `TypeLayout` that behaves intuitively.
            // The layout of `bool`s is not actually observable in any part of the api.
            {
                type_layout_internal::new_type_layout(Some(size), align, Sem::Vector(*l, *t))
            }
            ir::SizedType::Matrix(c, r, t) => {
                type_layout_internal::new_type_layout(Some(size), align, Sem::Matrix(*c, *r, *t))
            }
            ir::SizedType::Array(sized, l) => Self::from_sized_array(rules, sized, *l),
            ir::SizedType::Atomic(t) => Self::from_sized_ty(rules, &ir::SizedType::Vector(ir::Len::X1, (*t).into())),
            // ir::SizedType guarantees that the TypeLayout is sized.
            ir::SizedType::Structure(s) => type_layout_internal::cast(TypeLayout::from_struct(rules, s)),
        }
    }

    /// Create a new `TypeLayout<restrict::Sized> of an array from an element type and an array length.
    pub fn from_sized_array(rules: TypeLayoutRules, element: &ir::SizedType, len: NonZeroU32) -> Self {
        type_layout_internal::new_type_layout(
            Some(byte_size_of_array(element, len)),
            align_of_array(element),
            TypeLayoutSemantics::Array(
                Rc::new(ElementLayout {
                    byte_stride: match rules {
                        TypeLayoutRules::Wgsl => stride_of_array(element),
                    },
                    ty: Self::from_sized_ty(rules, element).into(),
                }),
                Some(len.get()),
            ),
        )
    }

    /// Create a new `TypeLayout<restrict::Sized> of a sized from it's intermediate representation.
    pub fn from_sized_struct(rules: TypeLayoutRules, s: &ir::SizedStruct) -> Self {
        let mut fields = s.fields();

        // Gpu structs have at least one field
        let first_field = fields.next().unwrap();

        let options = |field: &ir::SizedField| {
            FieldOptions::new(field.name.clone(), field.custom_min_align, field.custom_min_size)
        };
        let layout = |field: &ir::SizedField| TypeLayout::from_sized_ty(rules, first_field.ty());

        let mut builder = StructLayoutBuilder::<constraint::Sized>::new_struct(
            s.name().clone(),
            options(first_field),
            layout(first_field),
        );

        for field in fields {
            builder = builder.extend(options(field), layout(field));
        }

        builder.finish()
    }

    /// Create a new `TypeLayout<constraint::Sized>` from a packed vector.
    pub fn from_packed_vector(rules: TypeLayoutRules, packed_vec: PackedVector) -> Self {
        type_layout_internal::new_type_layout(
            Some(u8::from(packed_vec.byte_size()) as u64),
            packed_vec.align(),
            TypeLayoutSemantics::PackedVector(packed_vec),
        )
    }
}

impl<T: TypeConstraint> TypeLayout<T>
where
    TypeLayout<constraint::Sized>: From<TypeLayout<T>>,
{
    /// Get the byte size of the type the layout represents. This does not return an `Option`
    /// in contrast to [`TypeLayout::byte_size`].
    pub fn byte_size_sized(&self) -> u64 {
        // This type layout can infallibly be converted to TypeLayout<marker::Sized>
        // and sized types have a size.
        self.byte_size.unwrap()
    }
}


impl StructLayoutBuilder<constraint::Sized> {
    /// Construct a new `TypeLayoutBuilder<Sized>` and immediately add it's first field.
    /// Having at least one field is a requirement for a gpu struct.
    pub fn new_struct<T: ValidOrSized>(
        struct_options: impl Into<StructOptions>,
        field_options: impl Into<FieldOptions>,
        layout: TypeLayout<T>,
    ) -> StructLayoutBuilder<T> {
        let mut this = type_layout_internal::new_builder::<constraint::Sized>(struct_options);
        T::extend_builder(this, field_options.into(), layout)
    }

    /// Extends the struct layout builder with an additional field.
    /// Takes the field options and layout for the new field and returns
    /// the updated builder.
    pub fn extend<T: ValidOrSized>(
        mut self,
        options: impl Into<FieldOptions>,
        layout: TypeLayout<T>,
    ) -> StructLayoutBuilder<T> {
        T::extend_builder(self, options.into(), layout)
    }

    /// Finalizes the struct layout builder and returns the complete
    /// `TypeLayout<constraint::Sized>` for the sized struct.
    pub fn finish(self) -> TypeLayout<constraint::Sized> { type_layout_internal::finish(self) }
}

/// A trait that [`constraint::Basic`] and [`constraint::Sized`] implement.
/// It is used to make `StructLayoutBuilder::<contraint::Sized>::extend(field_layout: TypeLayout<T>)`
/// return a `StructLayoutBuilder<T>`.
pub trait ValidOrSized: TypeConstraint {
    /// Extends the struct layout builder by a new field.
    fn extend_builder(
        builder: StructLayoutBuilder<constraint::Sized>,
        options: FieldOptions,
        layout: TypeLayout<Self>,
    ) -> StructLayoutBuilder<Self>;
}

impl ValidOrSized for constraint::Basic {
    fn extend_builder(
        builder: StructLayoutBuilder<constraint::Sized>,
        options: FieldOptions,
        layout: TypeLayout<Self>,
    ) -> StructLayoutBuilder<Self> {
        StructLayoutBuilder::__new(builder.struct_builder, options, layout)
    }
}
impl ValidOrSized for constraint::Sized {
    fn extend_builder(
        mut builder: StructLayoutBuilder<constraint::Sized>,
        options: FieldOptions,
        layout: TypeLayout<Self>,
    ) -> StructLayoutBuilder<Self> {
        type_layout_internal::extend(&mut builder, options, layout);
        builder
    }
}

impl TryFrom<TypeLayout<constraint::Basic>> for TypeLayout<constraint::Sized> {
    type Error = ();

    fn try_from(layout: TypeLayout<constraint::Basic>) -> Result<Self, Self::Error> {
        if layout.byte_size().is_some() {
            Ok(type_layout_internal::cast(layout))
        } else {
            Err(())
        }
    }
}
