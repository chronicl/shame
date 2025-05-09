use crate::ir;

use super::*;

impl TypeLayout<marker::Sized> {
    pub fn struct_builder<T: ValidOrSized>(
        struct_options: impl Into<StructOptions>,
        field_options: impl Into<FieldOptions>,
        layout: TypeLayout<T>,
    ) -> StructLayoutBuilder<T> {
        StructLayoutBuilder::new_struct(struct_options, field_options, layout)
    }

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
                unsafe_type_layout::new_type_layout(Some(size), align, Sem::Vector(*l, *t))
            }
            ir::SizedType::Matrix(c, r, t) => {
                unsafe_type_layout::new_type_layout(Some(size), align, Sem::Matrix(*c, *r, *t))
            }
            ir::SizedType::Array(sized, l) => Self::from_sized_array(rules, sized, *l),
            ir::SizedType::Atomic(t) => Self::from_sized_ty(rules, &ir::SizedType::Vector(ir::Len::X1, (*t).into())),
            // ir::SizedType guarantees that the TypeLayout is sized.
            ir::SizedType::Structure(s) => unsafe_type_layout::cast(TypeLayout::from_struct(rules, s)),
        }
    }

    pub fn from_sized_array(rules: TypeLayoutRules, element: &ir::SizedType, len: NonZeroU32) -> Self {
        unsafe_type_layout::new_type_layout(
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

    pub fn from_sized_struct(rules: TypeLayoutRules, s: &ir::SizedStruct) -> Self {
        let mut fields = s.fields();

        // Gpu structs have at least one field
        let first_field = fields.next().unwrap();

        let options = |field: &ir::SizedField| {
            FieldOptions::new(field.name.clone(), field.custom_min_align, field.custom_min_size)
        };
        let layout = |field: &ir::SizedField| TypeLayout::from_sized_ty(rules, first_field.ty());

        let mut builder = StructLayoutBuilder::<marker::Sized>::new_struct(
            s.name().clone(),
            options(first_field),
            layout(first_field),
        );

        for field in fields {
            builder = builder.extend(options(field), layout(field));
        }

        builder.finish()
    }
}

impl<T: TypeRestriction> TypeLayout<T>
where
    TypeLayout<marker::Sized>: From<TypeLayout<T>>,
{
    // TOOD(chronicl) find better name
    pub fn byte_size_sized(&self) -> u64 {
        // This type layout can infallibly be converted to TypeLayout<marker::Sized>
        // and sized types have a size.
        self.byte_size.unwrap()
    }
}


impl StructLayoutBuilder<marker::Sized> {
    /// Construct a new `TypeLayoutBuilder<Sized>` and immediately add it's first field.
    /// Having at least one field is a requirement for a gpu struct.
    pub fn new_struct<T: ValidOrSized>(
        struct_options: impl Into<StructOptions>,
        field_options: impl Into<FieldOptions>,
        layout: TypeLayout<T>,
    ) -> StructLayoutBuilder<T> {
        let mut this = unsafe_type_layout::new_builder::<marker::Sized>(struct_options);
        T::extend_builder(this, field_options.into(), layout)
    }

    pub fn extend<T: ValidOrSized>(
        mut self,
        options: impl Into<FieldOptions>,
        layout: TypeLayout<T>,
    ) -> StructLayoutBuilder<T> {
        T::extend_builder(self, options.into(), layout)
    }

    pub fn finish(self) -> TypeLayout<marker::Sized> { unsafe_type_layout::finish(self) }
}


pub trait ValidOrSized: TypeRestriction {
    fn extend_builder(
        builder: StructLayoutBuilder<marker::Sized>,
        options: FieldOptions,
        layout: TypeLayout<Self>,
    ) -> StructLayoutBuilder<Self>;
}

impl ValidOrSized for marker::Valid {
    fn extend_builder(
        builder: StructLayoutBuilder<marker::Sized>,
        options: FieldOptions,
        layout: TypeLayout<Self>,
    ) -> StructLayoutBuilder<Self> {
        StructLayoutBuilder::__new(builder.struct_builder, options, layout)
    }
}
impl ValidOrSized for marker::Sized {
    fn extend_builder(
        mut builder: StructLayoutBuilder<marker::Sized>,
        options: FieldOptions,
        layout: TypeLayout<Self>,
    ) -> StructLayoutBuilder<Self> {
        unsafe_type_layout::extend(&mut builder, options, layout);
        builder
    }
}

impl TryFrom<TypeLayout<marker::Valid>> for TypeLayout<marker::Sized> {
    type Error = ();

    fn try_from(layout: TypeLayout<marker::Valid>) -> Result<Self, Self::Error> {
        if layout.byte_size().is_some() {
            Ok(unsafe_type_layout::cast(layout))
        } else {
            Err(())
        }
    }
}
