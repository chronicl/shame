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
        T::extend_builder(this, layout, field_options.into())
    }

    pub fn extend<T: ValidOrSized>(
        mut self,
        options: impl Into<FieldOptions>,
        layout: TypeLayout<T>,
    ) -> StructLayoutBuilder<T> {
        T::extend_builder(self, layout, options.into())
    }

    pub fn finish(self) -> TypeLayout<marker::Sized> { unsafe_type_layout::finish(self) }
}


pub trait ValidOrSized: TypeRestriction {
    fn extend_builder(
        builder: StructLayoutBuilder<marker::Sized>,
        layout: TypeLayout<Self>,
        options: FieldOptions,
    ) -> StructLayoutBuilder<Self>;
}

impl ValidOrSized for marker::Valid {
    fn extend_builder(
        builder: StructLayoutBuilder<marker::Sized>,
        layout: TypeLayout<Self>,
        options: FieldOptions,
    ) -> StructLayoutBuilder<Self> {
        StructLayoutBuilder::__new(builder.struct_builder, layout, options)
    }
}
impl ValidOrSized for marker::Sized {
    fn extend_builder(
        mut builder: StructLayoutBuilder<marker::Sized>,
        layout: TypeLayout<Self>,
        options: FieldOptions,
    ) -> StructLayoutBuilder<Self> {
        unsafe_type_layout::extend(&mut builder, layout, options);
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
