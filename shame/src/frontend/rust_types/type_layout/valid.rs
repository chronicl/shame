use super::*;

impl StructLayoutBuilder<marker::Valid> {
    pub(super) fn __new(
        struct_builder: StructLayoutBuilderErased,
        last_field: TypeLayout<marker::Valid>,
        options: FieldOptions,
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
