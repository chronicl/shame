use super::*;

// TODO(chronicl) ensure that fields have unique names by appending numbers for duplicate field names.
// also make name fields private.
impl SizedStruct {
    /// Creates a new `SizedStruct` with one field.
    ///
    /// To add additional fields to it, use [`SizedStruct::extend`] or [`SizedStruct::extend_unsized`].
    pub fn new(name: impl Into<CanonName>, field_options: impl Into<FieldOptions>, ty: impl Into<SizedType>) -> Self {
        Self {
            name: name.into(),
            fields: vec![SizedField::new(field_options, ty)],
        }
    }

    /// Adds a sized field to the struct.
    pub fn extend(mut self, field_options: impl Into<FieldOptions>, ty: impl Into<SizedType>) -> Self {
        self.fields.push(SizedField::new(field_options, ty));
        self
    }

    /// Adds a runtime sized array field to the struct. This can only be the last
    /// field of a struct, which is ensured by transitioning to an UnsizedStruct.
    pub fn extend_unsized(
        self,
        name: impl Into<CanonName>,
        custom_min_align: Option<U32PowerOf2>,
        element_ty: impl Into<SizedType>,
    ) -> UnsizedStruct {
        UnsizedStruct {
            name: self.name,
            sized_fields: self.fields,
            last_unsized: RuntimeSizedArrayField::new(name, custom_min_align, element_ty),
        }
    }

    /// Adds either a `SizedType` or a `RuntimeSizedArray` field to the struct.
    ///
    /// Returns a `HostshareableType`, because the `Self` may either stay
    /// a `SizedStruct` or become an `UnsizedStruct` depending on the field's type.
    pub fn extend_sized_or_array(
        self,
        name: impl Into<CanonName>,
        custom_min_size: Option<u64>,
        custom_min_align: Option<U32PowerOf2>,
        field: SizedOrArray,
    ) -> CpuShareableType {
        match field {
            SizedOrArray::Sized(ty) => self
                .extend(
                    FieldOptions {
                        name: name.into(),
                        custom_min_size,
                        custom_min_align,
                    },
                    ty,
                )
                .into(),
            SizedOrArray::RuntimeSizedArray(element_ty) => {
                self.extend_unsized(name, custom_min_align, element_ty).into()
            }
        }
    }

    pub fn fields(&self) -> &[SizedField] { &self.fields }
}

pub enum SizedOrArray {
    Sized(SizedType),
    RuntimeSizedArray(SizedType),
}


impl<T: Into<CanonName>> From<(T, SizedType)> for SizedField {
    fn from(value: (T, SizedType)) -> Self {
        Self {
            name: value.0.into(),
            ty: value.1,
            custom_min_size: None,
            custom_min_align: None,
        }
    }
}

impl<T: Into<CanonName>> From<(T, SizedType)> for RuntimeSizedArrayField {
    fn from(value: (T, SizedType)) -> Self {
        Self {
            name: value.0.into(),
            array: RuntimeSizedArray { element: value.1 },
            custom_min_align: None,
        }
    }
}
