use super::*;

impl LayoutType {
    /// Fallibly creates a new `TypeLayoutRecipe` of a struct.
    ///
    /// An error is returned if the following rules aren't followed:
    /// - There must be at least one field.
    /// - None of the fields must be an `UnsizedStruct`.
    /// - Only the last field may be unsized (a runtime sized array).
    pub fn struct_from_parts(
        struct_name: impl Into<CanonName>,
        fields: impl IntoIterator<Item = (FieldOptions, LayoutType)>,
        repr: Repr,
    ) -> Result<Self, StructFromPartsError> {
        use StructFromPartsError::*;

        enum Field {
            Sized(SizedField),
            Unsized(RuntimeSizedArrayField),
        }

        let mut fields = fields
            .into_iter()
            .map(|(options, ty)| {
                Ok(match ty {
                    LayoutType::Sized(s) => Field::Sized(SizedField::new(options, s)),
                    LayoutType::RuntimeSizedArray(a) => Field::Unsized(RuntimeSizedArrayField::new(
                        options.name,
                        options.custom_min_align,
                        a.element,
                    )),
                    LayoutType::UnsizedStruct(_) => return Err(MustNotHaveUnsizedStructField),
                })
            })
            .peekable();

        let mut sized_fields = Vec::new();
        let mut last_unsized = None;
        while let Some(field) = fields.next() {
            let field = field?;
            match field {
                Field::Sized(sized) => sized_fields.push(sized),
                Field::Unsized(a) => {
                    last_unsized = Some(a);
                    if fields.peek().is_some() {
                        return Err(OnlyLastFieldMayBeUnsized);
                    }
                }
            }
        }

        let field_count = sized_fields.len() + last_unsized.is_some() as usize;
        if field_count == 0 {
            return Err(MustHaveAtLeastOneField);
        }

        if let Some(last_unsized) = last_unsized {
            Ok(UnsizedStruct {
                name: struct_name.into(),
                sized_fields,
                last_unsized,
                repr,
            }
            .into())
        } else {
            Ok(SizedStruct::new(struct_name, sized_fields, repr).into())
        }
    }
}

impl SizedStruct {
    pub fn new(name: impl Into<CanonName>, fields: Vec<SizedField>, repr: Repr) -> Self {
        Self {
            name: name.into(),
            fields,
            repr,
        }
    }
}

impl UnsizedStruct {
    pub fn new(
        name: impl Into<CanonName>,
        sized_fields: Vec<SizedField>,
        last_unsized: RuntimeSizedArrayField,
        repr: Repr,
    ) -> Self {
        Self {
            name: name.into(),
            sized_fields,
            last_unsized,
            repr,
        }
    }
}

#[allow(missing_docs)]
#[derive(thiserror::Error, Debug)]
pub enum StructFromPartsError {
    #[error("Struct must have at least one field.")]
    MustHaveAtLeastOneField,
    #[error("Only the last field of a struct may be unsized.")]
    OnlyLastFieldMayBeUnsized,
    #[error("A field of the struct is an unsized struct, which isn't allowed.")]
    MustNotHaveUnsizedStructField,
}

/// Options for the field of a struct.
///
/// If you only want to customize the field's name, you can convert most string types
/// to `FieldOptions` using `Into::into`. For methods that take `impl Into<FieldOptions>`
/// parameters you can just pass the string type directly.
#[derive(Debug, Clone)]
pub struct FieldOptions {
    /// Name of the field
    pub name: CanonName,
    /// Custom minimum align of the field.
    pub custom_min_align: Option<U32PowerOf2>,
    /// Custom mininum size of the field.
    pub custom_min_size: Option<u64>,
}

impl FieldOptions {
    /// Creates new `FieldOptions`.
    ///
    /// If you only want to customize the field's name, you can convert most string types
    /// to `FieldOptions` using `Into::into`. For methods that take `impl Into<FieldOptions>`
    /// parameters you can just pass the string type directly.
    pub fn new(
        name: impl Into<CanonName>,
        custom_min_align: Option<U32PowerOf2>,
        custom_min_size: Option<u64>,
    ) -> Self {
        Self {
            name: name.into(),
            custom_min_align,
            custom_min_size,
        }
    }
}

impl<T: Into<CanonName>> From<T> for FieldOptions {
    fn from(name: T) -> Self { Self::new(name, None, None) }
}
