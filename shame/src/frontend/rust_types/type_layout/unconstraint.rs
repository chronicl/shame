use super::*;

impl TypeLayout<constraint::Unconstraint> {
    /// Create a new `TypeLayout<constraint::Unconstraint>`. This type layout does not need
    /// to follow any rules and you can freely fill it up in any way you like.
    ///
    /// `TypeLayout<constraint::Unconstraint>::try_into_basic` can be used to try to convert
    /// an unconstraint type layout into one that is valid as a gpu layout.
    pub fn new(byte_size: Option<u64>, byte_align: u64, kind: TypeLayoutSemantics) -> Self {
        Self {
            byte_size,
            byte_align: byte_align.into(),
            kind,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn from_rust_sized<T: Sized>(kind: TypeLayoutSemantics) -> Self {
        Self::new(Some(size_of::<T>() as u64), align_of::<T>() as u64, kind)
    }

    pub(crate) fn first_line_of_display_with_ellipsis(&self) -> String {
        let string = format!("{}", self);
        string.split_once('\n').map(|(s, _)| format!("{s}…")).unwrap_or(string)
    }

    // TODO add proper Error handling to this and test it.
    #[allow(clippy::result_unit_err)]
    /// If possible, this method changes the layout of self, so that it fullfills all requirements to be
    /// a `TypeLayout<constraint::Basic>.
    pub fn try_to_make_basic(self, packed: bool, rules: TypeLayoutRules) -> Result<TypeLayout<constraint::Basic>, ()> {
        use TypeLayoutSemantics as TLS;

        let valid_layout: TypeLayout<constraint::Basic> = match self.kind {
            TLS::Array(ty, len) => {
                let basic: TypeLayout<constraint::Basic> = Rc::unwrap_or_clone(ty).ty.try_into_basic(packed, rules)?;
                let sized: TypeLayout<constraint::Sized> = basic.try_into()?;
                let len = match len {
                    Some(n) => Some(NonZeroU32::new(n).ok_or(())?),
                    None => None,
                };
                TypeLayout::from_array(rules, sized, len)
            }
            TLS::Matrix(c, r, ty) => TypeLayout::from_sized_ty(rules, &ir::SizedType::Matrix(c, r, ty)).into(),
            TLS::Vector(len, ty) => TypeLayout::from_sized_ty(rules, &ir::SizedType::Vector(len, ty)).into(),
            TLS::PackedVector(packed_vec) => TypeLayout::from_packed_vector(rules, packed_vec).into(),
            TLS::Structure(s) => {
                let mut s = Rc::unwrap_or_clone(s);

                let field_options_and_layout =
                    |field: FieldLayout| -> Result<(FieldOptions, TypeLayout<constraint::Basic>), ()> {
                        let align: Option<U32PowerOf2> = match field.custom_min_align.0 {
                            Some(align) => Some((align as u32).try_into().map_err(|_| ())?),
                            None => None,
                        };
                        let layout: TypeLayout<constraint::Basic> = field.ty.try_into_basic(packed, rules)?;

                        Ok((FieldOptions::new(field.name, align, field.custom_min_size.0), layout))
                    };

                let fields: Result<Vec<_>, ()> = s
                    .fields
                    .into_iter()
                    .map(|f| field_options_and_layout(f.field))
                    .collect();

                TypeLayout::struct_from_parts(StructOptions::new(s.name.0, packed, rules), fields?.into_iter())
                    .map_err(|_| ())?
            }
        };

        Ok(valid_layout)
    }

    #[allow(clippy::result_unit_err)]
    /// Tries to cast self to a `TypeLayout<constraint::Basic>. `packed` and `rules` are validated against.
    ///
    /// Very expensive call. Includes cloning and constructing a new `TypeLayout`.
    pub fn try_into_basic(self, packed: bool, rules: TypeLayoutRules) -> Result<TypeLayout<constraint::Basic>, ()> {
        // Given the information in self, we are constructing a valid layout, if the valid
        // layout coincides with self, then self is upgraded to TypeLayout<Basic>.
        let original_layout = self.clone();
        let valid_layout = self.try_to_make_basic(packed, rules)?;
        if valid_layout == original_layout {
            // We are casting the original layout and not returning the new valid_layout, because
            // the std::rc::Rc in the original should be the ones alive and not the new ones in valid_layout.
            Ok(type_layout_internal::cast(original_layout))
        } else {
            Err(())
        }
    }
}

impl StructLayoutBuilder<constraint::Unconstraint> {}
