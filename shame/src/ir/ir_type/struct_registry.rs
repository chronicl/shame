use thiserror::Error;

use super::{StoreType, Type};
use crate::{
    call_info,
    common::{format::numeral_suffix, iterator_ext::IteratorExt, po2::U32PowerOf2, pool::Key},
    ir::{
        CanonName, Repr, RuntimeSizedArrayField, StructKind, StructKindRef,
        ir_type::layout_type::{
            FieldOptions, LayoutType, RuntimeSizedArray, SizedField, SizedStruct, SizedType, StructKindVariant,
            UnsizedStruct,
        },
        recording::{Context, Ident},
    },
};
use crate::{
    common::pool::PoolRefMut,
    ir::recording::{CallInfo, Priority},
};

#[doc(hidden)]
#[derive(Default)]
pub struct StructRegistry {
    /// "topologically sorted" list of structure definitions
    ///
    /// if structure `b`'s fields reference a structure `a` in any way, `a` appears
    /// before `b` in this list.
    defs: Vec<(StructKind, StructDef)>,
}

impl StructRegistry {
    /// (no documentation yet)
    pub fn get(&self, s: StructKindRef<'_>) -> Option<&StructDef> {
        //TODO(release) this is quite inefficient because of the struct equals check on _every_ registered struct
        // consider using a different datastructure + representing the topological sort differently
        self.defs.iter().find_map(|(o, def)| (o.as_ref() == s).then_some(def))
    }

    /// returns an iterator that goes through the structs in a "topologically sorted" way.
    /// this means if structure `b`'s fields reference a structure `a` in any way, `a` appears
    /// before `b` in this list.
    pub fn iter_topo_sorted(&self) -> impl Iterator<Item = &StructDef> { self.defs.iter().map(|(_, def)| def) }

    /// only registers this one struct (if it wasn't registered before),
    /// not any structures that are used within that struct's fields.
    ///
    /// this function must not be public, because that can be used to violate
    /// the topological sort.
    /*not pub*/
    fn register_single_struct(
        &mut self,
        s: StructKindRef<'_>,
        idents: &mut PoolRefMut<Ident>,
        call_info: CallInfo,
    ) -> bool {
        if !self.contains(s) {
            if let Err(e) = check_struct_definition(s) {
                Context::try_with(call_info, |ctx| ctx.push_error(e.into()));
            }

            self.defs
                .push((s.to_owned(), StructDef::new_for_struct(s, idents, call_info)));
            true
        } else {
            false
        }
    }

    /// registers `s` as well as any other structs mentioned in the fields of
    /// this struct (recursively) if they aren't already registered.
    pub fn register_mentioned_structs_recursively(
        &mut self,
        s: StructKindRef<'_>,
        idents: &mut PoolRefMut<Ident>,
        call_info: CallInfo,
    ) {
        // Note: ORDER IMPORTANT
        // first iterate, then insert (this ensures topological sortedness)
        for field in s.sized_fields() {
            self.find_and_register_new_structs_used_in_sized_type(&field.ty, idents, call_info);
        }
        if let Some(array) = s.last_unsized() {
            self.find_and_register_new_structs_used_in_sized_type(array.element_ty(), idents, call_info);
        }
        self.register_single_struct(s, idents, call_info);
    }

    /// (no documentation yet)
    pub fn find_and_register_new_structs_used_in_type(
        &mut self,
        t: &Type,
        idents: &mut PoolRefMut<Ident>,
        call_info: CallInfo,
    ) {
        match t {
            Type::Unit => (),
            Type::Ptr(_, s, _) | Type::Ref(_, s, _) | Type::Store(s) => {
                self.find_and_register_new_structs_used_in_store_type(s, idents, call_info)
            }
        }
    }

    /// (no documentation yet)
    pub fn find_and_register_new_structs_used_in_store_type(
        &mut self,
        t: &StoreType,
        idents: &mut PoolRefMut<Ident>,
        call_info: CallInfo,
    ) {
        match t {
            StoreType::Layout(t) => match t {
                LayoutType::Sized(s) => self.find_and_register_new_structs_used_in_sized_type(s, idents, call_info),
                LayoutType::RuntimeSizedArray(a) => {
                    self.find_and_register_new_structs_used_in_sized_type(&a.element, idents, call_info)
                }
                LayoutType::UnsizedStruct(s) => {
                    self.register_mentioned_structs_recursively(s.into(), idents, call_info)
                }
            },
            StoreType::Handle(_) => (),
            StoreType::BindingArray(s, _) => {
                self.find_and_register_new_structs_used_in_store_type(s, idents, call_info)
            }
        }
    }

    /// (no documentation yet)
    pub fn find_and_register_new_structs_used_in_sized_type(
        &mut self,
        t: &SizedType,
        idents: &mut PoolRefMut<Ident>,
        call_info: CallInfo,
    ) {
        match t {
            SizedType::Vector(_) | SizedType::Matrix(_) | SizedType::Atomic(_) => (),
            SizedType::Array(a) => self.find_and_register_new_structs_used_in_sized_type(&a.element, idents, call_info),
            SizedType::Struct(s) => self.register_mentioned_structs_recursively(s.into(), idents, call_info),
        }
    }

    /// (no documentation yet)
    pub fn contains(&mut self, s: StructKindRef<'_>) -> bool { self.defs.iter().any(|(x, _)| x.as_ref() == s) }

    /// (no documentation yet)
    pub fn definitions(&self) -> &[(StructKind, StructDef)] { &self.defs }
}

/// the precise definition of a struct type wrt. actual `Ident`s rather than just
/// canonical names of fields etc.
pub struct StructDef {
    call_info: CallInfo,
    kind: StructKindVariant,
    name: CanonName,
    ident: Key<Ident>,
    sized_fields: Vec<(Key<Ident>, SizedField)>,
    last_unsized: Option<(Key<Ident>, RuntimeSizedArrayField)>,
}

impl StructDef {
    /// (no documentation yet)
    pub fn new_for_struct(s: StructKindRef<'_>, idents: &mut PoolRefMut<Ident>, call_info: CallInfo) -> Self {
        StructDef {
            call_info,
            kind: s.kind(),
            name: s.name().clone(),
            ident: Ident::auto_in_pool(s.name().to_string(), idents),
            sized_fields: s
                .sized_fields()
                .iter()
                .map(|f| (Ident::auto_in_pool(f.name.to_string(), idents), f.clone()))
                .collect(),
            last_unsized: s
                .last_unsized()
                .as_ref()
                .map(|f| (Ident::auto_in_pool(f.name.to_string(), idents), (*f).clone())),
        }
    }

    /// (no documentation yet)
    pub fn call_info(&self) -> CallInfo { self.call_info }

    /// (no documentation yet)
    pub fn canonical_name(&self) -> &CanonName { &self.name }

    /// (no documentation yet)
    pub fn ident(&self) -> Key<Ident> { self.ident }

    /// (no documentation yet)
    pub fn get_field_ident(&self, canonical_name: &CanonName) -> Option<&Key<Ident>> {
        self.sized_fields
            .iter()
            .find_map(|(ident, field)| (&field.name == canonical_name).then_some(ident))
            .or_else(|| {
                self.last_unsized
                    .as_ref()
                    .filter(|(_, field)| &field.name == canonical_name)
                    .map(|(ident, _)| ident)
            })
    }

    /// (no documentation yet)
    pub fn get_field_ident_by_index(&self, index: u32) -> Option<&Key<Ident>> {
        let index = index as usize;
        if index < self.sized_fields.len() {
            Some(&self.sized_fields[index].0)
        } else if index == self.sized_fields.len() {
            self.last_unsized.as_ref().map(|(ident, _)| ident)
        } else {
            None
        }
    }

    /// Iterator over (ident, align, size, ty).
    /// Clones each LayoutType, so this is relatively expensive.
    pub fn fields(&self) -> impl Iterator<Item = (&Key<Ident>, Option<U32PowerOf2>, Option<u64>, LayoutType)> + '_ {
        self.sized_fields
            .iter()
            .map(|(ident, field)| {
                (
                    ident,
                    field.custom_min_align,
                    field.custom_min_size,
                    LayoutType::Sized(field.ty.clone()),
                )
            })
            .chain(self.last_unsized.iter().map(|(ident, field)| {
                (
                    ident,
                    field.custom_min_align,
                    None,
                    LayoutType::RuntimeSizedArray(field.array.clone()),
                )
            }))
    }
}

#[allow(missing_docs)]
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum StructDefinitionError {
    #[error("{0} definitions require at least one field")]
    /// required by https://www.w3.org/TR/WGSL/#struct-types
    MustHaveAtLeastOneField(StructKind),
    #[error(transparent)]
    FieldNamesMustBeUnique(#[from] StructureFieldNamesMustBeUnique),
}

fn check_struct_definition(s: StructKindRef<'_>) -> Result<(), StructDefinitionError> {
    if s.sized_fields().is_empty() && s.last_unsized().is_none() {
        Err(StructDefinitionError::MustHaveAtLeastOneField(s.to_owned()))
    } else {
        check_for_duplicate_field_names(s).map_err(StructDefinitionError::FieldNamesMustBeUnique)
    }
}

/// an error created if a struct contains two or more fields of the same name
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("Field {} appears more than once in {}. Duplicate field names are not allowed in struct definitions.",
    self.field_name,
    self.s
)]
pub struct StructureFieldNamesMustBeUnique {
    pub first_occurence: usize,
    pub second_occurence: usize,
    pub s: StructKind,
    pub field_name: CanonName,
}

fn check_for_duplicate_field_names(s: StructKindRef<'_>) -> Result<(), StructureFieldNamesMustBeUnique> {
    // Brute force search > HashMap for the amount of fields
    // we'd usually deal with.
    let mut duplicate_fields = None;
    let sized_fields = s.sized_fields();
    'a: for (i, field1) in sized_fields.iter().enumerate() {
        for (j, field2) in sized_fields.iter().enumerate().skip(i + 1) {
            if field1.name == field2.name {
                duplicate_fields = Some((field1.name.clone(), i, j));
                break 'a;
            }
        }
        if let Some(last_unsized) = s.last_unsized() {
            if field1.name == last_unsized.name {
                duplicate_fields = Some((field1.name.clone(), i, sized_fields.len()));
                break 'a;
            }
        }
    }
    match duplicate_fields {
        Some((field_name, first_occurence, second_occurence)) => Err(StructureFieldNamesMustBeUnique {
            first_occurence,
            second_occurence,
            field_name,
            s: s.to_owned(),
        }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use crate::{self as sm};
    use sm::{ToGpuType, EncodingErrorKind};
    use sm::any::{AsAny, Repr, SizedStruct, StructDefinitionError, SizedField, Vector, ScalarType, Len, Any};

    macro_rules! assert_error_is_present {
        ($errors:expr, $error:pat) => {
            match $errors {
                Ok(_) => panic!("Expected an error due to duplicate field names"),
                Err(es) => {
                    let mut found = false;
                    for e in es.into_iter() {
                        match e.error {
                            $error => {
                                found = true;
                                break;
                            }
                            _ => {}
                        }
                    }
                    assert!(found, "Expected error {}", stringify!($error));
                }
            }
        };
    }

    #[test]
    fn test_struct_duplicate_field_name_error() {
        let mut encoder = sm::start_encoding(Default::default()).unwrap();
        let pipeline = encoder.new_compute_pipeline([1]);
        let s = SizedStruct::new(
            "DuplicateFieldStruct",
            vec![
                SizedField::new("a", Vector::new(ScalarType::F32, Len::X1)),
                SizedField::new("a", Vector::new(ScalarType::F32, Len::X1)),
            ],
            Repr::Wgsl,
        );
        let a = Any::new_struct(s, &[1.0f32.to_gpu().as_any(), 2.0f32.to_gpu().as_any()]);

        assert_error_is_present!(
            encoder.finish(),
            EncodingErrorKind::StructDefinitionError(StructDefinitionError::FieldNamesMustBeUnique(_))
        );
    }

    #[test]
    fn test_struct_empty_error() {
        let mut encoder = sm::start_encoding(Default::default()).unwrap();
        let pipeline = encoder.new_compute_pipeline([1]);
        let s = SizedStruct::new("A", vec![], Repr::Wgsl);
        let a = Any::new_struct(s, &[]);

        assert_error_is_present!(
            encoder.finish(),
            EncodingErrorKind::StructDefinitionError(StructDefinitionError::MustHaveAtLeastOneField(_))
        );
    }
}
