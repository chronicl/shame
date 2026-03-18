use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Display,
    ops::Deref,
    rc::Rc,
};

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

#[derive(Default)]
pub struct StructRegistry {
    /// "topologically sorted" list of structure definitions
    ///
    /// if structure `b`'s fields reference a structure `a` in any way, `a` appears
    /// before `b` in this list.
    defs: Vec<(StructKind, StructDef)>,
}

impl StructRegistry {
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

    pub fn contains(&mut self, s: StructKindRef<'_>) -> bool { self.defs.iter().any(|(x, _)| x.as_ref() == s) }

    pub fn definitions(&self) -> &[(StructKind, StructDef)] { &self.defs }
}

fn check_for_duplicate_field_names(
    sized_fields: &[SizedField],
    last_unsized: Option<&RuntimeSizedArrayField>,
) -> Result<(), StructureFieldNamesMustBeUnique> {
    // Brute force search > HashMap for the amount of fields
    // we'd usually deal with.
    let mut duplicate_fields = None;
    for (i, field1) in sized_fields.iter().enumerate() {
        for (j, field2) in sized_fields.iter().enumerate().skip(i + 1) {
            if field1.name == field2.name {
                duplicate_fields = Some((i, j));
                break;
            }
        }
        if let Some(last_unsized) = last_unsized {
            if field1.name == last_unsized.name {
                duplicate_fields = Some((i, sized_fields.len()));
                break;
            }
        }
    }
    match duplicate_fields {
        Some((first_occurence, second_occurence)) => Err(StructureFieldNamesMustBeUnique {
            first_occurence,
            second_occurence,
        }),
        None => Ok(()),
    }
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

    pub fn call_info(&self) -> CallInfo { self.call_info }

    pub fn canonical_name(&self) -> &CanonName { &self.name }

    pub fn ident(&self) -> Key<Ident> { self.ident }

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
pub enum StructureDefinitionError {
    #[error("{0} definitions require at least one field")]
    /// required by https://www.w3.org/TR/WGSL/#struct-types
    MustHaveAtLeastOneField(StructKindVariant),
    #[error(transparent)]
    FieldNamesMustBeUnique(#[from] StructureFieldNamesMustBeUnique),
}

/// an error created if a struct contains two or more fields of the same name
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{} and {} struct field have the same name. Field names must be unique within a structure definition",
    numeral_suffix(self.first_occurence + 1),
    numeral_suffix(self.second_occurence + 1)
)]
pub struct StructureFieldNamesMustBeUnique {
    pub first_occurence: usize,
    pub second_occurence: usize,
}

// TODO(chronicl) check every registered struct for duplicate field names
// and having at least one field
