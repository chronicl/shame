use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Display,
    ops::Deref,
    rc::Rc,
};

use thiserror::Error;

use super::{canon_name::CanonName, SizedType, StoreType, Type};
use crate::{
    any::layout::Repr,
    call_info,
    common::{format::numeral_suffix, iterator_ext::IteratorExt, po2::U32PowerOf2, pool::Key},
    ir::{
        ir_type::{FieldOptions, LayoutType},
        recording::{Context, Ident},
    },
};
use crate::{
    common::pool::PoolRefMut,
    ir::recording::{CallInfo, Priority},
};

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeSizedArray {
    pub element: SizedType,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeSizedArrayField {
    name: CanonName,
    pub custom_min_align: Option<U32PowerOf2>,
    pub array: RuntimeSizedArray,
}

impl RuntimeSizedArrayField {
    #[allow(missing_docs)] // runtime api
    pub fn new(
        name: impl Into<CanonName>,
        custom_min_align: Option<U32PowerOf2>,
        element_ty: impl Into<SizedType>,
    ) -> Self {
        Self {
            name: name.into(),
            custom_min_align,
            array: RuntimeSizedArray {
                element: element_ty.into(),
            },
        }
    }

    pub fn name(&self) -> &CanonName { &self.name }

    #[allow(missing_docs)] // runtime api
    pub fn element_ty(&self) -> &SizedType { &self.array.element }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SizedField {
    name: CanonName,
    pub custom_min_size: Option<u64>,
    pub custom_min_align: Option<U32PowerOf2>,
    pub ty: SizedType,
}

impl SizedField {
    pub fn new(options: impl Into<FieldOptions>, ty: SizedType) -> Self {
        let options = options.into();
        Self {
            name: options.name,
            custom_min_size: options.custom_min_size,
            custom_min_align: options.custom_min_align,
            ty,
        }
    }

    pub fn name(&self) -> &CanonName { &self.name }
}

/// A struct with a known fixed size.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SizedStruct {
    /// The canonical name of the struct.
    pub name: CanonName,
    // This is private to ensure a `SizedStruct` always has at least one field.
    fields: Vec<SizedField>,
    /// The representation/layout rules for this struct. See [`Repr`] for more details.
    pub repr: Repr,
}

impl SizedStruct {
    #[track_caller]
    pub fn new(
        name: impl Into<CanonName>,
        sized_fields_nonempty: Vec<SizedField>,
        repr: Repr,
    ) -> Result<Self, StructureDefinitionError> {
        if sized_fields_nonempty.is_empty() {
            Err(StructureDefinitionError::MustHaveAtLeastOneField(
                StructKindVariant::Sized,
            ))
        } else {
            check_for_duplicate_field_names(&sized_fields_nonempty, None)
                .map_err(StructureDefinitionError::FieldNamesMustBeUnique)?;
            let s = SizedStruct {
                name: name.into(),
                fields: sized_fields_nonempty.clone(),
                repr,
            };
            try_register_struct(call_info!(), StructKindRef::Sized(&s));
            Ok(s)
        }
    }

    #[track_caller]
    pub fn new_nonempty(
        name: impl Into<CanonName>,
        mut sized_fields_first: Vec<SizedField>,
        sized_fields_last: SizedField,
        repr: Repr,
    ) -> Result<Self, StructureFieldNamesMustBeUnique> {
        sized_fields_first.push(sized_fields_last);
        Self::new(name, sized_fields_first, repr).map_err(|e| match e {
            StructureDefinitionError::MustHaveAtLeastOneField(_) => {
                unreachable!("prevented by `new_nonempty` signature");
            }
            StructureDefinitionError::FieldNamesMustBeUnique(e) => e,
        })
    }

    /// The fields of this struct.
    pub fn fields(&self) -> &[SizedField] { &self.fields }

    pub fn apply_to_fields(&mut self, mut f: impl FnMut(&mut SizedField)) {
        for field in &mut self.fields {
            f(field);
        }
    }
}

/// A struct whose size is not known before shader runtime.
///
/// This struct has a runtime sized array as it's last field.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnsizedStruct {
    /// The canonical name of the struct.
    pub name: CanonName,
    /// Fixed-size fields that come before the unsized field
    sized_fields: Vec<SizedField>,
    /// Last runtime sized array field of the struct.
    last_unsized: RuntimeSizedArrayField,
    /// The representation/layout rules for this struct. See [`Repr`] for more details.
    pub repr: Repr,
}

impl UnsizedStruct {
    pub fn apply_to_sized_fields(&mut self, mut f: impl FnMut(&mut SizedField)) {
        for field in &mut self.sized_fields {
            f(field);
        }
    }

    pub fn apply_to_last_unsized(&mut self, mut f: impl FnOnce(&mut RuntimeSizedArrayField)) {
        f(&mut self.last_unsized);
    }
}


impl UnsizedStruct {
    pub fn new(
        name: impl Into<CanonName>,
        sized_fields: Vec<SizedField>,
        last_unsized: RuntimeSizedArrayField,
        repr: Repr,
    ) -> Result<Self, StructureFieldNamesMustBeUnique> {
        check_for_duplicate_field_names(&sized_fields, Some(&last_unsized))?;
        let s = UnsizedStruct {
            name: name.into(),
            sized_fields,
            last_unsized,
            repr,
        };
        try_register_struct(call_info!(), StructKindRef::Unsized(&s));
        Ok(s)
    }

    pub fn sized_fields(&self) -> &[SizedField] { &self.sized_fields }

    pub fn last_unsized(&self) -> &RuntimeSizedArrayField { &self.last_unsized }
}


#[derive(Debug, Clone)]
pub enum StructKind {
    Sized(SizedStruct),
    Unsized(UnsizedStruct),
}

impl StructKind {
    pub fn as_ref(&self) -> StructKindRef<'_> {
        match self {
            StructKind::Sized(s) => StructKindRef::Sized(s),
            StructKind::Unsized(s) => StructKindRef::Unsized(s),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructKindRef<'a> {
    Sized(&'a SizedStruct),
    Unsized(&'a UnsizedStruct),
}

impl<'a> From<&'a SizedStruct> for StructKindRef<'a> {
    fn from(s: &'a SizedStruct) -> Self { StructKindRef::Sized(s) }
}

impl<'a> From<&'a UnsizedStruct> for StructKindRef<'a> {
    fn from(s: &'a UnsizedStruct) -> Self { StructKindRef::Unsized(s) }
}

impl StructKindRef<'_> {
    fn to_owned(&self) -> StructKind {
        match self {
            StructKindRef::Sized(s) => StructKind::Sized((*s).clone()),
            StructKindRef::Unsized(s) => StructKind::Unsized((*s).clone()),
        }
    }
}

impl StructKindRef<'_> {
    pub fn name(&self) -> &CanonName {
        match self {
            StructKindRef::Sized(s) => &s.name,
            StructKindRef::Unsized(s) => &s.name,
        }
    }

    pub fn sized_fields(&self) -> &[SizedField] {
        match self {
            StructKindRef::Sized(s) => s.fields(),
            StructKindRef::Unsized(s) => s.sized_fields(),
        }
    }

    pub fn last_unsized(&self) -> Option<&RuntimeSizedArrayField> {
        match self {
            StructKindRef::Sized(_) => None,
            StructKindRef::Unsized(s) => Some(s.last_unsized()),
        }
    }

    pub fn repr(&self) -> Repr {
        match self {
            StructKindRef::Sized(s) => s.repr,
            StructKindRef::Unsized(s) => s.repr,
        }
    }

    pub fn kind(&self) -> StructKindVariant {
        match self {
            StructKindRef::Sized(_) => StructKindVariant::Sized,
            StructKindRef::Unsized(_) => StructKindVariant::Unsized,
        }
    }
}

/// try register `struct_` if we're currently in a pipeline encoding,
/// otherwise the registration will happen later with a less useful `call_info`
fn try_register_struct(call_info: CallInfo, struct_: StructKindRef<'_>) {
    Context::try_with(call_info, |ctx| {
        ctx.struct_registry_mut().register_mentioned_structs_recursively(
            struct_,
            &mut ctx.pool_mut(),
            ctx.latest_user_caller(),
        );
    });
}


#[doc(hidden)] // internal
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StructKindVariant {
    Sized,
    Unsized,
}

impl Display for StructKindVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            StructKindVariant::Sized => "struct",
            StructKindVariant::Unsized => "unsized struct",
        })
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
            .find_map(|(ident, field)| (field.name() == canonical_name).then_some(ident))
            .or_else(|| {
                self.last_unsized
                    .as_ref()
                    .filter(|(_, field)| field.name() == canonical_name)
                    .map(|(ident, _)| ident)
            })
    }
}

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
            StoreType::LayoutType(t) => match t {
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
