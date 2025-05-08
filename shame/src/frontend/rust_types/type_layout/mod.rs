#![allow(missing_docs)]
// TODO(chronicl) remove allow(missing_docs)
use std::{
    fmt::{Debug, Display, Write},
    marker::PhantomData,
    num::NonZeroU32,
    ops::Deref,
    rc::Rc,
};

use crate::{
    any::U32PowerOf2,
    call_info,
    common::{
        ignore_eq::{IgnoreInEqOrdHash, InEqOrd},
        prettify::set_color,
    },
    ir::{
        self,
        ir_type::{
            align_of_array, byte_size_of_array, round_up, stride_of_array, AlignedType, CanonName, LenEven,
            PackedVectorByteSize, ScalarTypeFp, ScalarTypeInteger,
        },
        recording::Context,
        Len, SizedType, Type,
    },
    GpuAligned, GpuLayout, GpuSized,
};
use thiserror::Error;

use super::{mem, type_traits};

mod eq;
mod maybe_invalid;
mod sized;
mod valid;
mod vertex;

pub use eq::*;
pub use maybe_invalid::*;
pub use sized::*;
pub use valid::*;
pub use vertex::*;

/// The type contained in the bytes of a `TypeLayout`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeLayoutSemantics {
    /// `vec<T, L>`
    Vector(ir::Len, ir::ScalarType),
    /// special compressed vectors for vertex attribute types
    ///
    /// see the [`crate::packed`] module
    PackedVector(ir::PackedVector),
    /// `mat<T, Cols, Rows>`, first `Len2` is cols, 2nd `Len2` is rows
    Matrix(ir::Len2, ir::Len2, ScalarTypeFp),
    /// `Array<T>` and `Array<T, Size<N>>`
    Array(Rc<ElementLayout>, Option<u32>), // not NonZeroU32, since for rust `CpuLayout`s the array size may be 0.
    /// structures which may be empty and may have an unsized last field
    Structure(Rc<StructLayout>),
}

/// The memory layout of a type.
///
/// This models only the layout, not other characteristics of the types.
/// For example an `Atomic<vec<u32, x1>>` is treated like a regular `vec<u32, x1>` layout wise.
///
/// The `PartialEq + Eq` implementation of `TypeLayout` is designed to answer the question
/// "do these two types have the same layout" so that uploading a type to the gpu
/// will result in no memory errors.
///
/// a layout comparison looks like this:
/// ```
/// assert!(f32::cpu_layout() == vec<f32, x1>::gpu_layout());
/// // or, more explicitly
/// assert_eq!(
///     <f32 as CpuLayout>::cpu_layout(),
///     <vec<f32, x1> as GpuLayout>::gpu_layout(),
/// );
/// ```
///
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TypeLayout<T: TypeRestriction = marker::Valid> {
    /// size in bytes (Some), or unsized (None)
    pub byte_size: Option<u64>,
    /// the byte alignment
    ///
    /// top level alignment is not considered relevant in some checks, but relevant in others (vertex array elements)
    pub byte_align: IgnoreInEqOrdHash<u64>,
    /// the type contained in the bytes of this type layout
    pub kind: TypeLayoutSemantics,
    _phantom: PhantomData<T>,
}

// TODO(chronicl) consider refactoring to it's own type which implements from and into TypeLayout<marker::MaybeInvalid>
/// TypeLayout for cpu types. Is not necessarily a valid layout for a gpu type.
pub type CpuTypeLayout = TypeLayout<marker::MaybeInvalid>;

use marker::TypeRestriction;
// TODO(chronicl) would be nice to have a different name for this module
pub mod marker {
    use super::*;

    pub trait TypeRestriction: Clone + PartialEq + Eq + std::hash::Hash {}

    macro_rules! type_restriction {
        ($($restriction:ident),*) => {
            $(
                #[derive(Clone, PartialEq, Eq, Hash)]
                pub struct $restriction;
                impl TypeRestriction for $restriction {}
            )*
        };
    }

    type_restriction!(
        MaybeInvalid,    // Any type layout, may be invalid
        Valid,           // GpuLayout, Any valid gpu type layout
        Sized,           // GpuSized
        Vertex,          // VertexLayout
        VertexAttribute  // VertexAttribute
    );
    // TODO(chronicl)
    // Storage, // StorageLayout
    // Uniform, // UniformLayout

    macro_rules! impl_from_into {
        ($from:ident -> $($into:ident),*) => {
            $(
                impl From<TypeLayout<$from>> for TypeLayout<$into> {
                    fn from(layout: TypeLayout<$from>) -> Self { unsafe_type_layout::cast(layout) }
                }
            )*
        };
    }

    impl_from_into!(Valid           -> MaybeInvalid);
    impl_from_into!(Sized           -> MaybeInvalid, Valid);
    impl_from_into!(Vertex          -> MaybeInvalid, Valid, Sized);
    impl_from_into!(VertexAttribute -> MaybeInvalid, Valid, Sized, Vertex);
}

// TODO(chronicl) could consider to add another generic which determines whether at least one field
// has been added, so that the `new` method may avoid taking the first field immediately.
pub struct StructLayoutBuilder<T: TypeRestriction = marker::Valid> {
    struct_builder: StructLayoutBuilderErased,
    // This is used by TypeLayoutBuilder<marker::Valid>, which only has one public method `finish`,
    // which inserts this last, potentially unsized field and finishes the build.
    // Building a TypeLayout<marker::Valid> always starts with a StructLayoutBuilder<marker::Sized>
    // and converts automatically to a StructLayoutBuilder<marker::Valid> when extending
    // by a field that is TypeLayout<marker::Valid> (potentially unsized).
    last_maybe_unsized_field: Option<(TypeLayout, FieldOptions)>,
    _phantom: PhantomData<T>,
}

// TODO(chronicl) probably don't use the word unsafe here, but something like dangerous.
/// This module is not unsafe in the typical rust sense. It offers helper
/// methods that do not adhere to the restriction of `TypeLayout<T: TypeRestriction>.
/// The caller must uphold these restrictions themselves.
/// The main purpose of this module is to avoid repetition.
pub(in super::super::rust_types) mod unsafe_type_layout {
    use std::{marker::PhantomData, rc::Rc};
    use crate::{GpuAligned, GpuSized};
    use super::*;

    pub fn new_builder<T: TypeRestriction>(options: impl Into<StructOptions>) -> StructLayoutBuilder<T> {
        StructLayoutBuilder {
            struct_builder: StructLayoutBuilderErased::new(options.into()),
            last_maybe_unsized_field: None,
            _phantom: PhantomData,
        }
    }

    pub fn new_type_layout<T: TypeRestriction>(
        byte_size: Option<u64>,
        byte_align: u64,
        kind: TypeLayoutSemantics,
    ) -> TypeLayout<T> {
        TypeLayout {
            byte_size,
            byte_align: byte_align.into(),
            kind,
            _phantom: PhantomData,
        }
    }

    pub fn new_type_layout_struct<T: TypeRestriction>(
        byte_size: Option<u64>,
        byte_align: u64,
        s: StructLayout,
    ) -> TypeLayout<T> {
        TypeLayout {
            byte_size,
            byte_align: byte_align.into(),
            kind: TypeLayoutSemantics::Structure(Rc::new(s)),
            _phantom: PhantomData,
        }
    }

    pub fn extend<R: TypeRestriction>(
        builder: &mut StructLayoutBuilder<R>,
        layout: TypeLayout<marker::Sized>,
        options: FieldOptions,
    ) {
        builder.struct_builder.extend(layout, options);
    }

    pub fn finish<R: TypeRestriction>(builder: StructLayoutBuilder<R>) -> TypeLayout<R> {
        let (byte_size, byte_align, s) = builder.struct_builder.finish();
        new_type_layout_struct(Some(byte_size), byte_align, s)
    }

    pub fn finish_maybe_unsized<R: TypeRestriction>(
        builder: StructLayoutBuilder<R>,
        layout: TypeLayout,
        options: impl Into<FieldOptions>,
    ) -> TypeLayout<R> {
        let (byte_size, byte_align, s) = builder.struct_builder.finish_maybe_unsized(layout, options.into());
        new_type_layout_struct(byte_size, byte_align, s)
    }

    pub fn cast<From: TypeRestriction, Into: TypeRestriction>(layout: TypeLayout<From>) -> TypeLayout<Into> {
        TypeLayout {
            byte_size: layout.byte_size,
            byte_align: layout.byte_align,
            kind: layout.kind,
            _phantom: PhantomData,
        }
    }
}

/// (no documentation - chronicl)
#[derive(Debug, Clone)]
pub struct StructOptions {
    pub name: CanonName,
    pub packed: bool,
    pub rules: TypeLayoutRules,
}

impl StructOptions {
    pub fn new(name: impl Into<CanonName>, packed: bool, rules: TypeLayoutRules) -> Self {
        Self {
            name: name.into(),
            packed,
            rules,
        }
    }
}

impl<T: Into<CanonName>> From<T> for StructOptions {
    fn from(name: T) -> Self { Self::new(name, false, TypeLayoutRules::Wgsl) }
}


/// (no documentation - chronicl)
#[derive(Debug, Clone)]
pub struct FieldOptions {
    pub name: CanonName,
    pub(crate) custom_min_align: Option<u64>,
    pub custom_min_size: Option<u64>,
}

impl FieldOptions {
    /// Creates new `FieldOptions`.
    ///
    /// If you only want to customize the name, you can convert most string types
    /// to `FieldOptions` using `Into::into`.
    pub fn new(
        name: impl Into<CanonName>,
        custom_min_align: Option<U32PowerOf2>,
        custom_min_size: Option<u64>,
    ) -> Self {
        Self {
            name: name.into(),
            custom_min_align: custom_min_align.map(Into::into),
            custom_min_size,
        }
    }
}

impl<T: Into<CanonName>> From<T> for FieldOptions {
    fn from(name: T) -> Self { Self::new(name, None, None) }
}

/// Builds a TypeLayout<marker::Valid> and is wrapped by all StructLayoutBuilder<T>,
/// which impose further restrictions on the building process, so that they may
/// produce more strict TypeLayouts, such as StructLayoutBuilder<Sized> -> TypeLayout<Sized>.
struct StructLayoutBuilderErased {
    next_offset_min: u64,
    align: u64,
    packed: bool,
    rules: TypeLayoutRules,
    s: StructLayout,
}

impl StructLayoutBuilderErased {
    fn new(options: StructOptions) -> Self {
        Self {
            next_offset_min: 0,
            align: 1,
            packed: options.packed,
            rules: options.rules,
            s: StructLayout {
                name: options.name.into(),
                fields: Vec::new(),
            },
        }
    }

    fn extend(&mut self, layout: TypeLayout<marker::Sized>, options: FieldOptions) {
        let field_size = FieldLayout::calculate_byte_size(layout.byte_size_sized(), options.custom_min_size);
        let field_align = FieldLayout::calculate_align(layout.align(), options.custom_min_align);
        let field_offset = self.next_field_offset(field_align, options.custom_min_align);
        self.next_offset_min = field_offset + field_size;
        self.align = self.align.max(field_align);

        self.s.fields.push(FieldLayoutWithOffset {
            field: FieldLayout::new(layout.into(), options),
            rel_byte_offset: field_offset,
        });
    }

    /// Returns (byte_size, byte_align, StructLayout)
    // wgsl spec:
    //   roundUp(AlignOf(S), justPastLastMember)
    //   where justPastLastMember = OffsetOfMember(S,N) + SizeOfMember(S,N)
    //
    // self.next_offset_min is justPastLastMember already.
    fn finish(self) -> (u64, u64, StructLayout) { (round_up(self.align, self.next_offset_min), self.align, self.s) }

    /// Returns (byte_size, byte_align, StructLayout), where byte_size is None
    /// if the struct is unsized, that is, when `last_field` is unsized.
    fn finish_maybe_unsized(mut self, layout: TypeLayout, options: FieldOptions) -> (Option<u64>, u64, StructLayout) {
        let field = FieldLayout::new(layout, options);
        let field_offset = self.next_field_offset(field.align(), field.custom_min_align.0);
        let align = self.align.max(field.align());

        // wgsl spec:
        //   roundUp(AlignOf(S), justPastLastMember)
        //   where justPastLastMember = OffsetOfMember(S,N) + SizeOfMember(S,N)
        let size = field
            .byte_size()
            .map(|field_size| round_up(align, field_offset + field_size));

        self.s.fields.push(FieldLayoutWithOffset {
            field,
            rel_byte_offset: field_offset,
        });

        (size, align, self.s)
    }

    /// field_align should already respect field_custom_min_align.
    /// field_custom_min_align is used to overwrite packing if self is packed.
    fn next_field_offset(&self, field_align: u64, field_custom_min_align: Option<u64>) -> u64 {
        match self.rules {
            TypeLayoutRules::Wgsl => match (self.packed, field_custom_min_align) {
                (true, None) => self.next_offset_min,
                (true, Some(custom_align)) => round_up(custom_align, self.next_offset_min),
                (false, _) => round_up(field_align, self.next_offset_min),
            },
        }
    }
}

/// a sized or unsized struct type with 0 or more fields
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructLayout {
    pub name: IgnoreInEqOrdHash<CanonName>,
    pub fields: Vec<FieldLayoutWithOffset>,
}

impl StructLayout {
    /// this exists, because if in the future a refactor happens that separates
    /// fields into sized and unsized fields, the intention of this function is
    /// clear
    fn all_fields(&self) -> &[FieldLayoutWithOffset] { &self.fields }

    /// returns a `(byte_size, byte_alignment, struct_layout)` tuple
    #[doc(hidden)]
    pub fn from_ir_struct(rules: TypeLayoutRules, s: &ir::Struct) -> (Option<u64>, u64, StructLayout) {
        // Could replace all of this with TypeLayout::from_struct and then
        // taking the StructLayout from TypeLayout::kind, but that would
        // need an unreachable!.
        let mut total_byte_size = None;
        let struct_layout = StructLayout {
            name: s.name().clone().into(),
            fields: {
                let mut offset = 0;
                let mut fields = Vec::new();
                for field in s.sized_fields() {
                    fields.push(FieldLayoutWithOffset {
                        field: FieldLayout {
                            name: field.name.clone(),
                            ty: TypeLayout::from_sized_ty(rules, &field.ty).into(),
                            custom_min_size: field.custom_min_size.into(),
                            custom_min_align: field.custom_min_align.map(u64::from).into(),
                        },
                        rel_byte_offset: match rules {
                            TypeLayoutRules::Wgsl => {
                                let rel_byte_offset = round_up(field.align(), offset);
                                offset = rel_byte_offset + field.byte_size();
                                rel_byte_offset
                            }
                        },
                    })
                }
                if let Some(unsized_array) = s.last_unsized_field() {
                    fields.push(FieldLayoutWithOffset {
                        field: FieldLayout {
                            name: unsized_array.name.clone(),
                            custom_min_align: unsized_array.custom_min_align.map(u64::from).into(),
                            custom_min_size: None.into(),
                            ty: TypeLayout::from_array(rules, &unsized_array.element_ty, None).into(),
                        },
                        rel_byte_offset: round_up(unsized_array.align(), offset),
                    })
                } else {
                    total_byte_size = Some(s.min_byte_size());
                }
                fields
            },
        };
        (total_byte_size, s.align(), struct_layout)
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldLayoutWithOffset {
    pub field: FieldLayout,
    pub rel_byte_offset: u64, // this being relative is used in TypeLayout::byte_size
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ElementLayout {
    pub byte_stride: u64,
    /// The elment layout must be marker::MaybeInvalid, because that's the most it can
    /// be while supporting all possible markers. TypeLayouts with stricter markers
    /// should have methods that make their element layout accessible with stricter marker layouts.
    pub ty: TypeLayout<marker::MaybeInvalid>,
}

/// the layout rules used when calculating the byte offsets and alignment of a type
#[derive(Debug, Clone, Copy)]
pub enum TypeLayoutRules {
    /// wgsl type layout rules, see https://www.w3.org/TR/WGSL/#memory-layouts
    Wgsl,
    // reprC,
    // Std140,
    // Std430,
    // Scalar,
}

#[allow(missing_docs)]
#[derive(Error, Debug, Clone)]
pub enum TypeLayoutError {
    #[error("An array cannot contain elements of an unsized type {elements}")]
    ArrayOfUnsizedElements { elements: TypeLayout },
    #[error("the type `{0}` has no defined {1:?} layout in shaders")]
    LayoutUndefined(Type, TypeLayoutRules),
    #[error("in `{parent_name}` at field `{field_name}`: {error}")]
    AtField {
        parent_name: CanonName,
        field_name: CanonName,
        error: Rc<TypeLayoutError>,
    },
    #[error("in element of array: {0}")]
    InArrayElement(Rc<TypeLayoutError>),
}

impl TypeLayout {
    pub fn struct_from_parts(
        struct_options: impl Into<StructOptions>,
        mut fields: impl ExactSizeIterator<Item = (FieldOptions, TypeLayout)>,
    ) -> Result<TypeLayout, StructLayoutError> {
        let struct_options = struct_options.into();
        let num_fields = fields.len();

        let make_sized = |layout: TypeLayout, field_index: usize| {
            layout
                .try_into_sized()
                .map_err(|layout| StructLayoutError::UnsizedFieldMustBeLast {
                    struct_name: struct_options.name.clone(),
                    unsized_field_index: field_index,
                    num_fields,
                })
        };

        let (options, layout) = fields
            .next()
            .ok_or_else(|| StructLayoutError::HasNoFields(struct_options.name.clone()))?;
        let layout = make_sized(layout, 0)?;
        let mut builder = StructLayoutBuilder::new_struct(struct_options.clone(), options, layout);

        if num_fields == 1 {
            return Ok(builder.finish().into());
        }
        for (i, (options, layout)) in fields.enumerate() {
            let field_index = i + 1;
            let is_last_field = field_index == num_fields - 1;
            if is_last_field {
                return Ok(builder.extend(options, layout).finish());
            } else {
                let layout = make_sized(layout, i)?;
                builder = builder.extend(options, layout);
            }
        }
        // TODO(chronicl) this is actually unreachable, but should replace this with different builder api anyway
        Ok(builder.finish().into())
    }


    pub fn try_into_sized(self) -> Result<TypeLayout<marker::Sized>, TypeLayout> {
        if self.byte_size().is_some() {
            Ok(unsafe_type_layout::cast(self))
        } else {
            Err(self)
        }
    }

    pub fn from_ty(rules: TypeLayoutRules, ty: &ir::Type) -> Result<Self, TypeLayoutError> {
        match ty {
            Type::Unit | Type::Ptr(_, _, _) | Type::Ref(_, _, _) => {
                Err(TypeLayoutError::LayoutUndefined(ty.clone(), rules))
            }
            Type::Store(ty) => Self::from_store_ty(rules, ty),
        }
    }

    pub fn from_array(rules: TypeLayoutRules, element: &ir::SizedType, len: Option<NonZeroU32>) -> Self {
        unsafe_type_layout::new_type_layout(
            len.map(|n| byte_size_of_array(element, n)),
            align_of_array(element),
            TypeLayoutSemantics::Array(
                Rc::new(ElementLayout {
                    byte_stride: match rules {
                        TypeLayoutRules::Wgsl => stride_of_array(element),
                    },
                    ty: TypeLayout::from_sized_ty(rules, element).into(),
                }),
                len.map(NonZeroU32::get),
            ),
        )
    }

    pub fn from_struct(rules: TypeLayoutRules, s: &ir::Struct) -> Self {
        let sized_fields = s.sized_fields();

        let array_options =
            |field: &ir::RuntimeSizedArrayField| FieldOptions::new(field.name.clone(), field.custom_min_align, None);

        if sized_fields.is_empty() {
            // ir::Struct always has at least one field and if there are no sized_fields (checked above),
            // then there must be an unsized field.
            let array = s.last_unsized_field().as_ref().unwrap();
            let array_layout = TypeLayout::from_array(rules, &array.element_ty, None);
            StructLayoutBuilder::new_struct(s.name().clone(), array_options(array), array_layout).finish()
        } else {
            // checked above that at least one sized field exists
            let first_field = &sized_fields[0];

            let options = |field: &ir::SizedField| {
                FieldOptions::new(field.name.clone(), field.custom_min_align, field.custom_min_size)
            };
            let layout =
                |field: &ir::SizedField| -> TypeLayout<marker::Sized> { TypeLayout::from_sized_ty(rules, field.ty()) };

            let mut builder = StructLayoutBuilder::<marker::Sized>::new_struct(
                s.name().clone(),
                options(first_field),
                layout(first_field),
            );

            for field in &sized_fields[1..] {
                builder = builder.extend(options(field), layout(field));
            }

            if let Some(array) = s.last_unsized_field() {
                let array_layout = TypeLayout::from_array(rules, &array.element_ty, None);
                let builder = builder.extend(array_options(array), array_layout);
                builder.finish()
            } else {
                builder.finish().into()
            }
        }
    }

    pub fn from_store_ty(rules: TypeLayoutRules, ty: &ir::StoreType) -> Result<Self, TypeLayoutError> {
        match ty {
            ir::StoreType::Sized(sized) => Ok(TypeLayout::from_sized_ty(rules, sized).into()),
            ir::StoreType::Handle(handle) => Err(TypeLayoutError::LayoutUndefined(ir::Type::Store(ty.clone()), rules)),
            ir::StoreType::RuntimeSizedArray(element) => Ok(TypeLayout::from_array(rules, element, None)),
            ir::StoreType::BufferBlock(s) => Ok(TypeLayout::from_struct(rules, s)),
        }
    }


    pub fn from_aligned_type(rules: TypeLayoutRules, ty: &AlignedType) -> Self {
        match ty {
            AlignedType::Sized(sized) => TypeLayout::from_sized_ty(rules, sized).into(),
            AlignedType::RuntimeSizedArray(element) => Self::from_array(rules, element, None),
        }
    }
}

impl TypeLayout<marker::Sized> {
    pub fn from_sized_ty(rules: TypeLayoutRules, ty: &ir::SizedType) -> Self {
        pub use TypeLayoutSemantics as Sem;
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

impl<T: TypeRestriction> TypeLayout<T> {
    pub fn byte_size(&self) -> Option<u64> { self.byte_size }

    pub fn align(&self) -> u64 { *self.byte_align }

    pub fn layout_eq<R: TypeRestriction>(&self, other: &TypeLayout<R>) -> bool {
        self.byte_size == other.byte_size && self.kind == other.kind
    }

    /// Although all TypeLayout<T> always implement Into<TypeLayout<marker::MaybeInvalid> this method
    /// is offered to avoid having to declare that as a bound when handling generic TypeLayout<T>.
    pub fn into_maybe_invalid(self) -> TypeLayout<marker::MaybeInvalid> { unsafe_type_layout::cast(self) }

    /// a short name for this `TypeLayout`, useful for printing inline
    pub fn short_name(&self) -> String {
        match &self.kind {
            TypeLayoutSemantics::Vector { .. } |
            TypeLayoutSemantics::PackedVector { .. } |
            TypeLayoutSemantics::Matrix { .. } => format!("{}", self),
            TypeLayoutSemantics::Array(element_layout, n) => match n {
                Some(n) => format!("array<{}, {n}>", element_layout.ty.short_name()),
                None => format!("array<{}, runtime-sized>", element_layout.ty.short_name()),
            },
            TypeLayoutSemantics::Structure(s) => s.name.to_string(),
        }
    }

    pub(crate) fn writeln<W: Write>(&self, indent: &str, colored: bool, f: &mut W) -> std::fmt::Result {
        self.write(indent, colored, f)?;
        writeln!(f)
    }

    //TODO(low prio) try to figure out a cleaner way of writing these.
    pub(crate) fn write<W: Write>(&self, indent: &str, colored: bool, f: &mut W) -> std::fmt::Result {
        let tab = "  ";
        let use_256_color_mode = false;
        let color = |f_: &mut W, hex| match colored {
            true => set_color(f_, Some(hex), use_256_color_mode),
            false => Ok(()),
        };
        let reset = |f_: &mut W| match colored {
            true => set_color(f_, None, use_256_color_mode),
            false => Ok(()),
        };

        use TypeLayoutSemantics as Sem;

        match &self.kind {
            Sem::Vector(l, t) => match l {
                Len::X1 => write!(f, "{t}")?,
                l => write!(f, "{t}x{}", u64::from(*l))?,
            },
            Sem::PackedVector(c) => write!(f, "{}", c)?,
            Sem::Matrix(c, r, t) => write!(f, "{}", ir::SizedType::Matrix(*c, *r, *t))?,
            Sem::Array(t, n) => {
                let stride = t.byte_stride;
                write!(f, "array<")?;
                t.ty.write(&(indent.to_string() + tab), colored, f)?;
                if let Some(n) = n {
                    write!(f, ", {n}")?;
                }
                write!(f, ">  stride={stride}")?;
            }
            Sem::Structure(s) => {
                writeln!(f, "struct {} {{", s.name)?;
                {
                    let indent = indent.to_string() + tab;
                    for field in &s.fields {
                        let offset = field.rel_byte_offset;
                        let field = &field.field;
                        write!(f, "{indent}{offset:3} {}: ", field.name)?;
                        field.ty.write(&(indent.to_string() + tab), colored, f)?;
                        if let Some(size) = field.ty.byte_size {
                            let size = size.max(field.custom_min_size.unwrap_or(0));
                            write!(f, " size={size}")?;
                        } else {
                            write!(f, " size=?")?;
                        }
                        writeln!(f, ",")?;
                    }
                }
                write!(f, "{indent}}}")?;
                write!(f, " align={}", self.byte_align)?;
                if let Some(size) = self.byte_size {
                    write!(f, " size={size}")?;
                } else {
                    write!(f, " size=?")?;
                }
            }
        };
        Ok(())
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldLayout {
    pub name: CanonName,
    pub custom_min_size: IgnoreInEqOrdHash<Option<u64>>, // whether size/align is custom doesn't matter for the layout equality.
    pub custom_min_align: IgnoreInEqOrdHash<Option<u64>>,
    /// The fields layout must be marker::MaybeInvalid, because that's the most it can
    /// be while supporting all possible markers. TypeLayouts with stricter markers
    /// should have methods that make their fields accessible with stricter marker layouts.
    /// For example TypeLayout<Vertex> should have a method for making it's fields accessible
    /// as TypeLayout<VertexAttribute> TODO(chronicl).
    pub ty: TypeLayout<marker::MaybeInvalid>,
}

impl FieldLayout {
    fn new(ty: TypeLayout, options: impl Into<FieldOptions>) -> Self {
        let options = options.into();
        Self {
            name: options.name,
            custom_min_size: options.custom_min_size.into(),
            custom_min_align: options.custom_min_align.into(),
            ty: ty.into(),
        }
    }
    fn byte_size(&self) -> Option<u64> {
        self.ty
            .byte_size()
            .map(|byte_size| Self::calculate_byte_size(byte_size, self.custom_min_size.0))
    }
    /// The alignment of the field with `custom_min_align` taken into account.
    fn align(&self) -> u64 { Self::calculate_align(self.ty.align(), self.custom_min_align.0) }
    fn calculate_byte_size(byte_size: u64, custom_min_size: Option<u64>) -> u64 {
        byte_size.max(custom_min_size.unwrap_or(0))
    }
    fn calculate_align(align: u64, custom_min_align: Option<u64>) -> u64 { align.max(custom_min_align.unwrap_or(1)) }
}

#[allow(missing_docs)]
#[derive(Error, Debug)]
pub enum StructLayoutError {
    #[error(
        "field #{unsized_field_index} in struct `{struct_name}` with {num_fields} is unsized. Only the last field may be unsized."
    )]
    UnsizedFieldMustBeLast {
        struct_name: CanonName,
        unsized_field_index: usize,
        num_fields: usize,
    },
    #[error("struct `{0}` must have at least one field to be a valid GPU struct.")]
    HasNoFields(CanonName),
}

impl StructLayout {
    /// returns a `(byte_size, byte_alignment, struct_layout)` tuple or an error
    ///
    /// this was created for the `#[derive(GpuLayout)]` macro to support the
    /// non-GpuType `PackedVec` for gpu_repr(packed) and non-packed.
    ///
    // TODO(low prio) find a way to merge all struct layout calculation functions in this codebase. This is very redundand.
    pub(crate) fn new(
        rules: TypeLayoutRules,
        packed: bool,
        name: CanonName,
        fields: impl ExactSizeIterator<Item = FieldLayout>,
    ) -> Result<(Option<u64>, u64, StructLayout), StructLayoutError> {
        let mut total_byte_size = None;
        let mut total_align = 1;
        let num_fields = fields.len();
        let struct_layout = StructLayout {
            name: name.clone().into(),
            fields: {
                let mut offset_so_far = 0;
                let mut fields_with_offset = Vec::new();
                for (i, field) in fields.enumerate() {
                    let is_last = i + 1 == num_fields;
                    fields_with_offset.push(FieldLayoutWithOffset {
                        field: field.clone(),
                        rel_byte_offset: match rules {
                            TypeLayoutRules::Wgsl => {
                                let field_offset = match (packed, *field.custom_min_align) {
                                    (true, None) => offset_so_far,
                                    (true, Some(custom_align)) => round_up(custom_align, offset_so_far),
                                    (false, _) => round_up(field.align(), offset_so_far),
                                };
                                match (field.byte_size(), is_last) {
                                    (Some(field_size), _) => {
                                        offset_so_far = field_offset + field_size;
                                        Ok(())
                                    }
                                    (None, true) => Ok(()),
                                    (None, false) => Err(StructLayoutError::UnsizedFieldMustBeLast {
                                        struct_name: name.clone(),
                                        unsized_field_index: i,
                                        num_fields,
                                    }),
                                }?;
                                field_offset
                            }
                        },
                    });
                    total_align = total_align.max(field.align());
                    if is_last {
                        // wgsl spec:
                        //   roundUp(AlignOf(S), justPastLastMember)
                        //   where justPastLastMember = OffsetOfMember(S,N) + SizeOfMember(S,N)

                        // if the last field size is None (= unsized), just_past_last is None (= unsized)
                        let just_past_last = field.byte_size().map(|_| offset_so_far);
                        total_byte_size = just_past_last.map(|just_past_last| round_up(total_align, just_past_last));
                    }
                }
                fields_with_offset
            },
        };
        Ok((total_byte_size, total_align, struct_layout))
    }
}

impl<T: TypeRestriction> Display for TypeLayout<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let colored = Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages).unwrap_or(false);
        self.write("", colored, f)
    }
}

impl<T: TypeRestriction> Debug for TypeLayout<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.write("", false, f) }
}
