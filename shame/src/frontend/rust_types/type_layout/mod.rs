//! Everything related to type layouts.

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
mod sized;
mod unconstraint;
mod valid;
mod vertex;

pub use eq::*;
pub use unconstraint::*;
pub use sized::*;
pub use valid::*;
pub use vertex::*;

/// The type contained in the bytes of a `TypeLayout`.
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
/// assert!(f32::cpu_layout() == vec<f32, x1>::gpu_layout().into_unconstraint());
/// // or, more explicitly
/// assert_eq!(
///     <f32 as CpuLayout>::cpu_layout(),
///     <vec<f32, x1> as GpuLayout>::gpu_layout(),
/// );
/// ```
///
#[derive(Clone, Eq, Hash)]
pub struct TypeLayout<T: TypeConstraint = constraint::Basic> {
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

impl<L: TypeConstraint, R: TypeConstraint> PartialEq<TypeLayout<R>> for TypeLayout<L> {
    fn eq(&self, other: &TypeLayout<R>) -> bool { self.byte_size() == other.byte_size() && self.kind == other.kind }
}

/// TypeLayout for cpu types. Is not necessarily a valid layout for a gpu type.
pub type TypeLayoutUnconstraint = TypeLayout<constraint::Unconstraint>;

use constraint::TypeConstraint;
/// Module for all restrictions on `TypeLayout<T: TypeRestriction>`.
pub mod constraint {
    use super::*;

    /// Type restriction of a `TypeLayout<T: TypeRestriction>`. This allows type layouts
    /// to have guarantees based on their type restriction. For example a
    /// `TypeLayout<restriction::Vertex> can be used in vertex buffers.
    ///
    /// The following constraints exist:
    ///
    /// - `MaybeInvalid   ` Any type layout
    /// - `Valid          ` Any valid gpu layout
    /// - `Sized          ` Any sized gpu layout
    /// - `Vertex         ` A gpu layout that can be used in vertex buffers.
    /// - `VertexAttribute` Layout for the fields of structs that can be used in vertex buffers.
    pub trait TypeConstraint: Clone + PartialEq + Eq {}

    macro_rules! type_restriction {
        ($($constraint:ident),*) => {
            $(
                /// A type restriction for `TypeLayout<T: TypeRestriction>`.
                /// See [`TypeRestriction`] documentation for
                #[derive(Clone, PartialEq, Eq, Hash)]
                pub struct $constraint;
                impl TypeConstraint for $constraint {}
            )*
        };
    }

    type_restriction!(
        Unconstraint,    // Any type layout, may be invalid
        Basic,           // GpuLayout, Any valid gpu type layout
        Sized,           // GpuSized
        Vertex,          // VertexLayout
        VertexAttribute  // VertexAttribute
    );

    macro_rules! impl_from_into {
        ($from:ident -> $($into:ident),*) => {
            $(
                impl From<TypeLayout<$from>> for TypeLayout<$into> {
                    fn from(layout: TypeLayout<$from>) -> Self { type_layout_internal::cast(layout) }
                }
            )*
        };
    }

    impl_from_into!(Basic           -> Unconstraint);
    impl_from_into!(Sized           -> Unconstraint, Basic);
    impl_from_into!(Vertex          -> Unconstraint, Basic, Sized);
    impl_from_into!(VertexAttribute -> Unconstraint, Basic, Sized, Vertex);
}

/// Builder for the `TypeLayout<T>` of a struct. The builder methods guarantee that the
/// constructed layout fullfills all requirements to be `TypeLayout<T: TypeConstraint>`.
// TODO(chronicl) add example here for TypeLayout<Vertex> once VertexBufferAny is merged.
pub struct StructLayoutBuilder<T: TypeConstraint = constraint::Basic> {
    struct_builder: StructLayoutBuilderErased,
    // This is used by TypeLayoutBuilder<marker::Valid>, which only has one public method `finish`,
    // which inserts this last, potentially unsized field and finishes the build.
    // Building a TypeLayout<marker::Valid> always starts with a StructLayoutBuilder<marker::Sized>
    // and converts automatically to a StructLayoutBuilder<marker::Valid> when extending
    // by a field that is TypeLayout<marker::Valid> (potentially unsized).
    last_maybe_unsized_field: Option<(TypeLayout, FieldOptions)>,
    _phantom: PhantomData<T>,
}

/// This module offers helper methods that do not adhere to the restriction of `TypeLayout<T: TypeRestriction>.
/// The caller must uphold these restrictions themselves.
/// The main purpose of this module is to avoid repetition.
pub(in super::super::rust_types) mod type_layout_internal {
    use std::{marker::PhantomData, rc::Rc};
    use crate::{GpuAligned, GpuSized};
    use super::*;

    pub fn new_builder<T: TypeConstraint>(options: impl Into<StructOptions>) -> StructLayoutBuilder<T> {
        StructLayoutBuilder {
            struct_builder: StructLayoutBuilderErased::new(options.into()),
            last_maybe_unsized_field: None,
            _phantom: PhantomData,
        }
    }

    pub fn new_type_layout<T: TypeConstraint>(
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

    pub fn new_type_layout_struct<T: TypeConstraint>(
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

    pub fn extend<R: TypeConstraint>(
        builder: &mut StructLayoutBuilder<R>,
        options: FieldOptions,
        layout: TypeLayout<constraint::Sized>,
    ) {
        builder.struct_builder.extend(options, layout);
    }

    pub fn finish<R: TypeConstraint>(builder: StructLayoutBuilder<R>) -> TypeLayout<R> {
        let (byte_size, byte_align, s) = builder.struct_builder.finish();
        new_type_layout_struct(Some(byte_size), byte_align, s)
    }

    pub fn finish_maybe_unsized<R: TypeConstraint>(
        builder: StructLayoutBuilder<R>,
        layout: TypeLayout,
        options: impl Into<FieldOptions>,
    ) -> TypeLayout<R> {
        let (byte_size, byte_align, s) = builder.struct_builder.finish_maybe_unsized(options.into(), layout);
        new_type_layout_struct(byte_size, byte_align, s)
    }

    pub fn cast<From: TypeConstraint, Into: TypeConstraint>(layout: TypeLayout<From>) -> TypeLayout<Into> {
        TypeLayout {
            byte_size: layout.byte_size,
            byte_align: layout.byte_align,
            kind: layout.kind,
            _phantom: PhantomData,
        }
    }
}

/// Options for creating a new struct `TypeLayout<T>`.
///
/// If you only want to customize the struct's name, you can convert most string types
/// to `StructOptions` using `Into::into`, but most methods take `impl Into<StructOptions>`,
/// meaning you can just pass the string type directly.
#[derive(Debug, Clone)]
pub struct StructOptions {
    /// Name of the struct
    pub name: CanonName,
    /// Whether the struct should be packed
    pub packed: bool,
    /// Which layout rules the struct follows
    pub rules: TypeLayoutRules,
}

impl StructOptions {
    /// Creates new StructOptions.
    ///
    /// If you only want to customize the struct's name, you can convert most string types
    /// to `StructOptions` using `Into::into`, but most methods take `impl Into<StructOptions>`,
    /// meaning you can just pass the string type directly.
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


/// Options for the field of a struct.
///
/// If you only want to customize the field's name, you can convert most string types
/// to `FieldOptions` using `Into::into`, but most methods take `impl Into<StructOptions>`,
/// meaning you can just pass the string type directly.
#[derive(Debug, Clone)]
pub struct FieldOptions {
    /// Name of the field
    pub name: CanonName,
    /// Custom minimum align of the field.
    /// If the struct is packed and `custom_min_align` is Some,
    /// makes the field non-packed and aligns it to `custom_min_align`.
    pub(crate) custom_min_align: Option<u64>,
    /// Custom mininum size of the field.
    pub custom_min_size: Option<u64>,
}

impl FieldOptions {
    /// Creates new `FieldOptions`.
    ///
    /// If you only want to customize the field's name, you can convert most string types
    /// to `FieldOptions` using `Into::into`, but most methods take `impl Into<StructOptions>`,
    /// meaning you can just pass the string type directly.
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

/// `LayoutCalculator` helps calculate the size, align and the field offsets of a gpu struct.
#[derive(Debug, Clone)]
pub struct LayoutCalculator {
    next_offset_min: u64,
    align: u64,
    packed: bool,
    rules: TypeLayoutRules,
}

impl LayoutCalculator {
    /// Creates a new `LayoutCalculator`, which calculates the size and
    /// the field offsets of a structs layout.
    pub fn new(packed: bool, rules: TypeLayoutRules) -> Self {
        Self {
            next_offset_min: 0,
            align: 1,
            packed,
            rules,
        }
    }

    /// Returns the field's offset in the struct.
    pub fn extend(
        &mut self,
        size: u64,
        align: u64,
        custom_min_size: Option<u64>,
        custom_min_align: Option<u64>,
    ) -> u64 {
        let size = FieldLayout::calculate_byte_size(size, custom_min_size);
        let align = FieldLayout::calculate_align(align, custom_min_align);

        let offset = self.next_field_offset(align, custom_min_align);
        self.next_offset_min = offset + size;
        self.align = self.align.max(align);

        offset
    }

    /// Returns (byte size, byte align, last field offset).
    ///
    /// `self` is consumed, so that no further fields may be extended, because
    /// only the last field may be unsized.
    pub fn extend_maybe_unsized(
        mut self,
        size: Option<u64>,
        align: u64,
        custom_min_size: Option<u64>,
        custom_min_align: Option<u64>,
    ) -> (Option<u64>, u64, u64) {
        if let Some(size) = size {
            let offset = self.extend(size, align, custom_min_size, custom_min_align);
            (Some(self.byte_size()), self.align(), offset)
        } else {
            let (offset, align) = self.extend_unsized(align, custom_min_align);
            (None, align, offset)
        }
    }


    /// Returns (byte align, last field offset).
    ///
    /// `self` is consumed, so that no further fields may be extended, because
    /// only the last field may be unsized.
    pub fn extend_unsized(mut self, align: u64, custom_min_align: Option<u64>) -> (u64, u64) {
        let align = FieldLayout::calculate_align(align, custom_min_align);

        let offset = self.next_field_offset(align, custom_min_align);
        self.align = self.align.max(align);

        (offset, self.align)
    }

    /// Returns the byte size of the struct.
    // wgsl spec:
    //   roundUp(AlignOf(S), justPastLastMember)
    //   where justPastLastMember = OffsetOfMember(S,N) + SizeOfMember(S,N)
    //
    // self.next_offset_min is justPastLastMember already.
    pub fn byte_size(&self) -> u64 { round_up(self.align, self.next_offset_min) }

    /// Returns the align of the struct.
    pub fn align(&self) -> u64 { self.align }

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

/// Builds a TypeLayout<marker::Valid> and is wrapped by all StructLayoutBuilder<T>,
/// which impose further restrictions on the building process, so that they may
/// produce more strict TypeLayouts, such as StructLayoutBuilder<Sized> -> TypeLayout<Sized>.
struct StructLayoutBuilderErased {
    calc: LayoutCalculator,
    s: StructLayout,
}

impl StructLayoutBuilderErased {
    fn new(options: StructOptions) -> Self {
        Self {
            calc: LayoutCalculator::new(options.packed, options.rules),
            s: StructLayout {
                name: options.name.into(),
                fields: Vec::new(),
            },
        }
    }

    fn extend(&mut self, options: FieldOptions, layout: TypeLayout<constraint::Sized>) {
        let field_offset = self.calc.extend(
            layout.byte_size_sized(),
            layout.align(),
            options.custom_min_size,
            options.custom_min_align,
        );

        self.s.fields.push(FieldLayoutWithOffset {
            field: FieldLayout::new(layout.into(), options),
            rel_byte_offset: field_offset,
        });
    }

    /// Returns (byte_size, byte_align, StructLayout).
    fn finish(self) -> (u64, u64, StructLayout) { (self.calc.byte_size(), self.calc.align(), self.s) }

    /// Returns (byte_size, byte_align, StructLayout), where byte_size is None
    /// if the struct is unsized, that is, when `last_field` is unsized.
    fn finish_maybe_unsized(mut self, options: FieldOptions, layout: TypeLayout) -> (Option<u64>, u64, StructLayout) {
        let (size, align, offset) = self.calc.extend_maybe_unsized(
            layout.byte_size(),
            layout.align(),
            options.custom_min_size,
            options.custom_min_align,
        );

        self.s.fields.push(FieldLayoutWithOffset {
            field: FieldLayout::new(layout, options),
            rel_byte_offset: offset,
        });

        (size, align, self.s)
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
    pub fn from_struct_ir(rules: TypeLayoutRules, s: &ir::Struct) -> (Option<u64>, u64, StructLayout) {
        let mut builder =
            StructLayoutBuilderErased::new(StructOptions::new(s.name().clone(), false, TypeLayoutRules::Wgsl));

        for field in s.sized_fields() {
            builder.extend(
                FieldOptions::new(field.name.clone(), field.custom_min_align, field.custom_min_size),
                TypeLayout::from_sized_ty(rules, &field.ty),
            );
        }

        if let Some(array) = s.last_unsized_field() {
            builder.finish_maybe_unsized(
                FieldOptions::new(array.name.clone(), array.custom_min_align, None),
                TypeLayout::from_array_ir(rules, &array.element_ty, None),
            )
        } else {
            let (size, align, s) = builder.finish();
            (Some(size), align, s)
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldLayoutWithOffset {
    pub field: FieldLayout,
    pub rel_byte_offset: u64, // this being relative is used in TypeLayout::byte_size
}

/// Describes the layout of the elements of an array.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ElementLayout {
    /// Stride of the elements
    pub byte_stride: u64,
    /// The elment layout must be marker::MaybeInvalid, because that's the most it can
    /// be while supporting all possible markers. TypeLayouts with stricter markers
    /// should have methods that make their element layout accessible with stricter marker layouts.
    pub ty: TypeLayout<constraint::Unconstraint>,
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

impl<T: TypeConstraint> TypeLayout<T> {
    /// Byte size of the represented type.
    pub fn byte_size(&self) -> Option<u64> { self.byte_size }

    /// Align of the represented type.
    pub fn align(&self) -> u64 { *self.byte_align }

    /// Although all TypeLayout<T> always implement Into<TypeLayout<marker::MaybeInvalid>, this method
    /// is offered to avoid having to declare that as a bound when handling generic TypeLayout<T>.
    pub fn into_unconstraint(self) -> TypeLayout<constraint::Unconstraint> { type_layout_internal::cast(self) }

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
    // whether size/align is custom doesn't matter for the layout equality.
    pub custom_min_size: IgnoreInEqOrdHash<Option<u64>>,
    pub custom_min_align: IgnoreInEqOrdHash<Option<u64>>,
    /// The fields layout must be constraint::Unconstraint,
    /// because that's the most it can be while supporting all possible constraints.
    pub ty: TypeLayout<constraint::Unconstraint>,
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

impl<T: TypeConstraint> Display for TypeLayout<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let colored = Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages).unwrap_or(false);
        self.write("", colored, f)
    }
}

impl<T: TypeConstraint> Debug for TypeLayout<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.write("", false, f) }
}
