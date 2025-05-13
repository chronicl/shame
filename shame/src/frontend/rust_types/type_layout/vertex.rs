use crate::{
    any::{Attrib, Location, VertexAttribFormat, VertexBufferLookupIndex},
    common::iterator_ext::try_collect,
    frontend::{
        any::render_io::{VertexAttribute, VertexAttributes},
        rust_types::{
            len::LenEven,
            packed_vec::{self, PackedScalarType},
        },
    },
    ir::{self, ir_type::stride_of_array_from_element_align_size, PackedVector, ScalarType},
    packed::PackedVec,
    NoBools,
};

use super::*;

impl TypeLayout<constraint::Vertex> {
    /// Create a new struct layout builder. The produced struct will be useable
    /// in vertex buffers.
    pub fn struct_builder_vertex(
        struct_options: impl Into<StructOptions>,
        field_options: impl Into<FieldOptions>,
        layout: TypeLayout<constraint::VertexAttribute>,
    ) -> StructLayoutBuilder<constraint::Vertex> {
        StructLayoutBuilder::new_vertex(struct_options, field_options, layout)
    }

    /// Locations are assigned incrementally starting with `first_location`.
    pub fn as_vertex_attributes(&self) -> VertexAttributes {
        use TypeLayoutSemantics as TLS;

        let stride = stride_of_array_from_element_align_size(self.align(), self.byte_size_sized());

        let attribs: Box<[VertexAttribute]> = match &self.kind {
            // TypeLayout<Vertex> is either a type implementing VertexAttribute or a struct of
            // VertexAttributes. A VertexAttribute is either a Vector with non-bool scalar type
            // or a packed vector.
            TLS::Matrix(..) | TLS::Array(..) | TLS::Vector(_, ir::ScalarType::Bool) => unreachable!(),
            TLS::Vector(len, non_bool) => [VertexAttribute {
                offset: 0,
                format: VertexAttribFormat::Fine(*len, *non_bool),
            }]
            .into(),
            TLS::PackedVector(packed_vector) => [VertexAttribute {
                offset: 0,
                format: VertexAttribFormat::Coarse(*packed_vector),
            }]
            .into(),
            TLS::Structure(rc) => rc
                .fields
                .iter()
                .map(|f| {
                    VertexAttribute {
                        offset: f.rel_byte_offset,
                        format: match f.field.ty.kind {
                            TLS::Matrix(..) |
                            TLS::Array(..) |
                            TLS::Structure(..) |
                            TLS::Vector(_, ir::ScalarType::Bool) => unreachable!(), // see above
                            TLS::Vector(len, non_bool) => VertexAttribFormat::Fine(len, non_bool),
                            TLS::PackedVector(packed_vector) => VertexAttribFormat::Coarse(packed_vector),
                        },
                    }
                })
                .collect(),
        };

        VertexAttributes { stride, attribs }
    }

    /// Returns the amount of vertex attributes this type layout contains.
    pub fn vertex_attribute_count(&self) -> usize {
        match &self.kind {
            TypeLayoutSemantics::Structure(s) => s.fields.len(),
            _ => 1,
        }
    }
}

impl TypeLayout<constraint::VertexAttribute> {
    /// Create a new `TypeLayout<constraint::VertexAttribute>` from a vector.
    pub fn from_vec<T: crate::ScalarType, L: crate::Len>(rules: TypeLayoutRules) -> Self
    where
        crate::vec<T, L>: NoBools,
    {
        // any non-bool vec is a VertexAttribut
        type_layout_internal::cast(TypeLayout::from_sized_ty(
            rules,
            &ir::SizedType::Vector(L::LEN, T::SCALAR_TYPE),
        ))
    }

    /// Create a new `TypeLayout<constraint::VertexAttribute>` from a packed vector.
    pub fn from_packed_vector_vertex_attr<T: PackedScalarType, L: LenEven>(rules: TypeLayoutRules) -> Self
    where
        PackedVec<T, L>: NoBools,
    {
        let packed_vec = packed_vec::get_type_description::<L, T>();
        type_layout_internal::new_type_layout(
            Some(u8::from(packed_vec.byte_size()) as u64),
            packed_vec.align(),
            TypeLayoutSemantics::PackedVector(packed_vec),
        )
    }
}

// Does not have a finish_maybe_unsized method,
// because it's fields must be at most 16 byte so it can't be unsized.
impl StructLayoutBuilder<constraint::Vertex> {
    /// Construct a new `TypeLayoutBuilder<Vertex>` and immediately add it's first field.
    /// Having at least one field is a requirement of `TypeLayout<Vertex>`.
    pub fn new_vertex(
        struct_options: impl Into<StructOptions>,
        field_options: impl Into<FieldOptions>,
        layout: TypeLayout<constraint::VertexAttribute>,
    ) -> Self {
        let mut this = type_layout_internal::new_builder::<constraint::Vertex>(struct_options);
        this.extend(field_options, layout)
    }

    /// Extends the struct layout builder with an additional field.
    /// Takes the field options and layout for the new field and returns
    /// the updated builder.
    pub fn extend(mut self, options: impl Into<FieldOptions>, layout: TypeLayout<constraint::VertexAttribute>) -> Self {
        type_layout_internal::extend(&mut self, options.into(), layout.into());
        self
    }

    /// Finalizes the struct layout builder and returns the complete
    /// `TypeLayout<constraint::Vertex>` for the struct, which can be used in vertex buffers.
    pub fn finish(self) -> TypeLayout<constraint::Vertex> { type_layout_internal::finish(self) }
}

impl TryFrom<TypeLayout<constraint::Basic>> for TypeLayout<constraint::Vertex> {
    type Error = ();

    fn try_from(layout: TypeLayout<constraint::Basic>) -> Result<Self, Self::Error> {
        use TypeLayoutSemantics as TLS;

        let is_compatible = match &layout.kind {
            // TypeLayout<Vertex> is either a type implementing VertexAttribute or a struct of
            // VertexAttributes. A VertexAttribute is either a Vector with non-bool scalar type
            // or a packed vector.
            TLS::Matrix(..) | TLS::Array(..) | TLS::Vector(_, ir::ScalarType::Bool) => false,
            TLS::Vector(_, _) | TLS::PackedVector(_) => true,
            TLS::Structure(rc) => rc.fields.iter().all(|f| match f.field.ty.kind {
                TLS::Vector(_, ir::ScalarType::Bool) | TLS::Matrix(..) | TLS::Array(..) | TLS::Structure(..) => false,
                TLS::Vector(_, _) | TLS::PackedVector(_) => true,
            }),
        };

        if layout.byte_size().is_some() && is_compatible {
            Ok(type_layout_internal::cast(layout))
        } else {
            Err(())
        }
    }
}
