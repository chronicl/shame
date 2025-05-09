use crate::{
    any::{Attrib, Location, VertexAttribFormat, VertexBufferLayout, VertexBufferLookupIndex},
    common::iterator_ext::try_collect,
    ir::{self, ir_type::stride_of_array_from_element_align_size},
};

use super::*;

impl TypeLayout<marker::Vertex> {
    pub fn struct_builder_vertex(
        struct_options: impl Into<StructOptions>,
        field_options: impl Into<FieldOptions>,
        layout: TypeLayout<marker::VertexAttribute>,
    ) -> StructLayoutBuilder<marker::Vertex> {
        StructLayoutBuilder::new_vertex(struct_options, field_options, layout)
    }

    // This is just placeholder code for now, for demonstration purposes.
    pub fn as_vertex_buffer_layout(&self) -> VertexBufferLayout {
        let stride = {
            let size = self.byte_size_sized();
            stride_of_array_from_element_align_size(self.align(), size)
        };
        use TypeLayoutSemantics as TLS;

        let mut location_counter = 0;
        let mut next_location = || {
            location_counter += 1;
            location_counter - 1
        };

        let attribs: Box<[Attrib]> = match &self.kind {
            // TypeLayout<Vertex> is either a type implementing VertexAttribute or a struct of
            // VertexAttributes. A VertexAttribute is either a Vector with non-bool scalar type
            // or a packed vector.
            TLS::Matrix(..) | TLS::Array(..) | TLS::Vector(_, ir::ScalarType::Bool) => unreachable!(),
            TLS::Vector(len, non_bool) => [Attrib {
                offset: 0,
                location: Location(next_location()),
                format: VertexAttribFormat::Fine(*len, *non_bool),
            }]
            .into(),
            TLS::PackedVector(packed_vector) => [Attrib {
                offset: 0,
                location: Location(next_location()),
                format: VertexAttribFormat::Coarse(*packed_vector),
            }]
            .into(),
            TLS::Structure(rc) => rc
                .fields
                .iter()
                .map(|f| {
                    Attrib {
                        offset: f.rel_byte_offset,
                        location: Location(next_location()),
                        format: match f.field.ty.kind {
                            TLS::Vector(_, ir::ScalarType::Bool) => unreachable!(), // see above
                            TLS::Vector(len, non_bool) => VertexAttribFormat::Fine(len, non_bool),
                            TLS::PackedVector(packed_vector) => VertexAttribFormat::Coarse(packed_vector),
                            TLS::Matrix(..) | TLS::Array(..) | TLS::Structure(..) => unreachable!(), // see above
                        },
                    }
                })
                .collect(),
        };

        VertexBufferLayout {
            // This is not something we should determine here, just placeholder code for different
            // VertexBufferLayout struct without this field. TODO(chronicl)
            lookup: VertexBufferLookupIndex::VertexIndex,
            stride,
            attribs,
        }
    }
}

// Does not have a finish_maybe_unsized method,
// because it's fields must be at most 16 byte so it can't be unsized.
impl StructLayoutBuilder<marker::Vertex> {
    /// Construct a new `TypeLayoutBuilder<Vertex>` and immediately add it's first field.
    /// Having at least one field is a requirement of `TypeLayout<Vertex>`.
    pub fn new_vertex(
        struct_options: impl Into<StructOptions>,
        field_options: impl Into<FieldOptions>,
        layout: TypeLayout<marker::VertexAttribute>,
    ) -> Self {
        let mut this = unsafe_type_layout::new_builder::<marker::Vertex>(struct_options);
        this.extend(field_options, layout)
    }

    pub fn extend(mut self, options: impl Into<FieldOptions>, layout: TypeLayout<marker::VertexAttribute>) -> Self {
        unsafe_type_layout::extend(&mut self, options.into(), layout.into());
        self
    }

    pub fn finish(self) -> TypeLayout<marker::Vertex> { unsafe_type_layout::finish(self) }
}

impl TryFrom<TypeLayout<marker::Valid>> for TypeLayout<marker::Vertex> {
    type Error = ();

    fn try_from(layout: TypeLayout<marker::Valid>) -> Result<Self, Self::Error> {
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
            Ok(unsafe_type_layout::cast(layout))
        } else {
            Err(())
        }
    }
}
