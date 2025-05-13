#![allow(unused, clippy::no_effect)]
use shame::results::Len;
use shame::results::ScalarType;
use shame::results::VertexAttribFormat;
use shame::results::VertexBufferLookupIndex;
use shame::FromAnys;
use shame::VertexAttribute;
use shame::VertexLayout;
use shame as sm;
use shame::prelude::*;
use shame::aliases::*;
use sm::any;
use sm::any::Any;

fn main() { make_pipeline(false).unwrap(); }

fn make_pipeline(has_uv: bool) -> Result<sm::results::RenderPipeline, sm::EncodingErrors> {
    let mut encoder = sm::start_encoding(sm::Settings::default())?;
    let mut drawcall = encoder.new_render_pipeline(sm::Indexing::BufferU16);

    // There is multiple ways to obtain data from vertex buffers

    // Fully typed api
    #[derive(sm::GpuLayout)]
    struct Vertex {
        position: f32x3,
        normal: f32x3,
        uv: f32x2,
    }

    let buffer: sm::VertexBuffer<Vertex> = drawcall.vertices.buffers.next();
    let vertex: Vertex = buffer.at(drawcall.vertices.index);

    // Attribute iter api
    let buffer: sm::VertexBufferDynamic = drawcall.vertices.buffers.next_dynamic();
    let mut vertex: sm::VertexAttributeIter = buffer.at(drawcall.vertices.index).attribute_iter(false);
    let position: f32x3 = vertex.next();
    // specify locations of the attributes manually
    let normal: f32x3 = vertex.at(4);
    // can be used to have optional vertex attributes
    let uv: Option<f32x2> = has_uv.then(|| vertex.next());

    // Any api
    // Api for full control over vertex buffer creation.
    // You have to manually specify the stride of the vertex buffer elements.
    let stride = Vertex::gpu_layout_vertex().as_vertex_attributes().stride;
    let lookup = VertexBufferLookupIndex::VertexIndex;
    let locations_and_attributes = [
        (
            any::Location(6),
            any::VertexAttribute {
                offset: 0,
                format: VertexAttribFormat::Fine(Len::X3, ScalarType::F32),
            },
        ),
        (
            any::Location(7),
            any::VertexAttribute {
                offset: 16,
                format: f32x3::vertex_attrib_format(),
            },
        ),
        (
            any::Location(8),
            any::VertexAttribute {
                offset: 32,
                format: f32x2::vertex_attrib_format(),
            },
        ),
    ];

    // We need to obtain a key, which makes us the owner of the vertex buffer
    // at the slot that we pass here. This has to be done in order to avoid undetected
    // clashes with the other vertex buffer apis.
    let slot_key = Any::vertex_buffer_new(2);
    let anys: Vec<Any> = Any::vertex_buffer_extend(&slot_key, lookup, stride, locations_and_attributes);
    let mut anys = anys.into_iter();

    let position: f32x3 = FromAnys::from_anys([anys.next().unwrap()].into_iter());
    let normal: f32x3 = FromAnys::from_anys([anys.next().unwrap()].into_iter());
    let uv: f32x2 = FromAnys::from_anys([anys.next().unwrap()].into_iter());

    // needed to be able to finish encoding
    let primitive = drawcall
        .vertices
        .assemble(f32x4::zero(), sm::Draw::triangle_list(sm::Ccw));
    primitive.rasterize(sm::Accuracy::Relaxed);

    encoder.finish()
}
