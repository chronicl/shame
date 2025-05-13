#![allow(dead_code, unused)]
//! Demonstration of the TypeLayout and TypeLayout Builder API.

use shame::{
    any::{self, U32PowerOf2},
    f32x1, f32x3, f32x4,
    type_layout::*,
    Array, GpuLayout, GpuSized, VertexAttribute, VertexLayout,
};

fn main() {
    // We'll start by replicating this derived GpuLayout with the error free TypeLayout Builder API
    #[derive(GpuLayout)]
    struct A {
        a: f32x4,
        b: f32x3,
        c: Array<f32x1>,
    }

    // TypeLayouts take a generic parameter. Here we are using `GpuSized::gpu_layout_sized`
    // to obtain a TypeLayout<constraint::Sized>, which signifies that sm::f32x4 is sized.
    let f32x4_layout: TypeLayout<constraint::Sized> = f32x4::gpu_layout_sized();
    // TypeLayout::struct_builder immediately takes the first field of the struct, because
    // a gpu struct is required to have at least one field.
    let mut builder = TypeLayout::struct_builder("A", "a", f32x4_layout);

    // We can extend the TypeLayout with more fields
    builder = builder.extend("b", f32x3::gpu_layout_sized());

    // `GpuLayout::gpu_layout` only returns a TypeLayout<constraint:::Basic>,
    // which doesn't guarantee it's corresponding type is sized.
    let unsized_array_layout: TypeLayout<constraint::Basic> = Array::<f32x1>::gpu_layout();

    // Only the last field of a gpu layout may be sized.
    // Our builder respects this by transitioning to a `StructLayoutBuilder<constraint::Basic>`
    // when we extend by a `TypeLayout<constraint::Basic>`, which only has one method: `finish`.
    let builder = builder.extend("c", unsized_array_layout);
    // builder.extend("d", f32x3::gpu_layout()) is a compilation error here

    let a: TypeLayout<constraint::Basic> = builder.finish();

    assert_eq!(A::gpu_layout(), a);

    // Or in short
    let _ = TypeLayout::struct_builder("A", "a", f32x4::gpu_layout_sized())
        .extend("b", f32x3::gpu_layout_sized())
        .extend("c", Array::<f32x1>::gpu_layout())
        .finish();

    // Let's replicate a more complex example
    #[derive(shame::GpuLayout)]
    #[gpu_repr(packed)]
    struct B {
        a: f32x3,
        b: f32x3,
        #[align(16)]
        c: f32x3,
    }

    println!("{}", B::gpu_layout());

    // The arguments to struct_builder and extend are actually impl Into<StructOptions>
    // and impl Into<FieldOptions>, so that we can do the following instead
    let packed = true;
    let struct_options = StructOptions::new("B", packed, TypeLayoutRules::Wgsl);
    let c_options = FieldOptions::new("c", Some(U32PowerOf2::_16), None);
    let b = TypeLayout::struct_builder(struct_options, "a", f32x3::gpu_layout_sized())
        .extend("b", f32x3::gpu_layout_sized())
        .extend(c_options, f32x3::gpu_layout_sized())
        .finish();

    assert_eq!(B::gpu_layout_sized(), b);

    // Now we'll look at some other constraints.

    // constraint::Vertex
    // A vertex buffer can only be filled with very specific types. In shame
    // those types implement the VertexLayout trait, which is automatically derived
    // for a struct when deriving GpuLayout and all requirements are fullfilled.
    // Superficially, the requirements are that all fields must implement VertexAttribute,
    // another shame trait. TypeLayoutBuilder<constraint::Vertex> let's you create a TypeLayout<constraint::Vertex>
    // in a type safe way, by only taking fields that are TypeLayout<constraint::VertexAttribute>.
    #[derive(shame::GpuLayout)]
    struct C {
        a: f32x3,
        b: f32x1,
    }

    let c = TypeLayout::struct_builder_vertex("C", "a", f32x3::gpu_layout_vertex_attribute())
        .extend("b", f32x1::gpu_layout_vertex_attribute())
        .finish();

    assert_eq!(&C::gpu_layout_vertex(), &c);

    // And we can use special methods that are only available on TypeLayout<constraint::Vertex>, like
    let vertex_buffer_layout = c.as_vertex_buffer_layout();
    // which can be used to obtain a VertexBufferAny (not on this branch) TODO(chronicl)
    // ...

    // Same can be done for the type layouts required for storage and uniform buffers (WIP)

    // In case you don't know your field layouts ahead of runtime,
    // TypeLayout::struct_from_parts let's you fallibly construct a TypeLayout<constraint::Basic>
    // You can then try to convert this layout to any other layout using rust's `TryInto::try_into`.
    let c = TypeLayout::struct_from_parts(
        "C",
        [("a".into(), f32x3::gpu_layout()), ("b".into(), f32x1::gpu_layout())].into_iter(),
    )
    .unwrap();
    let c_sized: TypeLayout<constraint::Sized> = c.clone().try_into().unwrap();
    let c_vertex: TypeLayout<constraint::Vertex> = c.try_into().unwrap();


    // Non-struct types can be constructed from their intermediate representations,
    // which can be found in shame::any.
    let vec_layout = TypeLayout::from_sized_ty(
        TypeLayoutRules::Wgsl,
        &any::SizedType::Vector(any::Len::X3, any::ScalarType::F32),
    );
    assert_eq!(f32x3::gpu_layout_sized(), vec_layout);
}
