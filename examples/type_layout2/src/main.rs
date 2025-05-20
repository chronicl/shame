#![allow(dead_code, unused)]
//! Demonstration of the TypeLayout and TypeLayout Builder API.

use shame::{
    any::{self, U32PowerOf2},
    cs::AsCpuShareableSized,
    f32x1, f32x2, f32x3, f32x4,
    type_layout2::*,
    Array, GpuLayout, GpuSized, VertexAttribute, VertexLayout,
};

fn main() {
    // We'll start by building a `TypeLayout<Vertex>`, which can be used for ... nothing really
    #[derive(GpuLayout)]
    struct Vertex {
        position: f32x3,
        normal: f32x3,
        uv: f32x1,
    }

    // TypeLayout::vertex_builder immediately takes the first field of the struct, because
    // structs need to have at least one field.
    let mut builder =
        TypeLayout::vertex_builder("Vertex", "position", f32x4::vertex_attrib_format(), cs::Repr::Storage)
            .extend("normal", f32x3::vertex_attrib_format())
            .extend("uv", f32x1::vertex_attrib_format())
            .finish();


    // Now we'll replicate the layout of this struct
    #[derive(GpuLayout)]
    struct A {
        a: f32x4,
        b: f32x3,
        c: Array<f32x1>,
    }

    // Be default structs are #[gpu_repr(Storage)], which means that it follows
    // the wgsl storage alignment rules (std430). To obtain a corresponding TypeLayout<Storage>
    // we first need to build a `CpuShareableType`, in our case an `UnsizedStruct`.
    let unsized_struct = cs::UnsizedStruct {
        name: "A".into(),
        sized_fields: vec![
            cs::SizedField::new("a", cs::Vector::new(cs::ScalarType::F32, cs::Len::X4)),
            cs::SizedField::new("b", f32x3::cpu_shareable_sized()),
        ],
        last_unsized: cs::RuntimeSizedArrayField::new("c", None, f32x1::cpu_shareable_sized()),
    };
    // And now we can get the `TypeLayout<Storage>`.
    let s_layout = TypeLayout::<constraint::Storage>::new_layout_for(unsized_struct.clone());
    // For now TypeLayout::<constraint::Uniform>::new_layout_for only accepts sized types,
    // however TypeLayout::<constraint::Uniform>::new_layout_for_unchecked allows to obtain the
    // the uniform layout of an unsized cpu-shareable. Using this with wgsl as your target language will cause an error.
    let u_layout = TypeLayout::<constraint::Uniform>::new_layout_for_unchecked(unsized_struct);
    // This struct has differently aligned fields with the different alignment rules. The array
    // has an alignment of 4 with storage alignment and an alignment of 16 with uniform alignment.
    assert_ne!(s_layout, u_layout);

    #[derive(GpuLayout)]
    struct B {
        a: f32x4,
        b: f32x3,
        c: f32x1,
    }

    // Sized structs require a builder to ensure it always contains at least one field.
    let mut sized_struct = cs::SizedStruct::new("B", "b", f32x4::cpu_shareable_sized())
        .extend("b", f32x3::cpu_shareable_sized())
        .extend("c", f32x1::cpu_shareable_sized());
    // Since this struct is sized we can use TypeLayout::<constraint::Uniform>::new_layout_for.
    let u_layout = TypeLayout::<constraint::Uniform>::new_layout_for(sized_struct.clone());
    let s_layout = TypeLayout::<constraint::Storage>::new_layout_for(sized_struct);
    // And this time they are equal, despite different alignment rules.
    assert_eq!(s_layout, u_layout);
    // We can also check whether constraint::Storage could also be used as constraint::Uniform
    // via `TryFrom::try_from`.
    // Which in this case will succeed, but if it doesn't we get a very nice error message about
    // why the layout is not compatible with the uniform alignment rules (WIP).
    let u_layout = TypeLayout::<constraint::Uniform>::try_from(&s_layout).unwrap();

    // Let's replicate a more complex example with implicit field size and align.
    #[derive(shame::GpuLayout)]
    struct C {
        a: f32x2,
        #[size(16)]
        b: f32x1,
        #[align(16)]
        c: f32x2,
    }

    let mut sized_struct = cs::SizedStruct::new("C", "a", f32x3::cpu_shareable_sized())
        .extend(FieldOptions::new("b", None, Some(16)), f32x3::cpu_shareable_sized())
        .extend(
            FieldOptions::new("c", Some(U32PowerOf2::_16), None),
            f32x1::cpu_shareable_sized(),
        );
    let layout = TypeLayout::<constraint::Storage>::new_layout_for(sized_struct);
    assert!(layout.byte_align.as_u32() == 16);

    // Let's end on a pretty error message
    let mut sized_struct = cs::SizedStruct::new("D", "a", f32x2::cpu_shareable_sized())
        // This has align of 4 for storage, and align of 16 for uniform
        .extend("b", Array::<f32x1, shame::Size<1>>::cpu_shareable_sized());
    let s_layout = TypeLayout::<constraint::Storage>::new_layout_for(sized_struct);
    let result = TypeLayout::<constraint::Uniform>::try_from(&s_layout);
    match result {
        Err(e) => println!("This error is a showcase:\n{}", e),
        Ok(u_layout) => println!("It unexpectedly worked, ohh no."),
    }
}
